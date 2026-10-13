"""Immutable, VAE-free evidence bundles for the bounded paired-prefix experiment.

This schema is deliberately separate from DiagnosticSave: a nonzero-sigma prefix
is never a completed image or a resumable LATENT output.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

import torch
from safetensors.torch import load_file, save_file

from .experimental_artifacts import validate_experimental_prefix
from .masks import _validate_partition
from .sampling import tensor_hash

SCHEMA = 1
ROLES = ('target', 'protected', 'background')
BRANCHES = ('base', 'slider')
OBSERVER_BLOCKS = (8, 10, 18, 23, 26)
TAPS = {f'step_{s}.block_{b}' for s in (1, 2, 3) for b in OBSERVER_BLOCKS}
TRACE_KEYS = ('trace_inputs', 'trace_sigmas', 'trace_base_predictions', 'trace_slider_predictions', 'trace_predictions')
VALIDATION_KEYS = {'initial_noise', 'initial_latent', 'full_sigmas', 'used_sigmas', *TRACE_KEYS}
IDENTITY_KEYS = ('seed', 'steps', 'cfg', 'strength', 'sampler', 'scheduler', 'mix_scope', 'mask_mode',
                 'diagnostic_level', 'grid', 'token_positions', 'conditioning_sha256', 'lora_sha256',
                 'implementation_source_sha256', 'environment')
HASH_KEYS = ('conditioning_sha256', 'lora_sha256', 'implementation_source_sha256',
             'initial_noise_sha256', 'initial_latent_sha256', 'full_sigmas_sha256')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=bad_constant)


def _finite(value, name):
    if not isinstance(value, torch.Tensor) or not value.numel() or not torch.isfinite(value).all():
        raise ValueError('Expected nonempty finite tensor: ' + name)


def _same(a, b, name):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
        raise ValueError('Exact tensor mismatch: ' + name)


def _validate(probe, bank):
    """Validate completeness before publication and again on every load."""
    if not isinstance(probe, dict) or not isinstance(bank, dict):
        raise ValueError('Expected pilot probe and mask bank')
    try:
        report = probe['report']
        mode = report['pilot_mode']
        n = {'prefix': 3, 'full_reference': 8}.get(mode)
        if (report['pilot_schema_version'] != SCHEMA or n is None or report['steps'] != 8
                or report['executed_steps'] != n or report['branch_nfe'] != {'base': n, 'slider': n}
                or report['total_model_nfe'] != 2*n):
            raise ValueError('Wrong pilot schema, mode or NFE counts')
        from .paired_prefix import validate_pilot_options
        validate_pilot_options(steps=report['steps'], cfg=report['cfg'], strength=report['strength'],
            mix_scope=report['mix_scope'], mask_mode=report['mask_mode'], diagnostic_level=report['diagnostic_level'], pilot_mode=mode)
        if type(report['seed']) is not int or not 0 <= report['seed'] <= 0xffffffffffffffff:
            raise ValueError('Invalid bounded pilot seed')
        if any(key not in report for key in IDENTITY_KEYS):
            raise ValueError('Missing pilot identity')
        for key in HASH_KEYS:
            if not isinstance(report.get(key), str) or not re.fullmatch('[0-9a-f]{64}', report[key]):
                raise ValueError('Missing or invalid provenance hash: ' + key)
        if not isinstance(report['environment'], dict) or not report['environment']:
            raise ValueError('Missing runtime environment identity')
        if (report['cfg'] != 1 or report['sampler'] != 'euler' or report['scheduler'] != 'simple'
                or report['mix_scope'] != 'target_mask' or report['mask_mode'] != 'manual'
                or report['diagnostic_level'] != 'audit'):
            raise ValueError('Unsupported bounded pilot controls')
        if (report.get('prediction_selection', {}).get('dilate_radius') != 0
                or report.get('mask_generation', {}).get('fill_holes_max_area') != 0
                or report.get('mask_generation', {}).get('mask_dilate_radius') != 0):
            raise ValueError('Pilot requires zero postprocessing radii')
        t = probe['validation_tensors']
        if set(t) != VALIDATION_KEYS:
            raise ValueError('Missing or unexpected validation tensors; no partial/final latent is permitted')
        for key, value in t.items():
            _finite(value, key)
        for key in ('initial_noise', 'initial_latent', 'full_sigmas'):
            if tensor_hash(t[key]) != report[key+'_sha256']:
                raise ValueError('Provenance tensor hash mismatch: ' + key)
        full, used = t['full_sigmas'], t['used_sigmas']
        if (full.ndim != 1 or len(full) != 9 or full[-1] != 0 or not (full[:-1] > 0).all()
                or not (full[:-1] > full[1:]).all()):
            raise ValueError('Expected full nine-value decreasing Euler schedule ending at zero')
        _same(used, full[:n+1], 'used sigma prefix')
        _same(t['trace_sigmas'], full[:n], 'trace sigmas')
        if mode == 'prefix' and used[-1] <= 0:
            raise ValueError('Prefix must stop at a nonzero sigma')
        latent_shape = t['initial_latent'].shape
        if t['initial_noise'].shape != latent_shape:
            raise ValueError('Noise/latent shape mismatch')
        for key in TRACE_KEYS:
            expected = (n,) if key == 'trace_sigmas' else (n, *latent_shape)
            if tuple(t[key].shape) != expected:
                raise ValueError('Wrong trace shape or count: ' + key)
        predictions = probe['predictions']
        if set(predictions) != {'base', 'slider', 'mix'}:
            raise ValueError('Missing paired predictions')
        for branch, name in (('base', 'trace_base_predictions'), ('slider', 'trace_slider_predictions'), ('mix', 'trace_predictions')):
            _same(predictions[branch], t[name], 'prediction/' + branch)
        masks = _validate_partition(bank)
        if list(bank['grid']) != report['grid']:
            raise ValueError('Mask/report grid mismatch')
        for key in ('selection_target_mask', 'effective_image_mask'):
            _same(bank[key], masks['target'], key)
        _same(bank['selection_added_mask'], torch.zeros_like(masks['target']), 'selection_added_mask')
        branches = probe['branches']
        if set(branches) != set(BRANCHES) or set(branches['base']) != set(branches['slider']):
            raise ValueError('Missing or inconsistent paired branches')
        if (set(branches['base']) != TAPS or report.get('observer_blocks') != list(OBSERVER_BLOCKS)
                or report.get('observer_steps') != [1, 2, 3]):
            raise ValueError('Missing or changed original observer instrumentation taps')
        for branch, taps in branches.items():
            for key, tap in taps.items():
                match = re.fullmatch(r'step_([123])\.block_(\d+)', key)
                if not match:
                    raise ValueError('Unsupported tap name')
                step, block = map(int, match.groups())
                algorithm = tap['algorithm']
                config, positions = algorithm['config'], algorithm['positions']
                if (config['collect_step'] != step or config['collect_block'] != block
                        or list(algorithm['grid']) != report['grid'] or set(positions) != set(ROLES)
                        or any(not positions[r] for r in ROLES)):
                    raise ValueError('Invalid tap algorithm identity')
                evidence = tap['evidence_maps']
                required = {f'{key}.{r}.{metric}' for r in ROLES for metric in
                            ('head_scores', 'token_scores', 'head_allowed_fraction', 'token_allowed_fraction')}
                if step == 3 and block in (18, 23):
                    required.update('capture.'+name for name in ('attention_output', 'img_q', 'img_k'))
                if not isinstance(evidence, dict) or not required <= set(evidence):
                    raise ValueError('Missing raw tap evidence')
                for name, value in evidence.items():
                    _finite(value, branch+'/'+key+'/'+name)
                if not isinstance(tap['banks'], dict):
                    raise ValueError('Missing tap banks')
        frames = probe['frames']
        expected = {(b, s) for b in BRANCHES for s in (1, 2, 3)}
        if len(frames) != 6 or {(f['branch'], f['step']) for f in frames} != expected:
            raise ValueError('Missing actual-input observation frames')
        for frame in frames:
            i = frame['step']-1
            if frame['input_sha256'] != tensor_hash(t['trace_inputs'][i]) or frame['sigma'] != float(full[i]):
                raise ValueError('Observation frame differs from actual input/sigma')
            rows = frame['taps']
            blocks = {int(k.split('block_')[1]) for k in branches[frame['branch']] if k.startswith(f'step_{i+1}.')}
            if len(rows) != len(blocks) or {row['block'] for row in rows} != blocks:
                raise ValueError('Incomplete per-frame tap audit rows')
            observation_fields = {'img_q', 'img_k', 'txt_k', 'attention_input', 'attention_output'}
            for row in rows:
                before, after = row['tensor_sha256_before'], row['tensor_sha256_after']
                if (row['step'] != i+1 or row['sigma'] != float(full[i]) or row['tensor_audit_enabled'] is not True
                        or row['image_tokens'] != report['grid'][0]*report['grid'][1]
                        or type(row['text_tokens']) is not int or row['text_tokens'] < 1
                        or set(before) != observation_fields or before != after
                        or row['block_input_before_sha256'] != row['block_input_after_sha256']):
                    raise ValueError('Mutated or incomplete owned observation tap audit')
                digests = [*before.values(), row['block_input_before_sha256']]
                if any(not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in digests):
                    raise ValueError('Missing observation integrity hashes')
        bootstrap = {'grid': report['grid'], 'masks': probe['bootstrap_masks']}
        _validate_partition(bootstrap)
        for role in ROLES:
            if report['reference_partition_sha256'][role] != tensor_hash(bootstrap['masks'][role]):
                raise ValueError('Bootstrap partition hash mismatch')
        from .paired_prefix import build_r0_masks
        rebuilt, evidence = build_r0_masks(probe, bootstrap)
        retained = probe['r0_tensors']
        if set(evidence) != set(retained):
            raise ValueError('Missing or unexpected R0 evidence tensors')
        for key, value in evidence.items():
            _same(value, retained[key], 'rebuilt R0/' + key)
        for role in ROLES:
            _same(rebuilt['masks'][role], masks[role], 'rebuilt R0 mask/' + role)
            if report['r0_partition_sha256'][role] != tensor_hash(masks[role]):
                raise ValueError('R0 partition hash mismatch')
        for key in ('selection_target_mask', 'selection_added_mask', 'effective_image_mask'):
            _same(rebuilt[key], bank[key], 'rebuilt R0 selection/' + key)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('Malformed or incomplete prefix pilot evidence') from error


def _flatten(value, tensors, path=''):
    if isinstance(value, torch.Tensor):
        _finite(value, path)
        if any(word in path for word in ('final_latent', 'partial_latent')):
            raise ValueError('No completed or partial output latent may be saved')
        tensors[path] = value.detach().cpu().contiguous().clone()
        return {'__tensor__': path}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or '/' in key or key == '__tensor__':
                raise ValueError('Unsafe artifact field name')
            result[key] = _flatten(item, tensors, path+'/'+key if path else key)
        return result
    if isinstance(value, (list, tuple)):
        return [_flatten(item, tensors, path+'/'+str(i)) for i, item in enumerate(value)]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError('Unsupported artifact metadata type at ' + path)


def _inflate(value, tensors, used):
    if isinstance(value, dict):
        if set(value) == {'__tensor__'}:
            key = value['__tensor__']
            if key not in tensors or key in used:
                raise ValueError('Invalid or duplicate tensor reference')
            used.add(key)
            return tensors[key]
        return {key: _inflate(item, tensors, used) for key, item in value.items()}
    if isinstance(value, list):
        return [_inflate(item, tensors, used) for item in value]
    return value


def save_prefix_artifacts(directory, filename_prefix, probe, mask_bank):
    """Exclusively publish one checked bundle; never overwrite an earlier run."""
    validate_experimental_prefix(filename_prefix)
    _validate(probe, mask_bank)
    selected = {key: probe[key] for key in ('report', 'frames', 'branches', 'predictions', 'validation_tensors', 'bootstrap_masks', 'r0_tensors')}
    selected['mask_bank'] = mask_bank
    tensors = {}
    tree = _flatten(selected, tensors)
    _json(tree)  # Fail on NaN metadata before touching the filesystem.
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError('Output directory must already exist')
    artifact_id = uuid.uuid4().hex
    bundle = root / f'{filename_prefix}_{artifact_id}'
    bundle.mkdir(mode=0o700, exist_ok=False)
    try:
        path = bundle/'evidence.safetensors'
        save_file(tensors, str(path), metadata={'pilot_schema_version': str(SCHEMA), 'artifact_id': artifact_id})
        manifest = {'pilot_schema_version': SCHEMA, 'complete': True, 'artifact_id': artifact_id,
            'tensors': path.name, 'file_sha256': {path.name: _sha(path)}, 'tree': tree,
            'tensor_manifest': {key: {'sha256': tensor_hash(value), 'shape': list(value.shape), 'dtype': str(value.dtype)}
                                for key, value in tensors.items()},
            'publication': 'exclusive bundle, rollback on handled failure; not crash-atomic'}
        with (bundle/'manifest.json').open('x', encoding='utf-8') as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
            handle.write('\n')
        with (bundle/'manifest.sha256').open('x', encoding='ascii') as handle:
            handle.write(_sha(bundle/'manifest.json')+'\n')
    except BaseException:
        shutil.rmtree(bundle)
        raise
    return {'manifest': f'{bundle.name}/manifest.json', 'bundle': bundle.name, 'artifact_id': artifact_id,
            'manifest_sha256': _sha(bundle/'manifest.json')}


def load_prefix_artifacts(manifest_path):
    """Check every file/tensor hash, shape, dtype, and required evidence field."""
    path = Path(manifest_path).resolve()
    try:
        if path.name != 'manifest.json' or (path.parent/'manifest.sha256').read_text(encoding='ascii').strip() != _sha(path):
            raise ValueError('Changed manifest hash')
        manifest = _read_json(path)
        if manifest.get('pilot_schema_version') != SCHEMA or manifest.get('complete') is not True:
            raise ValueError('Incomplete or unsupported prefix artifact')
        name = manifest['tensors']
        if name != 'evidence.safetensors' or set(manifest['file_sha256']) != {name}:
            raise ValueError('Unsafe artifact path or file manifest')
        tensor_path = path.parent/name
        if tensor_path.is_symlink() or _sha(tensor_path) != manifest['file_sha256'][name]:
            raise ValueError('Changed artifact file hash')
        tensors = load_file(str(tensor_path), device='cpu')
        if set(tensors) != set(manifest['tensor_manifest']):
            raise ValueError('Incomplete tensor manifest')
        for key, value in tensors.items():
            row = manifest['tensor_manifest'][key]
            _finite(value, key)
            if (row['sha256'] != tensor_hash(value) or row['shape'] != list(value.shape) or row['dtype'] != str(value.dtype)):
                raise ValueError('Changed tensor hash, shape or dtype: ' + key)
            if any(word in key for word in ('final_latent', 'partial_latent')):
                raise ValueError('Unexpected output latent')
        used = set()
        restored = _inflate(manifest['tree'], tensors, used)
        if used != set(tensors):
            raise ValueError('Unreferenced artifact tensors')
        _validate(restored, restored['mask_bank'])
    except (OSError, KeyError, TypeError, IndexError) as error:
        raise ValueError('Malformed or missing prefix artifact file') from error
    return manifest, tensors


def _compare_replays(reference_path, prefix_path, root, pilots):
    from PIL import Image
    import numpy as np
    from .paired_observation import checked_diagnostic
    paths = [Path(reference_path).resolve(), Path(prefix_path).resolve()]
    if paths[0] == paths[1] or paths[0].samefile(paths[1]):
        raise ValueError('Two distinct independent replay reports are required')
    approved_root = Path(root).resolve() if root is not None else Path(os.path.commonpath([str(p.parent) for p in paths]))
    if approved_root == Path(approved_root.anchor):
        raise ValueError('Replay artifacts need a shared parent or explicit diagnostic root')
    checked = [checked_diagnostic(path, root=approved_root) for path in paths]
    if checked[0][0].get('artifact_id') == checked[1][0].get('artifact_id'):
        raise ValueError('Two distinct independent replay artifact IDs are required')
    for (report, tensors), pilot in zip(checked, pilots):
        if (report.get('steps') != 8 or report.get('branch_nfe') != {'base': 8, 'slider': 8}
                or report.get('total_model_nfe') != 16 or report.get('diagnostic_level') != 'audit'
                or 'image' not in report.get('artifact_sha256', {}) or 'final_latent' not in report.get('tensor_manifest', {})):
            raise ValueError('Replay is not a complete checked paired eight-step image')
        for key in ('seed', 'cfg', 'strength', 'sampler', 'scheduler', 'mix_scope', 'mask_mode',
                    'lora_sha256', 'initial_noise_sha256', 'initial_latent_sha256', 'full_sigmas_sha256',
                    'implementation_source_sha256', 'environment', 'conditioning_sha256', 'token_positions'):
            if report.get(key) != pilot['report'][key]:
                raise ValueError('Replay/pilot identity mismatch: ' + key)
        for role in ROLES:
            _same(tensors.get('reference_'+role+'_mask'), pilot['mask_bank']['masks'][role], 'replay mask/' + role)
        if (report.get('prediction_selection', {}).get('dilate_radius') != 0
                or report.get('mask_generation', {}).get('fill_holes_max_area') != 0
                or report.get('mask_generation', {}).get('mask_dilate_radius') != 0
                or report.get('sampler_nfe') != 8):
            raise ValueError('Replay selection or sampler count mismatch')
        _same(tensors.get('effective_image_mask'), pilot['mask_bank']['masks']['target'], 'replay effective selection')
        if set(tensors) != set(report['tensor_manifest']):
            raise ValueError('Incomplete replay tensor manifest')
        for name, value in tensors.items():
            _finite(value, 'replay/'+name)
            row = report['tensor_manifest'][name]
            if row.get('shape') != list(value.shape) or row.get('dtype') != str(value.dtype):
                raise ValueError('Replay tensor shape/dtype manifest mismatch')
        rows = report.get('step_trace')
        if not isinstance(rows, list) or len(rows) != 8:
            raise ValueError('Replay needs complete eight-step traces')
        for key in TRACE_KEYS:
            value = tensors.get(key)
            _finite(value, 'replay/' + key)
            expected = (8,) if key == 'trace_sigmas' else (8, *tensors['final_latent'].shape)
            if tuple(value.shape) != expected:
                raise ValueError('Replay trace shape/count mismatch')
        _same(tensors['trace_sigmas'], pilot['validation_tensors']['full_sigmas'][:-1], 'replay sigmas')
        for key in ('trace_inputs', 'trace_base_predictions', 'trace_slider_predictions'):
            _same(tensors[key][0], pilot['validation_tensors'][key][0], 'replay first ' + key)
        for i, row in enumerate(rows):
            if (row.get('eval_index') != i or row.get('sigma') != float(tensors['trace_sigmas'][i])
                    or row.get('input_sha256') != tensor_hash(tensors['trace_inputs'][i])
                    or row.get('prediction_sha256') != tensor_hash(tensors['trace_predictions'][i])):
                raise ValueError('Replay actual-input/prediction trace mismatch')
    (a, at), (b, bt) = checked
    if not a.get('generation') or a.get('generation') != b.get('generation'):
        raise ValueError('Replay generation provenance mismatch')
    for key in (*TRACE_KEYS, 'final_latent'):
        _same(at[key], bt[key], 'replay/' + key)
    images = []
    for path, (report, _) in zip(paths, checked):
        with Image.open(path.parent/report['artifacts']['image']) as image:
            images.append(np.asarray(image.convert('RGB')).copy())
    if images[0].shape != images[1].shape or not np.array_equal(*images):
        raise ValueError('Replay decoded RGB mismatch')
    return {'final_latent_equal': True, 'decoded_rgb_equal': True,
            'reports': [{'path': str(p), 'sha256': _sha(p)} for p in paths]}


def compare_prefix_artifacts(reference_manifest, prefix_manifest, *, reference_replay=None, prefix_replay=None, diagnostic_root=None):
    """Require exact tensor parity. Pilot-only evidence cannot establish image parity."""
    if (reference_replay is None) != (prefix_replay is None):
        raise ValueError('Supply both replay reports as a pair')
    loaded = [load_prefix_artifacts(path) for path in (reference_manifest, prefix_manifest)]
    manifests, tensors = zip(*loaded)
    trees = [_inflate(manifest['tree'], values, set()) for manifest, values in loaded]
    reference, prefix = trees
    if reference['report']['pilot_mode'] != 'full_reference' or prefix['report']['pilot_mode'] != 'prefix':
        raise ValueError('Expected full_reference then prefix artifacts')
    for key in IDENTITY_KEYS:
        if _json(reference['report'][key]) != _json(prefix['report'][key]):
            raise ValueError('Pilot identity mismatch: ' + key)
    if _json(reference['report'].get('source_identity')) != _json(prefix['report'].get('source_identity')):
        raise ValueError('Supplied frozen source identity mismatch')
    if _json(reference['report'].get('r0_cpu_runtime')) != _json(prefix['report'].get('r0_cpu_runtime')):
        raise ValueError('R0 CPU runtime identity mismatch')
    for group in ('branches', 'mask_bank', 'r0_tensors', 'bootstrap_masks'):
        left, right = [manifest['tree'].get(group) for manifest in manifests]
        if _json(left) != _json(right):
            raise ValueError('Pilot evidence metadata mismatch: ' + group)
    for name in ('initial_noise', 'initial_latent', 'full_sigmas'):
        _same(reference['validation_tensors'][name], prefix['validation_tensors'][name], name)
    for name in TRACE_KEYS:
        _same(reference['validation_tensors'][name][:3], prefix['validation_tensors'][name], name)
    for group in ('branches/', 'mask_bank/', 'r0_tensors/', 'bootstrap_masks/'):
        a = {key for key in tensors[0] if key.startswith(group)}
        b = {key for key in tensors[1] if key.startswith(group)}
        if a != b:
            raise ValueError('Pilot evidence tensor set mismatch: ' + group)
        for key in sorted(a):
            _same(tensors[0][key], tensors[1][key], key)
    if _json(reference['frames']) != _json(prefix['frames']):
        raise ValueError('Actual-input observation frames differ')
    result = {'pilot_schema_version': SCHEMA, 'status': 'pilot_parity_only', 'pilot_parity': True,
        'full_pipeline_parity': None, 'required_tap_count': 30, 'extraction_tap_count': 18,
        'compared_tap_count': sum(len(taps) for taps in reference['branches'].values()),
        'reference_nfe': 16, 'prefix_nfe': 6,
        'nfe': {'reference_route': 34, 'prefix_route': 24, 'cold_paired_validation': 56},
        'timing_claim': None,
        'nfe_scope': 'nominal uncached diffusion model forwards; excludes VAE and non-model work',
        'source_identity_scope': 'supplied_frozen_provenance_not_runtime_verified_checkpoint_or_style',
        'manifests': [{'path': str(Path(path).resolve()), 'sha256': _sha(path)} for path in (reference_manifest, prefix_manifest)]}
    if reference_replay is not None:
        result['replay_parity'] = _compare_replays(reference_replay, prefix_replay, diagnostic_root, trees)
        result.update(status='full_pipeline_parity', full_pipeline_parity=True)
    return result
