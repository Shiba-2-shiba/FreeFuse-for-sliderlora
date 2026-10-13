"""Build explicit API graphs and compare retained prefix artifacts, entirely offline."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slider_fuse.experimental_artifacts import validate_experimental_prefix
from slider_fuse.prefix_artifacts import compare_prefix_artifacts


def _node(graph, ident, kind, inputs):
    graph[str(ident)] = {'class_type': kind, 'inputs': inputs}


def build_validation_graph(case_config, initial_mask_manifest=None, *, trial_id, prefix, include_replays=False, mode='compare'):
    """Copy only the explicit loaders; no inherited OFF output can execute."""
    validate_experimental_prefix(prefix)
    if type(trial_id) is not int or not 0 <= trial_id <= 0x7fffffff:
        raise ValueError('trial_id must be a nonnegative 31-bit integer')
    if mode not in ('compare', 'fast'):
        raise ValueError('mode must be compare or fast')
    include_replays = include_replays or mode == 'fast'
    old = case_config['base_graph']
    expected = {'1': 'UNETLoader', '2': 'LoraLoaderModelOnly', '3': 'CLIPLoader',
                '4': 'Krea2SliderFuseEncode', '5': 'ConditioningZeroOut', '7': 'EmptySD3LatentImage'}
    for ident, kind in expected.items():
        if old[ident]['class_type'] != kind:
            raise ValueError('Unsupported explicit loader ' + ident)
    if old['2']['inputs']['lora_name'] == case_config['slider_path']:
        raise ValueError('Target Slider must not be loaded upstream')
    seed = case_config['seed']
    if type(seed) is not int or not 0 <= seed <= 0xffffffffffffffff:
        raise ValueError('Invalid frozen seed')
    graph = {ident: copy.deepcopy(old[ident]) for ident in expected}
    subjects = copy.deepcopy(old['6']['inputs'])
    for field in ('target_mask', 'protected_mask'):
        subjects.pop(field, None)
    subjects['prompt_info'] = ['4', 1]
    if initial_mask_manifest is not None:
        _node(graph, '60', 'Krea2SliderFuseRetainedMask', {'report_path': str(initial_mask_manifest)})
        subjects.update(target_mask=['60', 0], protected_mask=['60', 1])
        _node(graph, '6', 'Krea2SliderFuseSubjects', subjects)
        subject_id = '6'
    else:
        _node(graph, '6', 'Krea2SliderFuseSubjects', copy.deepcopy(subjects))
        config = copy.deepcopy(case_config['collector_config'])
        if (config.get('collect_step') != 2 or config.get('collect_block') != 18
                or config.get('selected_steps') != [1, 2] or config.get('selected_blocks') != [16, 18]):
            raise ValueError('Cold bootstrap requires frozen steps 1/2, blocks 16/18')
        collect = {'model': ['2', 0], 'positive': ['4', 0], 'negative': ['5', 0], 'prompt_info': ['4', 1],
            'subjects': ['6', 0], 'latent': ['7', 0], 'seed': seed, 'steps': 8, 'trial_id': trial_id,
            'config_json': json.dumps(config, allow_nan=False), 'background_phrase': case_config['background_phrase'],
            'background_occurrence': 0, 'audit_tensors': True}
        _node(graph, '8', 'Krea2SliderFuseExperimentalMaskCollect', collect)
        _node(graph, '18', 'Krea2SliderFuseExperimentalMaskSave', {'suite': ['8', 0], 'filename_prefix': prefix+'_bootstrap'})
        _node(graph, '60', 'Krea2SliderFuseExperimentalMaskSelect', {'suite': ['8', 0], 'variant': 'multi_proto_bg'})
        _node(graph, '61', 'Krea2SliderFuseMaskPreview', {'mask_bank': ['60', 0]})
        subjects.update(target_mask=['61', 0], protected_mask=['61', 1])
        _node(graph, '62', 'Krea2SliderFuseSubjects', subjects)
        subject_id = '62'
    source_identity = {'case_id': case_config.get('case_id'),
        'asset_sha256': copy.deepcopy(case_config.get('asset_sha256', {})),
        'base_graph_sha256': hashlib.sha256(json.dumps(old, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()}
    original = old['21']['inputs']
    allowed = {'strength', 'collect_step', 'collect_block', 'top_k_ratio', 'temperature'}
    common = {key: copy.deepcopy(value) for key, value in original.items() if key in allowed}
    common.update(model=['2', 0], positive=['4', 0], negative=['5', 0], prompt_info=['4', 1],
        subjects=[subject_id, 0], latent=['7', 0], lora_name=case_config['slider_path'], strength=4.,
        seed=seed, steps=8, cfg=1., trial_id=trial_id, mix_scope='target_mask', mask_mode='manual',
        diagnostic_level='audit', fill_holes_max_area=0, mask_dilate_radius=0, selection_dilate_radius=0)
    if include_replays:
        if old['9']['class_type'] != 'VAELoader':
            raise ValueError('Expected explicit final replay VAELoader')
        graph['9'] = copy.deepcopy(old['9'])
    modes = ((100, 'full_reference'), (200, 'prefix')) if mode == 'compare' else ((200, 'prefix'),)
    for offset, mode in modes:
        pid = str(offset)
        pilot = copy.deepcopy(common)
        pilot.update(pilot_mode=mode, background_phrase=case_config['background_phrase'],
                     source_identity_json=json.dumps(source_identity, sort_keys=True, allow_nan=False))
        _node(graph, pid, 'Krea2SliderFusePairedPrefixPilot', pilot)
        _node(graph, offset+5, 'Krea2SliderFusePrefixPilotSave',
              {'probe': [pid, 1], 'mask_bank': [pid, 0], 'filename_prefix': prefix+'_'+mode})
        if include_replays:
            preview, subject, sampler = str(offset+10), str(offset+11), str(offset+20)
            _node(graph, preview, 'Krea2SliderFuseMaskPreview', {'mask_bank': [pid, 0]})
            sub = copy.deepcopy(subjects)
            sub.update(target_mask=[preview, 0], protected_mask=[preview, 1])
            _node(graph, subject, 'Krea2SliderFuseSubjects', sub)
            replay = copy.deepcopy(common)
            replay['subjects'] = [subject, 0]
            _node(graph, sampler, 'Krea2SliderFusePredictionMixSampler', replay)
            _node(graph, offset+21, 'VAEDecode', {'samples': [sampler, 0], 'vae': ['9', 0]})
            _node(graph, offset+22, 'Krea2SliderFuseDiagnosticSave', {'images': [str(offset+21), 0],
                'latent': [sampler, 0], 'diagnostics': [sampler, 2], 'filename_prefix': prefix+'_'+mode+'_replay'})
    return graph


def _write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build', help='Write API graph JSON without submitting it')
    build.add_argument('--case-config', required=True, type=Path)
    build.add_argument('--case-id', help='Select one case from a frozen experiment.json')
    build.add_argument('--bootstrap-manifest', type=Path, help='Optional retained mask; omission builds a cold two-forward bootstrap')
    build.add_argument('--mode', choices=['compare', 'fast'], default='compare')
    build.add_argument('--trial-id', required=True, type=int)
    build.add_argument('--prefix', required=True)
    build.add_argument('--with-replays', action='store_true')
    build.add_argument('--output', required=True, type=Path)
    compare = commands.add_parser('compare', help='Fail closed unless exact retained evidence agrees')
    compare.add_argument('--reference', required=True, type=Path)
    compare.add_argument('--prefix', required=True, type=Path)
    compare.add_argument('--reference-replay', type=Path)
    compare.add_argument('--prefix-replay', type=Path)
    compare.add_argument('--diagnostic-root', type=Path)
    compare.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'build':
            case = json.loads(args.case_config.read_text(encoding='utf-8'))
            if args.case_id:
                matches = [c for c in case.get('cases', []) if c.get('case_id') == args.case_id]
                if len(matches) != 1:
                    raise ValueError('Frozen case ID must select exactly one case')
                case = matches[0]
            graph = build_validation_graph(case, args.bootstrap_manifest, trial_id=args.trial_id,
                prefix=args.prefix, include_replays=args.with_replays, mode=args.mode)
            _write_new(args.output, {'prompt': graph})
            print(json.dumps({'status': 'built_not_submitted', 'output': str(args.output)}))
        else:
            result = compare_prefix_artifacts(args.reference, args.prefix, reference_replay=args.reference_replay,
                prefix_replay=args.prefix_replay, diagnostic_root=args.diagnostic_root)
            if args.output:
                _write_new(args.output, result)
            print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'failed', 'error': str(error)}), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
