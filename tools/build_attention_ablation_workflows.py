"""Build the focused background-on prototype and fixed-mask strength experiment.

Default: two diagnostic cases, two matched collectors and three edits per case.
No completed OFF image, new segmentation method or production-node change.
Cold generation is stdlib-only. Optional saved-mask reuse requires complete
historical artifacts, exact PNG copies and separately retained asset identities.
Only this generator's named JSON files are replaced (atomic per file, not a
multi-file transaction); inference and model downloads are never performed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
import sys
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
STEM = 'krea2_female_slider_attention_ablation'
CASES = {'park': 444444, 'woman_front_strong_overlap': 42}
COLLECT = 'Krea2SliderFuseExperimentalMaskCollect'
SELECT = 'Krea2SliderFuseExperimentalMaskSelect'
SAVE = 'Krea2SliderFuseExperimentalMaskSave'
SAMPLER = 'Krea2SliderFusePredictionMixSampler'


def load_tool(filename):
    spec = importlib.util.spec_from_file_location('ablation_' + Path(filename).stem, ROOT / 'tools' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add(api, sid, node_type, inputs, title):
    api[str(sid)] = {'class_type': node_type, 'inputs': inputs, '_meta': {'title': title}}


def make_ui(api, template_ui, notes, rows):
    """Use existing node/widget schemas and reconstruct every link from the API."""
    templates = {n['type']: n for n in template_ui['nodes'] if n['type'] != 'Note'}
    templates['LoadImageMask'] = {'size': [420, 250], 'inputs': [],
        'outputs': [{'name': 'MASK', 'type': 'MASK', 'links': []}],
        'widgets_values_named': {'image': '', 'channel': 'red'}}
    order, active, visited = [], set(), set()
    def visit(sid):
        if sid in active:
            raise ValueError('Cycle in ablation graph')
        if sid in visited:
            return
        active.add(sid)
        for value in api[sid]['inputs'].values():
            if isinstance(value, list):
                visit(value[0])
        active.remove(sid)
        visited.add(sid)
        order.append(sid)
    for sid in api:
        visit(sid)
    setup = {'1': [0, 40], '2': [0, 260], '3': [470, 40], '4': [470, 310],
        '5': [1060, 40], '6': [1060, 260], '7': [1510, 40], '9': [1510, 260],
        '8': [0, 860], '10': [600, 880], '11': [1100, 860], '12': [1700, 880]}
    nodes = {}
    offsets = {0: [0, 0], 1: [480, 0], 10: [920, 0], 11: [1360, 0], 12: [1970, 0],
        13: [2410, 0], 14: [1970, 390], 15: [2410, 390], 16: [2850, 390],
        17: [2410, 720], 18: [2850, 720]}
    for i, sid in enumerate(order):
        entry = api[sid]
        template = templates[entry['class_type']]
        node = deepcopy(template)
        node.update(id=int(sid), type=entry['class_type'], flags={}, mode=0, order=i,
            title=entry['_meta']['title'], properties={'Node name for S&R': entry['class_type']})
        if sid in setup:
            node['pos'] = setup[sid]
        else:
            row, off = divmod(int(sid) - 100, 30)
            x, y = offsets[off]
            node['pos'] = [x, 1810 + row * 1160 + y]
        for port in node['inputs']:
            port['link'] = None
        for port in node['outputs']:
            port['links'] = []
        named = {name: entry['inputs'][name] for name in template['widgets_values_named']}
        values = list(named.values())
        if entry['class_type'] in (COLLECT, SAMPLER):
            values.insert(list(named).index('seed') + 1, 'fixed')
        node.update(widgets_values_named=named, widgets_values=values)
        nodes[sid] = node
    links = []
    for sid, node in nodes.items():
        for slot, port in enumerate(node['inputs']):
            value = api[sid]['inputs'].get(port['name'])
            if value is None:
                continue
            src, out = value
            lid = len(links) + 1
            port['link'] = lid
            nodes[src]['outputs'][out]['links'].append(lid)
            links.append([lid, int(src), out, int(sid), slot, port['type']])
    for i, text in enumerate(notes):
        nodes[str(9000 + i)] = {'id': 9000 + i, 'type': 'Note', 'pos': [i * 1130, -530],
            'size': [1080, 470], 'flags': {}, 'order': len(order) + i, 'mode': 0,
            'inputs': [], 'outputs': [], 'properties': {}, 'widgets_values': [text],
            'title': 'Focused ablation instructions'}
    return {'last_node_id': 9000 + len(notes) - 1, 'last_link_id': len(links),
        'nodes': list(nodes.values()), 'links': links,
        'groups': [{'title': row['label'], 'bounding': [-30, 1750 + ((int(row['sampler_node']) - 111) // 30) * 1160, 3350, 1110],
            'color': '#526b87', 'font_size': 24, 'flags': {}} for i, row in enumerate(rows)],
        'config': {}, 'extra': {'ds': {'scale': .32, 'offset': [120, 640]}}, 'version': .4}


def metric(reason='not_evaluated'):
    return {'value': None, 'reason': reason, 'evidence': []}


def record_for(row):
    return dict(label=row['label'], strength=row['strength'], requested_prototypes=row['requested_prototypes'],
        achieved_prototypes_by_role={'target': None, 'protected': None, 'background': None},
        achieved_count_source='len(report.algorithm.variants.multi_proto_bg.prototype_indices[role]); requested k is an upper bound',
        image_path=None, edit_diagnostic_json_path=None, final_latent_sha256=None,
        reference_partition_sha256=None, effective_image_mask_sha256=None,
        pose_gate={'status': 'not_evaluated', 'passed': None, 'human_reviewer': None, 'evidence': [], 'notes': None},
        **{key: metric() for key in ('duplicate_anatomy', 'target_edit_achievement', 'natural_body_proportions',
            'male_face', 'male_eyeglasses', 'male_height', 'male_build', 'small_arm_hand_adjustment',
            'background_change', 'old_silhouette_residue')},
        preservation_vs_off=metric('not_evaluable_without_verified_matched_existing_OFF_reference; do not generate an extra OFF'),
        failure_reason=None)


def build_case(case_id, seed, trial_id, reuse=None):
    prior = load_tool('build_attention_validation_workflows.py')
    # Reuse established templates and controls without rewriting historical files.
    with tempfile.TemporaryDirectory() as temporary:
        prior.write_workflows(temporary, case_id=case_id, seed=seed, trial_id=trial_id)
        old = prior.read_json(Path(temporary) / (prior.EDITED + '_api.json'))
        template_ui = prior.read_json(Path(temporary) / (prior.EDITED + '.json'))
    api = {sid: deepcopy(old[sid]) for sid in ('1', '2', '3', '4', '5', '6', '7', '9')}
    prefix = f'slider_attention_ablation/{case_id}/seed{seed}/trial{trial_id}/'
    short_case = 'park' if case_id == 'park' else 'strong'
    configs = {}
    for k, collector, saver in ((1, '8', '10'), (3, '11', '12')):
        inputs = deepcopy(old['8']['inputs'])
        config = deepcopy(prior.CONFIG)
        config['prototypes'] = k
        configs[str(k)] = config
        inputs.update(config_json=json.dumps(config, indent=2), audit_tensors=True)
        add(api, collector, COLLECT, inputs, f'Background ON: requested k={k}; matched two-forward capture')
        add(api, saver, SAVE, dict(suite=[collector, 0],
            filename_prefix=f'attn_ablate_{short_case}_s{seed}_t{trial_id}_bg_k{k}'), f'Save k{k} suite and observation hashes')
    rows = []
    for b, k, strength, collector, subject in ((100, 1, 4., '8', '110'),
        (130, 3, 4., '11', '140'), (160, 3, 2., '11', '140')):
        label = f'bg_k{k}_s{int(strength)}'
        if b != 160:
            add(api, b, SELECT, dict(suite=[collector, 0], variant='multi_proto_bg'), f'{label}: background-on candidate, configured k{k}')
            add(api, b + 1, 'Krea2SliderFuseMaskPreview', dict(mask_bank=[str(b), 0]), f'k{k} background-on source masks')
            inputs = deepcopy(api['6']['inputs'])
            inputs.update(target_mask=[str(b + 1), 0], protected_mask=[str(b + 1), 1])
            add(api, subject, 'Krea2SliderFuseSubjects', inputs, f'k{k} masks; shared unchanged across strengths')
        sampler = deepcopy(old['201']['inputs'])
        sampler.update(subjects=[subject, 0], strength=strength, seed=seed, trial_id=trial_id)
        add(api, b + 11, SAMPLER, sampler, f'{label}: manual replay +{int(strength)}, 8 steps, all radii zero')
        add(api, b + 12, 'VAEDecode', dict(samples=[str(b + 11), 0], vae=['9', 0]), label + ': edited image')
        add(api, b + 13, 'Krea2SliderFuseDiagnosticSave', dict(images=[str(b + 12), 0],
            latent=[str(b + 11), 0], diagnostics=[str(b + 11), 2], filename_prefix=prefix + label), label + ': image, latent, audit')
        add(api, b + 14, 'Krea2SliderFuseMaskPreview', dict(mask_bank=[str(b + 11), 1]), label + ': effective bank')
        for off, slot, name in ((15, 7, 'effective_prediction_mask'), (17, 8, 'selection_added_mask')):
            add(api, b + off, 'MaskToImage', dict(mask=[str(b + 14), slot]), label + ': ' + name)
            add(api, b + off + 1, 'SaveImage', dict(images=[str(b + off), 0], filename_prefix=prefix + label + '_' + name), label + ': save ' + name)
        rows.append(dict(label=label, variant='multi_proto_bg', requested_prototypes=k, background_competitor=True,
            strength=strength, collector_node=collector, subjects_node=subject, sampler_node=str(b + 11),
            save_node=str(b + 13), effective_prediction_mask_save_node=str(b + 16), selection_added_mask_save_node=str(b + 18)))
    if reuse is not None:
        # Reuse record is produced only by the strict verifier, never read as an
        # unchecked "approved" flag. The current graph adopts verified assets.
        gen = reuse['generation_config']
        expected = dict(prompt=api['4']['inputs']['prompt'], seed=seed, steps=8, cfg=1., strength=4.,
            sampler='euler', scheduler='simple', batch_size=1, denoise=1.)
        if any(gen.get(k) != v for k, v in expected.items()):
            raise ValueError('Historical +4 generation does not match selected case/seed/controls')
        api['1']['inputs'].update(unet_name=gen['checkpoint'], weight_dtype=gen['weight_dtype'])
        style = gen['style_loras'][0]
        api['2']['inputs'].update(lora_name=style['lora_name'], strength_model=style['strength'])
        api['3']['inputs'].update(clip_name=gen['clip_name'], type=gen['clip_type'], device=gen['clip_device'])
        api['7']['inputs'].update(width=gen['width'], height=gen['height'], batch_size=gen['batch_size'])
        api['9']['inputs']['vae_name'] = gen['vae_name']
        for key in ('background_phrase', 'background_occurrence'):
            if reuse['runtime_invariants'].get(key) != api['8']['inputs'][key]:
                raise ValueError('Historical background role differs from selected case: ' + key)
        if reuse['frozen_suite_config'] != configs['3']:
            raise ValueError('Historical suite config differs from frozen k3 controls')
        for sid in ('11', '12', *map(str, range(141, 149))):
            api.pop(sid, None)
        add(api, '130', 'LoadImageMask', dict(image=reuse['masks']['target']['filename'], channel='red'), 'Verified saved k3 target PNG, no alpha inversion')
        add(api, '131', 'LoadImageMask', dict(image=reuse['masks']['protected']['filename'], channel='red'), 'Verified saved k3 protected PNG, unchanged grid')
        api['140']['inputs'].update(target_mask=['130', 0], protected_mask=['131', 0])
        for node in api.values():
            if node['class_type'] == SAMPLER:
                node['inputs']['lora_name'] = gen['slider_lora']
        rows = [row for row in rows if row['label'] != 'bg_k3_s4']
        rows[-1]['collector_node'] = None
    mode = 'cold' if reuse is None else 'verified_saved_reuse'
    stem = STEM + '_' + case_id + ('' if reuse is None else '_reuse')
    pose = ['Exactly two people with distinct heads, full-body framing and both pairs of feet visible.',
        'Younger target appearance is an intended slider effect, not itself a pose or quality failure.',
        'A human checks actual images before any pose-conditioned score; mask overlap and prompt wording do not prove pose.']
    pose += (["Woman in the foreground; man behind her. The woman's body substantially occludes the man's torso and his face remains visible.",
        'Side-by-side placement or shoulder contact alone is composition_not_met; keep every failure.']
        if case_id != 'park' else ['Preserve the park scene, side-by-side arrangement and both complete figures.'])
    notes = ['FOCUSED EXPERIMENT: background ON in both k1 and k3. Variant names remain multi_proto_bg; titles report actual requested k. Existing centroid_control has background OFF and is not the k1 control here. Only prototypes changes in collection config. Both collectors use the same original model/prompt/latent/seed/full schedule. audit_tensors=true adds hashing/transfer time; compare primary observation hashes before attributing a result to k.',
        ('COLD: two matched 2-forward collectors + three partial-mask edits = 52 model NFE per case; both cases104. k3 +2 and +4 share exactly one Subjects/mask source. No completed OFF run. Save both suites and all edit diagnostics.' if reuse is None else 'SAVED REUSE: one k1 2-forward collector + k1/+4 and saved-k3/+2 =34 model NFE per case,68 for both cases. Historical k3/+4 is external evidence. Only verified unchanged PNGs are loaded via red channel. Check actual new runtime hashes before accepting the comparison; preflight is not execution proof.'),
        'Pose gate first, then separate duplicate anatomy, intended age/size edit and natural proportions. Check male face/eyeglasses/height/build separately from allowed small arm/hand motion. No matched existing OFF means preservation-vs-OFF stays not evaluable. Seeds123/777 have already been inspected; register genuinely unseen seeds after freezing settings. See docs/attention-ablation.md.']
    manifest = dict(name='Background-on k1/k3 and fixed-k3-mask strength ablation', generator='tools/build_attention_ablation_workflows.py',
        mode=mode, selected_case=dict(case_id=case_id, seed=seed, trial_id=trial_id),
        workflows={'ui': stem + '.json', 'api': stem + '_api.json'}, variants=rows, collection_configs=configs,
        validation=dict(gpu_execution='not_run', native_comfy='not_run', image_quality='not_evaluated', pose_gate='not_evaluated'),
        cost_accounting=dict(expected_cold_model_nfe=52, expected_cold_two_cases_model_nfe=104,
            conditional_reuse_model_nfe=34, conditional_reuse_two_cases_model_nfe=68,
            expected_this_graph_model_nfe=52 if reuse is None else 34, actual_model_nfe=None,
            actual_wall_seconds=None, peak_memory_bytes=None, cache_status='not_recorded',
            assumptions='Cold uncached active outputs, valid nonempty partial masks, 8-step two-branch edits. Actual counters override nominal values. NFE excludes VAE, feature hooks, audit hashes, transfers, algorithms and saving.',
            memory_rule='Retain runtime memory.scope; process-lifetime or resident-model peaks are not incremental VRAM. Do not sum peaks. Missing measurements stay null.'),
        output_prefixes={sid: n['inputs']['filename_prefix'] for sid, n in api.items() if 'filename_prefix' in n['inputs']},
        pose_gate_criteria=pose,
        holdout_policy=dict(previously_seen_seeds=[42, 123, 777, 444444], confirmed_unseen_seeds=[],
            instruction='Freeze config, then choose and record new seeds not previously inspected in any tuning. Check the run log; a new number is not proof it is unseen. Keep every attempted seed/failure and use both cases.'),
        mask_semantics=dict(baseline='Stored baseline uncertain_mask is the whole residual routing background. It is not zero and is not equivalent to new confidence/margin-based unknown cells; do not rank confidence across these definitions.',
            background='Routing background = explicit background competitor cells + uncertain cells for bg candidates. Neither is a human ground-truth annotation.'),
        reuse_verification=reuse,
        evaluation_record_template=dict(case_id=case_id, seed=seed, trial_id=trial_id, queued_api_path=None,
            queued_api_sha256=None, collection_run_ids={'k1': None, 'k3': None}, suite_manifest_paths={'k1': None, 'k3': None},
            collection_evidence={label: dict.fromkeys(('normalized_config', 'initial_noise_sha256',
                'initial_latent_sha256', 'full_sigmas_sha256', 'used_sigmas_sha256', 'token_positions',
                'conditioning', 'primary_tap', 'source_partition_sha256', 'implementation')) for label in ('k1', 'k3')},
            observation_parity=dict(status='not_checked', primary_step=2, primary_block=18, evidence=[],
                required=['normalized_config except prototypes', 'seed/prompt/token_positions/conditioning including metadata',
                    'initial_noise_sha256', 'initial_latent_sha256', 'full_sigmas_sha256', 'used_sigmas_sha256',
                    'primary tensor_sha256_before and tensor_sha256_after', 'attention_mask_sha256 and shape'],
                rule='Check actual hashes, not seeds alone. Missing historical tensor audit is unverified, not equal.'),
            asset_identity=dict(status='same_graph_sources_pending_runtime_evidence' if reuse is None else 'see_reuse_verification', evidence=[]),
            strength_pair_mask_parity=dict(status='not_checked', required=['all three reference_partition_sha256 values', 'effective_image_mask_sha256'],
                rule='Equal source masks alone do not prove equal effective selection. Require all fill/dilation controls zero.'),
            actual_model_nfe=None, wall_seconds=None, peak_memory_bytes=None, cache_status='not_recorded',
            mask_accuracy=metric('not_evaluable_without_independent_human_verified_same_grid_labels'),
            variant_results=[record_for(row) for row in rows],
            reused_k3_s4=None if reuse is None else dict(source=reuse['sources'], evaluation=record_for(dict(label='bg_k3_s4', strength=4., requested_prototypes=3)))),
        evaluation_order=['Verify runtime provenance, primary observation equality, prototype counts and mask/selection hashes.',
            'Apply human pose gate to each image before pose-conditioned quality scoring. Retain composition_not_met cases.',
            'Score edit achievement separately from natural proportions and duplicate anatomy; reduced effect is not automatically improved editing.',
            'Compare male face/eyeglasses/height/build while recording small plausible arm/hand changes separately.',
            'Freeze decisions before genuinely unseen-seed followup; identical k1/k3 masks are a valid null result.'],
        limitations=['This index is a generated plan, not a runtime or quality certificate. UI edits require a new queued API provenance record.',
            'Two separately collected observations are matched by controls, not automatically proven identical; enabled tensor audit permits checking.',
            'Requested k3 can realize fewer prototypes; use runtime prototype_indices lengths per role.',
            'No extra completed OFF generation; old edited baseline +4 is not OFF. Absent matched existing OFF cannot certify preservation.',
            'No equal-area allocation, uniform dilation, propagation retuning, architecture changes or automatic quality ranking.'])
    return {stem + '_api.json': api, stem + '.json': make_ui(api, template_ui, notes, rows), stem + '.index.json': manifest}


def write_workflows(output_dir=None, case_id='all', seed=None, trial_id=0, *, reuse_suite=None,
                    reuse_edit=None, input_dir=None, reuse_target=None, reuse_protected=None, asset_record=None):
    if case_id not in ('all', *CASES):
        raise ValueError('Unknown diagnostic case')
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**64 - 1):
        raise ValueError('seed must be an unsigned 64-bit integer')
    if case_id == 'all' and seed is not None:
        raise ValueError('Select one case when overriding seed')
    if type(trial_id) is not int or not 0 <= trial_id <= 2**31 - 1:
        raise ValueError('trial_id must be a 31-bit nonnegative integer')
    reuse_args = [reuse_suite, reuse_edit, input_dir, reuse_target, reuse_protected, asset_record]
    reuse = None
    if any(v is not None for v in reuse_args):
        if case_id == 'all' or any(v is None for v in reuse_args):
            raise ValueError('Saved reuse requires one case and all six reuse arguments; otherwise use the cold graph')
        # Script execution outside the checkout puts only tools/ on sys.path.
        # The verifier uses the repository's existing artifact validators.
        sys.path.insert(0, str(ROOT))
        reuse = load_tool('verify_attention_reuse.py').verify_reuse(*reuse_args)
    payloads = {}
    for case in (CASES if case_id == 'all' else [case_id]):
        payloads.update(build_case(case, CASES[case] if seed is None else seed, trial_id, reuse))
    rendered = {name: json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n' for name, value in payloads.items()}
    destination = Path(output_dir) if output_dir is not None else ROOT / 'workflows'
    destination.mkdir(parents=True, exist_ok=True)
    pending = []
    try:
        for name, content in rendered.items():
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='\n', dir=destination, prefix='.ablation-', delete=False) as stream:
                pending.append((Path(stream.name), destination / name))
                stream.write(content)
        for temporary, target in pending:
            os.replace(temporary, target)
    finally:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)
    return [str(destination / name) for name in payloads]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', dest='case_id', choices=('all', *CASES), default='all')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--trial-id', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'workflows')
    for arg in ('reuse-suite', 'reuse-edit', 'input-dir', 'reuse-target', 'reuse-protected', 'asset-record'):
        parser.add_argument('--' + arg)
    try:
        files = write_workflows(**vars(parser.parse_args()))
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, 'Ablation workflow generation failed: ' + str(error) + '\n')
    print(json.dumps({'status': 'generated_not_gpu_validated', 'files': files}, indent=2))


if __name__ == '__main__':
    main()
