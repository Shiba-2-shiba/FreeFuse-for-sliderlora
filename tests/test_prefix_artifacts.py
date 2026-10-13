"""Exact, fail-closed offline evidence checks for the bounded paired pilot."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from slider_fuse.sampling import tensor_hash

ROOT = Path(__file__).resolve().parents[1]


def artifact_module():
    path = ROOT / 'slider_fuse/prefix_artifacts.py'
    assert path.is_file(), 'prefix artifact module is missing'
    from slider_fuse import prefix_artifacts
    return prefix_artifacts


def runner():
    path = ROOT / 'docs/run_prefix_pilot_validation.py'
    assert path.is_file(), 'offline prefix runner is missing'
    spec = importlib.util.spec_from_file_location('prefix_runner_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(mode='prefix'):
    n = 3 if mode == 'prefix' else 8
    shape = (1, 4, 2, 3)
    sigmas = torch.linspace(1., 0., 9)
    tensors = {'initial_noise': torch.ones(shape), 'initial_latent': torch.zeros(shape),
        'full_sigmas': sigmas, 'used_sigmas': sigmas[:n+1].clone(),
        'trace_inputs': torch.ones(n, *shape), 'trace_sigmas': sigmas[:n].clone(),
        'trace_base_predictions': torch.zeros(n, *shape),
        'trace_slider_predictions': torch.ones(n, *shape), 'trace_predictions': torch.full((n, *shape), .5)}
    masks = {'target': torch.tensor([[[1., 1., 0.], [0., 0., 0.]]]),
             'protected': torch.tensor([[[0., 0., 1.], [1., 0., 0.]]]),
             'background': torch.tensor([[[0., 0., 0.], [0., 1., 1.]]])}
    bank = {'grid': [2, 3], 'masks': masks, 'selection_target_mask': masks['target'].clone(),
            'selection_added_mask': torch.zeros(1, 2, 3), 'effective_image_mask': masks['target'].clone()}
    report = {'pilot_schema_version': 1, 'pilot_mode': mode, 'observer_blocks': [8, 10, 18, 23, 26], 'observer_steps': [1, 2, 3], 'steps': 8, 'executed_steps': n,
        'branch_nfe': {'base': n, 'slider': n}, 'total_model_nfe': n*2,
        'seed': 42, 'cfg': 1., 'strength': 4., 'sampler': 'euler', 'scheduler': 'simple',
        'mix_scope': 'target_mask', 'mask_mode': 'manual', 'diagnostic_level': 'audit',
        'grid': [2, 3], 'token_positions': {'target': [0], 'protected': [1]},
        'conditioning_sha256': 'a'*64, 'lora_sha256': 'b'*64,
        'implementation_source_sha256': 'c'*64, 'environment': {'torch': torch.__version__},
        'prediction_selection': {'dilate_radius': 0},
        'mask_generation': {'fill_holes_max_area': 0, 'mask_dilate_radius': 0}}
    for key in ('initial_noise', 'initial_latent', 'full_sigmas'):
        report[key+'_sha256'] = tensor_hash(tensors[key])
    report['reference_partition_sha256'] = {r: tensor_hash(v) for r, v in masks.items()}
    from slider_fuse.experimental_masks import validate_experimental_config
    from slider_fuse.paired_prefix import build_r0_masks
    cfg = validate_experimental_config({'prototypes': 1, 'top_k_ratio': 1., 'seed_confidence': .65, 'seed_margin': .15})
    features = torch.eye(3).repeat_interleave(2, 0)
    positions = {'target': [0], 'protected': [1], 'background': [2]}
    branches = {}
    frames = []
    for branch in ('base', 'slider'):
        branches[branch] = {}
        for step in (1, 2, 3):
            frames.append({'branch': branch, 'step': step, 'input_sha256': tensor_hash(tensors['trace_inputs'][step-1]),
                           'sigma': float(sigmas[step-1]), 'taps': []})
            for block in (8, 10, 18, 23, 26):
                key = f'step_{step}.block_{block}'
                evidence = {f'{key}.{role}.{metric}': (features[:, i][None]*5 if metric.endswith('_scores') else torch.ones(1, 6))
                    for i, role in enumerate(positions) for metric in ('head_scores', 'token_scores', 'head_allowed_fraction', 'token_allowed_fraction')}
                if step == 3 and block in (18, 23):
                    evidence.update({'capture.'+name: features.clone() for name in ('attention_output', 'img_q', 'img_k')})
                branches[branch][key] = {'algorithm': {'grid': [2, 3], 'positions': positions,
                    'config': dict(cfg, collect_step=step, collect_block=block)},
                    'evidence_maps': evidence, 'banks': {}}
    probe = {'report': report, 'frames': frames, 'branches': branches,
             'predictions': {'base': tensors['trace_base_predictions'].clone(),
                 'slider': tensors['trace_slider_predictions'].clone(), 'mix': tensors['trace_predictions'].clone()},
             'validation_tensors': tensors}
    for frame in frames:
        for block in (8, 10, 18, 23, 26):
            frame['taps'].append({'step': frame['step'], 'block': block, 'sigma': frame['sigma'],
                'text_tokens': 3, 'image_tokens': 6, 'query_heads': 1, 'key_value_heads': 1, 'gqa_expanded_heads': 1,
                'tensor_audit_enabled': True, 'attention_mask_shape': None, 'attention_mask_sha256': None,
                'block_input_before_sha256': 'e'*64, 'block_input_after_sha256': 'e'*64,
                'tensor_sha256_before': {name: 'f'*64 for name in ('img_q', 'txt_k', 'img_k', 'attention_output', 'attention_input')},
                'tensor_sha256_after': {name: 'f'*64 for name in ('img_q', 'txt_k', 'img_k', 'attention_output', 'attention_input')}})
    probe['bootstrap_masks'] = copy.deepcopy(masks)
    bank, probe['r0_tensors'] = build_r0_masks(probe, bank)
    report['r0_partition_sha256'] = {r: tensor_hash(v) for r, v in bank['masks'].items()}
    return probe, bank


def save(tmp_path, mode='prefix', change=None):
    probe, bank = fixture(mode)
    if change:
        change(probe, bank)
    saved = artifact_module().save_prefix_artifacts(tmp_path, mode, probe, bank)
    return tmp_path / saved['manifest']


def test_round_trip_preserves_owned_tensors_and_never_saves_partial_latent(tmp_path):
    m = artifact_module()
    p, bank = fixture()
    original = p['validation_tensors']['trace_inputs'].clone()
    saved = m.save_prefix_artifacts(tmp_path, 'pilot', p, bank)
    path = tmp_path / saved['manifest']
    manifest, tensors = m.load_prefix_artifacts(path)
    assert manifest['complete'] is True
    assert torch.equal(tensors['validation_tensors/trace_inputs'], original)
    assert not any('final_latent' in key or 'partial_latent' in key for key in tensors)
    p['validation_tensors']['trace_inputs'].zero_()
    assert torch.equal(m.load_prefix_artifacts(path)[1]['validation_tensors/trace_inputs'], original)
    second = m.save_prefix_artifacts(tmp_path, 'pilot', *fixture())
    assert second['manifest'] != saved['manifest']


def test_exact_prefix_parity_is_not_full_pipeline_success(tmp_path):
    m = artifact_module()
    result = m.compare_prefix_artifacts(save(tmp_path, 'full_reference'), save(tmp_path))
    assert result['pilot_parity'] is True
    assert result['full_pipeline_parity'] is None
    assert result['status'] == 'pilot_parity_only'
    assert result['required_tap_count'] == 30
    assert result['extraction_tap_count'] == 18
    assert result['nfe'] == {'reference_route': 34, 'prefix_route': 24, 'cold_paired_validation': 56}


@pytest.mark.parametrize('problem', ['missing_tap', 'missing_raw', 'nan', 'wrong_count', 'missing_identity',
    'short_trace', 'bad_sigma', 'wrong_hash', 'partial_latent', 'bad_selection', 'wrong_prediction', 'bad_frame'])
def test_malformed_bundle_rejected_before_write(tmp_path, problem):
    m = artifact_module(); p, bank = fixture()
    if problem == 'missing_tap': del p['branches']['base']['step_1.block_8']
    elif problem == 'missing_raw': p['branches']['base']['step_1.block_8']['evidence_maps'] = {}
    elif problem == 'nan': p['validation_tensors']['trace_inputs'][0, 0, 0, 0, 0] = float('nan')
    elif problem == 'wrong_count': p['report']['branch_nfe']['base'] = 2
    elif problem == 'missing_identity': del p['report']['conditioning_sha256']
    elif problem == 'short_trace': p['validation_tensors']['trace_inputs'] = p['validation_tensors']['trace_inputs'][:2]
    elif problem == 'bad_sigma': p['validation_tensors']['used_sigmas'][-1] = 0
    elif problem == 'wrong_hash': p['report']['initial_noise_sha256'] = '0'*64
    elif problem == 'partial_latent': p['validation_tensors']['partial_latent'] = torch.zeros(1)
    elif problem == 'bad_selection': bank['selection_added_mask'][0, 0, 0] = 1
    elif problem == 'wrong_prediction': p['predictions']['mix'][0] = 2
    elif problem == 'bad_frame': p['frames'][0]['input_sha256'] = '0'*64
    with pytest.raises(ValueError): m.save_prefix_artifacts(tmp_path, 'bad', p, bank)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('field', ['conditioning', 'raw', 'trace', 'mask', 'environment', 'config'])
def test_exact_mismatch_fails_closed(tmp_path, field):
    reference = save(tmp_path, 'full_reference')
    def change(p, bank):
        if field == 'conditioning': p['report']['conditioning_sha256'] = 'd'*64
        elif field == 'raw': p['branches']['base']['step_2.block_10']['evidence_maps']['step_2.block_10.target.head_scores'][0, 0] += 1
        elif field == 'trace':
            p['validation_tensors']['trace_base_predictions'][0] += 1
            p['predictions']['base'][0] += 1
        elif field == 'mask':
            bank['masks']['target'][0, 1, 2] = 1; bank['masks']['background'][0, 1, 2] = 0
            bank['selection_target_mask'] = bank['masks']['target'].clone()
            bank['effective_image_mask'] = bank['masks']['target'].clone()
        elif field == 'environment': p['report']['environment']['torch'] = 'different'
        elif field == 'config': p['branches']['base']['step_2.block_10']['algorithm']['config']['temperature'] = 1
    with pytest.raises(ValueError, match='[Mm]ismatch|differ'):
        prefix = save(tmp_path, change=change)
        artifact_module().compare_prefix_artifacts(reference, prefix)


def test_modified_files_and_path_traversal_are_rejected(tmp_path):
    m = artifact_module(); path = save(tmp_path)
    manifest = json.loads(path.read_text())
    tensor_path = path.parent / manifest['tensors']
    tensor_path.write_bytes(tensor_path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='[Cc]hanged|hash'): m.load_prefix_artifacts(path)
    path = save(tmp_path)
    manifest = json.loads(path.read_text()); manifest['tensors'] = '../escape.safetensors'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError): m.load_prefix_artifacts(path)


def test_one_replay_report_is_never_sufficient(tmp_path):
    m = artifact_module(); ref = save(tmp_path, 'full_reference'); prefix = save(tmp_path)
    with pytest.raises(ValueError, match='both|pair'):
        m.compare_prefix_artifacts(ref, prefix, reference_replay=tmp_path/'missing.json')


def case():
    graph = {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'model', 'weight_dtype': 'default'}},
        '2': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['1', 0], 'lora_name': 'style', 'strength_model': 1.}},
        '3': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'clip', 'type': 'krea2', 'device': 'default'}},
        '4': {'class_type': 'Krea2SliderFuseEncode', 'inputs': {'clip': ['3', 0], 'prompt': 'woman man room'}},
        '5': {'class_type': 'ConditioningZeroOut', 'inputs': {'conditioning': ['4', 0]}},
        '6': {'class_type': 'Krea2SliderFuseSubjects', 'inputs': {'prompt_info': ['4', 1], 'target_phrase': 'woman', 'protected_phrase': 'man'}},
        '7': {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': 1024, 'height': 1024, 'batch_size': 1}},
        '9': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'vae'}},
        '21': {'class_type': 'Krea2SliderFusePredictionMixSampler', 'inputs': {'model': ['2', 0], 'positive': ['4', 0],
            'negative': ['5', 0], 'prompt_info': ['4', 1], 'subjects': ['6', 0], 'latent': ['7', 0], 'strength': 4.}},
        '999': {'class_type': 'SaveImage', 'inputs': {'filename_prefix': 'inherited_OFF'}}}
    return {'case_id': 'park__seed_42', 'seed': 42, 'slider_path': 'slider', 'background_phrase': 'room', 'base_graph': graph,
            'collector_config': {'collect_step': 2, 'collect_block': 18, 'selected_steps': [1, 2], 'selected_blocks': [16, 18]}}


def test_workflow_builder_is_pure_and_has_no_vae_or_inherited_outputs(tmp_path):
    m = runner(); c = case(); old = copy.deepcopy(c)
    graph = m.build_validation_graph(c, tmp_path/'bootstrap.json', trial_id=5, prefix='prefix_test')
    assert c == old
    assert set(n['class_type'] for n in graph.values()).isdisjoint({'VAELoader', 'VAEDecode', 'SaveImage', 'Krea2SliderFuseDiagnosticSave'})
    pilots = [n for n in graph.values() if n['class_type'] == 'Krea2SliderFusePairedPrefixPilot']
    assert {p['inputs']['pilot_mode'] for p in pilots} == {'prefix', 'full_reference'}
    assert all(p['inputs']['steps'] == 8 and p['inputs']['latent'] == ['7', 0] and p['inputs']['seed'] == 42 for p in pilots)
    savers = [n for n in graph.values() if n['class_type'] == 'Krea2SliderFusePrefixPilotSave']
    assert len(savers) == 2 and all(set(n['inputs']) == {'probe', 'mask_bank', 'filename_prefix'} for n in savers)
    assert 'inherited_OFF' not in str(graph)
    source = (ROOT/'docs/run_prefix_pilot_validation.py').read_text()
    assert 'run_mask_source_comparison_live' not in source


def test_optional_replays_start_from_original_seed_and_only_replay_is_decoded(tmp_path):
    graph = runner().build_validation_graph(case(), tmp_path/'bootstrap.json', trial_id=8, prefix='test', include_replays=True)
    replay_ids = {sid for sid, n in graph.items() if n['class_type'] == 'Krea2SliderFusePredictionMixSampler'}
    assert len(replay_ids) == 2
    for sid in replay_ids:
        inputs = graph[sid]['inputs']
        assert inputs['latent'] == ['7', 0] and inputs['seed'] == 42 and inputs['steps'] == 8
        subject = graph[inputs['subjects'][0]]['inputs']
        preview = graph[subject['target_mask'][0]]
        assert preview['class_type'] == 'Krea2SliderFuseMaskPreview'
        assert subject['protected_mask'] == [subject['target_mask'][0], 1]
    decodes = [n['inputs']['samples'] for n in graph.values() if n['class_type'] == 'VAEDecode']
    assert len(decodes) == 2 and all(link[0] in replay_ids and link[1] == 0 for link in decodes)


def replay_report(tmp_path, mode, *, pixel=0, latent_value=0):
    from PIL import Image
    p, bank = fixture(mode)
    folder = tmp_path/mode; folder.mkdir()
    traces = fixture('full_reference')[0]['validation_tensors']
    tensors = {**{k: traces[k] for k in ('trace_inputs', 'trace_sigmas', 'trace_predictions', 'trace_base_predictions', 'trace_slider_predictions')},
               'effective_image_mask': bank['effective_image_mask'], 'final_latent': torch.full((1, 4, 2, 3), float(latent_value)),
               **{'reference_'+r+'_mask': bank['masks'][r] for r in bank['masks']}}
    save_file(tensors, str(folder/'replay.safetensors'))
    Image.new('RGB', (4, 4), (pixel, pixel, pixel)).save(folder/'replay.png')
    report = dict(p['report'], artifact_id=mode+'_artifact', steps=8, executed_steps=8, branch_nfe={'base': 8, 'slider': 8},
        total_model_nfe=16, sampler_nfe=8, complete=True, step_trace=[{'eval_index': i, 'sigma': float(traces['trace_sigmas'][i]),
            'input_sha256': tensor_hash(traces['trace_inputs'][i]), 'prediction_sha256': tensor_hash(traces['trace_predictions'][i])} for i in range(8)], generation={'checkpoint': 'model', 'seed': 42},
        artifacts={'tensors': 'replay.safetensors', 'image': 'replay.png'},
        artifact_sha256={kind: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                        for kind, name in [('tensors', 'replay.safetensors'), ('image', 'replay.png')]},
        tensor_manifest={name: {'sha256': tensor_hash(value), 'shape': list(value.shape), 'dtype': str(value.dtype)}
                         for name, value in tensors.items()})
    path = folder/'replay.json'; path.write_text(json.dumps(report))
    return path


def test_full_pipeline_requires_verified_exact_final_latent_and_decoded_rgb(tmp_path):
    m = artifact_module(); ref = save(tmp_path, 'full_reference'); prefix = save(tmp_path)
    a = replay_report(tmp_path, 'full_reference'); b = replay_report(tmp_path, 'prefix')
    result = m.compare_prefix_artifacts(ref, prefix, reference_replay=a, prefix_replay=b)
    assert result['full_pipeline_parity'] is True and result['status'] == 'full_pipeline_parity'
    assert result['replay_parity']['decoded_rgb_equal'] is True


@pytest.mark.parametrize('problem', ['image', 'latent', 'identity', 'changed_file', 'missing_image_hash', 'source', 'environment', 'nan'])
def test_replay_cannot_upgrade_bad_evidence_to_pipeline_success(tmp_path, problem):
    m = artifact_module(); ref = save(tmp_path, 'full_reference'); prefix = save(tmp_path)
    a = replay_report(tmp_path, 'full_reference')
    b = replay_report(tmp_path, 'prefix', pixel=1 if problem == 'image' else 0,
                      latent_value=1 if problem == 'latent' else (float('nan') if problem == 'nan' else 0))
    report = json.loads(b.read_text())
    if problem == 'identity': report['seed'] = 43
    if problem == 'source': report['implementation_source_sha256'] = 'd'*64
    if problem == 'environment': report['environment']['torch'] = 'different'
    if problem == 'missing_image_hash': del report['artifact_sha256']['image']
    if problem == 'changed_file': (b.parent/'replay.png').write_bytes(b'changed')
    b.write_text(json.dumps(report))
    with pytest.raises(ValueError): m.compare_prefix_artifacts(ref, prefix, reference_replay=a, prefix_replay=b)


def test_additional_instrumentation_taps_and_r0_evidence_are_compared(tmp_path):
    m = artifact_module(); ref = save(tmp_path, 'full_reference'); prefix = save(tmp_path)
    assert m.compare_prefix_artifacts(ref, prefix)['compared_tap_count'] == 30
    def bad(p, bank):
        p['r0_tensors']['base.scores'][0, 0] += 1
    with pytest.raises(ValueError, match='mismatch'): m.compare_prefix_artifacts(ref, save(tmp_path, change=bad))


def test_supplied_source_identity_is_retained_and_cannot_change(tmp_path):
    def add(p, bank): p['report']['source_identity'] = {'case_id': 'case', 'asset_sha256': {'base': 'a'*64}}
    m = artifact_module(); ref = save(tmp_path, 'full_reference', add); prefix = save(tmp_path, change=add)
    result = m.compare_prefix_artifacts(ref, prefix)
    assert result['source_identity_scope'] == 'supplied_frozen_provenance_not_runtime_verified_checkpoint_or_style'
    def bad(p, bank): add(p, bank); p['report']['source_identity']['case_id'] = 'other'
    with pytest.raises(ValueError, match='identity'): m.compare_prefix_artifacts(ref, save(tmp_path, change=bad))


def test_builder_passes_frozen_source_identity_and_cli_never_submits(tmp_path, capsys):
    m = runner(); c = case()
    source = tmp_path/'case.json'; source.write_text(json.dumps(c))
    target = tmp_path/'graph.json'
    assert m.main(['build', '--case-config', str(source), '--bootstrap-manifest', str(tmp_path/'bootstrap.json'),
        '--trial-id', '5', '--prefix', 'frozen', '--output', str(target)]) == 0
    graph = json.loads(target.read_text())['prompt']
    pilots = [node for node in graph.values() if node['class_type'] == 'Krea2SliderFusePairedPrefixPilot']
    identities = [json.loads(node['inputs']['source_identity_json']) for node in pilots]
    assert identities[0] == identities[1] and identities[0]['case_id'] == c['case_id']
    assert len(identities[0]['base_graph_sha256']) == 64
    assert 'built_not_submitted' in capsys.readouterr().out
    assert m.main(['compare', '--reference', str(tmp_path/'missing'), '--prefix', str(tmp_path/'missing')]) == 2


@pytest.mark.parametrize('problem', ['raw_features', 'bootstrap', 'bootstrap_hash', 'r0_wrong', 'r0_missing'])
def test_missing_or_miscomputed_r0_cannot_be_saved(tmp_path, problem):
    p, bank = fixture()
    if problem == 'raw_features': del p['branches']['base']['step_3.block_18']['evidence_maps']['capture.attention_output']
    if problem == 'bootstrap': del p['bootstrap_masks']
    if problem == 'bootstrap_hash': p['report']['reference_partition_sha256']['target'] = '0'*64
    if problem == 'r0_wrong': p['r0_tensors']['anchor'].zero_()
    if problem == 'r0_missing': del p['r0_tensors']
    with pytest.raises(ValueError): artifact_module().save_prefix_artifacts(tmp_path, 'bad', p, bank)
    assert not list(tmp_path.iterdir())


def test_cold_fast_workflow_has_shared_two_forward_bootstrap_and_only_one_replay():
    graph = runner().build_validation_graph(case(), trial_id=7, prefix='cold', mode='fast')
    classes = [node['class_type'] for node in graph.values()]
    assert classes.count('Krea2SliderFuseExperimentalMaskCollect') == 1
    assert classes.count('Krea2SliderFusePairedPrefixPilot') == 1
    assert classes.count('Krea2SliderFusePredictionMixSampler') == 1
    assert classes.count('VAEDecode') == 1
    assert 'Krea2SliderFuseRetainedMask' not in classes
    collect = next(node['inputs'] for node in graph.values() if node['class_type'] == 'Krea2SliderFuseExperimentalMaskCollect')
    assert collect['seed'] == 42 and collect['steps'] == 8 and collect['model'] == ['2', 0]
    assert json.loads(collect['config_json'])['selected_steps'] == [1, 2]
    assert all(node['inputs']['pilot_mode'] == 'prefix' for node in graph.values() if node['class_type'] == 'Krea2SliderFusePairedPrefixPilot')


@pytest.mark.parametrize('problem', ['before_after', 'tensor_before_after', 'missing_rows', 'missing_hash'])
def test_observer_mutation_or_incomplete_tap_audit_cannot_pass(tmp_path, problem):
    p, bank = fixture()
    if problem == 'before_after': p['frames'][0]['taps'][0]['block_input_after_sha256'] = '1'*64
    if problem == 'tensor_before_after': p['frames'][0]['taps'][0]['tensor_sha256_after']['img_q'] = '1'*64
    if problem == 'missing_rows': p['frames'][0]['taps'].pop()
    if problem == 'missing_hash': del p['frames'][0]['taps'][0]['tensor_sha256_before']['img_q']
    with pytest.raises(ValueError): artifact_module().save_prefix_artifacts(tmp_path, 'bad', p, bank)


def test_same_replay_path_cannot_supply_two_independent_replays(tmp_path):
    a = save(tmp_path, 'full_reference'); b = save(tmp_path)
    replay = replay_report(tmp_path, 'prefix')
    with pytest.raises(ValueError, match='distinct|independent'):
        artifact_module().compare_prefix_artifacts(a, b, reference_replay=replay, prefix_replay=replay)


def test_replay_conditioning_is_bound_to_pilot(tmp_path):
    a = save(tmp_path, 'full_reference'); b = save(tmp_path)
    replays = [replay_report(tmp_path, mode) for mode in ('full_reference', 'prefix')]
    for path in replays:
        report = json.loads(path.read_text()); report['conditioning_sha256'] = 'e'*64
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='identity'):
        artifact_module().compare_prefix_artifacts(a, b, reference_replay=replays[0], prefix_replay=replays[1])


def test_replays_must_match_pilot_first_native_branch_predictions(tmp_path):
    from safetensors.torch import load_file
    a = save(tmp_path, 'full_reference'); b = save(tmp_path)
    replays = [replay_report(tmp_path, mode) for mode in ('full_reference', 'prefix')]
    for path in replays:
        report = json.loads(path.read_text()); tensor_path = path.parent/report['artifacts']['tensors']
        tensors = load_file(str(tensor_path)); tensors['trace_base_predictions'][0] += 1
        save_file(tensors, str(tensor_path))
        report['tensor_manifest']['trace_base_predictions']['sha256'] = tensor_hash(tensors['trace_base_predictions'])
        report['artifact_sha256']['tensors'] = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='first|prediction'):
        artifact_module().compare_prefix_artifacts(a, b, reference_replay=replays[0], prefix_replay=replays[1])


def test_both_routes_missing_original_instrumentation_blocks_still_fail(tmp_path):
    p, bank = fixture()
    for branch in p['branches'].values():
        for key in list(branch):
            if key.endswith(('block_23', 'block_26')): del branch[key]
    for frame in p['frames']:
        frame['taps'] = [row for row in frame['taps'] if row['block'] in (8, 10, 18)]
    with pytest.raises(ValueError, match='tap|observer|instrumentation'):
        artifact_module().save_prefix_artifacts(tmp_path, 'missing', p, bank)


def test_recorded_r0_cpu_execution_identity_cannot_differ(tmp_path):
    def first(p, bank): p['report']['r0_cpu_runtime'] = {'torch_num_threads': 1}
    def second(p, bank): p['report']['r0_cpu_runtime'] = {'torch_num_threads': 2}
    reference = save(tmp_path, 'full_reference', first)
    prefix = save(tmp_path, change=second)
    with pytest.raises(ValueError, match='runtime|identity'):
        artifact_module().compare_prefix_artifacts(reference, prefix)


@pytest.mark.parametrize('field,value', [('strength', 0), ('strength', True), ('strength', float('inf')),
                                       ('seed', -1), ('seed', True), ('seed', 2**64)])
def test_invalid_bounded_sampler_controls_are_rejected(tmp_path, field, value):
    p, bank = fixture(); p['report'][field] = value
    with pytest.raises(ValueError): artifact_module().save_prefix_artifacts(tmp_path, 'invalid', p, bank)
    assert not list(tmp_path.iterdir())
