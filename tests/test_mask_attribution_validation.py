import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'docs' / 'run_mask_attribution_validation.py'


def runner():
    assert MODULE.is_file(), 'attribution runner not implemented'
    spec = importlib.util.spec_from_file_location('attribution_runner_test', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freeze_rejects_existing_output_without_touching_it(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    target = tmp_path/'out'; target.mkdir(); (target/'marker').write_text('keep')
    with pytest.raises(ValueError, match='exist'):
        runner().freeze_experiment(source, target, 1)
    assert (target/'marker').read_text() == 'keep'


def test_freeze_rejects_missing_evidence_before_creating_output(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    target = tmp_path/'out'
    with pytest.raises((ValueError, FileNotFoundError)):
        runner().freeze_experiment(source, target, 1)
    assert not target.exists()


def test_frozen_source_validation_accepts_json_path_strings_and_detects_changes(tmp_path):
    import hashlib
    path=tmp_path/'evidence';path.write_bytes(b'original')
    experiment={'source_files':{str(path):hashlib.sha256(b'original').hexdigest()}}
    runner().validate_frozen_sources(experiment)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed frozen source'):
        runner().validate_frozen_sources(experiment)


def test_runner_reads_retained_json_path_strings(tmp_path):
    path=tmp_path/'report.json';path.write_text('{"complete":true}')
    assert runner().read(str(path)) == {'complete':True}


def test_candidate_builder_reads_json_path_strings(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE.parent))
    path=tmp_path/'report.json';path.write_text('{"complete":true}')
    spec=importlib.util.spec_from_file_location('attribution_builder_test', MODULE.with_name('build_paired_mask_candidates.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.read(str(path)) == {'complete':True}


@pytest.mark.parametrize('key,value',[('sampler','heun'),('scheduler','normal'),('cfg',2),('steps',9)])
def test_source_controls_reject_unsupported_sampler_schedule_cfg_and_steps(key,value):
    report={'sampler':'euler','scheduler':'simple','cfg':1,'steps':8,'effective_slider_strength':4,
            'grid':[64,64],'prediction_selection':{'dilate_radius':0},
            'mask_generation':{'fill_holes_max_area':0,'mask_dilate_radius':0}}
    report[key]=value
    with pytest.raises(ValueError,match='controls'):
        runner().validate_generation_controls(report)


def test_stage_graph_pins_runtime_conditions_and_uses_retained_candidates(tmp_path):
    m = runner()
    graph = {str(n): {'class_type': 'TestSource', 'inputs': {}} for n in (1,2,3,4,5,7,9)}
    old = {'6': {'inputs': {'target_phrase': 'woman', 'protected_phrase': 'man'}},
           '21': {'inputs': {'seed': 42, 'steps': 8, 'cfg': 1, 'strength': 4,
                  'sampler_name': 'euler', 'scheduler': 'simple', 'latent': ['7', 0],
                  'positive': ['4', 0], 'negative': ['5', 0], 'prompt_info': ['4', 1],
                  'lora_name': 'slider', 'selection_dilate_radius': 0,
                  'fill_holes_max_area': 0, 'mask_dilate_radius': 0}}}
    prompt, candidates = m.build_stage_graph(graph, old, [{'label': 'S0', 'status': 'ok', 'manifest': str(tmp_path/'mask.json')}],
                                            case='overlap', trial_id=123, stamp='run', stage='factorial')
    inputs = prompt[candidates[0]['sampler_node']]['inputs']
    assert inputs['model'] == ['2',0] and inputs['trial_id'] == 123
    assert inputs['mask_mode'] == 'manual' and inputs['mix_scope'] == 'target_mask'
    assert inputs['seed'] == 42 and inputs['steps'] == 8 and inputs['strength'] == 4
    assert all(inputs[k] == 0 for k in ('selection_dilate_radius','fill_holes_max_area','mask_dilate_radius'))
    assert prompt['103']['class_type'] == 'Krea2SliderFuseRetainedMask'


def test_stage_rejects_duplicate_and_failed_candidates():
    m = runner()
    for entries in ([{'label':'S0','status':'failed'}], [{'label':'S0','status':'ok','manifest':'x'}]*2):
        with pytest.raises(ValueError): m.build_stage_graph({}, {}, entries, case='x', trial_id=1, stamp='x', stage='factorial')
