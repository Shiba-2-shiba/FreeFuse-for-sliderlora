import copy
import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'docs/run_seed_bootstrap_validation.py'


def runner():
    assert MODULE.is_file(), 'seed bootstrap runner not implemented'
    spec = importlib.util.spec_from_file_location('seed_bootstrap_test', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case():
    graph = {str(i): {'class_type': 'TestSource', 'inputs': {}} for i in (1,2,3,4,5,7,9)}
    graph['2']['class_type'] = 'LoraLoaderModelOnly'  # Legitimate style, not target Slider.
    graph['2']['inputs'] = {'model': ['1',0], 'lora_name': 'style', 'strength_model': 1.}
    graph['6'] = {'class_type': 'Krea2SliderFuseSubjects', 'inputs': {'target_phrase': 'woman', 'protected_phrase': 'man', 'prompt_info': ['4',1]}}
    graph['8'] = {'class_type': 'Krea2SliderFuseExperimentalMaskCollect', 'inputs': {
        'model': ['2',0], 'subjects': ['6',0], 'seed': 42, 'latent': ['7',0], 'positive': ['4',0], 'negative': ['5',0], 'background_phrase': 'room'}}
    graph['21'] = {'class_type': 'Krea2SliderFusePredictionMixSampler', 'inputs': {
        'model': ['2',0], 'subjects': ['20',0], 'seed': 42, 'steps': 8, 'cfg': 1., 'strength': 4.,
        'latent': ['7',0], 'positive': ['4',0], 'negative': ['5',0], 'prompt_info': ['4',1], 'lora_name': 'slider'}}
    graph['60'] = {'class_type': 'Krea2SliderFuseRetainedMask', 'inputs': {'report_path': 'old_D.json'}}
    return {'case_id': 'park__seed_123', 'prompt_case': 'park', 'seed': 123, 'base_graph': graph,
            'collector_config': {'collect_step': 2, 'collect_block': 18, 'selected_steps':[1,2], 'selected_blocks':[16,18]},
            'slider_path': 'slider', 'asset_sha256': {}, 'role_positions': {}, 'background_phrase': 'room'}


def test_bootstrap_and_pilot_have_no_d_mask_dependency(tmp_path):
    m, c = runner(), case()
    bootstrap = m.build_bootstrap_graph(c, trial_id=14, prefix='test_123')
    pilot = m.build_pilot_graph(c, tmp_path/'new_B.json', trial_id=15, prefix='test_124')
    assert sum(n['class_type']=='LoraLoaderModelOnly' for n in bootstrap.values()) == 1
    assert not any(n['class_type']=='Krea2SliderFuseRetainedMask' for n in bootstrap.values())
    assert bootstrap['8']['inputs']['model'] == ['2',0]
    assert 'target_mask' not in bootstrap['6']['inputs']
    assert pilot['60']['inputs']['report_path'] == str(tmp_path/'new_B.json')
    assert pilot['100']['inputs']['mix_scope'] == 'target_mask'
    assert pilot['100']['class_type'] == 'Krea2SliderFuseObservedPredictionMix'
    assert 'old_D' not in str(bootstrap) + str(pilot)


def test_holdout_seed_is_applied_without_mutating_prompt_template(tmp_path):
    m, c = runner(), case()
    original = copy.deepcopy(c)
    graphs = [m.build_bootstrap_graph(c, trial_id=14, prefix='case123'), m.build_pilot_graph(c,tmp_path/'B.json',trial_id=15,prefix='case123p')]
    assert graphs[0]['8']['inputs']['seed'] == graphs[1]['100']['inputs']['seed'] == 123
    assert c == original
    ids = {m.case_identity(prompt, seed) for prompt in ('park','overlap') for seed in (123,777)}
    assert len(ids) == 4


def passing_gates(variant):
    return {'cases': {c: {'A-'+variant: {'gates':{f'G{i}':'pass' for i in range(6)}, 'major_anatomy_defect':False}}
                     for c in ('park__seed_444444', 'woman_front_strong_overlap__seed_42')}}


def test_unknown_gate_and_failed_selected_variant_block_holdout():
    m = runner()
    g = passing_gates('R2')
    assert m.evaluate_holdout_eligibility({'selected_variant':'R2'}, g)['eligible']
    for bad in ('unknown','fail','not_applicable'):
        g['cases']['park__seed_444444']['A-R2']['gates']['G1'] = bad
        assert not m.evaluate_holdout_eligibility({'selected_variant':'R2'}, g)['eligible']
    g = passing_gates('R0')
    assert not m.evaluate_holdout_eligibility({'selected_variant':'R2'}, g)['eligible']
    assert m.evaluate_holdout_eligibility({'selected_variant':None}, g)['mode'] == 'path_only'
    for bad in (True, None):
        g['cases']['park__seed_444444']['A-R0']['major_anatomy_defect'] = bad
        assert not m.evaluate_holdout_eligibility({'selected_variant':None}, g)['eligible']


def test_resume_reuses_verified_run_and_rejects_changed_identity(tmp_path):
    m = runner()
    output=tmp_path/'output.json'; output.write_text('{"complete":true}')
    artifact=tmp_path/'artifacts.json'
    m.write_new(artifact, {'identity':{'case_id':'x','phase':'bootstrap','variant':'base','attempt_id':1},
        'status':'complete','file_sha256':{str(output):m.sha(output)},'generation_nfe':2})
    identity={'case_id':'x','phase':'bootstrap','variant':'base','attempt_id':1}
    assert m.verified_resume(artifact, identity)['generation_nfe']==2
    with pytest.raises(ValueError,match='identity'):
        m.verified_resume(artifact,dict(identity,case_id='other'))
    output.write_text('changed')
    with pytest.raises(ValueError,match='Changed'):
        m.verified_resume(artifact,identity)


def test_freeze_rejects_existing_directory_before_reading_source(tmp_path):
    m=runner(); dest=tmp_path/'existing';dest.mkdir()
    with pytest.raises(ValueError, match='existing'):
        m.freeze_experiment(tmp_path/'absent', dest, 1, 'http://127.0.0.1:8000')


def test_invalid_trial_and_unsafe_prefix_are_rejected(tmp_path):
    m,c=runner(),case()
    for trial in (-1, 2**31, True):
        with pytest.raises(ValueError):m.build_bootstrap_graph(c,trial_id=trial,prefix='safe')
    with pytest.raises(ValueError):m.build_pilot_graph(c,tmp_path/'B.json',trial_id=1,prefix='../bad')


def test_annotation_rejects_wrong_rgb_and_overlapping_roles(tmp_path):
    m=runner();image=tmp_path/'rgb';image.write_bytes(b'actual RGB')
    reference={'image':str(image),'image_sha256':m.sha(image),'branch':'base'}
    spec={'image_sha256':'different','regions':{'target_face':[[1,1,2,2]]}}
    with pytest.raises(ValueError,match='RGB'):m.regions_from_rectangles(reference,spec)
    spec['image_sha256']=m.sha(image)
    spec['regions']['protected_face']=[[1,1,2,2]]
    with pytest.raises(ValueError,match='overlap'):m.regions_from_rectangles(reference,spec)
    del spec['regions']['protected_face']
    result=m.regions_from_rectangles(reference,spec)
    assert sum(result['target_face'])==1 and result['target_full'] is None


def test_gates_require_numeric_evidence_and_image_identity():
    m=runner();review={'case_id':'case','label':'A-R0','image_sha256':'rgb','gates':{f'G{i}':'pass' for i in range(1,6)},'major_anatomy_defect':False,'notes':'examined RGB'}
    with pytest.raises(ValueError,match='image'):m.combine_gate_review(review,'other',False)
    row=m.combine_gate_review(review,'rgb',False)
    assert row['gates']['G0']=='fail' and row['gates']['G1']=='pass'
    assert m.combine_gate_review(review,'rgb',True)['gates']['G0']=='pass'


def test_completed_holdout_resume_never_submits_again(tmp_path,monkeypatch):
    m=runner();p=tmp_path/'experiment.json';template=case();case_row={'case_id':'park__seed_123','prompt_case':'park','seed':123}
    m.write_new(p,{'output_root':str(tmp_path),'cases':[template],'holdout_cases':[case_row]})
    m.write_new(tmp_path/'selection.json',{'selected_variant':None})
    gates=passing_gates('R0');gates['selection_sha256']=m.sha(tmp_path/'selection.json');m.write_new(tmp_path/'gates.json',gates)
    frozen={'selection_sha256':m.sha(tmp_path/'selection.json'),'gates_sha256':m.sha(tmp_path/'gates.json'),
            'cases':[case_row],'eligibility':m.evaluate_holdout_eligibility({'selected_variant':None},gates)}
    m.write_new(tmp_path/'holdout-conditions.json',frozen)
    evidence=tmp_path/'evidence';evidence.write_text('stable')
    for folder,phase,variant,nfe in [('bootstrap','bootstrap','base',2),('pilot_A','pilot','A',16),
            ('replay_A-R0','replay','A-R0',16),('trace_pilot_A','traces','pilot_A',0),('trace_replay_A-R0','traces','replay_A-R0',0)]:
        dest=tmp_path/case_row['case_id']/folder;dest.mkdir(parents=True)
        m.write_new(dest/'artifacts.json',{'status':'complete','identity':m._identity(case_row,phase,variant),
            'file_sha256':{str(evidence):m.sha(evidence)},'generation_nfe':nfe})
    saved={'status':'executed','conditions_sha256':m.sha(tmp_path/'holdout-conditions.json'),
           'cases':[{**case_row,'status':'generated','generation_nfe':34,'assessment':'awaiting_annotation'}],'generation_nfe':34}
    m.write_new(tmp_path/'holdout.json',saved)
    def forbid(*args,**kwargs):raise AssertionError('completed holdout submitted again')
    monkeypatch.setattr(m,'run_bootstrap_case',forbid)
    assert m.run_holdout(p)==saved


def test_ambiguous_submission_intent_is_never_posted_again(tmp_path,monkeypatch):
    m=runner();graph={'100':{'class_type':'Output','inputs':{}}}
    m.write_new(tmp_path/'queued_api.json',{'prompt':graph})
    m.write_new(tmp_path/'submission-intent.json',{'prompt_id':'12345678-1234-4234-8234-123456789abc',
        'queued_api_sha256':m.sha(tmp_path/'queued_api.json')})
    monkeypatch.setattr(m.rt,'preflight',lambda *a:{'output_node_count':1})
    monkeypatch.setattr(m.rt,'format_preflight_issues',lambda *a:[])
    def request(url,**kwargs):
        if kwargs.get('method')=='POST':raise AssertionError('ambiguous submission was retried')
        return {} if '/history/' in url else {'queue_running':[],'queue_pending':[]}
    monkeypatch.setattr(m.rt,'request_json',request)
    with pytest.raises(ValueError,match='Unresolved submission'):
        m.execute_graph({'server':'http://example.invalid'},tmp_path,graph)


def test_submission_intent_recovers_known_history_without_post(tmp_path,monkeypatch):
    m=runner();graph={'100':{'class_type':'Output','inputs':{}}};pid='12345678-1234-4234-8234-123456789abc'
    m.write_new(tmp_path/'queued_api.json',{'prompt':graph})
    m.write_new(tmp_path/'submission-intent.json',{'prompt_id':pid,'queued_api_sha256':m.sha(tmp_path/'queued_api.json')})
    monkeypatch.setattr(m.rt,'preflight',lambda *a:{'output_node_count':1})
    monkeypatch.setattr(m.rt,'format_preflight_issues',lambda *a:[])
    entry={'prompt':[1,pid,graph,{},['100']],'status':{'status_str':'success','completed':True,'messages':[]}}
    def request(url,**kwargs):
        assert kwargs.get('method')!='POST'
        return {pid:entry}
    monkeypatch.setattr(m.rt,'request_json',request)
    assert m.execute_graph({'server':'http://example.invalid'},tmp_path,graph)==entry
    assert m.read(tmp_path/'submission.json')['recovered_from_intent'] is True
