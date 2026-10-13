"""CPU boundary evidence only; no native GPU parity or timing claim."""
from contextlib import nullcontext
import copy
import importlib.util
from types import SimpleNamespace

import pytest
import torch

from test_native_pair import pair_environment, guider_environment, create_guider


def prefix_module():
    from slider_fuse import paired_prefix
    return paired_prefix


class EulerBoundary:
    """Comfy sampler-shaped double with deliberately nonidentity output transforms."""
    def __init__(self):
        self.calls = []
        self.states = []

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        self.calls.append((sigmas.clone(), copy.deepcopy(extra_args)))
        # max_denoise and initial noise scaling depend on first sigma, not length.
        x = noise * sigmas[0] + latent_image
        for sigma, following in zip(sigmas[:-1], sigmas[1:]):
            p = model.predict_noise(x, sigma[None], **extra_args)
            x = x + (x-p) / sigma * (following-sigma)
            self.states.append(x.clone())
        return x * 7 + sigmas[-1]  # Deliberate inverse-noise scaling.


def native_boundary(env, monkeypatch, underlying, *, prefix):
    guider, report, recorder = create_guider(env)
    base = env[0]
    base.model.process_latent_out = lambda x: x + 11
    def inner(noise, latent_image, device, sampler, sigmas, denoise_mask,
              callback, disable_pbar, seed, latent_shapes=None):
        options = {'transformer_options': {'sample_sigmas': sigmas.clone()}}
        output = sampler.sample(guider, sigmas, {'model_options': options, 'seed': seed},
                                callback, noise, latent_image, denoise_mask, disable_pbar)
        return base.model.process_latent_out(output)
    monkeypatch.setattr(guider, 'inner_sample', inner)
    sampler = prefix_module().PrefixEulerSampler(underlying) if prefix else underlying
    return guider, report, recorder, sampler


def test_prefix_preserves_full_model_schedule_and_three_actual_updates(guider_environment, monkeypatch):
    schedule = torch.tensor([1., .94, .82, .69, .55, .4, .23, .1, 0.])
    noise = torch.arange(4).float().reshape(1,1,2,2)+1
    latent = torch.full_like(noise, .2)
    outputs = []
    for is_prefix in (False, True):
        underlying = EulerBoundary()
        guider, report, recorder, sampler = native_boundary(guider_environment, monkeypatch, underlying, prefix=is_prefix)
        value = guider.sample(noise, latent, sampler, schedule, seed=42)
        tensors = guider.finish_report()
        outputs.append((value, tensors, underlying))
        count = 3 if is_prefix else 8
        assert report['branch_nfe'] == {'base':count, 'slider':count}
        assert len(underlying.states) == count
        used, args = underlying.calls[0]
        assert torch.equal(used, schedule[:4] if is_prefix else schedule)
        assert torch.equal(args['model_options']['transformer_options']['sample_sigmas'], schedule)
        assert guider_environment[0].model.current_patcher is None
    for name in ('trace_inputs','trace_sigmas','trace_base_predictions','trace_slider_predictions','trace_predictions'):
        assert torch.equal(outputs[0][1][name][:3], outputs[1][1][name])
    assert torch.equal(outputs[0][2].states[2], outputs[1][2].states[2])
    assert not torch.equal(outputs[1][0], outputs[1][2].states[-1])
    assert schedule[3] > 0 and schedule[-1] == 0


@pytest.mark.parametrize('bad', [torch.linspace(1,0,4), torch.tensor([1., .9,.8,0.,.6,.5,.4,.2,0.]),
                               torch.tensor([1.,.9,.8,.7,.6,.5,.4,.2,float('nan')]),
                               torch.tensor([1.,.9,.8,.7,.6,.5,.4,.2,.1])])
def test_prefix_sampler_rejects_invalid_original_schedule(bad):
    called = []
    sampler = prefix_module().PrefixEulerSampler(SimpleNamespace(sample=lambda *a:called.append(True)))
    with pytest.raises(ValueError, match='schedule'):
        sampler.sample(None, bad, {}, None, torch.ones(1))
    assert not called


def test_prefix_failure_cleans_owned_patcher_and_conditioning(guider_environment, monkeypatch):
    class Broken(EulerBoundary):
        def sample(self, model, sigmas, extra_args, callback, noise, *args):
            model.predict_noise(noise, sigmas[0:1], **extra_args)
            raise RuntimeError('prefix failure')
    guider, report, _, sampler = native_boundary(guider_environment, monkeypatch, Broken(), prefix=True)
    with pytest.raises(RuntimeError, match='prefix failure'):
        guider.sample(torch.ones(1,1,2,2), torch.zeros(1,1,2,2), sampler, torch.linspace(1,0,9), seed=42)
    assert guider_environment[0].model.current_patcher is None
    assert ('cleanup_models',1) in guider_environment[2]


def make_probe():
    from slider_fuse.experimental_masks import validate_experimental_config
    cfg = validate_experimental_config({'prototypes':1, 'top_k_ratio':1., 'seed_confidence':.65, 'seed_margin':.15})
    features = torch.eye(3).repeat_interleave(2,0)
    logits = features * 5
    branches = {}
    for branch in ('base','slider'):
        taps = {}
        for step in (1,2,3):
            for block in (8,10,18):
                prefix = f'step_{step}.block_{block}'
                maps = {}
                for i, role in enumerate(('target','protected','background')):
                    maps[prefix+'.'+role+'.token_scores'] = logits[:,i][None]
                    maps[prefix+'.'+role+'.token_allowed_fraction'] = torch.ones(1,6)
                if step == 3 and block == 18:
                    maps['capture.attention_output'] = features.clone()
                taps[prefix] = {'evidence_maps':maps, 'algorithm':{'config':copy.deepcopy(cfg), 'grid':[2,3]}}
        branches[branch] = taps
    target = torch.tensor([[[1.,1.,0.],[0.,0.,0.]]])
    protected = torch.tensor([[[0.,0.,1.],[1.,0.,0.]]])
    bank = {'grid':(2,3), 'masks':{'target':target, 'protected':protected, 'background':1-target-protected}}
    return {'branches':branches, 'report':{'grid':[2,3]}}, bank


def test_r0_masks_match_legacy_anchor_oracle_and_fixed_bootstrap_protection():
    from slider_fuse.anchored_masks import anchored_candidate, compose_paired_masks
    from slider_fuse.experimental_masks import _candidate_bank
    probe, bootstrap = make_probe()
    original = copy.deepcopy(probe)
    bank, evidence = prefix_module().build_r0_masks(probe, bootstrap)
    tap = probe['branches']['base']['step_3.block_18']
    features = tap['evidence_maps']['capture.attention_output'][None]
    cfg = tap['algorithm']['config']
    logits = features[0] * 5
    reference = _candidate_bank(anchored_candidate(features, logits, logits.softmax(-1), cfg), (2,3), cfg)
    expected = compose_paired_masks(reference['masks']['target'],reference['masks']['target'],
                                   bootstrap['masks']['target'],bootstrap['masks']['protected'],include_pilot=False)
    for role in ('target','protected','background'):
        assert torch.equal(bank['masks'][role], expected[role])
    assert torch.equal(bank['selection_target_mask'],bank['masks']['target'])
    assert not bank['selection_added_mask'].any()
    assert torch.equal(evidence['anchor'],logits.softmax(-1))
    assert torch.equal(probe['branches']['base']['step_3.block_18']['evidence_maps']['capture.attention_output'],
                       original['branches']['base']['step_3.block_18']['evidence_maps']['capture.attention_output'])


def test_r0_missing_branch_and_masked_text_links_fail_closed():
    probe, bootstrap = make_probe()
    del probe['branches']['slider']
    with pytest.raises(ValueError, match='branches'):
        prefix_module().build_r0_masks(probe,bootstrap)
    probe, bootstrap = make_probe()
    probe['branches']['base']['step_3.block_8']['evidence_maps']['step_3.block_8.target.token_allowed_fraction'][0,0]=0
    with pytest.raises(ValueError, match='text links'):
        prefix_module().build_r0_masks(probe,bootstrap)


@pytest.mark.parametrize('changed', [dict(steps=3),dict(steps=True),dict(cfg=2.),dict(strength=0.),
    dict(mix_scope='all'),dict(mask_mode='auto'),dict(diagnostic_level='summary'),dict(selection_dilate_radius=1),
    dict(pilot_mode='typo')])
def test_pilot_controls_are_bounded_before_any_comfy_import(changed):
    options=dict(steps=8,cfg=1.,strength=4.,mix_scope='target_mask',mask_mode='manual',diagnostic_level='audit',
                 fill_holes_max_area=0,mask_dilate_radius=0,selection_dilate_radius=0,pilot_mode='prefix')
    options.update(changed)
    with pytest.raises(ValueError):
        prefix_module().validate_pilot_options(**options)

from test_runtime import environment, setup_run
from test_prediction_mix_auto import mix_runtime


@pytest.mark.parametrize('pilot_steps', [3,8])
def test_private_pilot_orchestration_keeps_original_schedule_and_discards_output(mix_runtime,environment,tmp_path,monkeypatch,pilot_steps):
    from slider_fuse import native_pair
    model, positive, info, subjects, path = setup_run(tmp_path)
    schedule = torch.linspace(1,0,9)
    scheduled_steps=[]; outer_schedules=[]
    def schedule_factory(base,steps,*args,**kwargs):
        scheduled_steps.append(steps)
        return SimpleNamespace(sigmas=schedule.clone())
    monkeypatch.setattr(environment[0]['samplers'],'KSampler',schedule_factory)
    maker = native_pair.make_prediction_mix_guider
    def wrap_maker(*args,observer=None,**kwargs):
        guider=maker(*args,**kwargs)
        original=guider.sample
        def sample(noise,image,sampler,sigmas,seed=None):
            outer_schedules.append(sigmas.clone())
            def run(model_wrap,used,extra,callback,n,lat,mask,disabled):
                return original(n,lat,'euler',used,seed=seed)
            delegate=SimpleNamespace(sample=run)
            if isinstance(sampler,prefix_module().PrefixEulerSampler):
                sampler.sampler=delegate
            else:sampler=delegate
            return sampler.sample(None,sigmas,{},None,noise,image,None,False)
        guider.sample=sample
        return guider
    monkeypatch.setattr(native_pair,'make_prediction_mix_guider',wrap_maker)
    observed=[]
    observer=SimpleNamespace(bind=lambda *a:observed.append(a[-1].clone()))
    latent={'samples':torch.zeros(1,4,4,4)}
    result,bank,payload=native_pair.sample_krea2_prediction_mix(model,positive,positive,info,subjects,
        latent,str(path),strength=4.,seed=42,steps=8,cfg=1.,mask_mode='manual',collect_block=0,
        _observer=observer,_pilot_steps=pilot_steps)
    assert result is None
    assert scheduled_steps==[8]
    assert torch.equal(outer_schedules[0],schedule) and torch.equal(observed[0],schedule)
    assert torch.equal(mix_runtime[0][2],schedule[:pilot_steps+1])
    assert payload.report['steps']==8 and payload.report['executed_steps']==pilot_steps
    assert payload.report['total_model_nfe']==pilot_steps*2
    assert payload.report['pilot_mode']==('prefix' if pilot_steps==3 else 'full_reference')
    assert not any(k.startswith('final_latent') for k in payload.report)
    assert 'diagnostic_schema_version' not in payload.report
    assert 'final_latent' not in payload.extra_tensors
    assert torch.equal(payload.extra_tensors['initial_latent'],latent['samples'])
    assert torch.equal(payload.extra_tensors['full_sigmas'],schedule)
    assert torch.equal(payload.extra_tensors['used_sigmas'],schedule[:pilot_steps+1])
    assert len(environment[2])==1


def test_private_pilot_rejects_unsupported_options_before_model_execution(environment,tmp_path):
    from slider_fuse import native_pair
    model, positive, info, subjects, path=setup_run(tmp_path)
    for options in ({'_pilot_steps':2},{'_pilot_steps':3,'steps':3},{'_pilot_steps':3,'strength':0.},
                    {'_pilot_steps':3,'mix_scope':'none'},{'_pilot_steps':3,'_observer':None}):
        kwargs=dict(strength=4.,seed=42,steps=8,_observer=object());kwargs.update(options)
        with pytest.raises(ValueError):
            native_pair.sample_krea2_prediction_mix(model,positive,positive,info,subjects,
                {'samples':torch.zeros(1,4,4,4)},str(path),**kwargs)
    assert not environment[1] and not environment[2]


def test_public_mask_pilot_owns_only_declared_trace_tensors_and_bootstrap(monkeypatch):
    from slider_fuse import native_pair, conditioning, paired_observation
    from slider_fuse.diagnostics import DiagnosticPayload
    from slider_fuse.sampling import tensor_hash
    template, bootstrap=make_probe()
    # Exercise the real wrapper and real observer finalizer with synthetic owned taps.
    for branch in template['branches']:
        for step in (1,2,3):
            for block in (23,26):
                template['branches'][branch][f'step_{step}.block_{block}']=copy.deepcopy(template['branches'][branch][f'step_{step}.block_18'])
    returned=[]
    def sample(*args,_observer,_pilot_steps,**kwargs):
        n=_pilot_steps;shape=(1,4,4,6)
        _observer.accumulators={b:SimpleNamespace(taps=t) for b,t in template['branches'].items()}
        _observer.frames=[{'branch':b,'step':s} for b in ('base','slider') for s in (1,2,3)]
        trace=torch.ones(n,*shape);schedule=torch.linspace(1,0,9)
        extra={'trace_inputs':trace,'trace_sigmas':schedule[:n], 'trace_base_predictions':trace,
            'trace_slider_predictions':trace,'trace_predictions':trace,
            'initial_noise':torch.ones(shape),'initial_latent':torch.zeros(shape),
            'full_sigmas':schedule,'used_sigmas':schedule[:n+1],
            **{'reference_'+r+'_mask':v for r,v in bootstrap['masks'].items()}}
        report={'grid':[2,3],'branch_nfe':{'base':n,'slider':n}}
        returned.append(extra)
        return None,bootstrap,DiagnosticPayload(report,trace[0],bootstrap['masks']['target'],extra)
    monkeypatch.setattr(native_pair,'sample_krea2_prediction_mix',sample)
    monkeypatch.setattr(conditioning,'resolve_phrase',lambda *args:(2,))
    subjects=(SimpleNamespace(id='target',positions=(0,)),SimpleNamespace(id='protected',positions=(1,)))
    bank,probe=prefix_module().sample_paired_pilot(object(),[],[],object(),subjects,{},'slider',
        background_phrase='room',strength=4.,seed=42,source_identity_json='{"case_id":"synthetic"}')
    expected={'initial_noise','initial_latent','full_sigmas','used_sigmas','trace_inputs','trace_sigmas',
              'trace_base_predictions','trace_slider_predictions','trace_predictions'}
    assert set(probe['validation_tensors'])==expected
    assert set(probe['bootstrap_masks'])=={'target','protected','background'}
    assert probe['report']['r0_cpu_runtime']=={'torch_num_threads':torch.get_num_threads(), 'torch_num_interop_threads':torch.get_num_interop_threads()}
    assert probe['report']['source_identity']=={'case_id':'synthetic'}
    assert probe['report']['source_identity_verification']=='supplied_frozen_provenance_not_runtime_verified'
    returned[0]['initial_noise'].zero_()
    assert probe['validation_tensors']['initial_noise'].any()
    probe['bootstrap_masks']['target'].zero_()
    assert bootstrap['masks']['target'].any()
    assert set(bank['masks'])=={'target','protected','background'}


@pytest.mark.parametrize('value',['[]','null','{"score":NaN}','{"case_id":"x","case_id":"y"}'])
def test_public_pilot_rejects_invalid_source_identity_before_generation(value):
    with pytest.raises(ValueError):
        prefix_module().sample_paired_pilot(None,None,None,None,None,None,None,
            background_phrase='room',source_identity_json=value)


def test_r0_does_not_reintroduce_bootstrap_only_target_support():
    probe, bootstrap=make_probe()
    # A false bootstrap-only cell is outside both paired target supports.
    bootstrap['masks']['target'][0,1,2]=1
    bootstrap['masks']['background'][0,1,2]=0
    bank,evidence=prefix_module().build_r0_masks(probe,bootstrap)
    assert bootstrap['masks']['target'][0,1,2]==1
    assert bank['masks']['target'][0,1,2]==0
    assert bank['masks']['background'][0,1,2]==1
    assert evidence['composition.removed_from_pilot'][0,1,2]==1
    assert torch.equal(bank['masks']['protected'],bootstrap['masks']['protected'])


def test_pilot_does_not_accept_boolean_cfg():
    with pytest.raises(ValueError,match='CFG1'):
        prefix_module().validate_pilot_options(cfg=True)


def test_public_pilot_rejects_nonstring_identity():
    with pytest.raises(ValueError,match='JSON'):
        prefix_module().sample_paired_pilot(None,None,None,None,None,None,None,
            background_phrase='room',source_identity_json=None)


def test_private_pilot_rejects_unexpected_nfe_and_still_unloads(mix_runtime,environment,tmp_path,monkeypatch):
    from slider_fuse import native_pair
    model,positive,info,subjects,path=setup_run(tmp_path)
    monkeypatch.setattr(environment[0]['samplers'],'KSampler',lambda *a,**k:SimpleNamespace(sigmas=torch.linspace(1,0,9)))
    original=native_pair.make_prediction_mix_guider
    monkeypatch.setattr(native_pair,'make_prediction_mix_guider',lambda *a,observer=None,**k:original(*a,**k))
    with pytest.raises(RuntimeError,match='expected Euler steps'):
        native_pair.sample_krea2_prediction_mix(model,positive,positive,info,subjects,
            {'samples':torch.zeros(1,4,4,4)},str(path),strength=4.,seed=42,steps=8,
            collect_block=0,_pilot_steps=3,_observer=SimpleNamespace(bind=lambda *a:None))
    assert len(environment[2])==1


def test_normal_sampler_constructor_keeps_original_inference_context(mix_runtime,environment,tmp_path,monkeypatch):
    from test_prediction_mix_auto import run
    seen=[]
    def make_sampler(name):
        seen.append(torch.is_inference_mode_enabled())
        return name
    monkeypatch.setattr(environment[0]['samplers'],'sampler_object',make_sampler)
    result,_,payload=run(tmp_path,auto=False)
    assert seen==[True]
    assert 'samples' in result and payload.report['steps']==2
    assert 'pilot_mode' not in payload.report
