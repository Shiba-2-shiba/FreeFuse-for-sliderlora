from contextlib import contextmanager
import importlib.util

import pytest
import torch

from test_native_pair import pair_environment


def test_observer_sees_selected_branch_same_input_and_always_closes(pair_environment):
    from slider_fuse.native_pair import NativePairRunner
    base, slider, events, _ = pair_environment
    base.model.diffusion_model=base.model
    records=[]
    class Observer:
        @contextmanager
        def capture(self, core, branch, step, x, sigma):
            records.append(("begin",branch,step,core.current_patcher.factor,x.clone()))
            try: yield
            finally: records.append(("end",branch,step))
    runner=NativePairRunner(base,slider,observer=Observer());x=torch.ones(1,1,2,2)
    try:
        a=runner.predict("base",x,torch.ones(1),positive=[],negative=[],model_options={},seed=42)
        b=runner.predict("slider",x,torch.ones(1),positive=[],negative=[],model_options={},seed=42)
        assert torch.equal(a,x*2) and torch.equal(b,x*6)
        assert torch.equal(records[0][4],records[2][4]) and records[0][3]==2 and records[2][3]==6
        assert records[1]==("end","base",1) and records[3]==("end","slider",1)
    finally:runner.close()


def test_observer_cleanup_on_forward_failure(pair_environment):
    from slider_fuse.native_pair import NativePairRunner
    base,slider,events,samplers=pair_environment;closed=[]
    base.model.diffusion_model=base.model
    class Observer:
        @contextmanager
        def capture(self,*args):
            try:yield
            finally:closed.append(True)
    samplers.sampling_function=lambda *a,**kw:(_ for _ in ()).throw(RuntimeError("forward failed"))
    runner=NativePairRunner(base,slider,observer=Observer())
    with pytest.raises(RuntimeError,match="forward failed"):
        try:runner.predict("base",torch.ones(1,1,2,2),torch.ones(1),positive=[],negative=[],model_options={},seed=1)
        finally:runner.close()
    assert closed==[True] and base.model.current_patcher is None


def test_trace_reader_rejects_path_escape_and_changed_bytes(tmp_path):
    assert importlib.util.find_spec("slider_fuse.paired_observation") is not None,"paired observation module missing"
    from slider_fuse.paired_observation import checked_diagnostic
    import json
    report=tmp_path/'bad.json';report.write_text(json.dumps({'complete':True,'artifacts':{'tensors':'../escape.safetensors'},'artifact_sha256':{'tensors':'x'}}))
    with pytest.raises(ValueError):checked_diagnostic(report,tmp_path)


def test_anchor_seeds_can_correct_false_late_protected_seed():
    assert importlib.util.find_spec("slider_fuse.anchored_masks") is not None,"anchored masks module missing"
    from slider_fuse.anchored_masks import anchored_candidate
    from slider_fuse.experimental_masks import _candidate,_candidate_bank,validate_experimental_config
    features=torch.tensor([[[1.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,1.]]])
    late=torch.tensor([[5.,0.,0.],[0.,5.,0.],[0.,5.,0.],[0.,5.,0.],[0.,0.,5.],[0.,0.,5.]])
    early=late.clone();early[1]=torch.tensor([5.,0.,0.])
    cfg=validate_experimental_config({'prototypes':1,'top_k_ratio':1.,'seed_confidence':.65,'seed_margin':.15})
    old=_candidate_bank(_candidate(features,late,('target','protected','background'),cfg,1),(2,3),cfg)
    new=_candidate_bank(anchored_candidate(features,late,early.softmax(-1),cfg),(2,3),cfg)
    assert old['masks']['protected'].flatten()[1]==1
    assert new['masks']['target'].flatten()[1]==1 and new['masks']['protected'].flatten()[1]==0


def test_decoder_applies_latent_transform_once_and_flattens_still_video():
    from slider_fuse import paired_observation as m
    assert hasattr(m,'decode_prediction'),"prediction decoder missing"
    from types import SimpleNamespace
    calls=[]
    model=SimpleNamespace(model=SimpleNamespace(process_latent_out=lambda x:(calls.append(True),x+1)[1]))
    vae=SimpleNamespace(decode=lambda x:x[:,None].permute(0,1,3,4,2))
    x=torch.zeros(1,3,2,2)
    a=m.decode_prediction(model,vae,x);b=m.decode_prediction(model,vae,x,processed=True)
    assert a.shape==(2,2,3) and torch.all(a==1) and torch.all(b==0)
    assert calls==[True] and not x.any()


def test_shape_refinement_can_unlock_nonseed_but_preserves_anchor_seeds():
    from slider_fuse.experimental_masks import _propagate,validate_experimental_config
    cfg=validate_experimental_config({'propagation_strength':1.,'affinity_threshold':0.})
    scores=torch.tensor([[1.,0.,0.],[.1,.8,.1],[0.,1.,0.],[0.,0.,1.]])
    candidate={'scores':scores,'roles':('target','protected','background'),'seeds':{
        'target':torch.tensor([True,False,False,False]),'protected':torch.tensor([False,False,True,False]),
        'background':torch.tensor([False,False,False,True])},'growth_allowed':torch.ones(4,3,dtype=torch.bool)}
    q=torch.tensor([[[[20.,0.]],[[20.,0.]],[[20.,0.]],[[20.,0.]]]])
    k=torch.tensor([[[[20.,0.]],[[0.,0.]],[[0.,0.]],[[0.,0.]]]])
    updated,info=_propagate(candidate,q,k,None,cfg,lock_confident=False)
    assert updated['scores'][1].argmax()==0
    assert torch.equal(updated['scores'][[0,2,3]],scores[[0,2,3]]) and info['seed_changes']==0


def test_retained_candidate_rejects_incomplete_and_changed_tensors(tmp_path):
    from slider_fuse import paired_observation as m
    assert hasattr(m,'read_retained_masks'),"checked retained mask reader missing"
    import json,hashlib
    from safetensors.torch import save_file
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps({'anchored_mask_schema':1,'complete':False}))
    with pytest.raises(ValueError,match='Incomplete'):m.read_retained_masks(path,tmp_path)
    target=torch.tensor([[[1.,0.],[1.,0.]]]);protected=1-target
    t=tmp_path/'maps.safetensors';save_file({'target':target,'protected':protected,'background':torch.zeros_like(target)},str(t))
    path.write_text(json.dumps({'anchored_mask_schema':1,'complete':True,'tensors':t.name,'tensors_sha256':hashlib.sha256(t.read_bytes()).hexdigest()}))
    masks=m.read_retained_masks(path,tmp_path);assert torch.equal(masks['target'],target)
    t.write_bytes(t.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='Changed'):m.read_retained_masks(path,tmp_path)
