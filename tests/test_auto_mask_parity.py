from dataclasses import replace

import torch

from test_runtime import environment,setup_run
from slider_fuse.sampling import sample_krea2
from slider_fuse.mask_collection import collect_auto_mask


def test_legacy_auto_and_new_collection_have_identical_raw_maps_and_partition(environment,tmp_path,monkeypatch):
    torch.manual_seed(81)
    model,positive,info,subjects,path=setup_run(tmp_path)
    subjects=tuple(replace(s,manual_mask=None) for s in subjects)
    modules=environment[0]
    # An untrained random toy core can legitimately produce an empty subject mask.
    # Use orthogonal concepts to test numerical parity, not segmentation quality.
    with torch.no_grad():
        for name in ("wq","wk","wv","gate","wo"):
            layer=getattr(model.model.diffusion_model.blocks[0].attn,name)
            layer.weight.copy_(torch.eye(4));layer.bias.zero_()
    positive[0][0].zero_();positive[0][0][0,0,0]=1;positive[0][0][0,1,1]=1
    fixed_noise=torch.zeros(1,4,4,4);fixed_noise[:,0,:,:2]=1;fixed_noise[:,1,:,2:]=1
    monkeypatch.setattr(modules["sample"],"prepare_noise",lambda *a:fixed_noise.clone())
    # The small boundary model uses a real noisy input at each collected step.
    def sample(patcher,noise,steps,cfg,sampler,scheduler,positive,negative,latent,**kwargs):
        for items in patcher.injections.values():
            for injection in items:injection.inject(patcher)
        x=noise.clone()
        for sigma,next_sigma in zip(kwargs["sigmas"][:-1],kwargs["sigmas"][1:]):
            args={"input":x,"timestep":sigma.reshape(1),"c":{"c_crossattn":positive[0][0]},"cond_or_uncond":[0]}
            p=patcher.model_options["model_function_wrapper"](
                lambda x,t,**c:patcher.model.diffusion_model(x,t,c["c_crossattn"]),args)
            p=p.repeat_interleave(2,-2).repeat_interleave(2,-1)
            x=x+(x-p)/sigma*(next_sigma-sigma)
        return x
    monkeypatch.setattr(modules["sample"],"sample",sample)
    image=torch.zeros(1,4,4,4)
    _,legacy,_=sample_krea2(model,positive,positive,info,subjects,{"samples":image},str(path),
        strength=4.,seed=42,steps=2,cfg=1.,mask_mode="auto",collect_step=1,collect_block=0,
        top_k_ratio=.3,temperature=4000.)
    noise=modules["sample"].prepare_noise(image,42,None)
    current,_=collect_auto_mask(model,positive,positive,info,subjects,image,noise,torch.tensor([1.,.5,0.]),
        grid=(2,2),seed=42,steps=2,cfg=1.,collect_step=1,collect_block=0,top_k_ratio=.3,
        temperature=4000.,fill_holes_max_area=0,mask_dilate_radius=0)
    for name in ("target","protected","background"):
        assert torch.equal(legacy["masks"][name],current["masks"][name])
    for name in ("target","protected"):
        assert torch.equal(legacy["raw_maps"][name].reshape(1,-1),current["raw_maps"][name].reshape(1,-1))
