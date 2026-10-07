"""ComfyUI boundary doubles; these are NOT native ComfyUI integration evidence."""
import copy
import sys
import types

import pytest
import torch
from torch import nn

from slider_fuse.conditioning import build_info, make_subjects
from slider_fuse.sampling import sample_krea2


class Attention(nn.Module):
    def __init__(self):
        super().__init__(); self.heads = self.kvheads = 2
        for name in ("wq", "wk", "wv", "gate", "wo"):
            setattr(self, name, nn.Linear(4, 4))
    def qknorm(self, q, k): return q, k
    def forward(self, x, freqs=None, mask=None, transformer_options=None):
        q, k, v = [getattr(self, n)(x).reshape(1, -1, 2, 2).transpose(1, 2) for n in ("wq", "wk", "wv")]
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, -1, 4)
        return self.wo(out * self.gate(x).sigmoid())


class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.attn = Attention(); self.prenorm = nn.Identity()
    def mod(self, vec):
        zeros = vec[:, None] * 0
        return (zeros, zeros, zeros, zeros, zeros, zeros)
    def forward(self, x, vec, freqs, mask=None, transformer_options=None):
        return x + self.attn(x, freqs, mask, transformer_options=transformer_options)


class TinyKrea(nn.Module):
    def __init__(self):
        super().__init__(); self.patch = 2; self.txtlayers = 12
        self.txtfusion = nn.Identity(); self.blocks = nn.ModuleList([Block()])
    def forward(self, x, sigma, context):
        text = self.txtfusion(context)
        img = x[:, :, ::2, ::2].flatten(2).transpose(1, 2)
        result = self.blocks[0](torch.cat([text, img], 1), sigma[:, None].expand(1, 4), None)
        return result[:, text.shape[1]:].transpose(1, 2).reshape(1, 4, 2, 2)


class Patcher:
    def __init__(self, core):
        self.model = types.SimpleNamespace(diffusion_model=core)
        self.model_options = {}; self.load_device = torch.device("cpu"); self.injections = {}
        self.patches = {}; self.forced_hooks = None; self.hook_patches = {}
    def clone(self):
        other = Patcher(self.model.diffusion_model)
        other.model_options = copy.deepcopy(self.model_options)
        return other
    def set_injections(self, name, items): self.injections[name] = items


@pytest.fixture
def environment(monkeypatch):
    calls = []; unloads = []
    comfy = types.ModuleType("comfy"); comfy.__path__ = []
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    modules = {}
    for name in ("sample", "samplers", "model_management", "patcher_extension", "utils"):
        module = types.ModuleType("comfy." + name)
        monkeypatch.setitem(sys.modules, module.__name__, module); setattr(comfy, name, module); modules[name] = module
    for name in ("ldm", "ldm.krea2", "ldm.flux", "ldm.modules"):
        module = types.ModuleType("comfy." + name); module.__path__ = []
        monkeypatch.setitem(sys.modules, module.__name__, module)
    native = types.ModuleType("comfy.ldm.krea2.model"); native.SingleStreamDiT = TinyKrea
    monkeypatch.setitem(sys.modules, native.__name__, native)
    math = types.ModuleType("comfy.ldm.flux.math"); math.apply_rope = lambda q, k, f: (q, k)
    monkeypatch.setitem(sys.modules, math.__name__, math)
    attention = types.ModuleType("comfy.ldm.modules.attention")
    attention.optimized_attention_masked = lambda q,k,v,heads,**kwargs: torch.nn.functional.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(1,-1,heads*q.shape[-1])
    monkeypatch.setitem(sys.modules, attention.__name__, attention)
    modules["sample"].fix_empty_latent_channels = lambda model, value, *args: value
    modules["sample"].prepare_noise = lambda value, seed, batch: torch.randn(value.shape, generator=torch.Generator().manual_seed(seed))
    modules["samplers"].KSampler = lambda *args, **kwargs: types.SimpleNamespace(sigmas=torch.tensor([1., .5, 0.]))
    modules["patcher_extension"].PatcherInjection = lambda inject, eject: types.SimpleNamespace(inject=inject, eject=eject)
    modules["model_management"].unload_model_and_clones = lambda patcher, **kwargs: (unloads.append(patcher), [i.eject(patcher) for items in patcher.injections.values() for i in items])
    def sample(patcher, noise, steps, cfg, sampler, scheduler, positive, negative, latent, **kwargs):
        for items in patcher.injections.values():
            for injection in items: injection.inject(patcher)
        calls.append((noise.clone(), latent.clone(), kwargs["sigmas"].clone()))
        for sigma in kwargs["sigmas"][:-1]:
            payload = {"input": latent, "timestep": sigma.reshape(1), "c": {"c_crossattn": positive[0][0]}, "cond_or_uncond": [0]}
            patcher.model_options["model_function_wrapper"](lambda x,t,**c: patcher.model.diffusion_model(x,t,c["c_crossattn"]), payload)
        return latent + .1
    modules["sample"].sample = sample
    return modules, calls, unloads


def setup_run(tmp_path):
    from dataclasses import replace
    from safetensors.torch import save_file
    core = TinyKrea(); patcher = Patcher(core)
    info = build_info("woman man", [(151644,1.),(90,1.),(151644,1.),(872,1.),(198,1.),(1,1.),(2,1.)],
                      lambda ids: "".join({1:"woman",2:" man"}[i] for i in ids))
    positive = [[torch.randn(1, 2, 4), {}]]; info = replace(info, _conditioning=positive)
    target = torch.zeros(1,4,4); target[:,:,:2] = 1
    protected = 1-target
    subjects = make_subjects(info,"woman","man",target_mask=target,protected_mask=protected)
    file = tmp_path / "slider.safetensors"
    data = {}
    for name in ("wq","wk","wv","gate","wo"):
        prefix = "lora_unet_blocks_0_attn_"+name
        data[prefix+".lora_down.weight"] = torch.ones(2,4)*.1
        data[prefix+".lora_up.weight"] = torch.ones(4,2)*.1
        data[prefix+".alpha"] = torch.tensor(2.)
    save_file(data,str(file))
    return patcher,positive,info,subjects,file


def test_manual_runtime_reports_all_keys_and_restores(environment, tmp_path):
    _,calls,unloads = environment
    model,positive,info,subjects,file = setup_run(tmp_path)
    core = model.model.diffusion_model
    originals = {name:m.forward for name,m in core.named_modules()}
    result,bank,report = sample_krea2(model,positive,positive,info,subjects,{"samples":torch.zeros(1,4,4,4)},str(file),
        strength=1.,seed=42,steps=2,cfg=1.,mask_mode="manual",collect_step=1,collect_block=0,top_k_ratio=.3,temperature=4000.)
    assert len(calls)==1 and len(unloads)==1
    assert report["matched_modules"]==5 and report["reached_modules"]==5
    assert report["adapter_groups"]=={"target":1,"protected":0,"background":0}
    assert not core.txtfusion._forward_hooks
    for name,module in core.named_modules(): assert module.forward==originals[name]
    assert not model.model_options


def test_runtime_exception_unloads_and_restores(environment,tmp_path):
    modules,calls,unloads=environment
    model,positive,info,subjects,file=setup_run(tmp_path)
    core=model.model.diffusion_model; original=core.blocks[0].attn.wq.forward
    def fail(*args,**kwargs): raise RuntimeError("intentional")
    modules["sample"].sample=fail
    with pytest.raises(RuntimeError,match="intentional"):
        sample_krea2(model,positive,positive,info,subjects,{"samples":torch.zeros(1,4,4,4)},str(file),
            strength=1.,seed=42,steps=2,cfg=1.,mask_mode="manual",collect_step=1,collect_block=0,top_k_ratio=.3,temperature=4000.)
    assert len(unloads)==1 and core.blocks[0].attn.wq.forward==original
    assert not core.txtfusion._forward_hooks


def test_auto_runtime_collects_both_subjects_with_no_protected_adapter(environment,tmp_path,monkeypatch):
    from slider_fuse import sampling
    from slider_fuse.masks import manual_masks
    _,calls,unloads=environment
    model,positive,info,subjects,file=setup_run(tmp_path)
    from dataclasses import replace
    auto_subjects=tuple(replace(s,manual_mask=None) for s in subjects)
    observed=[]
    def masks(maps,grid):
        observed.append(set(maps))
        bank=manual_masks(subjects[0].manual_mask,subjects[1].manual_mask,grid)
        bank["mode"]="auto"
        return bank
    monkeypatch.setattr(sampling,"generate_masks",masks)
    _,bank,report=sample_krea2(model,positive,positive,info,auto_subjects,{"samples":torch.zeros(1,4,4,4)},str(file),
        strength=1.,seed=42,steps=2,cfg=1.,mask_mode="auto",collect_step=1,collect_block=0,top_k_ratio=.3,temperature=4000.)
    assert observed==[{"target","protected"}] and len(calls)==2
    assert torch.equal(calls[0][0],calls[1][0]) and torch.equal(calls[0][1],calls[1][1])
    assert report["phase1_nfe"]==1 and report["phase2_nfe"]==2
    assert report["observation"]["sigma"]==1.
    assert report["observation"]["cap_len"]==2
    assert report["adapter_groups"]["protected"]==0


def test_observe_only_is_output_and_rng_neutral(environment):
    from slider_fuse.attention import AttentionCollector
    from slider_fuse.lora import RoutingState
    core=TinyKrea(); image=torch.randn(1,4,4,4); context=torch.randn(1,2,4)
    sigma=torch.ones(1); baseline=core(image,sigma,context)
    collector=AttentionCollector({"target":(0,),"protected":(1,)},(2,2),1,0,.3,4000.,torch.tensor([1.,0.]))
    state=RoutingState(); collector.active=True; collector.begin_forward(sigma,2); collector.install(core,state)
    rng=torch.random.get_rng_state()
    try:
        actual=core(image,sigma,context)
        assert torch.equal(actual,baseline)
        assert torch.equal(rng,torch.random.get_rng_state())
        assert set(collector.maps)=={"target","protected"}
    finally: collector.remove()
    assert not core.blocks[0]._forward_pre_hooks


@pytest.mark.parametrize("kind",["injection","active","skip","hook","pre_hook","forward"])
def test_foreign_injections_and_hooks_are_rejected_without_any_mutation(environment,tmp_path,kind):
    _,calls,unloads=environment
    model,positive,info,subjects,file=setup_run(tmp_path)
    core=model.model.diffusion_model; module=core.blocks[0].attn.wq
    handle=None
    if kind=="injection": model.injections["foreign"]=[types.SimpleNamespace(inject=lambda p:None,eject=lambda p:None)]
    if kind=="active": model.is_injected=True
    if kind=="skip": model.skip_injection=True
    if kind=="hook": handle=module.register_forward_hook(lambda m,a,o:o)
    if kind=="pre_hook": handle=module.register_forward_pre_hook(lambda m,a:None)
    if kind=="forward":
        original=module.forward
        module.forward=lambda x:original(x)
    before=module.forward; before_hooks=dict(module._forward_hooks); before_pre=dict(module._forward_pre_hooks)
    try:
        with pytest.raises(ValueError,match="unsupported"):
            sample_krea2(model,positive,positive,info,subjects,{"samples":torch.zeros(1,4,4,4)},str(file),
                strength=1.,seed=42,steps=2,cfg=1.,mask_mode="manual",collect_step=1,collect_block=0,top_k_ratio=.3,temperature=4000.)
        assert not calls and not unloads
        assert module.forward is before or module.forward==before
        assert dict(module._forward_hooks)==before_hooks and dict(module._forward_pre_hooks)==before_pre
    finally:
        if handle is not None: handle.remove()


def test_phase_two_uses_processed_mask_and_returns_cpu_snapshots(environment,tmp_path,monkeypatch):
    from dataclasses import replace
    from slider_fuse import sampling
    _,calls,_=environment
    model,positive,info,subjects,file=setup_run(tmp_path)
    auto_subjects=tuple(replace(s,manual_mask=None) for s in subjects)
    target=torch.tensor([[[1.,0.],[0.,0.]]]);protected=torch.tensor([[[0.,0.],[0.,1.]]])
    original={"grid":(2,2),"mode":"auto","masks":{"target":target,"protected":protected,"background":1-target-protected}}
    monkeypatch.setattr(sampling,"generate_masks",lambda maps,grid:original)
    used=[]
    class ObservedHook(sampling.SliderHook):
        def forward(self,*args,**kwargs):
            if self.state.phase=="route":used.append(self.state.mask.clone())
            return super().forward(*args,**kwargs)
    monkeypatch.setattr(sampling,"SliderHook",ObservedHook)
    _,bank,report=sample_krea2(model,positive,positive,info,auto_subjects,{"samples":torch.zeros(1,4,4,4)},str(file),
        strength=1.,seed=42,steps=2,cfg=1.,mask_mode="auto",collect_step=1,collect_block=0,top_k_ratio=.3,temperature=4000.,
        fill_holes_max_area=0,mask_dilate_radius=1)
    expected=1-protected
    assert used and all(torch.equal(mask,expected) for mask in used)
    assert torch.equal(bank["masks"]["target"],expected)
    assert torch.equal(bank["original_masks"]["target"],target)
    assert bank["added_target_mask"].sum()==2
    assert report["mask_postprocess"]["dilated_token_count"]==2
    assert report["mask_postprocess"]["dilation_protected_blocked_count"]==1
    assert all(m.device.type=="cpu" and not m.requires_grad for m in bank["original_masks"].values())
    assert bank["added_target_mask"].device.type=="cpu"
    assert torch.equal(calls[0][0],calls[1][0]) and torch.equal(calls[0][1],calls[1][1])
    assert torch.equal(original["masks"]["target"],target)


def test_manual_postprocess_rejects_before_model_mutation(environment,tmp_path):
    _,calls,unloads=environment
    model,positive,info,subjects,file=setup_run(tmp_path)
    with pytest.raises(ValueError,match="auto"):
        sample_krea2(model,positive,positive,info,subjects,{"samples":torch.zeros(1,4,4,4)},str(file),
            strength=1.,seed=42,steps=2,cfg=1.,mask_mode="manual",collect_step=1,collect_block=0,top_k_ratio=.3,temperature=4000.,
            fill_holes_max_area=8)
    assert not calls and not unloads
