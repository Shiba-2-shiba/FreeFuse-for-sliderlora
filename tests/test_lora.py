from types import SimpleNamespace

import pytest
import torch
from torch import nn

from slider_fuse.lora import Adapter, RoutingState, SliderHook, load_adapters, core_guard, compare_probe_outputs


def small_model():
    model = nn.Module()
    block = nn.Module(); block.attn = nn.Module()
    for name in ("wq", "wk", "wv", "gate", "wo"):
        setattr(block.attn, name, nn.Linear(3, 4, bias=False))
    model.blocks = nn.ModuleList([block])
    return model


def weights():
    result = {}
    for name in ("wq", "wk", "wv", "gate", "wo"):
        prefix = "lora_unet_blocks_0_attn_" + name
        result[prefix + ".lora_down.weight"] = torch.ones(2, 3)
        result[prefix + ".lora_up.weight"] = torch.ones(4, 2)
        result[prefix + ".alpha"] = torch.tensor(1.)
    return result


def test_all_five_layers_map_and_alpha_is_applied_once():
    adapters = load_adapters(weights(), small_model())
    assert len(adapters) == 5
    assert adapters["blocks.0.attn.gate"].scale == .5


@pytest.mark.parametrize("mode", ["unknown", "missing", "shape", "nonfinite"])
def test_no_keys_silently_discarded(mode):
    data = weights()
    if mode == "unknown": data["lora_unet_txtmlp_1.lora_up.weight"] = torch.ones(4, 2)
    if mode == "missing": data.pop("lora_unet_blocks_0_attn_wq.lora_up.weight")
    if mode == "shape": data["lora_unet_blocks_0_attn_wq.lora_down.weight"] = torch.ones(2, 8)
    if mode == "nonfinite": data["lora_unet_blocks_0_attn_wq.alpha"] = torch.tensor(float('nan'))
    with pytest.raises(ValueError): load_adapters(data, small_model())


@pytest.mark.parametrize("strength", [-1., 0., .7])
def test_only_target_image_tokens_receive_delta(strength):
    torch.manual_seed(5)
    base = nn.Linear(3, 4, bias=False); x = torch.randn(1, 6, 3)
    adapter = Adapter(torch.randn(2, 3), torch.randn(4, 2), 1.)
    state = RoutingState(); state.cap_len = 2
    state.mask = torch.tensor([[[1., 0.], [0., 1.]]]); state.phase = "route"
    original = base.forward; expected = original(x)
    hook = SliderHook(base, adapter, strength, state, "blocks.0.attn.wq")
    hook.inject()
    try:
        result = base(x)
        delta = (x @ adapter.down.T @ adapter.up.T) * .5 * strength
        mask = torch.tensor([0., 0., 1., 0., 0., 1.]).reshape(1, 6, 1)
        torch.testing.assert_close(result, expected + delta * mask)
        assert torch.equal(result[:, [0, 1, 3, 4]], expected[:, [0, 1, 3, 4]])
    finally: hook.eject()
    assert base.forward == original


def test_zero_mask_matches_base_and_collection_is_off():
    base = nn.Linear(3, 4); x = torch.randn(1, 6, 3); expected = base(x)
    state = RoutingState(); state.cap_len = 2; state.mask = torch.zeros(1, 2, 2)
    hook = SliderHook(base, Adapter(torch.ones(2, 3), torch.ones(4, 2), 2.), 1., state, "wq")
    hook.inject()
    try:
        assert torch.equal(base(x), expected)
        state.phase = "route"
        assert torch.equal(base(x), expected)
    finally: hook.eject()


def test_all_one_image_mask_keeps_text_unchanged():
    base = nn.Linear(3, 4); x = torch.randn(1, 6, 3); expected = base(x)
    state = RoutingState(); state.cap_len = 2; state.mask = torch.ones(1, 2, 2); state.phase = "route"
    hook = SliderHook(base, Adapter(torch.ones(2, 3), torch.ones(4, 2), 2.), 1., state, "wo")
    hook.inject()
    try:
        result = base(x)
        assert torch.equal(result[:, :2], expected[:, :2])
        torch.testing.assert_close(result[:, 2:], expected[:, 2:] + x[:, 2:] @ torch.ones(3, 2) @ torch.ones(2, 4))
    finally: hook.eject()


def test_mask_length_mismatch_fails_and_restores():
    base = nn.Linear(3, 4); state = RoutingState(); state.cap_len = 1
    state.mask = torch.ones(1, 2, 2); state.phase = "route"
    hook = SliderHook(base, Adapter(torch.ones(2, 3), torch.ones(4, 2), 2.), 1., state, "wo")
    original = base.forward
    hook.inject()
    try:
        with pytest.raises(ValueError, match="sequence"):
            base(torch.ones(1, 6, 3))
    finally: hook.eject()
    assert base.forward == original


def test_shared_core_guard_rejects_before_mutation_and_releases_after_error():
    core = small_model(); other = small_model()
    a = SimpleNamespace(model=SimpleNamespace(diffusion_model=core))
    b = SimpleNamespace(model=SimpleNamespace(diffusion_model=core))
    assert a.model.diffusion_model is b.model.diffusion_model
    with pytest.raises(RuntimeError, match="intentional"):
        with core_guard(core):
            with pytest.raises(RuntimeError, match="already"):
                with core_guard(b.model.diffusion_model): pass
            with core_guard(other): pass
            raise RuntimeError("intentional")
    with core_guard(core): pass


def test_integer_base_storage_and_float_adapter_are_preserved():
    class IntegerBase(nn.Module):
        def __init__(self):
            super().__init__(); self.register_buffer("weight", torch.arange(12, dtype=torch.int8).reshape(4, 3))
            self.register_buffer("weight_scale", torch.tensor(.1)); self.quant_format="int8_tensorwise"
        def forward(self, x): return x @ (self.weight.float() * self.weight_scale).T
    base=IntegerBase(); packed=base.weight.clone(); scale=base.weight_scale.clone()
    x=torch.randn(1,6,3); expected=base(x)
    adapter=Adapter(torch.ones(2,3),torch.ones(4,2),2.)
    state=RoutingState(phase="route",cap_len=2,mask=torch.tensor([[[1.,0.],[0.,1.]]]))
    hook=SliderHook(base,adapter,1.,state,"wq");hook.inject()
    try:
        result=base(x)
        assert torch.equal(result[:,[0,1,3,4]],expected[:,[0,1,3,4]])
        assert torch.equal(base.weight,packed) and torch.equal(base.weight_scale,scale)
        assert adapter.down.dtype==torch.float32 and adapter.up.dtype==torch.float32
    finally:hook.eject()


def test_bfloat16_compute_does_not_change_adapter_source_precision():
    base=nn.Linear(3,4).to(torch.bfloat16);x=torch.randn(1,6,3,dtype=torch.bfloat16);expected=base(x)
    adapter=Adapter(torch.ones(2,3),torch.ones(4,2),2.)
    state=RoutingState(phase="route",cap_len=2,mask=torch.ones(1,2,2))
    hook=SliderHook(base,adapter,1.,state,"wo");hook.inject()
    try:
        result=base(x)
        assert result.dtype==torch.bfloat16 and torch.equal(result[:,:2],expected[:,:2])
        assert adapter.down.dtype==torch.float32
    finally:hook.eject()
    assert not adapter._cache


@pytest.mark.parametrize("kind",["zero_adapter","suppressed_delta","incorrect_scale"])
def test_native_probe_cannot_pass_an_ineffective_or_wrong_slider(kind):
    reference=torch.zeros(1,6,4); mask=torch.tensor([[[1.],[0.],[0.],[1.]]])
    delta=torch.ones(1,4,4)
    actual=reference.clone()
    if kind=="zero_adapter":delta.zero_()
    if kind=="incorrect_scale":actual[:,2:]=delta*mask*2
    assert not compare_probe_outputs(reference,reference.clone(),actual,delta,mask,2)["passed"]


def test_native_probe_expected_alpha_rank_routing():
    adapter=Adapter(torch.ones(2,3),torch.ones(4,2),1.)
    x=torch.ones(1,6,3); reference=torch.randn(1,6,4)
    mask=torch.tensor([[[1.],[0.],[0.],[1.]]]);delta=adapter.delta(x[:,2:])
    actual=reference.clone();actual[:,2:]+=delta*mask
    report=compare_probe_outputs(reference,reference.clone(),actual,delta,mask,2)
    assert report["passed"] and report["target_max_change"]==3.
    assert report["expected_max_error"]==0. and report["outside_max_error"]==0.
