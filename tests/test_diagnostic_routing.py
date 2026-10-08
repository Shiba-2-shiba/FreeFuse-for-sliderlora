import pytest
import torch
from torch import nn

from slider_fuse.lora import Adapter, RoutingState, SliderHook


@pytest.mark.parametrize("image_scope", ["target_mask", "all"])
@pytest.mark.parametrize("text_scope", ["none", "target_phrase", "all"])
@pytest.mark.parametrize("strength", [-1., 0., 4.])
def test_scopes_match_independent_full_sequence_formula(image_scope, text_scope, strength):
    torch.manual_seed(9)
    module = nn.Linear(4, 5)
    x = torch.randn(1, 7, 4)
    base = module(x)
    adapter = Adapter(torch.randn(2, 4), torch.randn(5, 2), 1.)
    mask = torch.tensor([[[1., 0.], [1., 0.]]])
    state = RoutingState(phase="route", cap_len=3, mask=mask.clone(),
                         image_scope=image_scope, text_scope=text_scope)
    state.configure_target_text(0. if text_scope == "none" else 1., (1,), (2,), 3)
    selector = torch.zeros(1, 7, 1)
    selector[:, 3:] = 1. if image_scope == "all" else mask.reshape(1, 4, 1)
    if text_scope == "all": selector[:, :3] = 1.
    elif text_scope == "target_phrase": selector[:, 1] = 1.
    expected = base + adapter.delta(x) * strength * selector
    original = module.forward
    hook = SliderHook(module, adapter, strength, state, "blocks.0.attn.wq")
    hook.inject()
    try:
        actual = module(x)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        excluded = (selector == 0).expand_as(base)
        assert torch.equal(actual[excluded], base[excluded])
        assert torch.equal(state.mask, mask)
        assert state.text_linear_calls == int(bool(strength) and text_scope != "none")
        assert state.target_text_calls == int(bool(strength) and text_scope == "target_phrase")
    finally:
        hook.eject()
    assert module.forward == original


@pytest.mark.parametrize("image_scope,text_scope", [("bad", "none"), ("all", "bad")])
def test_invalid_scope_rejected(image_scope, text_scope):
    with pytest.raises(ValueError, match="scope"):
        RoutingState(image_scope=image_scope, text_scope=text_scope)


def test_scope_cache_and_clear_do_not_retain_all_routing():
    state = RoutingState(cap_len=2, mask=torch.tensor([[[1., 0.]]]), image_scope="all", text_scope="all")
    state.configure_target_text(1., (0,), (1,), 2)
    output = torch.zeros(1, 4, 3)
    assert state.effective_image_mask(output).sum() == 2
    assert state.target_text_indices(output).tolist() == [0, 1]
    state.clear()
    assert state.image_scope == "target_mask" and state.text_scope == "target_phrase"
    assert not state._mask_cache and not state._text_index_cache


def test_all_text_still_requires_correct_runtime_boundary():
    state = RoutingState(cap_len=4, mask=torch.ones(1, 1, 2), text_scope="all")
    state.configure_target_text(1., (0,), (1,), 3)
    with pytest.raises(ValueError, match="boundary"):
        state.target_text_indices(torch.zeros(1, 6, 2))
