import pytest
import torch
from torch import nn

from slider_fuse.lora import Adapter, RoutingState, SliderHook


def state_for(scale, mask=None):
    state=RoutingState(phase="route",cap_len=3,mask=torch.tensor([[[1.,0.],[0.,1.]]]) if mask is None else mask)
    state.configure_target_text(scale,(1,),(2,),3)
    return state


@pytest.mark.parametrize("scale",[0.,.5,1.])
@pytest.mark.parametrize("strength",[-1.,0.,1.25])
def test_only_target_text_and_target_image_receive_expected_delta(scale,strength):
    torch.manual_seed(42)
    module=nn.Linear(4,5); x=torch.randn(1,7,4); baseline=module(x)
    adapter=Adapter(torch.randn(2,4),torch.randn(5,2),1.)
    state=state_for(scale)
    hook=SliderHook(module,adapter,strength,state,"blocks.0.attn.wq");original=module.forward
    expected=baseline.clone()
    expected[:,3:]=baseline[:,3:]+adapter.delta(x[:,3:])*strength*state.mask.reshape(1,4,1)
    expected[:,1:2]=baseline[:,1:2]+adapter.delta(x[:,1:2])*strength*scale
    hook.inject()
    try:
        actual=module(x)
        torch.testing.assert_close(actual,expected)
        assert torch.equal(actual[:,[0,2,4,5]],baseline[:,[0,2,4,5]])
        assert torch.equal(actual[:,3:],expected[:,3:])
        assert state.target_text_calls==(1 if scale and strength else 0)
    finally:hook.eject()
    assert module.forward==original


@pytest.mark.parametrize("phase",["off","collect"])
def test_target_text_is_disabled_outside_route(phase):
    module=nn.Linear(4,5);x=torch.randn(1,7,4);baseline=module(x)
    state=state_for(1.);state.phase=phase
    hook=SliderHook(module,Adapter(torch.ones(2,4),torch.ones(5,2),2.),4.,state,"gate")
    hook.inject()
    try:assert torch.equal(module(x),baseline) and state.target_text_calls==0
    finally:hook.eject()


def test_empty_image_mask_disables_target_text_as_well():
    module=nn.Linear(4,5);x=torch.randn(1,7,4);baseline=module(x)
    state=state_for(1.,torch.zeros(1,2,2))
    hook=SliderHook(module,Adapter(torch.ones(2,4),torch.ones(5,2),2.),4.,state,"wo")
    hook.inject()
    try:assert torch.equal(module(x),baseline) and state.target_text_calls==0
    finally:hook.eject()


@pytest.mark.parametrize("scale",[True,-.1,1.1,float("nan"),float("inf"),None,"0.5"])
def test_invalid_target_text_scales_are_rejected(scale):
    with pytest.raises(ValueError):state_for(scale)


@pytest.mark.parametrize("target,protected",[((),(2,)),((1,1),(2,)),((3,),(2,)),((-1,),(2,)),((True,),(2,)),
    ((1,),(1,)),((1,),()),((1,),(3,)),((1,),(2,2))])
def test_invalid_phrase_positions_fail_without_global_or_clamped_fallback(target,protected):
    with pytest.raises(ValueError):RoutingState().configure_target_text(.5,target,protected,3)


def test_clear_removes_text_configuration_and_cached_indices():
    module=nn.Linear(4,5);x=torch.randn(1,7,4);state=state_for(.5)
    hook=SliderHook(module,Adapter(torch.ones(2,4),torch.ones(5,2),2.),4.,state,"wk")
    hook.inject()
    try:module(x)
    finally:hook.eject()
    assert state._text_index_cache and state.target_text_calls==1
    state.clear()
    assert not state._text_index_cache and not state.target_text_positions and not state.protected_text_positions
    assert state.target_text_scale==0. and state.phase=="off"
    assert state.target_text_calls==1  # diagnostic survives auto phase cleanup


def test_runtime_shorter_text_boundary_cannot_route_into_image_rows():
    module=nn.Linear(4,5);state=state_for(1.);state.cap_len=1
    hook=SliderHook(module,Adapter(torch.ones(2,4),torch.ones(5,2),2.),4.,state,"wv")
    hook.inject()
    try:
        with pytest.raises(ValueError,match="text"):
            module(torch.ones(1,5,4))
    finally:hook.eject()
