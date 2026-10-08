import torch
from torch import nn

from slider_fuse.lora import Adapter, RoutingState, SliderHook


def test_observer_has_no_rng_or_numerical_side_effect_and_records_actual_delta():
    from slider_fuse.diagnostics import DiagnosticRecorder
    torch.manual_seed(15)
    module = nn.Linear(4, 4)
    adapter = Adapter(torch.ones(2, 4) * .1, torch.ones(4, 2) * .1, 2.)
    state = RoutingState(phase="route", cap_len=2, mask=torch.tensor([[[1., 0.]]]), text_scope="all")
    state.configure_target_text(1., (0,), (1,), 2)
    hook = SliderHook(module, adapter, 4., state, "blocks.0.attn.wq")
    x = torch.randn(1, 4, 4)
    hook.inject()
    try:
        expected = module(x)
        rng = torch.random.get_rng_state()
        recorder = DiagnosticRecorder({"blocks.0.attn.wq"})
        state.recorder = recorder
        actual = module(x)
        assert torch.equal(actual, expected)
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert recorder.adapter_stats[0]["image"]["actual_delta_rms"] > 0
        assert recorder.adapter_stats[0]["text"]["actual_delta_rms"] > 0
        module(x)
        assert len(recorder.adapter_stats) == 1
    finally:
        hook.eject()
