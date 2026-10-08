import pytest
import torch
from torch import nn

from slider_fuse.diagnostics import DiagnosticRecorder
from slider_fuse.lora import Adapter, RoutingState, SliderHook


def setup_audit():
    module = nn.Linear(3, 3)
    adapter = Adapter(torch.ones(1, 3) * .1, torch.ones(3, 1) * .1, 1.)
    state = RoutingState(phase="route", cap_len=3, mask=torch.tensor([[[1., 0.]]]))
    state.configure_target_text(1., (0,), (1,), 3)
    state.audit_partition = {"target": state.mask, "protected": 1 - state.mask,
                             "background": torch.zeros_like(state.mask)}
    recorder = DiagnosticRecorder({"blocks.0.attn.wq"}, level="audit")
    state.recorder = recorder
    return module, adapter, state, recorder


def test_audit_measures_excluded_rows_on_every_call():
    module, adapter, state, recorder = setup_audit()
    x = torch.ones(1, 5, 3)
    hook = SliderHook(module, adapter, 4., state, "blocks.0.attn.wq")
    hook.inject()
    try:
        for step in range(2):
            recorder.begin_step(step, torch.tensor([1. - step * .5]))
            module(x)
        report, _ = recorder.finalize()
        assert len(recorder.adapter_stats) == 1
        assert len(report["linear_audit"]) == 2
        for row in report["linear_audit"]:
            assert row["image"]["unselected"]["max_abs"] == 0
            assert row["text"]["protected_phrase"]["changed_elements"] == 0
            assert row["text"]["other"]["changed_elements"] == 0
            assert row["text"]["target_phrase"]["max_abs"] > 0
            assert row["image"]["background"]["max_abs"] is None
    finally:
        hook.eject()


def test_audit_detects_protected_row_corruption():
    _, adapter, state, recorder = setup_audit()
    base = torch.zeros(1, 5, 3)
    result = base.clone()
    result[:, 1] = 2.
    result[:, 4] = 3.
    recorder.begin_step(0, torch.tensor([1.]))
    recorder.record_linear("blocks.2.attn.wv", base, base, result, None, None, state, adapter, 4.)
    report, _ = recorder.finalize()
    row = report["linear_audit"][0]
    assert row["text"]["protected_phrase"]["rms"] == 2.
    assert row["image"]["unselected"]["rms"] == 3.
    assert row["image"]["unselected"]["element_count"] == 3


def test_audit_outputs_rng_and_trace_are_independent():
    module, adapter, state, recorder = setup_audit()
    x = torch.ones(1, 5, 3)
    state.recorder = None
    hook = SliderHook(module, adapter, 4., state, "blocks.0.attn.wq")
    hook.inject()
    try:
        expected = module(x)
        rng = torch.random.get_rng_state().clone()
        state.recorder = recorder
        recorder.begin_step(0, torch.tensor([1.]))
        assert torch.equal(module(x), expected)
        assert torch.equal(torch.random.get_rng_state(), rng)
        prediction = torch.ones(1, 3, 2, 2)
        recorder.record_step(prediction, prediction * 2, torch.tensor([1.]))
        prediction.zero_()
        _, tensors = recorder.finalize()
        assert tensors["trace_predictions"].sum() == 12
    finally:
        hook.eject()


def test_invalid_level_is_rejected():
    with pytest.raises(ValueError, match="level"):
        DiagnosticRecorder(set(), level="unknown")
