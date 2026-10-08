"""Shared-core ComfyUI boundary doubles; real checks live in native_checks.py."""
import sys
import types
from contextlib import nullcontext

import pytest
import torch


@pytest.fixture
def pair_environment(monkeypatch):
    core = types.SimpleNamespace(current_patcher=None, factor=2.)
    events = []
    class Patcher:
        def __init__(self, factor):
            self.model = core; self.factor = factor
            self.model_options = {}; self.load_device = torch.device("cpu")
        def pre_run(self):
            events.append(("pre", self.factor)); core.current_patcher = self
        def cleanup(self):
            events.append(("clean", self.factor)); core.current_patcher = None
    base, slider = Patcher(2.), Patcher(6.)
    def load(models, **kwargs):
        assert len(models) == 1
        events.append(("load", models[0].factor)); core.factor = models[0].factor
    def predict(model, x, sigma, uncond, cond, cfg, model_options, seed):
        assert model.current_patcher is not None
        assert model.current_patcher.factor == model.factor
        events.append(("predict", model.factor))
        return x * model.factor
    comfy = types.ModuleType("comfy"); comfy.__path__ = []
    mm = types.ModuleType("comfy.model_management"); mm.load_models_gpu = load
    samplers = types.ModuleType("comfy.samplers"); samplers.sampling_function = predict
    helpers = types.ModuleType("comfy.sampler_helpers"); helpers.estimate_memory = lambda *a: (0, 0)
    for module in (comfy, mm, samplers, helpers):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    comfy.model_management = mm; comfy.samplers = samplers; comfy.sampler_helpers = helpers
    return base, slider, events, samplers


@pytest.fixture
def guider_environment(pair_environment):
    base, slider, events, samplers = pair_environment
    core = base.model
    core.process_latent_out = lambda x: x
    base.model_dtype = lambda: torch.float32
    class Guider:
        def __init__(self, patcher):
            self.model_patcher = patcher; self.model_options = {}; self.cfg = 1.
        def set_cfg(self, cfg):
            self.cfg = cfg
        def set_conds(self, positive, negative):
            self.conds = {"positive": positive, "negative": negative}
        def sample(self, noise, latent_image, sampler, sigmas, seed=None):
            events.append(("outer", 1))
            return self.outer_sample(noise, latent_image, sampler, sigmas, seed=seed)
        def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
                         callback=None, disable_pbar=False, seed=None, latent_shapes=None):
            raise AssertionError("base outer lifecycle must not be nested")
        def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask,
                         callback, disable_pbar, seed, latent_shapes=None):
            events.append(("process_conds", 1))
            x = noise.clone()
            for sigma, following in zip(sigmas[:-1], sigmas[1:]):
                p = self.predict_noise(x, sigma.reshape(1), model_options={}, seed=seed)
                x = x + (x - p) / sigma * (following - sigma)
            return x
    samplers.CFGGuider = Guider
    samplers.cast_to_load_options = lambda *a, **k: None
    samplers.sampler_object = lambda name: name
    helpers = sys.modules["comfy.sampler_helpers"]
    def prepare(patcher, shape, conds, options):
        events.append(("prepare", 1))
        return patcher.model, conds, []
    helpers.prepare_sampling = prepare
    helpers.cleanup_models = lambda *args: events.append(("cleanup_models", 1))
    sys.modules["comfy.model_management"].cuda_device_context = lambda device: nullcontext()
    return pair_environment


def create_guider(env, *, mode="both", text_len=2):
    from slider_fuse.native_pair import make_prediction_mix_guider
    from slider_fuse.diagnostics import DiagnosticRecorder
    base, slider, events, _ = env
    recorder = DiagnosticRecorder(set(), level="audit")
    report = {"sampler_nfe": 0, "grid": [2, 2], "text_token_count": text_len}
    guider = make_prediction_mix_guider(base, slider, torch.tensor([[[1., 0.], [1., 0.]]]),
        patch=1, recorder=recorder, report=report, mode=mode)
    guider.set_conds([{"model_conds": {"c_crossattn": types.SimpleNamespace(cond=torch.ones(1, 2, 3))}}], [])
    return guider, report, recorder


def test_pair_runs_one_loop_with_exact_selected_output(guider_environment):
    guider, report, recorder = create_guider(guider_environment)
    guider.sample(torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), "euler", torch.linspace(1, 0, 9), seed=42)
    tensors = guider.finish_report()
    assert report["sampler_nfe"] == 8 and report["branch_nfe"] == {"base": 8, "slider": 8}
    assert torch.equal(recorder.first_prediction, torch.tensor([[[[6., 2.], [6., 2.]]]]))
    assert tensors["trace_predictions"].shape == (8, 1, 1, 2, 2)
    assert all(r["base_excluded"]["max_abs"] == 0 and r["slider_selected"]["max_abs"] == 0
               for r in report["prediction_mix_audit"])
    events = guider_environment[2]
    for marker in ("prepare", "outer", "process_conds", "cleanup_models"):
        assert events.count((marker, 1)) == 1
    assert guider_environment[0].model.current_patcher is None


@pytest.mark.parametrize("mode,counts,value", [("base", {"base": 8, "slider": 0}, 2.),
                                              ("slider", {"base": 0, "slider": 8}, 6.)])
def test_endpoints_skip_unneeded_branch(guider_environment, mode, counts, value):
    guider, report, recorder = create_guider(guider_environment, mode=mode)
    guider.sample(torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), "euler", torch.linspace(1, 0, 9), seed=42)
    guider.finish_report()
    assert report["branch_nfe"] == counts
    assert torch.equal(recorder.first_prediction, torch.full((1, 1, 2, 2), value))


def test_bad_text_boundary_rejected_before_branch_forward(guider_environment):
    guider, _, _ = create_guider(guider_environment, text_len=3)
    with pytest.raises(ValueError, match="text boundary"):
        guider.sample(torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), "euler", torch.tensor([1., 0.]))
    assert not any(event[0] == "predict" for event in guider_environment[2])


def call(runner, branch, x):
    return runner.predict(branch, x, torch.tensor([1.]), positive=[], negative=[], model_options={}, seed=42)


def test_base_native_base_keeps_style_and_restores_slider(pair_environment):
    from slider_fuse.native_pair import NativePairRunner
    base, slider, events, _ = pair_environment
    runner = NativePairRunner(base, slider)
    x = torch.ones(1, 1, 2, 2)
    try:
        a = call(runner, "base", x)
        b = call(runner, "slider", x)
        c = call(runner, "base", x)
        assert torch.equal(a, x * 2) and torch.equal(b, x * 6) and torch.equal(c, a)
        assert events == [("load", 2.), ("pre", 2.), ("predict", 2.), ("clean", 2.),
                          ("load", 6.), ("pre", 6.), ("predict", 6.), ("clean", 6.),
                          ("load", 2.), ("pre", 2.), ("predict", 2.)]
    finally:
        runner.close()
    assert base.model.current_patcher is None


def test_failure_closes_active_branch_and_original_error_survives(pair_environment):
    from slider_fuse.native_pair import NativePairRunner
    base, slider, _, samplers = pair_environment
    runner = NativePairRunner(base, slider)
    def fail(*args, **kwargs):
        raise RuntimeError("native failure")
    samplers.sampling_function = fail
    with pytest.raises(RuntimeError, match="native failure"):
        try:
            call(runner, "slider", torch.ones(1, 1, 2, 2))
        finally:
            runner.close()
    assert base.model.current_patcher is None


def test_pre_run_identity_cannot_be_faked(pair_environment):
    from slider_fuse.native_pair import NativePairRunner
    base, slider, _, _ = pair_environment
    base.pre_run = lambda: None
    runner = NativePairRunner(base, slider)
    with pytest.raises(RuntimeError, match="current_patcher"):
        call(runner, "base", torch.ones(1, 1, 2, 2))
    runner.close()


def test_pair_inputs_and_returned_values_cannot_be_mutated(pair_environment):
    from slider_fuse.native_pair import NativePairRunner
    base, slider, _, samplers = pair_environment
    buffer = torch.zeros(1, 1, 2, 2)
    def predict(model, x, *args, **kwargs):
        x.mul_(model.factor)
        buffer.copy_(x)
        return buffer
    samplers.sampling_function = predict
    runner = NativePairRunner(base, slider)
    x = torch.ones_like(buffer)
    try:
        first = call(runner, "base", x)
        second = call(runner, "slider", x)
        assert torch.equal(first, torch.full_like(x, 2.))
        assert torch.equal(second, torch.full_like(x, 6.))
        assert torch.equal(x, torch.ones_like(x))
    finally:
        runner.close()
