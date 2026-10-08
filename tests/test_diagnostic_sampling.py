import sys
import types

import pytest
import torch

from test_runtime import environment, setup_run
from slider_fuse import sampling


@pytest.fixture
def native_loader(environment, monkeypatch):
    calls = []
    sd = types.ModuleType("comfy.sd")
    def load(model, clip, source, strength, clip_strength):
        from slider_fuse.lora import load_adapters
        clone = model.clone()
        adapters = load_adapters(source, model.model.diffusion_model)
        original = {}
        def inject(patcher):
            for name, adapter in adapters.items():
                module = patcher.model.diffusion_model.get_submodule(name)
                original[name] = module.weight.detach().clone()
                with torch.no_grad():
                    module.weight.add_((adapter.up @ adapter.down) * adapter.scale * strength)
        def eject(patcher):
            for name, weight in original.items():
                with torch.no_grad(): patcher.model.diffusion_model.get_submodule(name).weight.copy_(weight)
            original.clear()
        clone.injections["standard-test-double"] = [types.SimpleNamespace(inject=inject, eject=eject)]
        clone.patches = {"diffusion_model." + name + ".weight": [(strength,)] for name in adapters}
        calls.append((strength, clip_strength))
        return clone, None
    sd.load_lora_for_models = load
    monkeypatch.setitem(sys.modules, "comfy.sd", sd)
    monkeypatch.setattr(sys.modules["comfy"], "sd", sd, raising=False)
    return calls


def run(tmp_path, backend="hook", image_scope="target_mask", text_scope="none", strength=4., trial_id=0):
    run_path = tmp_path / str(len(list(tmp_path.iterdir())))
    run_path.mkdir()
    model, positive, info, subjects, file = setup_run(run_path)
    result = sampling.sample_krea2_diagnostic(model, positive, positive, info, subjects,
        {"samples": torch.zeros(1, 4, 4, 4)}, str(file), backend=backend,
        image_scope=image_scope, text_scope=text_scope, strength=strength,
        seed=42, steps=2, cfg=1., trial_id=trial_id)
    return result, model


@pytest.mark.parametrize("image_scope", ["target_mask", "all"])
@pytest.mark.parametrize("text_scope", ["none", "target_phrase", "all"])
def test_diagnostic_hook_records_real_scope_and_restores(environment, tmp_path, image_scope, text_scope):
    (latent, bank, payload), model = run(tmp_path, image_scope=image_scope, text_scope=text_scope)
    report = payload.report
    assert report["phase1_nfe"] == 0 and report["phase2_nfe"] == 2
    assert report["selected_image_row_count"] == (2 if image_scope == "target_mask" else 4)
    assert report["selected_text_row_count"] == {"none": 0, "target_phrase": 1, "all": 2}[text_scope]
    assert report["protected_text_direct_delta_policy"] == ("enabled" if text_scope == "all" else "zero")
    assert report["outside_target_direct_delta_policy"] == ("enabled" if image_scope == "all" else "zero")
    assert report["owned_hooks_removed"]
    assert len(report["adapter_stats"]) == 5
    # TinyKrea's double returns a patch-grid prediction; native Krea2 unpatches it.
    assert payload.first_prediction.shape == (1, 4, 2, 2)
    assert payload.effective_image_mask.sum() == report["selected_image_row_count"]
    assert bank["masks"]["protected"].sum() == 2
    assert not model.model_options and not model.injections
    assert not model.model.diffusion_model.txtfusion._forward_hooks


@pytest.mark.parametrize("backend", ["hook", "native"])
def test_zero_and_trial_do_not_change_noise_or_outputs(environment, native_loader, tmp_path, backend):
    from slider_fuse.sampling import tensor_hash
    torch.manual_seed(81)
    (a, _, pa), _ = run(tmp_path, backend=backend, image_scope="all", text_scope="all", strength=0.)
    torch.manual_seed(81)
    (b, _, pb), _ = run(tmp_path, backend=backend, image_scope="all", text_scope="all", strength=0., trial_id=1)
    assert torch.equal(a["samples"], b["samples"])
    assert torch.equal(pa.first_prediction, pb.first_prediction)
    assert pa.report["run_id"] != pb.report["run_id"]
    for key in ["initial_noise_sha256", "initial_latent_sha256", "full_sigmas_sha256"]:
        assert pa.report[key] == pb.report[key]
    assert pa.report["first_prediction_sha256"] == tensor_hash(pa.first_prediction)
    assert not native_loader


def test_native_is_full_sequence_only_and_never_installs_slider_hook(environment, native_loader, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("native installed SliderHook")
    monkeypatch.setattr(sampling, "SliderHook", forbidden)
    (_, _, payload), model = run(tmp_path, backend="native", image_scope="all", text_scope="all")
    assert native_loader == [(4., 0.)]
    assert payload.report["adapter_stats"] is None
    assert not model.model.diffusion_model.txtfusion._forward_hooks
    with pytest.raises(ValueError, match="native"):
        run(tmp_path, backend="native")


def test_diagnostic_exception_restores_owned_hooks(environment, tmp_path):
    modules, _, unloads = environment
    def fail(*args, **kwargs): raise RuntimeError("diagnostic interruption")
    modules["sample"].sample = fail
    with pytest.raises(RuntimeError, match="interruption"):
        run(tmp_path)
    assert len(unloads) == 1
    core = unloads[0].model.diffusion_model
    assert all("forward" not in module.__dict__ for module in core.modules())
    assert not core.txtfusion._forward_hooks


def test_fp32_whole_tiny_model_weight_merge_and_full_hook_agree(environment, native_loader, tmp_path):
    torch.manual_seed(81)
    (_, _, native), _ = run(tmp_path, backend="native", image_scope="all", text_scope="all")
    torch.manual_seed(81)
    (_, _, hook), _ = run(tmp_path, backend="hook", image_scope="all", text_scope="all")
    torch.testing.assert_close(native.first_prediction, hook.first_prediction, rtol=1e-5, atol=1e-6)
    assert native.report["conditioning_sha256"] == hook.report["conditioning_sha256"]
    assert native.report["first_input_sha256"] == hook.report["first_input_sha256"]


def test_native_invalid_lora_preserves_original_error_and_allows_clean_rerun(environment, native_loader, tmp_path, monkeypatch):
    import safetensors.torch
    original_load = safetensors.torch.load_file
    def fail(*args, **kwargs): raise ValueError("malformed slider source")
    monkeypatch.setattr(safetensors.torch, "load_file", fail)
    with pytest.raises(ValueError, match="malformed slider source"):
        run(tmp_path, backend="native", image_scope="all", text_scope="all")
    monkeypatch.setattr(safetensors.torch, "load_file", original_load)
    (_, _, payload), _ = run(tmp_path, backend="native", image_scope="all", text_scope="all")
    assert payload.report["owned_hooks_removed"]
