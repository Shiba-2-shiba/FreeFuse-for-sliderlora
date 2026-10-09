"""CPU ComfyUI boundary checks; native validation remains a separate tool."""
import importlib
from dataclasses import replace

import pytest
import torch

from test_runtime import environment, setup_run
from slider_fuse.sampling import tensor_hash


def helper():
    assert importlib.util.find_spec("slider_fuse.mask_collection") is not None, "Phase 1 helper is missing"
    return importlib.import_module("slider_fuse.mask_collection")


def collect(tmp_path, monkeypatch, *, fail=None):
    module = helper()
    model, positive, info, subjects, _ = setup_run(tmp_path)
    subjects = tuple(replace(s, manual_mask=None) for s in subjects)
    from slider_fuse import attention
    maps = {"target": torch.tensor([[1., 0., 1., 0.]]),
            "protected": torch.tensor([[0., 1., 0., 1.]])}
    if fail == "map": maps["target"][:] = float("nan")
    monkeypatch.setattr(attention, "similarity_maps", lambda *a, **k: maps)
    image = torch.zeros(1, 4, 4, 4)
    noise = torch.randn(image.shape, generator=torch.Generator().manual_seed(42))
    sigmas = torch.tensor([1., .5, 0.])
    rng = torch.random.get_rng_state()
    result = module.collect_auto_mask(model, positive, positive, info, subjects, image, noise, sigmas,
        grid=(2, 2), seed=42, steps=2, cfg=1., collect_step=1, collect_block=0,
        top_k_ratio=.3, temperature=4000., fill_holes_max_area=0, mask_dilate_radius=0)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert model.model_options == {} and model.injections == {}
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.model.diffusion_model.modules())
    return result, model, image, noise, sigmas


def test_collect_uses_only_base_and_prefix_sigmas(environment, tmp_path, monkeypatch):
    (_, calls, unloads) = environment
    (bank, report), model, image, noise, sigmas = collect(tmp_path, monkeypatch)
    assert len(calls) == 1 and len(unloads) == 1
    assert torch.equal(calls[0][0], noise) and torch.equal(calls[0][1], image)
    assert torch.equal(calls[0][2], sigmas[:2])
    assert bank["mode"] == "auto"
    assert torch.equal(bank["masks"]["target"], torch.tensor([[[1., 0.], [1., 0.]]]))
    assert bank["raw_maps"]["target"].shape == (1, 4)
    assert report["phase1_nfe"] == 1 and report["observation"]["block"] == 0
    assert report["collection_initial_noise_sha256"] == tensor_hash(noise)
    assert report["collection_initial_latent_sha256"] == tensor_hash(image)
    assert report["collection_full_sigmas_sha256"] == tensor_hash(sigmas)
    assert report["collection_used_sigmas_sha256"] == tensor_hash(sigmas[:2])


@pytest.mark.parametrize("failure", ["sample", "map", "install"])
def test_collection_failures_release_all_owned_state(environment, tmp_path, monkeypatch, failure):
    module = helper()
    if failure == "sample":
        def fail(*a, **k): raise RuntimeError("sampling failed")
        monkeypatch.setattr(environment[0]["sample"], "sample", fail)
    elif failure == "install":
        original = module.AttentionCollector.install
        def fail(self, *a):
            original(self, *a)
            raise RuntimeError("partial install failed")
        monkeypatch.setattr(module.AttentionCollector, "install", fail)
    with pytest.raises((ValueError, RuntimeError)):
        collect(tmp_path, monkeypatch, fail="map" if failure == "map" else None)
    assert environment[2]
    core = environment[2][-1].model.diffusion_model
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in core.modules())


def test_collection_preserves_base_prediction_and_rng(environment, tmp_path, monkeypatch):
    module = helper()
    model, positive, info, subjects, _ = setup_run(tmp_path)
    image = torch.zeros(1, 4, 4, 4); sigma = torch.ones(1)
    baseline = model.model.diffusion_model(image, sigma, positive[0][0])
    from slider_fuse import attention
    monkeypatch.setattr(attention, "similarity_maps", lambda *a, **k: {
        "target": torch.tensor([[1., 0., 1., 0.]]), "protected": torch.tensor([[0., 1., 0., 1.]])})
    module.collect_auto_mask(model, positive, positive, info,
        tuple(replace(s, manual_mask=None) for s in subjects), image, image.clone(), torch.tensor([1., .5, 0.]),
        grid=(2, 2), seed=42, steps=2, cfg=1., collect_step=1, collect_block=0,
        top_k_ratio=.3, temperature=4000., fill_holes_max_area=0, mask_dilate_radius=0)
    assert torch.equal(baseline, model.model.diffusion_model(image, sigma, positive[0][0]))
    assert model.model_options == {} and model.patches == {}


def test_cleanup_failure_does_not_hide_primary_or_leave_observation_hooks(environment, tmp_path, monkeypatch):
    module=helper()
    mm=environment[0]["model_management"];unload=mm.unload_model_and_clones
    def fail_after_unload(*a,**kw):
        unload(*a,**kw)
        raise RuntimeError("cleanup failed")
    monkeypatch.setattr(mm,"unload_model_and_clones",fail_after_unload)
    with pytest.raises(ValueError,match="NaN|Inf"):
        collect(tmp_path,monkeypatch,fail="map")
    core=environment[2][-1].model.diffusion_model
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in core.modules())
