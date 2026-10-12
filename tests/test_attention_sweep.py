"""Sweep keeps the original schedule and the existing per-tap k3 computation."""
import importlib
import importlib.util
import json

import pytest
import torch

from test_runtime import environment
from test_experimental_collection import prepared
from test_experimental_masks import fixture
from slider_fuse.experimental_masks import ExperimentalMaskAccumulator
from slider_fuse.sampling import tensor_hash


def module():
    assert importlib.util.find_spec("slider_fuse.attention_sweep") is not None, "Attention sweep is missing"
    return importlib.import_module("slider_fuse.attention_sweep")


def config(blocks=None):
    return {"collect_step": 3, "collect_block": 0, "selected_steps": [1, 2, 3],
            "selected_blocks": [0] if blocks is None else blocks, "prototypes": 1,
            "seed_confidence": .5, "seed_margin": 0., "mask_confidence": .5, "mask_margin": 0.}


def test_each_tap_matches_existing_k3_and_retains_only_cpu_evidence():
    sweep = module().SweepAccumulator(config([0, 1]), (3, 4), fixture()[1])
    data, positions = fixture()
    for step in (1, 2, 3):
        for block in (0, 1):
            before = {k: tensor_hash(v) for k, v in data.items()}
            sweep.observe(step=step, block=block, **data, attention_mask=None)
            assert before == {k: tensor_hash(v) for k, v in data.items()}
    taps, report = sweep.finalize()
    assert set(taps) == {f"step_{s}.block_{b}" for s in (1, 2, 3) for b in (0, 1)}
    assert report["missing_taps"] == []
    for step in (1, 2, 3):
        for block in (0, 1):
            cfg = {**config(), "collect_step": step, "collect_block": block,
                   "selected_steps": [step], "selected_blocks": [block]}
            original = ExperimentalMaskAccumulator(cfg, (3, 4), positions)
            original.observe(step=step, block=block, **data, attention_mask=None)
            expected, _ = original.finalize()
            tap = taps[f"step_{step}.block_{block}"]
            bank = tap["banks"]["multi_proto_bg"]
            for group in ("masks", "raw_maps", "seed_masks"):
                for role, tensor in bank[group].items():
                    assert torch.equal(tensor, expected["multi_proto_bg"][group][role])
                    assert tensor.device.type == "cpu" and not tensor.requires_grad
            assert tap["algorithm"]["variants"]["multi_proto_bg"]["status"] == "ok"


def test_three_forward_runtime_preserves_original_sigmas_inputs_rng_and_hooks(environment, tmp_path):
    kw = prepared(tmp_path, environment)
    kw["config_json"] = json.dumps(config())
    kw["audit_tensors"] = True
    core = kw["model"].model.diffusion_model
    image = kw["latent"]["samples"]
    before = tensor_hash(image)
    baseline = core(image, torch.ones(1), kw["positive"][0][0]).detach().clone()
    rng = torch.random.get_rng_state()
    sweep = module().collect_attention_sweep(**kw)
    assert len(environment[1]) == 1 and len(environment[2]) == 1
    assert torch.equal(environment[1][0][2], torch.tensor([1., .875, .75, .625]))
    assert sweep["report"]["phase1_nfe"] == 3 and sweep["report"]["phase2_nfe"] == 0
    assert sweep["report"]["completed_reference_image_generated"] is False
    assert set(sweep["taps"]) == {"step_1.block_0", "step_2.block_0", "step_3.block_0"}
    assert {(r["step"], r["block"]) for r in sweep["report"]["tap_manifest"]} == {(1, 0), (2, 0), (3, 0)}
    assert all(r["tensor_sha256_before"] == r["tensor_sha256_after"] for r in sweep["report"]["tap_manifest"])
    assert before == tensor_hash(image) and torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(baseline, core(image, torch.ones(1), kw["positive"][0][0]))
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in core.modules())
    assert not kw["model"].model_options and not kw["model"].injections


@pytest.mark.parametrize("override", [
    {"selected_steps": [1, 2]}, {"selected_steps": [1, 2, 3, 4]},
    {"selected_blocks": []}, {"selected_blocks": [0, 0]}, {"selected_blocks": [True]},
    {"selected_blocks": list(range(29))}, {"unknown": 1},
])
def test_bad_sweep_config_fails_before_sampling(environment, tmp_path, override):
    kw = prepared(tmp_path, environment)
    kw["config_json"] = json.dumps({**config(), **override})
    with pytest.raises(ValueError):
        module().collect_attention_sweep(**kw)
    assert not environment[1]


def test_failed_candidates_are_retained_and_selection_never_substitutes():
    mod = module()
    data, positions = fixture()
    data["img_q"] = torch.zeros_like(data["img_q"])
    accumulator = mod.SweepAccumulator(config(), (3, 4), positions)
    for step in (1, 2, 3):
        accumulator.observe(step=step, block=0, **data, attention_mask=None)
    taps, report = accumulator.finalize()
    assert len(taps) == 3 and report["missing_taps"] == []
    assert all(t["algorithm"]["variants"]["multi_proto_bg"]["status"] == "failed" for t in taps.values())
    sweep = {"sweep_schema_version": 1, "taps": {key: {"experiment_schema_version": 1,
             "banks": value["banks"], "report": {"algorithm": value["algorithm"]}} for key, value in taps.items()}}
    with pytest.raises(ValueError, match="failed"):
        mod.select_attention_sweep(sweep, 2, 0)


def test_sweep_exception_cleans_every_owned_hook(environment, tmp_path, monkeypatch):
    mod = module()
    kw = prepared(tmp_path, environment)
    kw["config_json"] = json.dumps(config())
    monkeypatch.setattr(mod.SweepAccumulator, "observe", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sweep probe failed")))
    with pytest.raises(RuntimeError, match="sweep probe failed"):
        mod.collect_attention_sweep(**kw)
    assert environment[2]
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in kw["model"].model.diffusion_model.modules())


def test_saver_keeps_every_tap_and_failed_status_with_verified_paths(tmp_path):
    import hashlib
    mod = module()
    data, positions = fixture()
    accumulator = mod.SweepAccumulator(config(), (3, 4), positions)
    for step in (1, 2, 3):
        accumulator.observe(step=step, block=0, **data, attention_mask=None)
    taps, _ = accumulator.finalize()
    sweep = {"sweep_schema_version": 1, "report": {"run_id": "shared", "phase1_nfe": 3},
             "taps": {key: {"experiment_schema_version": 1, "banks": value["banks"],
                      "report": {"algorithm": value["algorithm"]}, "evidence_maps": value["evidence_maps"]}
                      for key, value in taps.items()}}
    saved = mod.save_attention_sweep(tmp_path, "sweep", sweep)
    manifest_path = tmp_path / saved["manifest"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["complete"] and len(manifest["taps"]) == 3
    for key, item in manifest["taps"].items():
        path = (manifest_path.parent / item["manifest"]).resolve()
        assert path.is_relative_to(manifest_path.parent.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["manifest_sha256"]
        per_tap = json.loads(path.read_text())
        assert per_tap["complete"]
        assert per_tap["report"]["algorithm"]["variants"]["multi_proto_bg"]["status"] == "ok"


def test_missing_tap_cannot_be_finalized_or_selected():
    mod = module()
    accumulator = mod.SweepAccumulator(config(), (3, 4), fixture()[1])
    with pytest.raises(RuntimeError, match="Missing sweep taps"):
        accumulator.finalize()
    with pytest.raises(ValueError, match="not collected"):
        mod.select_attention_sweep({"sweep_schema_version": 1, "taps": {}}, 3, 0)
