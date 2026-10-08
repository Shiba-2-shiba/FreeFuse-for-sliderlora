import json

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file

from slider_fuse.diagnostics import DiagnosticPayload
from slider_fuse.sampling import tensor_hash


def graph():
    from pathlib import Path
    p = json.loads((Path(__file__).parents[1] / "workflows/krea2_female_slider_compare_manual_api.json").read_text())
    p["8"]["class_type"] = "Krea2SliderFuseDiagnosticSampler"
    p["8"]["inputs"].update(backend="hook", image_scope="target_mask", text_scope="none", strength=4., trial_id=0)
    p["11"] = {"class_type": "Krea2SliderFuseDiagnosticSave", "inputs": {
        "latent": ["8", 0], "images": ["10", 0], "diagnostics": ["8", 2], "filename_prefix": "test"}}
    return p


def payload():
    latent = torch.zeros(1, 4, 4, 4)
    mask = torch.tensor([[[1., 0.], [1., 0.]]])
    report = {"run_id": "test-run", "backend": "hook", "image_scope": "target_mask", "text_scope": "none",
        "strength": 4., "seed": 42, "steps": 8, "cfg": 1., "trial_id": 0,
        "prompt": graph()["4"]["inputs"]["prompt"], "final_latent_sha256": tensor_hash(latent),
        "effective_image_mask_sha256": tensor_hash(mask), "first_prediction_sha256": tensor_hash(latent)}
    return latent, DiagnosticPayload(report, latent.clone(), mask)


def test_saves_correlated_artifacts_and_never_overwrites(tmp_path):
    from slider_fuse.diagnostics import save_diagnostic_artifacts
    latent, data = payload()
    kwargs = dict(directory=tmp_path, basename="case_s42_00001_", latent=latent,
        images=torch.zeros(1, 8, 8, 3), payload=data, prompt=graph(), extra_pnginfo=None, save_node_id="11")
    a = save_diagnostic_artifacts(**kwargs)
    b = save_diagnostic_artifacts(**kwargs)
    assert a["artifact_id"] != b["artifact_id"]
    assert a["artifacts"]["image"] != b["artifacts"]["image"]
    persisted = json.loads((tmp_path / a["artifacts"]["report"]).read_text())
    assert persisted["complete"] is True
    assert persisted["generation"]["checkpoint"] == "intorealismAsian_k2JAVFLASHV1.safetensors"
    assert persisted["generation"]["style_loras"][0]["strength"] == .8
    im = Image.open(tmp_path / a["artifacts"]["image"])
    assert im.info["artifact_id"] == a["artifact_id"]
    tensors = load_file(str(tmp_path / a["artifacts"]["tensors"]))
    assert torch.equal(tensors["final_latent"], latent)
    mask = Image.open(tmp_path / a["artifacts"]["effective_mask"])
    assert mask.size == (2, 2)


@pytest.mark.parametrize("kind", ["latent", "mask", "prediction", "prompt", "images", "missing", "unknown", "prefix"])
def test_refuses_mismatched_or_unsafe_artifacts_before_writing(tmp_path, kind):
    from slider_fuse.diagnostics import save_diagnostic_artifacts
    latent, data = payload()
    prompt = graph(); basename = "case"
    if kind == "latent": latent = latent + 1
    if kind == "mask": data.effective_image_mask = 1 - data.effective_image_mask
    if kind == "prediction": data.first_prediction = data.first_prediction + 1
    if kind == "prompt": prompt["4"]["inputs"]["prompt"] = "different"
    if kind == "images": prompt["10"]["inputs"]["samples"] = ["7", 0]
    if kind == "missing": prompt = None
    if kind == "unknown": prompt["2"]["class_type"] = "UnknownModelTransform"
    if kind == "prefix": basename = "../outside"
    with pytest.raises(ValueError):
        save_diagnostic_artifacts(tmp_path, basename, latent, torch.zeros(1, 8, 8, 3), data, prompt, None, "11")
    assert not list(tmp_path.iterdir())


def test_failed_tensor_write_leaves_no_complete_manifest(tmp_path, monkeypatch):
    from slider_fuse import diagnostics
    import safetensors.torch
    latent, data = payload()
    monkeypatch.setattr(safetensors.torch, "save_file", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        diagnostics.save_diagnostic_artifacts(tmp_path, "case", latent, torch.zeros(1, 8, 8, 3), data, graph(), None, "11")
    assert not list(tmp_path.glob("*.json"))
