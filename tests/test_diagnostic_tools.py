import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file

from test_diagnostic_artifacts import graph, payload
from slider_fuse.diagnostics import save_diagnostic_artifacts, tensor_metrics


ROOT = Path(__file__).parents[1]


def tool():
    spec = importlib.util.spec_from_file_location("diagnostic_compare", ROOT / "tools/compare_slider_diagnostics.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def saved(tmp_path, run_id):
    latent, data = payload()
    data.report.update(run_id=run_id, initial_noise_sha256="noise", initial_latent_sha256="latent",
        full_sigmas_sha256="sigmas", conditioning_sha256="cond", first_input_sha256="input",
        first_timestep_sha256="sigma", first_input_shape=[1, 4, 4, 4], first_input_dtype="torch.float32",
        lora_sha256="lora", environment={"torch": "test", "comfy_revision": "test"},
        reference_partition_sha256={"target": "target", "protected": "protected", "background": "background"})
    result = save_diagnostic_artifacts(tmp_path, run_id, latent, torch.zeros(1, 8, 8, 3), data, graph(), None, "11")
    return tmp_path / result["artifacts"]["report"]


def test_saved_reports_compare_without_claiming_tolerance_pass(tmp_path):
    compare = tool()
    a, b = saved(tmp_path, "a"), saved(tmp_path, "b")
    result = compare.compare_reports([a, b])
    pair = result["pairs"][0]
    assert pair["final_latent"]["exact_equal"]
    assert pair["first_prediction"]["status"] == "measured_only"
    assert pair["rgb"]["mae"] == 0
    declared = compare.compare_reports([a, b], atol=1e-6, rtol=1e-5)
    assert declared["pairs"][0]["first_prediction"]["within_declared_tolerance"]


@pytest.mark.parametrize("kind", ["seed", "file", "cached", "missing", "mask", "revision"])
def test_invalid_comparison_is_not_success(tmp_path, kind):
    compare = tool()
    a, b = saved(tmp_path, "a"), saved(tmp_path, "b")
    r = json.loads(b.read_text())
    if kind == "seed": r["generation"]["seed"] = 777
    if kind == "file": (tmp_path / r["artifacts"]["tensors"]).write_bytes(b"broken")
    if kind == "cached": r["run_id"] = "a"
    if kind == "missing": del r["conditioning_sha256"]
    if kind == "mask": r["reference_partition_sha256"]["target"] = "different"
    if kind == "revision":
        r["environment"]["comfy_revision"] = None
        ra = json.loads(a.read_text()); ra["environment"]["comfy_revision"] = None; a.write_text(json.dumps(ra))
    b.write_text(json.dumps(r))
    with pytest.raises(ValueError): compare.compare_reports([a, b])


def test_standard_latent_version_is_explicit_and_metadata_pair_is_required(tmp_path):
    compare = tool()
    native = saved(tmp_path, "diag")
    prompt = json.loads((ROOT / "workflows/krea2_female_slider_global_reference_api.json").read_text())
    prompt["30"] = {"class_type": "SaveLatent", "inputs": {"samples": ["8", 0], "filename_prefix": "standard"}}
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    m = PngInfo(); m.add_text("prompt", json.dumps(prompt))
    png = tmp_path / "standard.png"; Image.new("RGB", (8, 8)).save(png, pnginfo=m)
    latent = tmp_path / "standard.latent"
    save_file({"latent_tensor": torch.zeros(1, 4, 4, 4), "latent_format_version_0": torch.tensor([])},
              str(latent), metadata={"prompt": json.dumps(prompt)})
    result = compare.compare_standard(latent, png, "standard_global", native)
    assert result["provenance_level"] == "prompt_and_explicit_pair"
    assert result["first_prediction"] is None
    assert result["final_latent"]["exact_equal"]
    assert result["status"] == "measured_only"
    declared = compare.compare_standard(latent, png, "standard_global", native, atol=1e-6, rtol=1e-5)
    assert declared["status"] == "within_tolerance"
    save_file({"latent_tensor": torch.zeros(1, 4, 4, 4)}, str(tmp_path / "unknown.latent"))
    with pytest.raises(ValueError): compare.compare_standard(tmp_path / "unknown.latent", png, "standard_global", native)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.])
def test_bad_tolerance_rejected(bad):
    with pytest.raises(ValueError): tensor_metrics(torch.zeros(1), torch.zeros(1), atol=bad, rtol=0.)


@pytest.mark.parametrize("filename", ["compare_slider_diagnostics.py", "probe_krea2_slider_parity.py"])
def test_help_is_available_without_comfy(filename):
    r = subprocess.run([sys.executable, "-B", str(ROOT / "tools" / filename), "--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "--output" in r.stdout
