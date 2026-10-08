import torch
import pytest
from safetensors.torch import load_file

from test_diagnostic_artifacts import graph, payload
from slider_fuse.diagnostics import save_diagnostic_artifacts
from slider_fuse.sampling import tensor_hash


def mix_payload():
    g = graph()
    g["8"]["class_type"] = "Krea2SliderFusePredictionMixSampler"
    g["8"]["inputs"].update(mix_scope="target_mask", diagnostic_level="audit")
    latent, data = payload()
    data.report.update(backend="prediction_mix", text_scope="all", mix_scope="target_mask",
                       diagnostic_schema_version=2, diagnostic_level="audit")
    data.extra_tensors = {"trace_inputs": torch.ones(8, 1, 4, 4, 4),
                          "trace_predictions": torch.zeros(8, 1, 4, 4, 4),
                          "trace_sigmas": torch.linspace(1, .1, 8)}
    data.report["step_trace"] = [{"eval_index": i, "sigma": float(data.extra_tensors["trace_sigmas"][i]),
        "input_sha256": tensor_hash(data.extra_tensors["trace_inputs"][i]),
        "prediction_sha256": tensor_hash(data.extra_tensors["trace_predictions"][i])} for i in range(8)]
    return g, latent, data


def test_mix_trace_artifacts_are_hashed_and_correlated(tmp_path):
    g, latent, data = mix_payload()
    report = save_diagnostic_artifacts(tmp_path, "mix", latent, torch.zeros(1, 8, 8, 3), data, g, None, "11")
    stored = load_file(str(tmp_path / report["artifacts"]["tensors"]))
    assert torch.equal(stored["trace_inputs"], data.extra_tensors["trace_inputs"])
    assert report["tensor_manifest"]["trace_predictions"]["sha256"] == tensor_hash(stored["trace_predictions"])


@pytest.mark.parametrize("kind", ["nan", "wrong_length", "reserved", "prediction"])
def test_invalid_trace_refused_before_any_write(tmp_path, kind):
    g, latent, data = mix_payload()
    if kind == "nan": data.extra_tensors["trace_inputs"][0] = float("nan")
    if kind == "wrong_length": data.extra_tensors["trace_sigmas"] = torch.ones(7)
    if kind == "reserved": data.extra_tensors["final_latent"] = latent
    if kind == "prediction": data.extra_tensors["trace_predictions"][0] = 1
    with pytest.raises(ValueError):
        save_diagnostic_artifacts(tmp_path, "mix", latent, torch.zeros(1, 8, 8, 3), data, g, None, "11")
    assert not list(tmp_path.iterdir())
