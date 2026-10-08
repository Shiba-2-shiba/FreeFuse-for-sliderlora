import json
import subprocess
import sys

import pytest
import torch

from test_diagnostic_tools import tool, saved
from test_prediction_mix_artifacts import mix_payload
from slider_fuse.diagnostics import save_diagnostic_artifacts
from slider_fuse.sampling import tensor_hash


def saved_v2(tmp_path, run_id, *, later_change=False):
    g, latent, data = mix_payload()
    if later_change:
        data.extra_tensors["trace_inputs"][1:] += 1
        data.extra_tensors["trace_predictions"][1:] += .1
        for i in range(1, 8):
            data.report["step_trace"][i].update(input_sha256=tensor_hash(data.extra_tensors["trace_inputs"][i]),
                prediction_sha256=tensor_hash(data.extra_tensors["trace_predictions"][i]))
    data.report.update(run_id=run_id, initial_noise_sha256="noise", initial_latent_sha256="latent",
        full_sigmas_sha256="sigmas", conditioning_sha256="cond",
        first_input_sha256=tensor_hash(data.extra_tensors["trace_inputs"][0]), first_timestep_sha256="sigma",
        first_input_shape=[1, 4, 4, 4], first_input_dtype="torch.float32", lora_sha256="lora",
        reference_partition_sha256={"target": "target", "protected": "protected", "background": "background"},
        environment={"comfy_revision": "test"}, implementation_revision="rev",
        implementation_source_sha256="source", prediction_space="comfy_cfg1_denoised",
        linear_audit=None, prediction_mix_audit=[], patch_size=2)
    result = save_diagnostic_artifacts(tmp_path, run_id, latent, torch.zeros(1, 8, 8, 3), data, g, None, "11")
    return tmp_path / result["artifacts"]["report"]


def test_different_later_inputs_are_not_same_input_predictions(tmp_path):
    a = saved_v2(tmp_path, "a")
    b = saved_v2(tmp_path, "b", later_change=True)
    pair = tool().compare_reports([a, b], include_trajectories=True)["pairs"][0]
    assert pair["trajectory"][0]["comparison_kind"] == "same_input_prediction_difference"
    assert pair["trajectory"][1]["comparison_kind"] == "trajectory_difference"
    assert pair["trajectory"][1]["prediction"]["mae"] > 0


def test_implementation_mismatch_is_not_same_input_comparison(tmp_path):
    a, b = saved_v2(tmp_path, "a"), saved_v2(tmp_path, "b")
    report = json.loads(b.read_text()); report["implementation_source_sha256"] = "different"
    b.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="implementation"):
        tool().compare_reports([a, b])


def test_direct_violation_is_reported_and_has_failing_exit(tmp_path):
    a, b = saved_v2(tmp_path, "a"), saved_v2(tmp_path, "b")
    report = json.loads(b.read_text())
    report["prediction_mix_audit"] = [{"eval_index": 0, "base_excluded": {
        "element_count": 10, "max_abs": 1., "changed_elements": 1, "nonfinite_count": 0},
        "slider_selected": {"element_count": 10, "max_abs": 0., "changed_elements": 0, "nonfinite_count": 0}}]
    b.write_text(json.dumps(report))
    compare = tool()
    assert compare.compare_reports([a, b])["status"] == "direct_routing_violation"
    result = subprocess.run([sys.executable, "-B", compare.__file__, str(a), str(b)], capture_output=True, text=True)
    assert result.returncode == 1


def test_legacy_common_metrics_still_load_without_fake_audit(tmp_path):
    a, b = saved(tmp_path, "a"), saved(tmp_path, "b")
    pair = tool().compare_reports([a, b], include_trajectories=True)["pairs"][0]
    assert pair["trajectory"] is None
    assert pair["direct_routing"][0]["status"] == "unavailable"


def test_trace_hash_tampering_rejected(tmp_path):
    a = saved_v2(tmp_path, "a")
    report = json.loads(a.read_text())
    report["tensor_manifest"]["trace_inputs"]["sha256"] = "wrong"
    a.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="manifest"):
        tool().load_report(a)


def test_base_endpoint_requires_explicit_comparison_and_proven_zero_slider_calls(tmp_path):
    a, b = saved_v2(tmp_path, "mix"), saved_v2(tmp_path, "reference")
    ra = json.loads(a.read_text()); rb = json.loads(b.read_text())
    ra.update(mix_scope="none", effective_slider_strength=0., branch_nfe={"base": 8, "slider": 0})
    rb.update(backend="native", strength=0.)
    rb["generation"]["strength"] = 0.
    a.write_text(json.dumps(ra)); b.write_text(json.dumps(rb))
    with pytest.raises(ValueError, match="generation"):
        tool().compare_reports([a, b])
    assert tool().compare_reports([a, b], endpoint_reference=True)["pairs"][0]["final_latent"]["exact_equal"]
    ra["branch_nfe"]["slider"] = 8
    a.write_text(json.dumps(ra))
    with pytest.raises(ValueError, match="endpoint"):
        tool().compare_reports([a, b], endpoint_reference=True)
