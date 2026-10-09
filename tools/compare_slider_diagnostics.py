"""Compare saved diagnostic runs without requiring ComfyUI or a GPU."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_report(path):
    import torch
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import load_file
    from slider_fuse.diagnostics import file_hash, validate_prefix, validate_audit_coverage, validate_mask_artifacts
    from slider_fuse.sampling import tensor_hash

    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("complete") is not True or report.get("provenance_level") != "diagnostic_run":
        raise ValueError("Incomplete or unsupported diagnostic manifest")
    revision = report.get("environment", {}).get("comfy_revision")
    if not isinstance(revision, str) or not revision or revision.lower() == "unknown":
        raise ValueError("invalid_comparison: ComfyUI revision is unavailable; use a Git checkout with identifiable revision")
    files = {}
    for kind in ("image", "effective_mask", "tensors"):
        name = report["artifacts"][kind]
        validate_prefix(name, flat=True)
        files[kind] = path.parent / name
        if file_hash(files[kind]) != report["artifact_sha256"][kind]:
            raise ValueError(f"Artifact file hash mismatch: {kind}")
    tensors = load_file(str(files["tensors"]), device="cpu")
    with safe_open(str(files["tensors"]), framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
    if any(metadata.get(k) != report[k] for k in ("artifact_id", "run_id")):
        raise ValueError("Tensor artifact ID mismatch")
    for kind in ("final_latent", "first_prediction"):
        if tensor_hash(tensors[kind]) != report[kind + "_sha256"]:
            raise ValueError(f"Tensor content hash mismatch: {kind}")
    im = Image.open(files["image"])
    mask = Image.open(files["effective_mask"])
    if any(im.info.get(k) != report[k] or mask.info.get(k) != report[k] for k in ("artifact_id", "run_id")):
        raise ValueError("PNG artifact ID mismatch")
    schema = report.get("diagnostic_schema_version", 1)
    if schema not in (1, 2, 3):
        raise ValueError("Unsupported diagnostic schema version")
    if schema in (2, 3):
        validate_audit_coverage(report)
        manifest = report.get("tensor_manifest", {})
        if set(manifest) != set(tensors):
            raise ValueError("Diagnostic tensor manifest keys mismatch")
        for name, value in tensors.items():
            entry = manifest[name]
            if (tensor_hash(value) != entry["sha256"] or list(value.shape) != entry["shape"]
                    or str(value.dtype) != entry["dtype"] or not torch.isfinite(value).all()):
                raise ValueError("Diagnostic tensor manifest mismatch: " + name)
        if "trace_predictions" in tensors:
            rows = report.get("step_trace", [])
            if len(rows) != report["steps"] or tensors["trace_predictions"].shape[0] != report["steps"]:
                raise ValueError("Diagnostic trace length mismatch")
            for i, row in enumerate(rows):
                if (row["eval_index"] != i or row["input_sha256"] != tensor_hash(tensors["trace_inputs"][i])
                        or row["prediction_sha256"] != tensor_hash(tensors["trace_predictions"][i])
                        or row["sigma"] != float(tensors["trace_sigmas"][i])):
                    raise ValueError("Diagnostic trace manifest mismatch")
        if schema == 3:
            mask_tensor = torch.frombuffer(bytearray(mask.convert("L").tobytes()), dtype=torch.uint8).float().reshape(1, mask.height, mask.width) / 255
            validate_mask_artifacts(report, tensors, mask_tensor)
    rgb = torch.frombuffer(bytearray(im.convert("RGB").tobytes()), dtype=torch.uint8).reshape(im.height, im.width, 3).double()
    return report, tensors, rgb


def direct_routing(report):
    from slider_fuse.diagnostics import validate_audit_coverage
    validate_audit_coverage(report)
    if report.get("diagnostic_schema_version", 1) not in (2, 3) or report.get("diagnostic_level") != "audit":
        return {"status": "unavailable", "measured_elements": 0, "violations": [], "note": "Audit not requested or legacy schema"}
    violations = []
    measured = 0
    for row in report.get("linear_audit") or []:
        for region in (row["image"]["unselected"], row["text"]["unselected"]):
            measured += region["element_count"]
            if region["changed_elements"] or region["nonfinite_count"]:
                violations.append({"eval_index": row["eval_index"], "module": row["module"], "region": region})
    for row in report.get("prediction_mix_audit") or []:
        for name in ("base_excluded", "slider_selected"):
            region = row[name]
            measured += region["element_count"]
            if region["changed_elements"] or region["nonfinite_count"]:
                violations.append({"eval_index": row["eval_index"], "region_name": name, "region": region})
    return {"status": "direct_routing_violation" if violations else "measured_zero" if measured else "unavailable",
            "measured_elements": measured, "violations": violations,
            "note": "Same-input direct deltas; does not establish protected image invariance."}


def trajectory_comparison(a, ta, b, tb):
    import torch
    from torch.nn import functional as F
    from slider_fuse.diagnostics import tensor_metrics
    from slider_fuse.prediction_mixing import prediction_mask
    required = {"trace_inputs", "trace_predictions", "trace_sigmas"}
    if not required <= set(ta) or not required <= set(tb):
        return None
    if ta["trace_sigmas"].shape != tb["trace_sigmas"].shape or not torch.equal(ta["trace_sigmas"], tb["trace_sigmas"]):
        raise ValueError("Trajectory sigma schedules differ")
    regions = {}
    reference = ta if "reference_target_mask" in ta else tb
    if "reference_target_mask" in reference:
        regions = {k: reference["reference_" + k + "_mask"] for k in ("target", "protected", "background")}
        target = regions["target"][None].float()
        outside = F.max_pool2d(target, 3, stride=1, padding=1) > 0
        inside = -F.max_pool2d(-target, 3, stride=1, padding=1) == 1
        regions["boundary"] = (outside & ~inside)[0]
    rows = []
    for i, (pa, pb) in enumerate(zip(ta["trace_predictions"], tb["trace_predictions"])):
        same_input = torch.equal(ta["trace_inputs"][i], tb["trace_inputs"][i])
        metrics = {}
        for name, token in regions.items():
            selector = prediction_mask(token, pa, patch=a["patch_size"]).expand_as(pa)
            metrics[name] = tensor_metrics(pa[selector], pb[selector]) if selector.any() else {"unavailable_reason": "empty region"}
        rows.append({"eval_index": i, "sigma": float(ta["trace_sigmas"][i]),
            "comparison_kind": "same_input_prediction_difference" if same_input else "trajectory_difference",
            "input": tensor_metrics(ta["trace_inputs"][i], tb["trace_inputs"][i]),
            "prediction": tensor_metrics(pa, pb), "regions": metrics or None})
    return rows


def _generation_for_comparison(report, endpoint_reference):
    generation = dict(report["generation"])
    if endpoint_reference and report.get("backend") == "prediction_mix":
        if report.get("mix_scope") == "none" or report["strength"] == 0:
            expected = {"base": report["steps"], "slider": 0}
            if report.get("branch_nfe") != expected or report.get("effective_slider_strength") != 0:
                raise ValueError("Unproven prediction-mix base endpoint")
            generation["strength"] = 0.
    return generation


def compare_reports(paths, *, atol=None, rtol=None, include_trajectories=False, endpoint_reference=False,
                    same_mask_reference=False):
    from slider_fuse.diagnostics import tensor_metrics
    if len(paths) < 2: raise ValueError("Specify at least two diagnostic reports")
    if same_mask_reference and (len(paths) != 2 or endpoint_reference):
        raise ValueError("Same-mask reference requires exactly one auto/manual pair")
    runs = [load_report(path) for path in paths]
    pairs = []
    invariants = ("lora_sha256", "initial_noise_sha256", "initial_latent_sha256", "full_sigmas_sha256",
                  "conditioning_sha256", "first_input_sha256", "first_timestep_sha256",
                  "first_input_shape", "first_input_dtype", "reference_partition_sha256", "environment")
    for i, j in itertools.combinations(range(len(runs)), 2):
        a, ta, ia = runs[i]; b, tb, ib = runs[j]
        if a["run_id"] == b["run_id"]:
            raise ValueError("Same cached run saved twice; change trial_id for a new trial")
        if _generation_for_comparison(a, endpoint_reference) != _generation_for_comparison(b, endpoint_reference):
            raise ValueError("invalid_comparison: generation conditions differ")
        for key in invariants:
            if key not in a or key not in b or a[key] != b[key]:
                raise ValueError(f"invalid_comparison: conditions differ or are missing: {key}")
        if ta["final_latent"].dtype != tb["final_latent"].dtype or ta["first_prediction"].dtype != tb["first_prediction"].dtype:
            raise ValueError("invalid_comparison: output dtypes differ")
        configs = [r.get("mask_generation") for r in (a, b)]
        from slider_fuse.diagnostics import prediction_selection_config
        selections = [r.get("prediction_selection", prediction_selection_config({})) for r in (a,b)]
        for index, r in enumerate((a, b)):
            if configs[index] is None:
                if r.get("mask_mode", "manual") != "manual":
                    raise ValueError("Legacy automatic mask provenance is unavailable")
                from slider_fuse.diagnostics import mask_generation_config
                configs[index] = mask_generation_config({"mask_mode": "manual"})
        if same_mask_reference:
            if (any(r.get("diagnostic_schema_version") != 3 or r.get("backend") != "prediction_mix" for r in (a,b))
                    or {c["mode"] for c in configs} != {"auto", "manual"}
                    or selections[0] != selections[1]
                    or a["mix_scope"] != b["mix_scope"]
                    or a["effective_image_mask_sha256"] != b["effective_image_mask_sha256"]):
                raise ValueError("Same-mask reference requires identical masks/scope and proven auto/manual sources")
        elif configs[0] != configs[1]:
            raise ValueError("invalid_comparison: mask generation settings differ")
        both_v2 = all(r.get("diagnostic_schema_version") in (2, 3) for r in (a, b))
        if both_v2:
            for key in ("implementation_revision", "implementation_source_sha256", "prediction_space"):
                if not a.get(key) and key != "implementation_revision":
                    raise ValueError("Missing implementation or prediction identity")
                if a.get(key) != b.get(key):
                    raise ValueError("invalid_comparison: implementation/prediction identity differs: " + key)
        pairs.append({"reports": [str(paths[i]), str(paths[j])], "run_ids": [a["run_id"], b["run_id"]],
            "comparison_kind": "same_mask_phase2_comparison" if same_mask_reference else
                               "selection_policy_comparison" if selections[0] != selections[1] else "same_conditions_comparison",
            "cases": [a.get("case_id"), b.get("case_id")],
            "repeat_trial": configs[0] == configs[1] and selections[0] == selections[1] and all(a.get(k) == b.get(k) for k in ("backend", "image_scope", "text_scope", "strength")),
            "first_prediction": tensor_metrics(ta["first_prediction"], tb["first_prediction"], atol=atol, rtol=rtol),
            "final_latent": tensor_metrics(ta["final_latent"], tb["final_latent"], atol=atol, rtol=rtol),
            "rgb": tensor_metrics(ia, ib), "direct_routing": [direct_routing(a), direct_routing(b)],
            "prediction_comparison_kind": "same_input_prediction_difference" if both_v2 else "legacy_prediction_space_unverified",
            "trajectory": trajectory_comparison(a, ta, b, tb) if include_trajectories and both_v2 else None,
            "trajectory_unavailable_reason": None if include_trajectories and both_v2 and
                "trace_predictions" in ta and "trace_predictions" in tb else "trace missing, legacy schema, or not requested",
            "branch_nfe": [a.get("branch_nfe"), b.get("branch_nfe")]})
    status = "measured_only" if atol is None else "within_tolerance" if all(
        pair[key]["within_declared_tolerance"] for pair in pairs for key in ("first_prediction", "final_latent")) else "mismatch"
    if any(r["status"] == "direct_routing_violation" for pair in pairs for r in pair["direct_routing"]):
        status = "direct_routing_violation"
    return {"status": status,
        "note": "RGB differences are not an age or local-effect score.", "pairs": pairs}


def compare_standard(latent_path, png_path, case, against, *, atol=None, rtol=None):
    import torch
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import load_file
    from slider_fuse.diagnostics import file_hash, generation_config, linked_node, tensor_metrics

    report, tensors, rgb = load_report(against)
    im = Image.open(png_path)
    with safe_open(str(latent_path), framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
    if not im.info.get("prompt") or not metadata.get("prompt"):
        raise ValueError("Standard PNG and latent must contain prompt metadata")
    prompt = json.loads(im.info["prompt"])
    if prompt != json.loads(metadata["prompt"]):
        raise ValueError("Standard PNG/latent prompt mismatch")
    # Require one explicit standard save pair, tracing both to the same KSampler.
    images = [key for key, node in prompt.items() if node["class_type"] == "SaveImage"]
    latents = [key for key, node in prompt.items() if node["class_type"] == "SaveLatent"]
    if len(images) != 1 or len(latents) != 1: raise ValueError("Ambiguous standard save pair")
    decode_id, _ = linked_node(prompt, images[0], "images", "VAEDecode")
    sampler_id, _ = linked_node(prompt, decode_id, "samples", "KSampler")
    latent_sampler_id, _ = linked_node(prompt, latents[0], "samples", "KSampler")
    if sampler_id != latent_sampler_id: raise ValueError("Standard save pair uses different Samplers")
    vae_id, _ = linked_node(prompt, decode_id, "vae", "VAELoader")
    config = generation_config(prompt, sampler_id, vae_id, standard_case=case)
    expected = dict(report["generation"])
    if case == "standard_zero" and expected["strength"] == 0.: expected["slider_lora"] = None
    if config != expected: raise ValueError("invalid_comparison: standard generation conditions differ")
    saved = load_file(str(latent_path), device="cpu")
    if "latent_format_version_0" not in saved or "latent_tensor" not in saved:
        raise ValueError("Unsupported standard latent format; expected latent_format_version_0")
    actual = saved["latent_tensor"]  # SaveLatent v0 stores raw samples, no scale conversion.
    if actual.dtype != tensors["final_latent"].dtype: raise ValueError("Standard latent dtype mismatch")
    irgb = torch.frombuffer(bytearray(im.convert("RGB").tobytes()), dtype=torch.uint8).reshape(im.height, im.width, 3).double()
    latent_metrics = tensor_metrics(tensors["final_latent"], actual, atol=atol, rtol=rtol)
    return {"status": latent_metrics["status"], "provenance_level": "prompt_and_explicit_pair", "standard_case": case,
        "generation": config, "first_prediction": None,
        "unavailable": ["first_prediction", "noise/sigma hashes", "shared run ID guarantee"],
        "standard_file_sha256": {"latent": file_hash(latent_path), "image": file_hash(png_path)},
        "latent_conversion": "ComfyUI SaveLatent version_0: raw latent_tensor, scale=1",
        "final_latent": latent_metrics,
        "rgb": tensor_metrics(rgb, irgb)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="*")
    parser.add_argument("--output", help="Write JSON comparison report")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--standard-latent")
    parser.add_argument("--standard-png")
    parser.add_argument("--standard-case", choices=["standard_zero", "standard_global"])
    parser.add_argument("--against")
    parser.add_argument("--include-trajectories", action="store_true")
    parser.add_argument("--same-mask-reference", action="store_true",
                        help="Compare schema 3 auto/manual Phase 2 with an identical reference partition and effective mask")
    parser.add_argument("--endpoint-reference", action="store_true",
                        help="Allow proven mix none/zero endpoint versus native zero; all other settings must match")
    args = parser.parse_args()
    try:
        if (args.atol is None) != (args.rtol is None): raise ValueError("Specify both --atol and --rtol")
        standard = [args.standard_latent, args.standard_png, args.standard_case, args.against]
        if any(standard) and args.same_mask_reference:
            raise ValueError("Same-mask reference applies only to diagnostic auto/manual reports")
        if any(standard):
            if not all(standard) or args.reports: raise ValueError("Standard comparison requires all four flags and no report list")
            result = compare_standard(*standard, atol=args.atol, rtol=args.rtol)
        else:
            result = compare_reports(args.reports, atol=args.atol, rtol=args.rtol,
                include_trajectories=args.include_trajectories, endpoint_reference=args.endpoint_reference,
                same_mask_reference=args.same_mask_reference)
    except (ValueError, KeyError, OSError, TypeError) as error:
        result = {"status": "invalid_comparison", "error": str(error)}
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output: Path(args.output).write_text(serialized, encoding="utf-8")
    print(serialized)
    if result.get("status") in ("invalid_comparison", "mismatch", "direct_routing_violation") or result.get("final_latent", {}).get("status") == "mismatch":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
