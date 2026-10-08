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
    from slider_fuse.diagnostics import file_hash, validate_prefix
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
    rgb = torch.frombuffer(bytearray(im.convert("RGB").tobytes()), dtype=torch.uint8).reshape(im.height, im.width, 3).double()
    return report, tensors, rgb


def compare_reports(paths, *, atol=None, rtol=None):
    from slider_fuse.diagnostics import tensor_metrics
    if len(paths) < 2: raise ValueError("Specify at least two diagnostic reports")
    runs = [load_report(path) for path in paths]
    pairs = []
    invariants = ("lora_sha256", "initial_noise_sha256", "initial_latent_sha256", "full_sigmas_sha256",
                  "conditioning_sha256", "first_input_sha256", "first_timestep_sha256",
                  "first_input_shape", "first_input_dtype", "reference_partition_sha256", "environment")
    for i, j in itertools.combinations(range(len(runs)), 2):
        a, ta, ia = runs[i]; b, tb, ib = runs[j]
        if a["run_id"] == b["run_id"]:
            raise ValueError("Same cached run saved twice; change trial_id for a new trial")
        for key in ("generation",) + invariants:
            if key not in a or key not in b or a[key] != b[key]:
                raise ValueError(f"invalid_comparison: conditions differ or are missing: {key}")
        if ta["final_latent"].dtype != tb["final_latent"].dtype or ta["first_prediction"].dtype != tb["first_prediction"].dtype:
            raise ValueError("invalid_comparison: output dtypes differ")
        pairs.append({"reports": [str(paths[i]), str(paths[j])], "run_ids": [a["run_id"], b["run_id"]],
            "cases": [a.get("case_id"), b.get("case_id")],
            "repeat_trial": all(a.get(k) == b.get(k) for k in ("backend", "image_scope", "text_scope", "strength")),
            "first_prediction": tensor_metrics(ta["first_prediction"], tb["first_prediction"], atol=atol, rtol=rtol),
            "final_latent": tensor_metrics(ta["final_latent"], tb["final_latent"], atol=atol, rtol=rtol),
            "rgb": tensor_metrics(ia, ib)})
    return {"status": "measured_only" if atol is None else "within_tolerance" if all(
        pair[key]["within_declared_tolerance"] for pair in pairs for key in ("first_prediction", "final_latent")) else "mismatch",
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
    args = parser.parse_args()
    try:
        if (args.atol is None) != (args.rtol is None): raise ValueError("Specify both --atol and --rtol")
        standard = [args.standard_latent, args.standard_png, args.standard_case, args.against]
        if any(standard):
            if not all(standard) or args.reports: raise ValueError("Standard comparison requires all four flags and no report list")
            result = compare_standard(*standard, atol=args.atol, rtol=args.rtol)
        else:
            result = compare_reports(args.reports, atol=args.atol, rtol=args.rtol)
    except (ValueError, KeyError, OSError, TypeError) as error:
        result = {"status": "invalid_comparison", "error": str(error)}
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output: Path(args.output).write_text(serialized, encoding="utf-8")
    print(serialized)
    if result.get("status") in ("invalid_comparison", "mismatch") or result.get("final_latent", {}).get("status") == "mismatch":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
