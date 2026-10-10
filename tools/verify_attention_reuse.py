"""Fail-closed, offline eligibility checks for the saved k3-bg/+4 reuse path.

No inference, copying, resizing, asset-record creation, or report rewriting occurs.
The original suite bundle and the complete +4 diagnostic bundle must be present.
Input masks must be byte-identical PNG copies inside the supplied ComfyUI input
folder; LoadImageMask must use the red channel without inversion.

Asset wrapper JSON (paths relative to this wrapper, or absolute local paths):
  {"schema_version": 1,
   "historical_record": {"path": "retained-assets.json", "sha256": "<64 hex>"},
   "current_paths": {"checkpoint": "...", "style_lora": "...", "clip": "...",
                     "vae": "...", "slider": "..."},
   "historical_binding": {"kind": "human_provenance_attestation",
       "attested_by": "operator", "statement": "Explain the retained evidence",
       "suite_manifest_sha256": "<64 hex>", "edit_report_sha256": "<64 hex>"}}
The separately retained historical record has schema_version:1 and assets mapping
those same five roles to {"name": "original graph asset name", "sha256": "<64 hex>"}.
It must contain hashes retained from the historical run, not hashes newly computed
and called historical. This tool verifies the retained record and current bytes,
but cannot independently verify when the record was created or which checkpoint,
style, CLIP or VAE the historical process loaded. The explicit human attestation
is a trust boundary, not a machine-verified historical weights claim. The Slider
hash additionally must equal the historical diagnostic's recorded lora_sha256.

Eligibility is not a completed comparison. New execution hashes, actual loaded
assets, observation parity, and visual/pose quality still need post-run checks.
The suite's weights_identity_verified field is preserved, never upgraded.
"""
from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re


VARIANT = "multi_proto_bg"
ASSET_ROLES = ("checkpoint", "style_lora", "clip", "vae", "slider")
HASH_FIELDS = ("initial_noise_sha256", "initial_latent_sha256", "full_sigmas_sha256")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value, label):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             "Missing or invalid SHA-256: " + label)
    return value


def _read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("Nonfinite JSON value: " + value)
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                       parse_constant=nonfinite)
    _require(isinstance(value, dict), "Expected a JSON object: " + str(path))
    return value


def _relative_png(root, value):
    name = str(value)
    _require(name and name == name.strip() and not any(ord(c) < 32 or ord(c) == 127 for c in name),
             "Mask filename must be a nonempty ComfyUI input-relative path without control characters")
    windows = PureWindowsPath(name)
    parts = name.split("/")
    _require(not windows.drive and not windows.is_absolute() and not name.startswith("/")
             and "\\" not in name and all(part not in ("", ".", "..") for part in parts)
             and PurePosixPath(name).suffix.lower() == ".png"
             and not any(c in name for c in ":\x00[]"),
             "Mask must be a ComfyUI input-relative PNG filename: " + name)
    path = (root / name).resolve()
    _require(path.is_relative_to(root), "Mask escapes the ComfyUI input folder")
    _require(path.is_file(), "Missing ComfyUI input mask: " + name)
    return name, path


def _bundle_file(manifest_path, reference):
    _require(isinstance(reference, str), "Missing suite artifact reference")
    parts = reference.split("/")
    _require(len(parts) == 2 and parts[0] == manifest_path.parent.name
             and parts[1] not in ("", ".", "..") and "\\" not in reference
             and ":" not in reference, "Suite artifact is outside its original bundle")
    path = (manifest_path.parent / parts[1]).resolve()
    _require(path.parent == manifest_path.parent, "Suite artifact escapes its bundle")
    return path


def _verify_assets(path, suite_path, edit_path, suite, generation, edit, file_hash):
    wrapper = _read_json(path)
    _require(type(wrapper.get("schema_version")) is int and wrapper["schema_version"] == 1,
             "Unsupported asset record schema")
    historical_ref = wrapper["historical_record"]
    retained_path = (path.parent / historical_ref["path"]).resolve()
    _require(retained_path != path, "Historical record must be a separate retained file")
    retained_hash = _digest(historical_ref.get("sha256"), "retained historical asset record")
    _require(file_hash(retained_path) == retained_hash, "Retained historical asset record hash mismatch")
    retained = _read_json(retained_path)
    _require(type(retained.get("schema_version")) is int and retained["schema_version"] == 1,
             "Unsupported retained historical asset schema")
    binding = wrapper["historical_binding"]
    _require(binding.get("kind") == "human_provenance_attestation",
             "An explicit human historical-provenance attestation is required")
    for key in ("attested_by", "statement"):
        _require(isinstance(binding.get(key), str) and binding[key].strip(),
                 "Historical provenance attestation needs " + key)
    for key, source in (("suite_manifest_sha256", suite_path), ("edit_report_sha256", edit_path)):
        _require(_digest(binding.get(key), key) == file_hash(source),
                 "Historical attestation does not bind this " + key)
    names = {"checkpoint": generation["checkpoint"], "style_lora": generation["style_loras"][0]["lora_name"],
             "clip": generation["clip_name"], "vae": generation["vae_name"], "slider": generation["slider_lora"]}
    _require(set(retained["assets"]) == set(ASSET_ROLES) and set(wrapper["current_paths"]) == set(ASSET_ROLES),
             "Supply retained identities and current local paths for exactly five asset roles")
    assets = {}
    for role in ASSET_ROLES:
        identity = retained["assets"][role]
        expected = _digest(identity.get("sha256"), "historical " + role)
        _require(identity.get("name") == names[role], "Historical asset name differs from generation_config: " + role)
        local = (path.parent / wrapper["current_paths"][role]).resolve()
        _require(file_hash(local) == expected, "Current asset differs from retained historical hash: " + role)
        if role == "slider":
            _require(expected == _digest(edit.get("lora_sha256"), "diagnostic Slider"),
                     "Slider hash differs from the historical diagnostic")
        assets[role] = {"name": names[role], "sha256": expected, "current_path": str(local)}
    return {"current_files_match_retained_hashes": True,
        "historical_assignment": "human_attested_not_independently_verified",
        "historical_provenance_attestation": deepcopy(binding),
        "trust_boundary": "The attestation assigns retained hashes to the historical runs; its truth and record age cannot be independently established. Current-file hashes do not prove historical or future loaded weights.",
        "suite_weights_identity_verified": suite["report"]["model"].get("weights_identity_verified"),
        "suite_weights_identity_reason": suite["report"]["model"].get("weights_identity_reason"),
        "retained_record": {"path": str(retained_path), "sha256": retained_hash},
        "wrapper": {"path": str(path), "sha256": file_hash(path)}, "assets": assets}


def _token_and_runtime_provenance(report, edit):
    count = report["text_token_count"]
    _require(type(count) is int and count > 0 and type(edit.get("text_token_count")) is int
             and count == edit["text_token_count"], "Suite/edit text token counts differ")
    positions = report["token_positions"]
    _require(isinstance(positions, dict) and set(positions) == {"target", "protected", "background"},
             "Suite must preserve target, protected and background token positions")
    observed = set()
    for role in ("target", "protected", "background"):
        values = positions[role]
        _require(isinstance(values, list) and values and all(type(v) is int and 0 <= v < count for v in values)
                 and len(set(values)) == len(values) and not observed.intersection(values),
                 "Invalid, overlapping or out-of-range token positions: " + role)
        observed.update(values)
        if role != "background":
            _require(values == edit["token_positions"].get(role), "Suite/edit subject token positions differ: " + role)
    phrase, occurrence = report["background_phrase"], report["background_occurrence"]
    _require(isinstance(phrase, str) and phrase.strip() and phrase in report["prompt"],
             "Missing or incompatible historical background_phrase")
    _require(type(occurrence) is int and occurrence >= 0, "Invalid historical background_occurrence")
    implementation = deepcopy(report.get("implementation"))
    edit_runtime = {key: deepcopy(edit.get(key)) for key in
        ("implementation_source_sha256", "implementation_revision", "implementation_dirty", "environment")}
    sources_recorded = (isinstance(implementation, dict)
        and all(isinstance(item.get("implementation_source_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", item["implementation_source_sha256"])
                and isinstance(item.get("implementation_revision"), str) and item["implementation_revision"]
                for item in (implementation, edit_runtime))
        and isinstance(edit_runtime["environment"], dict) and edit_runtime["environment"].get("comfy_revision"))
    suite_metadata = deepcopy(report["conditioning"].get("positive_metadata"))
    edit_metadata = deepcopy(edit.get("conditioning", {}).get("positive_metadata"))
    if suite_metadata is not None and edit_metadata is not None:
        _require(suite_metadata == edit_metadata, "Suite/edit positive conditioning metadata mismatch")
        metadata_status = "matching_recorded_values_requires_post_run_match"
    elif edit_metadata is None:
        metadata_status = "unverified_not_recorded_in_historical_edit"
    else:
        metadata_status = "unverified_not_recorded_in_historical_suite"
    return {"text_token_count": count, "token_positions": deepcopy(positions),
        "background_phrase": phrase, "background_occurrence": occurrence,
        "edit_runtime_provenance": edit_runtime, "suite_implementation": implementation,
        "runtime_provenance_status": "recorded_requires_post_run_match_or_review" if sources_recorded else "incomplete_unverified",
        "conditioning_metadata": {"suite": suite_metadata, "edit": edit_metadata, "status": metadata_status}}


def _verify_reuse(suite_manifest, edit_report, input_dir, target_image, protected_image, asset_record):
    # Imports remain lazy so the normal cold workflow generator stays stdlib-only.
    import torch
    from safetensors import safe_open
    from slider_fuse.diagnostics import file_hash
    from slider_fuse.experimental_masks import validate_experimental_config
    from slider_fuse.sampling import tensor_hash
    from tools.audit_mask_support import load_binary_png
    from tools.compare_slider_diagnostics import load_report

    suite_path, edit_path, asset_path = (Path(value).resolve() for value in (suite_manifest, edit_report, asset_record))
    input_root = Path(input_dir).resolve()
    _require(input_root.is_dir(), "ComfyUI input directory must already exist")
    suite = _read_json(suite_path)
    _require(suite.get("complete") is True and type(suite.get("experiment_schema_version")) is int
             and suite["experiment_schema_version"] == 1, "Incomplete or unsupported runtime suite manifest")
    report = suite["report"]
    _require(type(report.get("experiment_schema_version")) is int and report["experiment_schema_version"] == 1,
             "Unsupported runtime suite report schema")
    _require(report["model"].get("weights_identity_verified") is False,
             "Schema-1 suite weights_identity_verified must remain false; retained asset attestation is separate")
    config = report["normalized_config"]
    _require(config == validate_experimental_config(config) and config["prototypes"] == 3
             and config["collect_step"] == 2, "Reuse requires a fully normalized prototypes=3 suite config")
    _require(report["algorithm"]["variants"][VARIANT].get("status") == "ok",
             "Historical multi_proto_bg candidate did not succeed")
    for key, value in (("steps", 8), ("cfg", 1.), ("sampler", "euler"), ("scheduler", "simple"),
                       ("phase1_nfe", 2), ("phase2_nfe", 0)):
        _require(report.get(key) == value and not isinstance(report.get(key), bool), "Unexpected suite " + key)
    _require(report.get("completed_reference_image_generated") is False and report.get("slider_loaded") is False,
             "Suite must be a style-only two-forward prefix without an OFF image")
    grid = report["grid"]
    _require(isinstance(grid, list) and len(grid) == 2 and all(type(v) is int and v > 0 for v in grid),
             "Invalid suite mask grid")
    height, width = grid
    artifacts = suite["artifacts"]
    _require(_bundle_file(suite_path, artifacts["manifest"]) == suite_path, "Suite manifest reference mismatch")
    # Check every retained suite artifact, not just its name or PNG rendering.
    _require(isinstance(suite.get("file_sha256"), dict) and suite["file_sha256"], "Missing suite file hashes")
    for name, expected in suite["file_sha256"].items():
        path = _bundle_file(suite_path, suite_path.parent.name + "/" + name)
        _require(file_hash(path) == _digest(expected, "suite artifact " + name), "Suite artifact file hash mismatch: " + name)
    maps_path = _bundle_file(suite_path, artifacts["tensors"])
    _require(maps_path.name in suite["file_sha256"], "Missing suite tensor-file hash")

    # Reuse the established strict reader; do not weaken comparison validation.
    _read_json(edit_path)  # Reject ambiguous duplicate keys/nonfinite JSON first.
    edit, edit_tensors, _ = load_report(edit_path)
    _require(edit.get("diagnostic_schema_version") == 3 and edit.get("backend") == "prediction_mix",
             "Reuse requires the original schema-3 prediction-mix diagnostic bundle")
    for key, value in (("mask_mode", "manual"), ("mix_scope", "target_mask"), ("strength", 4.),
                       ("steps", 8), ("cfg", 1.), ("sampler", "euler"), ("scheduler", "simple")):
        _require(edit.get(key) == value and not isinstance(edit.get(key), bool), "Historical +4 diagnostic has incompatible " + key)
    _require(edit["mask_generation"]["fill_holes_max_area"] == 0
             and edit["mask_generation"]["mask_dilate_radius"] == 0
             and edit["prediction_selection"]["dilate_radius"] == 0, "All historical dilation/fill controls must be zero")
    _require(edit.get("grid") == grid, "Suite and historical edit mask grids differ")
    generation = edit["generation"]
    for key in ("seed", "steps", "cfg", "sampler", "scheduler", "prompt", "strength"):
        _require(key in generation and generation[key] == edit[key], "generation_config disagrees with diagnostic: " + key)
    _require(generation.get("batch_size") == 1 and generation.get("clip_type") == "krea2"
             and generation.get("denoise") == 1. and len(generation["style_loras"]) == 1,
             "Unsupported historical generation_config")
    for key in ("seed", "prompt"):
        _require(report.get(key) == edit.get(key), "Suite and historical edit differ: " + key)
    invariants = _token_and_runtime_provenance(report, edit)
    for key in HASH_FIELDS:
        expected = _digest(report.get(key), "suite " + key)
        _require(expected == _digest(edit.get(key), "edit " + key), "Suite/edit hash mismatch: " + key)
        invariants[key] = expected
    sigmas = torch.tensor(report["full_sigmas"], dtype=torch.float32)
    _require(sigmas.shape == (9,) and bool(torch.isfinite(sigmas).all()) and float(sigmas[-1]) == 0.
             and bool(torch.all(sigmas[:-1] > sigmas[1:])), "Invalid full eight-step sigma schedule")
    _require(tensor_hash(sigmas) == invariants["full_sigmas_sha256"], "Suite full sigma values/hash mismatch")
    _require(report["used_sigmas"] == report["full_sigmas"][:3]
             and tensor_hash(sigmas[:3]) == _digest(report.get("used_sigmas_sha256"), "suite prefix sigmas"),
             "Suite prefix is not the first two steps of the full sigma schedule")
    if "trace_sigmas" in edit_tensors:
        _require(torch.equal(edit_tensors["trace_sigmas"], sigmas[:-1]), "Historical trace/full sigma schedule mismatch")
    positive = _digest(report["conditioning"].get("positive_sha256"), "suite positive conditioning")
    _require(positive == _digest(edit.get("conditioning_sha256"), "edit positive conditioning"),
             "Suite/edit positive conditioning mismatch")
    if "conditioning" in edit:
        for key in set(report["conditioning"]) & set(edit["conditioning"]):
            _require(report["conditioning"][key] == edit["conditioning"][key], "Conditioning metadata mismatch: " + key)
    invariants.update(conditioning_sha256=positive, seed=edit["seed"], prompt=edit["prompt"],
                      grid=deepcopy(grid), full_sigmas=report["full_sigmas"])

    decoded, masks = {}, {}
    inputs = {"target": target_image, "protected": protected_image}
    with safe_open(str(maps_path), framework="pt", device="cpu") as stored:
        metadata = stored.metadata() or {}
        _require(metadata.get("artifact_id") == suite["artifact_id"] and metadata.get("experiment_schema_version") == "1",
                 "Suite tensor artifact identity mismatch")
        for role in ("target", "protected", "background"):
            key = VARIANT + ".masks." + role
            source = _bundle_file(suite_path, artifacts["mask_pngs"][key])
            expected_file = _digest(suite["file_sha256"].get(source.name), "suite " + role + " PNG")
            content = source.read_bytes()
            size, binary = load_binary_png(BytesIO(content))
            _require(size == (width, height), "Suite PNG differs from original mask grid: " + role)
            tensor = torch.tensor(list(binary), dtype=torch.float32).reshape(1, height, width)
            expected_tensor = _digest(suite["tensor_sha256"].get(key), "suite " + role + " tensor")
            _require(suite["tensor_shapes"].get(key) == [1, height, width]
                     and tensor_hash(tensor) == expected_tensor, "Suite decoded PNG/tensor hash mismatch: " + role)
            saved_tensor = stored.get_tensor(key)
            _require(tensor_hash(saved_tensor) == expected_tensor and torch.equal(saved_tensor, tensor),
                     "Suite persisted tensor/PNG mismatch: " + role)
            _require(expected_tensor == edit["reference_partition_sha256"].get(role),
                     "Historical edit reference partition differs: " + role)
            decoded[role] = tensor
            row = {"source_png": str(source), "file_sha256": expected_file, "tensor_sha256": expected_tensor,
                   "tensor_key": key, "grid": deepcopy(grid)}
            if role in inputs:
                filename, input_path = _relative_png(input_root, inputs[role])
                input_content = input_path.read_bytes()
                input_size, input_binary = load_binary_png(BytesIO(input_content))
                _require(input_size == size and input_binary == binary and input_content == content,
                         "Input mask must be an exact, noninverted copy of the suite PNG: " + role)
                row.update(filename=filename, input_image=filename, input_path=str(input_path),
                           channel="red", inverted=False)
            masks[role] = row
    _require(torch.equal(decoded["target"] + decoded["protected"] + decoded["background"],
                         torch.ones_like(decoded["target"])), "Suite masks are not an exclusive background-complement partition")
    _require(bool(decoded["target"].any()) and bool(decoded["protected"].any()), "Reuse needs nonempty target and protected masks")
    _require(masks["target"]["tensor_sha256"] == edit.get("effective_image_mask_sha256"),
             "Historical effective image mask differs from the saved target")
    assets = _verify_assets(asset_path, suite_path, edit_path, suite, generation, edit, file_hash)
    return {"schema_version": 1, "status": "eligible_for_runtime_comparison", "variant": VARIANT,
        "generation_config": deepcopy(generation), "frozen_suite_config": deepcopy(config), "masks": masks,
        "runtime_invariants": invariants, "asset_verification": assets,
        "new_execution_hashes_verified": False, "observation_parity": "unverified",
        "observation_parity_reason": "Historical suite inputs and artifacts were checked; this is not a comparison against a new collector execution or proof of equal primary observations.",
        "historical_tensor_audit_enabled": report.get("tensor_audit_enabled", False),
        "historical_tap_manifest": deepcopy(report.get("tap_manifest")),
        "post_run_requirements": ["Match new generation_config except intended Slider strength.",
            "Match initial noise, initial latent, full sigmas, positive conditioning and exact mask hashes.",
            "Confirm the runtime loaded the verified assets; current paths do not bind graph loaders.",
            "Match recorded suite/edit runtime source identities, revisions and environment (especially ComfyUI revision), or document an explicit compatibility review; missing evidence stays unverified.",
            "Match subject/background token positions, background phrase/occurrence and conditioning metadata where recorded; an absent historical metadata hash is not parity.",
            "Compare retained primary observation hashes when available before attributing differences causally to prototype count.",
            "Evaluate images and the human pose gate independently; mask equality proves neither quality nor person preservation."],
        "sources": {"suite_manifest": {"path": str(suite_path), "sha256": file_hash(suite_path),
                                        "run_id": report.get("run_id"), "artifact_id": suite["artifact_id"]},
                    "edit_report": {"path": str(edit_path), "sha256": file_hash(edit_path),
                                    "run_id": edit.get("run_id"), "artifact_id": edit.get("artifact_id")}}}


def verify_reuse(suite_manifest, edit_report, input_dir, target_image, protected_image, asset_record):
    """Return JSON-serializable eligibility evidence, or ValueError; never mutate inputs."""
    try:
        from safetensors import SafetensorError
        from PIL import Image
    except ImportError as error:
        raise ValueError("Historical reuse verification requires Pillow and safetensors") from error
    try:
        return _verify_reuse(suite_manifest, edit_report, input_dir, target_image, protected_image, asset_record)
    except (OSError, KeyError, TypeError, IndexError, AttributeError, RuntimeError,
            ImportError, SafetensorError, Image.DecompressionBombError) as error:
        raise ValueError("Historical reuse could not be verified: " + str(error)) from error
