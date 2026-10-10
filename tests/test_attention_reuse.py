"""Offline historical-reuse gates using real persisted artifacts, never inference."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
import torch
from PIL import Image

from slider_fuse.diagnostics import save_diagnostic_artifacts
from slider_fuse.experimental_artifacts import save_experimental_artifacts
from slider_fuse.experimental_masks import VARIANTS, validate_experimental_config
from slider_fuse.sampling import tensor_hash
from test_prediction_mix_schema3 import v3_payload


def verifier():
    path = Path(__file__).parents[1] / "tools/verify_attention_reuse.py"
    assert path.exists(), "Bounded saved-attention reuse verifier is missing"
    spec = importlib.util.spec_from_file_location("attention_reuse_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def make_history(tmp_path, *, prompt=None, seed=42, background_phrase="pale gray concrete wall"):
    graph, latent, payload = v3_payload(auto=False)
    if prompt is not None:
        graph["4"]["inputs"]["prompt"] = prompt
    graph["8"]["inputs"]["seed"] = seed
    payload.report.update(prompt=graph["4"]["inputs"]["prompt"], seed=seed, text_token_count=3,
                          implementation_source_sha256=hashlib.sha256(b"historical runtime source").hexdigest())
    current = tmp_path / "assets"
    current.mkdir()
    generation_names = {
        "checkpoint": graph["1"]["inputs"]["unet_name"],
        "style_lora": graph["2"]["inputs"]["lora_name"],
        "clip": graph["3"]["inputs"]["clip_name"],
        "vae": graph["9"]["inputs"]["vae_name"],
        "slider": graph["8"]["inputs"]["lora_name"],
    }
    identities, current_paths = {}, {}
    for role, name in generation_names.items():
        path = current / (role + ".bin")
        path.write_bytes(("retained original " + role).encode())
        identities[role] = {"name": name, "sha256": sha(path)}
        current_paths[role] = str(path)
    noise_hash = hashlib.sha256(b"noise").hexdigest()
    latent_hash = hashlib.sha256(b"latent").hexdigest()
    cond_hash = hashlib.sha256(b"positive").hexdigest()
    payload.report.update(initial_noise_sha256=noise_hash, initial_latent_sha256=latent_hash,
        conditioning_sha256=cond_hash, lora_sha256=identities["slider"]["sha256"],
        sampler="euler", scheduler="simple")
    saved_edit = save_diagnostic_artifacts(tmp_path, "historical_plus4", latent,
        torch.zeros(1, 8, 8, 3), payload, graph, None, "11")
    edit_path = tmp_path / saved_edit["artifacts"]["report"]
    masks = {role: payload.extra_tensors["reference_" + role + "_mask"].clone()
             for role in ("target", "protected", "background")}
    bank = {"grid": (2, 2), "masks": masks,
        "raw_maps": {role: value.reshape(1, -1) for role, value in masks.items()},
        "seed_masks": {}, "real_background_mask": masks["background"].clone(),
        "uncertain_mask": torch.zeros_like(masks["background"])}
    sigmas = torch.cat([payload.extra_tensors["trace_sigmas"], torch.zeros(1)])
    suite_report = {"experiment_schema_version": 1, "run_id": "original-collection", "grid": [2, 2],
        "seed": seed, "steps": 8, "trial_id": 0, "sampler": "euler", "scheduler": "simple", "cfg": 1.,
        "phase1_nfe": 2, "phase2_nfe": 0, "completed_reference_image_generated": False,
        "slider_loaded": False, "prompt": payload.report["prompt"],
        "background_phrase": background_phrase, "background_occurrence": 0, "text_token_count": 3,
        "token_positions": {"target": [0], "protected": [1], "background": [2]},
        "implementation": {"implementation_source_sha256": payload.report["implementation_source_sha256"],
                           "implementation_revision": "rev", "implementation_dirty": False},
        "normalized_config": validate_experimental_config({}),
        "initial_noise_sha256": noise_hash, "initial_latent_sha256": latent_hash,
        "full_sigmas_sha256": tensor_hash(sigmas), "used_sigmas_sha256": tensor_hash(sigmas[:3]),
        "full_sigmas": sigmas.tolist(), "used_sigmas": sigmas[:3].tolist(),
        "conditioning": {"positive_sha256": cond_hash, "positive_metadata": {}, "negative_used": False},
        "model": {"weights_identity_verified": False, "weights_identity_reason": "Not hashed at collection"},
        "algorithm": {"variants": {variant: {"status": "ok" if variant == "multi_proto_bg" else "failed"}
                                    for variant in VARIANTS}}}
    saved_suite = save_experimental_artifacts(tmp_path, "historical_suite", {
        "experiment_schema_version": 1, "report": suite_report,
        "banks": {"multi_proto_bg": bank}, "evidence_maps": {}})
    suite_path = tmp_path / saved_suite["manifest"]
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for role in ("target", "protected"):
        shutil.copyfile(tmp_path / saved_suite["mask_pngs"]["multi_proto_bg.masks." + role],
                        input_dir / (role + ".png"))
    retained = tmp_path / "retained-historical-assets.json"
    write_json(retained, {"schema_version": 1, "assets": identities})
    asset_path = tmp_path / "asset-verification.json"
    asset_record = {"schema_version": 1,
        "historical_record": {"path": retained.name, "sha256": sha(retained)},
        "current_paths": current_paths,
        "historical_binding": {"kind": "human_provenance_attestation", "attested_by": "fixture operator",
            "statement": "I retained these asset hashes from the original collection and +4 run.",
            "suite_manifest_sha256": sha(suite_path), "edit_report_sha256": sha(edit_path)}}
    write_json(asset_path, asset_record)
    return {"suite_manifest": suite_path, "edit_report": edit_path, "input_dir": input_dir,
        "target_image": "target.png", "protected_image": "protected.png", "asset_record": asset_path}


@pytest.fixture
def history(tmp_path):
    return make_history(tmp_path)


def update_document(history, key, mutate):
    path = history[key]
    doc = json.loads(path.read_text())
    mutate(doc)
    write_json(path, doc)
    if key in ("suite_manifest", "edit_report"):
        wrapper = json.loads(history["asset_record"].read_text())
        wrapper["historical_binding"][key + "_sha256"] = sha(path)
        write_json(history["asset_record"], wrapper)


def test_eligible_reuse_returns_frozen_provenance_and_preserves_trust_boundary(history):
    before = {key: history[key].read_bytes() for key in ("suite_manifest", "edit_report", "asset_record")}
    result = verifier().verify_reuse(**history)
    assert result["status"] == "eligible_for_runtime_comparison"
    assert result["generation_config"] == json.loads(before["edit_report"])["generation"]
    assert result["frozen_suite_config"]["prototypes"] == 3
    assert result["masks"]["target"]["input_image"] == "target.png"
    assert result["masks"]["protected"]["input_image"] == "protected.png"
    assert result["masks"]["target"]["tensor_sha256"] == json.loads(before["edit_report"])["effective_image_mask_sha256"]
    assert result["asset_verification"]["historical_assignment"] == "human_attested_not_independently_verified"
    assert result["asset_verification"]["current_files_match_retained_hashes"] is True
    assert result["asset_verification"]["suite_weights_identity_verified"] is False
    assert result["new_execution_hashes_verified"] is False
    assert result["runtime_invariants"]["initial_noise_sha256"] == json.loads(before["edit_report"])["initial_noise_sha256"]
    json.dumps(result, allow_nan=False)
    assert all(history[key].read_bytes() == value for key, value in before.items())


@pytest.mark.parametrize("key,change", [
    ("suite_manifest", lambda d: d.update(complete=False)),
    ("suite_manifest", lambda d: d.update(experiment_schema_version=2)),
    ("suite_manifest", lambda d: d["report"]["normalized_config"].update(prototypes=2)),
    ("suite_manifest", lambda d: d["report"]["normalized_config"].pop("mask_margin")),
    ("suite_manifest", lambda d: d["report"]["algorithm"]["variants"]["multi_proto_bg"].update(status="failed")),
    ("suite_manifest", lambda d: d["report"].update(phase1_nfe=8)),
    ("suite_manifest", lambda d: d["report"].update(initial_noise_sha256="0" * 64)),
    ("suite_manifest", lambda d: d["report"].update(initial_latent_sha256="0" * 64)),
    ("suite_manifest", lambda d: d["report"].update(full_sigmas_sha256="0" * 64)),
    ("suite_manifest", lambda d: d["report"]["full_sigmas"].__setitem__(3, .01)),
    ("suite_manifest", lambda d: d["report"].update(seed=43)),
    ("suite_manifest", lambda d: d["report"].update(prompt="a different scene")),
    ("suite_manifest", lambda d: d["report"]["conditioning"].update(positive_sha256="0" * 64)),
    ("suite_manifest", lambda d: d["report"].update(grid=[3, 2])),
    ("edit_report", lambda d: d.update(strength=2.)),
    ("edit_report", lambda d: d.update(mask_mode="auto")),
    ("edit_report", lambda d: d.update(mix_scope="all")),
    ("edit_report", lambda d: d["mask_generation"].update(fill_holes_max_area=1)),
    ("edit_report", lambda d: d["mask_generation"].update(mask_dilate_radius=1)),
    ("edit_report", lambda d: d["prediction_selection"].update(dilate_radius=1)),
    ("edit_report", lambda d: d["reference_partition_sha256"].update(background="0" * 64)),
    ("edit_report", lambda d: d.update(effective_image_mask_sha256="0" * 64)),
    ("edit_report", lambda d: d["generation"].update(seed=43)),
])
def test_rejects_changed_or_incomplete_historical_provenance(history, key, change):
    update_document(history, key, change)
    with pytest.raises(ValueError):
        verifier().verify_reuse(**history)


@pytest.mark.parametrize("kind", ["inverted", "resized", "nonbinary", "alpha", "colored", "different_bytes"])
def test_refuses_nonidentical_or_ambiguous_input_mask_without_repair(history, kind):
    path = history["input_dir"] / history["target_image"]
    with Image.open(path) as source:
        image = source.copy()
    if kind == "inverted":
        image = image.point(lambda v: 255 - v)
    elif kind == "resized":
        image = image.resize((4, 4), Image.Resampling.NEAREST)
    elif kind == "nonbinary":
        image.putpixel((0, 0), 127)
    elif kind == "alpha":
        image = image.convert("RGBA")
    elif kind == "colored":
        image = image.convert("RGB")
        image.putpixel((0, 0), (255, 0, 0))
    else:
        image = image.convert("RGB")  # Same decoded mask, not the same source file.
    image.save(path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        verifier().verify_reuse(**history)
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", ["../target.png", "/target.png", "C:\\input\\target.png", "target.jpg", "target.png [output]"])
def test_input_mask_names_must_be_input_relative_png(history, name):
    history["target_image"] = name
    with pytest.raises(ValueError):
        verifier().verify_reuse(**history)


@pytest.mark.parametrize("kind", ["missing_record", "missing_retained", "retained_hash", "current_hash",
    "slider_hash", "wrong_name", "no_attestation", "blank_attestation", "report_binding", "filenames_only"])
def test_asset_identity_is_fail_closed(history, kind):
    path = history["asset_record"]
    record = json.loads(path.read_text())
    if kind == "missing_record":
        path.unlink()
    elif kind == "missing_retained":
        (path.parent / record["historical_record"]["path"]).unlink()
    elif kind == "retained_hash":
        record["historical_record"]["sha256"] = "0" * 64
    elif kind == "current_hash":
        Path(record["current_paths"]["checkpoint"]).write_bytes(b"replacement checkpoint")
    elif kind == "slider_hash":
        update_document(history, "edit_report", lambda d: d.update(lora_sha256="0" * 64))
        record = json.loads(path.read_text())
    elif kind == "wrong_name":
        update_document(history, "edit_report", lambda d: d["generation"].update(checkpoint="other.safetensors"))
        record = json.loads(path.read_text())
    elif kind == "no_attestation":
        record.pop("historical_binding")
    elif kind == "blank_attestation":
        record["historical_binding"]["statement"] = " "
    elif kind == "report_binding":
        record["historical_binding"]["suite_manifest_sha256"] = "0" * 64
    elif kind == "filenames_only":
        record["historical_record"] = {"path": "checkpoint.safetensors"}
    if kind != "missing_record":
        write_json(path, record)
    with pytest.raises(ValueError):
        verifier().verify_reuse(**history)


@pytest.mark.parametrize("artifact", ["maps.safetensors", "multi_proto_bg.masks.target.png", "multi_proto_bg.masks.background.png"])
def test_rejects_tampered_suite_artifact_bytes(history, artifact):
    path = history["suite_manifest"].parent / artifact
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        verifier().verify_reuse(**history)


def test_missing_original_edit_bundle_is_not_eligible(history):
    report = json.loads(history["edit_report"].read_text())
    (history["edit_report"].parent / report["artifacts"]["tensors"]).unlink()
    with pytest.raises(ValueError):
        verifier().verify_reuse(**history)


@pytest.mark.parametrize("key,change", [
    ("suite_manifest", lambda d: d["report"]["algorithm"]["variants"].update(multi_proto_bg=[])),
    ("suite_manifest", lambda d: d["report"].update(model=[])),
    ("asset_record", lambda d: d.update(historical_binding=[])),
])
def test_malformed_nested_objects_raise_value_error(history, key, change):
    update_document(history, key, change)
    with pytest.raises(ValueError):
        verifier().verify_reuse(**history)


def test_does_not_accept_retrospectively_upgraded_suite_weight_identity(history):
    update_document(history, "suite_manifest", lambda d: d["report"]["model"].update(weights_identity_verified=True))
    with pytest.raises(ValueError, match="weights_identity_verified"):
        verifier().verify_reuse(**history)


@pytest.mark.parametrize("change", [
    lambda d: d["report"].pop("background_phrase"),
    lambda d: d["report"].update(background_phrase=""),
    lambda d: d["report"].update(background_occurrence=True),
    lambda d: d["report"].pop("token_positions"),
    lambda d: d["report"]["token_positions"].update(target=[1], protected=[0]),
    lambda d: d["report"]["token_positions"].update(background=[0]),
    lambda d: d["report"]["token_positions"].update(background=[3]),
    lambda d: d["report"].update(text_token_count=4),
])
def test_reuse_binds_the_original_subject_and_background_token_mapping(history, change):
    update_document(history, "suite_manifest", change)
    with pytest.raises(ValueError):
        verifier().verify_reuse(**history)


def test_returns_token_and_runtime_evidence_without_claiming_observation_or_metadata_parity(history):
    result = verifier().verify_reuse(**history)
    expected = json.loads(history["suite_manifest"].read_text())["report"]
    edit = json.loads(history["edit_report"].read_text())
    runtime = result["runtime_invariants"]
    assert runtime["token_positions"] == expected["token_positions"]
    assert runtime["background_phrase"] == "pale gray concrete wall"
    assert runtime["background_occurrence"] == 0
    assert runtime["edit_runtime_provenance"]["implementation_source_sha256"] == edit["implementation_source_sha256"]
    assert runtime["edit_runtime_provenance"]["environment"] == edit["environment"]
    assert runtime["suite_implementation"] == expected["implementation"]
    assert runtime["runtime_provenance_status"] == "recorded_requires_post_run_match_or_review"
    assert runtime["conditioning_metadata"]["suite"] == expected["conditioning"]["positive_metadata"]
    assert runtime["conditioning_metadata"]["edit"] is None
    assert runtime["conditioning_metadata"]["status"] == "unverified_not_recorded_in_historical_edit"
    assert any("runtime" in requirement and "review" in requirement for requirement in result["post_run_requirements"])


def test_missing_original_source_identity_is_explicitly_unverified(history):
    update_document(history, "edit_report", lambda d: d.pop("implementation_source_sha256"))
    update_document(history, "suite_manifest", lambda d: d["report"].pop("implementation"))
    result = verifier().verify_reuse(**history)
    assert result["runtime_invariants"]["runtime_provenance_status"] == "incomplete_unverified"
    assert result["runtime_invariants"]["suite_implementation"] is None


@pytest.mark.parametrize("filename", [" target.png", "bad\nmask.png", "bad\tmask.png", "bad\x7fmask.png"])
def test_existing_input_png_with_whitespace_or_controls_is_rejected(history, filename):
    source = history["input_dir"] / history["target_image"]
    destination = history["input_dir"] / filename
    try:
        shutil.copyfile(source, destination)
    except OSError:
        pytest.skip("This filesystem cannot create the invalid filename used by this regression test")
    history["target_image"] = filename
    with pytest.raises(ValueError, match="filename|relative"):
        verifier().verify_reuse(**history)


def test_malformed_tensor_container_raises_value_error_even_with_matching_file_hash(history):
    path = history["suite_manifest"].parent / "maps.safetensors"
    path.write_bytes(b"not a safetensors container")
    update_document(history, "suite_manifest", lambda d: d["file_sha256"].update({path.name: sha(path)}))
    with pytest.raises(ValueError):
        verifier().verify_reuse(**history)
