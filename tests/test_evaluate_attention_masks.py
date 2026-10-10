"""Offline, strict-grid target-mask evaluation without model or GPU imports."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image


TOOL = Path(__file__).resolve().parents[1] / "tools" / "evaluate_attention_masks.py"


def png(tmp_path, name, cells=(), size=(5, 4), mode="L"):
    image = Image.new("L", size, 0)
    for xy in cells:
        image.putpixel(xy, 255)
    path = tmp_path / (name + ".png")
    image.convert(mode).save(path)
    return path


def run(*arguments):
    assert TOOL.is_file(), "The attention-mask evaluation CLI is not implemented"
    return subprocess.run([sys.executable, str(TOOL), *map(str, arguments)],
                          capture_output=True, text=True)


def evaluate(*arguments):
    result = run(*arguments)
    assert result.stdout.strip(), result.stderr
    return result, json.loads(result.stdout)


def metric(report, key, name="trial"):
    return report["variants"][name]["metrics"][key]


def test_without_annotations_only_support_is_measured(tmp_path):
    candidate = png(tmp_path, "candidate", [(0, 0), (1, 1), (4, 3)])
    result, report = evaluate("--candidate", f"trial={candidate}")
    assert result.returncode == 0, result.stderr
    assert report["status"] == "evaluated"
    assert report["scope"] == "saved_target_mask_evaluation_only"
    assert report["grid"] == {"width": 5, "height": 4, "cells": 20,
                              "unit": "mask_grid_cell", "layout": "row_major_y_x"}
    variant = report["variants"]["trial"]
    assert variant["support"]["cells"] == 3
    assert variant["support"]["coverage_fraction"] == 3 / 20
    assert variant["support"]["components_8"]["count"] == 2
    assert variant["source"]["source_filename"] == candidate.name
    assert variant["source"]["sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert variant["source"]["grid"] == [5, 4]
    assert report["annotation_semantics"]["human_verified_visible_ground_truth"] is None
    for value in variant["metrics"].values():
        assert value["value"] is None
        assert value["reason"] == "annotation_not_supplied"
    assert report["limitations"]["protected_person_preservation"] == "not_established"
    assert report["limitations"]["visual_quality"] == "not_assessed"
    assert "rank" not in report and "quality_pass" not in report


def test_verified_annotations_report_separate_metrics_per_candidate(tmp_path):
    full = png(tmp_path, "full", [(0, 0), (1, 0), (0, 1), (1, 1)])
    torso = png(tmp_path, "torso", [(0, 0), (1, 0)])
    legs = png(tmp_path, "legs", [(0, 1), (1, 1)])
    protected = png(tmp_path, "protected", [(3, 0), (4, 0)])
    background = png(tmp_path, "background", [(3, 3), (4, 3)])
    candidate = png(tmp_path, "candidate", [(0, 0), (1, 0), (0, 1), (3, 0), (4, 3)])
    result, report = evaluate("--candidate", f"trial={candidate}", "--candidate", f"perfect={full}",
        "--target-full", full, "--target-torso", torso, "--target-legs", legs,
        "--protected-core", protected, "--background", background,
        "--annotations-human-verified", "true")
    assert result.returncode == 0, report
    assert list(report["variants"]) == ["trial", "perfect"]
    expected = {"target_full_recall": .75, "target_full_precision": .6,
                "target_full_iou": .5, "target_torso_recall": 1,
                "target_legs_recall": .5, "protected_core_intrusion_cells": 1,
                "protected_core_intrusion_fraction": .5,
                "background_false_positive_cells": 1, "background_false_positive_fraction": .5}
    for key, value in expected.items():
        assert metric(report, key)["value"] == value
        assert metric(report, key)["reason"] is None
    assert metric(report, "target_full_iou", "perfect")["value"] == 1
    assert metric(report, "protected_core_intrusion_cells", "perfect")["value"] == 0
    assert report["annotations"]["target_full"]["source_filename"] == full.name
    assert report["annotations"]["target_full"]["sha256"] == hashlib.sha256(full.read_bytes()).hexdigest()
    assert report["annotations"]["target_full"]["cells"] == 4
    semantics = report["annotation_semantics"]
    assert semantics["human_verified_visible_ground_truth"] is True
    assert semantics["basis"] == "user_declared_visible_ground_truth"
    assert semantics["declaration_independently_verified"] is False


def test_unverified_annotations_are_labelled_as_supplied_mask_overlap(tmp_path):
    candidate = png(tmp_path, "candidate", [(0, 0)])
    result, report = evaluate("--candidate", f"trial={candidate}", "--protected-core", candidate,
                             "--annotations-human-verified", "false")
    assert result.returncode == 0
    assert report["annotation_semantics"]["human_verified_visible_ground_truth"] is False
    assert report["annotation_semantics"]["basis"] == "supplied_unverified_annotation_overlap"
    assert metric(report, "protected_core_intrusion_fraction")["value"] == 1
    assert report["limitations"]["actual_male_leakage"] == "not_established_by_mask_overlap"


def test_annotations_require_explicit_true_or_false_declaration(tmp_path):
    candidate = png(tmp_path, "candidate")
    for declaration in ([], ["--annotations-human-verified", "yes"]):
        result = run("--candidate", f"trial={candidate}", "--target-full", candidate, *declaration)
        assert result.returncode == 2
        assert "annotations-human-verified" in result.stderr


def test_empty_candidate_is_valid_and_precision_is_undefined(tmp_path):
    candidate = png(tmp_path, "candidate")
    target = png(tmp_path, "target", [(0, 0)])
    result, report = evaluate("--candidate", f"trial={candidate}", "--target-full", target,
                             "--annotations-human-verified", "true")
    assert result.returncode == 0
    assert report["variants"]["trial"]["support"] == {
        "cells": 0, "coverage_fraction": 0, "components_8": {"count": 0, "largest": None}}
    assert metric(report, "target_full_recall")["value"] == 0
    assert metric(report, "target_full_iou")["value"] == 0
    assert metric(report, "target_full_precision")["value"] is None
    assert metric(report, "target_full_precision")["reason"] == "empty_candidate"


def test_empty_annotations_are_undefined_even_if_candidate_is_empty(tmp_path):
    empty = png(tmp_path, "empty")
    for candidate in (empty, png(tmp_path, "nonempty", [(0, 0)])):
        result, report = evaluate("--candidate", f"trial={candidate}", "--target-full", empty,
            "--target-torso", empty, "--target-legs", empty, "--protected-core", empty,
            "--background", empty, "--annotations-human-verified", "true")
        assert result.returncode == 0
        for value in report["variants"]["trial"]["metrics"].values():
            assert value["value"] is None
            assert value["reason"] == "empty_annotation"


def test_partial_target_labels_do_not_infer_full_body_metrics(tmp_path):
    candidate = png(tmp_path, "candidate", [(0, 0)])
    result, report = evaluate("--candidate", f"trial={candidate}", "--target-torso", candidate,
                             "--annotations-human-verified", "true")
    assert result.returncode == 0
    assert metric(report, "target_torso_recall")["value"] == 1
    assert metric(report, "target_full_recall")["value"] is None
    assert metric(report, "target_full_recall")["reason"] == "annotation_not_supplied"


@pytest.mark.parametrize("role", ["candidate", "target_full", "target_torso", "target_legs", "protected_core", "background"])
def test_mismatched_grids_are_rejected_without_resizing(tmp_path, role):
    candidate = png(tmp_path, "candidate", [(0, 0)])
    wrong = png(tmp_path, "wrong", [(0, 0)], size=(4, 5))
    options = ["--candidate", f"trial={candidate}"]
    if role == "candidate":
        options += ["--candidate", f"wrong={wrong}"]
    else:
        options += ["--" + role.replace("_", "-"), wrong, "--annotations-human-verified", "true"]
    result, report = evaluate(*options)
    assert result.returncode == 2
    assert report["status"] == "invalid"
    assert "grid_mismatch" in {error["code"] for error in report["errors"]}
    assert report["variants"] == {}


@pytest.mark.parametrize("left,right", [("target_full", "protected_core"),
    ("target_full", "background"), ("target_torso", "protected_core"),
    ("target_legs", "background"), ("protected_core", "background"),
    ("target_torso", "target_legs")])
def test_conflicting_visible_region_annotations_are_rejected(tmp_path, left, right):
    candidate = png(tmp_path, "candidate", [(0, 0)])
    result, report = evaluate("--candidate", f"trial={candidate}",
        "--" + left.replace("_", "-"), candidate, "--" + right.replace("_", "-"), candidate,
        "--annotations-human-verified", "true")
    assert result.returncode == 2
    assert "annotation_overlap" in {error["code"] for error in report["errors"]}
    assert report["variants"] == {}


@pytest.mark.parametrize("role", ["target_torso", "target_legs"])
def test_partial_target_must_be_subset_of_supplied_full_target(tmp_path, role):
    full = png(tmp_path, "full", [(0, 0)])
    part = png(tmp_path, "part", [(0, 0), (1, 0)])
    result, report = evaluate("--candidate", f"trial={full}", "--target-full", full,
        "--" + role.replace("_", "-"), part, "--annotations-human-verified", "false")
    assert result.returncode == 2
    assert "annotation_not_subset" in {error["code"] for error in report["errors"]}
    assert report["variants"] == {}


@pytest.mark.parametrize("spec", ["missing_separator", "=file.png", "trial=", "../trial=file.png",
    "/trial=file.png", "tr/ial=file.png", "tr\\ial=file.png", "trial name=file.png", "x" * 65 + "=file.png"])
def test_malformed_or_path_like_variant_names_get_a_cli_error(spec):
    result = run("--candidate", spec)
    assert result.returncode == 2
    assert "--candidate" in result.stderr
    assert "Traceback" not in result.stderr


def test_duplicate_candidate_names_are_rejected_not_silently_replaced(tmp_path):
    candidate = png(tmp_path, "candidate")
    result = run("--candidate", f"trial={candidate}", "--candidate", f"trial={candidate}")
    assert result.returncode == 2
    assert "duplicate" in result.stderr.lower()


def test_output_matches_stdout_and_source_hashes_are_for_unmodified_files(tmp_path):
    candidate = png(tmp_path, "candidate", [(0, 0)])
    target = png(tmp_path, "target", [(0, 0), (1, 1)])
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (candidate, target)}
    output = tmp_path / "evaluation.json"
    result, report = evaluate("--candidate", f"trial={candidate}", "--target-full", target,
                             "--annotations-human-verified", "true", "--output", output)
    assert result.returncode == 0
    assert json.loads(output.read_text()) == report
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}


@pytest.mark.parametrize("destination", ["existing", "input", "annotation", "symlink",
    "symlink_annotation", "dangling_symlink", "hardlink", "hardlink_annotation"])
def test_outputs_never_overwrite_existing_files_or_aliases(tmp_path, destination):
    candidate = png(tmp_path, "candidate", [(0, 0)])
    annotation = png(tmp_path, "annotation", [(0, 0), (1, 0)])
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep this unrelated file")
    output = tmp_path / "report.json"
    absent = tmp_path / "absent.json"
    if destination == "input":
        output = candidate
    elif destination == "annotation":
        output = annotation
    elif destination == "symlink":
        output.symlink_to(candidate)
    elif destination == "symlink_annotation":
        output.symlink_to(annotation)
    elif destination == "dangling_symlink":
        output.symlink_to(absent)
    elif destination == "hardlink":
        output.hardlink_to(candidate)
    elif destination == "hardlink_annotation":
        output.hardlink_to(annotation)
    else:
        output.write_text("keep this")
    before_files = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                    for path in (candidate, annotation, unrelated)}
    before_output = output.read_bytes() if output.exists() else None
    before_inode = output.lstat().st_ino
    before_link = output.readlink() if output.is_symlink() else None
    result, report = evaluate("--candidate", f"trial={candidate}", "--target-full", annotation,
        "--annotations-human-verified", "true", "--output", output)
    assert result.returncode == 2
    assert report["status"] == "invalid"
    assert "output_exists" in {error["code"] for error in report["errors"]}
    assert before_files == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before_files}
    assert (output.read_bytes() if output.exists() else None) == before_output
    assert output.lstat().st_ino == before_inode
    assert (output.readlink() if output.is_symlink() else None) == before_link
    assert not absent.exists()


def test_dangling_output_is_refused_before_platform_exclusive_open(tmp_path, monkeypatch, capsys):
    from tools.evaluate_attention_masks import main

    candidate = png(tmp_path, "candidate", [(0, 0)])
    annotation = png(tmp_path, "annotation", [(0, 0), (1, 0)])
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep this unrelated file")
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
              for path in (candidate, annotation, unrelated)}
    absent = tmp_path / "absent.json"
    output = tmp_path / "report.json"
    output.symlink_to(absent)
    original_open = Path.open
    output_opens = []

    def platform_open(path, mode="r", *args, **kwargs):
        if path == output:
            output_opens.append(mode)
            # Reproduce only the reported Windows dangling-link behavior. All
            # other opens, including input reads, retain their real semantics.
            if mode == "x" and path.is_symlink() and not path.exists():
                return original_open(path.resolve(), mode, *args, **kwargs)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", platform_open)
    exit_code = main(["--candidate", f"trial={candidate}", "--target-full", str(annotation),
        "--annotations-human-verified", "true", "--output", str(output)])
    report = json.loads(capsys.readouterr().out)
    assert not absent.exists(), "An existing dangling output link must never create its target"
    assert exit_code == 2
    assert report["status"] == "invalid"
    assert "output_exists" in {error["code"] for error in report["errors"]}
    assert output_opens == [], "Refuse the existing directory entry before trying to open it"
    assert output.is_symlink() and output.readlink() == absent
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before}


def test_output_creation_still_uses_exclusive_open(tmp_path, monkeypatch, capsys):
    from tools.evaluate_attention_masks import main

    candidate = png(tmp_path, "candidate", [(0, 0)])
    before_candidate = candidate.read_bytes()
    output = tmp_path / "report.json"
    original_open = Path.open
    output_modes = []

    def create_competing_file_then_open(path, mode="r", *args, **kwargs):
        if path == output:
            output_modes.append(mode)
            # A regular file can appear after the preflight check. This does
            # not model or promise protection against every symlink race.
            with original_open(path, "w", encoding="utf-8") as competing:
                competing.write("created by another writer")
        return original_open(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", create_competing_file_then_open)
        exit_code = main(["--candidate", f"trial={candidate}", "--output", str(output)])
    report = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert "output_exists" in {error["code"] for error in report["errors"]}
    assert output_modes == ["x"]
    assert output.read_text() == "created by another writer"
    assert candidate.read_bytes() == before_candidate


def test_missing_output_parent_returns_json_error(tmp_path):
    candidate = png(tmp_path, "candidate")
    result, report = evaluate("--candidate", f"trial={candidate}", "--output", tmp_path / "missing" / "report.json")
    assert result.returncode == 2
    assert "output_write_failed" in {error["code"] for error in report["errors"]}


def test_programmatic_entrypoint_requires_boolean_annotation_declaration(tmp_path):
    from tools.evaluate_attention_masks import evaluate_candidates

    candidate = png(tmp_path, "candidate", [(0, 0)])
    for declaration in (None, "false", 1):
        report, exit_code = evaluate_candidates({"trial": candidate}, {"target_full": candidate}, declaration)
        assert exit_code == 2
        assert "annotation_declaration_required" in {error["code"] for error in report["errors"]}


def test_programmatic_entrypoint_requires_candidates_and_known_annotation_roles(tmp_path):
    from tools.evaluate_attention_masks import evaluate_candidates

    candidate = png(tmp_path, "candidate")
    for candidates, annotations in (({}, {}), ({"../escape": candidate}, {}),
                                     ({"trial": candidate}, {"auto_protected": candidate})):
        report, exit_code = evaluate_candidates(candidates, annotations, True)
        assert exit_code == 2
        assert report["status"] == "invalid"
        assert report["errors"]


@pytest.mark.parametrize("mode", ["1", "L", "RGB"])
def test_binary_png_modes_are_accepted_without_losing_holes(tmp_path, mode):
    ring = [(x, y) for x in range(3) for y in range(3) if (x, y) != (1, 1)]
    candidate = png(tmp_path, "candidate", ring, mode=mode)
    result, report = evaluate("--candidate", f"trial={candidate}")
    assert result.returncode == 0
    assert report["variants"]["trial"]["support"]["cells"] == 8
    assert report["variants"]["trial"]["support"]["components_8"]["largest"]["cells"] == 8


@pytest.mark.parametrize("kind", ["missing", "corrupt", "bmp", "gray", "colored", "rgb_gray",
    "RGBA", "LA", "P", "I;16", "transparency", "animated"])
@pytest.mark.parametrize("role", ["candidate", "target_full"])
def test_strict_png_validation_is_applied_to_candidates_and_annotations(tmp_path, kind, role):
    valid = png(tmp_path, "valid", [(0, 0)])
    invalid = png(tmp_path, "invalid", [(0, 0)])
    with Image.open(invalid) as source:
        image = source.copy()
    if kind == "missing":
        invalid.unlink()
    elif kind == "corrupt":
        invalid.write_bytes(b"not a PNG")
    elif kind == "bmp":
        image.save(invalid, format="BMP")
    elif kind == "gray":
        image.putpixel((1, 1), 128)
        image.save(invalid)
    elif kind in ("colored", "rgb_gray"):
        image = image.convert("RGB")
        image.putpixel((1, 1), (255, 0, 0) if kind == "colored" else (128, 128, 128))
        image.save(invalid)
    elif kind == "transparency":
        image.save(invalid, transparency=0)
    elif kind == "animated":
        image.save(invalid, save_all=True, append_images=[Image.new("L", image.size, 255)], duration=100)
    else:
        image.convert(kind).save(invalid)
    before = invalid.read_bytes() if invalid.exists() else None
    options = ["--candidate", f"trial={invalid if role == 'candidate' else valid}"]
    if role == "target_full":
        options += ["--target-full", invalid, "--annotations-human-verified", "true"]
    result, report = evaluate(*options)
    assert result.returncode == 2
    assert "input_invalid" in {error["code"] for error in report["errors"]}
    assert report["variants"] == {}
    assert before == (invalid.read_bytes() if invalid.exists() else None)


def test_tool_import_does_not_import_torch_comfy_or_runtime():
    result = subprocess.run([sys.executable, "-c",
        "import sys; import tools.evaluate_attention_masks; "
        "assert not any(k == 'torch' or k.startswith(('torch.', 'comfy', 'slider_fuse')) for k in sys.modules)"],
        cwd=TOOL.parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_report_does_not_claim_same_grid_proves_scene_alignment(tmp_path):
    candidate = png(tmp_path, "candidate", [(0, 0)])
    _, report = evaluate("--candidate", f"trial={candidate}")
    assert report["limitations"]["scene_alignment"] == "not_verified_by_matching_dimensions"
