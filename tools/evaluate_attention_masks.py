r"""Evaluate explicitly saved TARGET mask PNGs offline, without inference or an OFF image.

Examples (run from the repository root):
  python tools/evaluate_attention_masks.py \
    --candidate baseline=baseline_target.png --candidate trial=trial_target.png
  python tools/evaluate_attention_masks.py --candidate trial=trial_target.png \
    --target-full visible_target.png --target-torso visible_torso.png \
    --target-legs visible_legs.png --protected-core visible_protected_core.png \
    --background known_background.png --annotations-human-verified true \
    --output new_evaluation.json

Annotation names specify visible, independently labelled regions on the exact
candidate grid. target-full means the entire visible target; torso/legs are
partial visible target regions. protected-core means the independently labelled
protected person's visible core; background means independently labelled known
background. Unknown/unlabelled cells are not automatically background. No anatomy
or identity is inferred, and automatic protected-mask overlap is not evidence of
actual male leakage. No full-body annotation is inferred from torso/legs.
Torso and legs must be disjoint and, when supplied, subsets of target-full.
Target, protected-core and background labels must be mutually disjoint.

Recall = candidate/annotation intersection divided by annotation cells. Target
precision = intersection divided by candidate cells; IoU = intersection/union.
Intrusion and background false-positive fractions use the corresponding labelled
region as denominator, not candidate size. Coverage = candidate cells/grid cells.

With any annotation, explicitly provide --annotations-human-verified true|false.
Use true only for labels manually drawn or human-verified against visible ground
truth. This records your declaration, which the tool cannot independently verify.
False permits descriptive supplied-mask overlap without a ground-truth claim.
Missing/empty annotations yield null metrics plus a reason, never perfect scores.
An empty candidate is valid; precision is then undefined.

Inputs are strictly binary PNGs: mode 1, L (0/255), or identical-channel RGB.
Alpha, transparency, palettes, animation, 16-bit and nonbinary values are rejected.
Grids must match exactly; nothing is resized, thresholded, filled or dilated.
Coverage and 8-connected components describe saved mask-grid cells, which are
attention tokens only if the saved PNG is the native token grid. Source filenames
and SHA-256 hashes identify the exact decoded files, not verified model provenance.
Source grid arrays are [width, height]; matrix indexing is row-major [y, x]. Equal
dimensions do not prove that files depict the same scene or are spatially aligned.

JSON describes each variant independently, with no ranking or quality pass/fail.
Exit 0 means evaluation completed, not that a mask is good. Exit 2 means invalid
input/output. --output creates a new file exclusively and never overwrites files.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
from itertools import combinations
import json
from pathlib import Path
import re
import sys

from PIL import Image

if __package__:
    from .audit_mask_support import load_binary_png, summarize
else:
    from audit_mask_support import load_binary_png, summarize


ANNOTATION_ROLES = ("target_full", "target_torso", "target_legs", "protected_core", "background")
CANDIDATE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def measured(value=None, reason=None):
    """Keep unavailable measurements JSON-null with an explicit reason."""
    return {"value": value, "reason": reason}


def evaluate_metrics(candidate, annotations):
    """Measure only explicit annotation support; never invent semantic labels."""
    metrics = {}
    candidate_cells = sum(candidate)
    for role in ANNOTATION_ROLES:
        mask = annotations.get(role)
        cells = sum(mask) if mask is not None else None
        reason = "annotation_not_supplied" if mask is None else "empty_annotation" if not cells else None
        overlap = sum(c and a for c, a in zip(candidate, mask)) if reason is None else None
        if role.startswith("target_"):
            metrics[role + "_recall"] = measured(overlap / cells) if reason is None else measured(reason=reason)
            if role == "target_full":
                metrics[role + "_precision"] = (
                    measured(reason=reason) if reason else
                    measured(overlap / candidate_cells) if candidate_cells else measured(reason="empty_candidate"))
                metrics[role + "_iou"] = (
                    measured(overlap / (candidate_cells + cells - overlap)) if reason is None else measured(reason=reason))
        else:
            prefix = "protected_core_intrusion" if role == "protected_core" else "background_false_positive"
            metrics[prefix + "_cells"] = measured(overlap, reason)
            metrics[prefix + "_fraction"] = measured(overlap / cells) if reason is None else measured(reason=reason)
    return metrics


def annotation_errors(annotations):
    """Reject conflicting visible labels without repairing or deriving masks."""
    errors = []
    for left, right in combinations(annotations, 2):
        # Full-target labels deliberately contain the partial target labels.
        if "target_full" in (left, right) and {left, right} <= {"target_full", "target_torso", "target_legs"}:
            continue
        overlap = sum(a and b for a, b in zip(annotations[left], annotations[right]))
        if overlap:
            errors.append({"code": "annotation_overlap", "annotations": [left, right],
                           "cells": overlap, "message": "Distinct visible regions must not overlap"})
    if "target_full" in annotations:
        for part in ("target_torso", "target_legs"):
            if part in annotations:
                outside = sum(p and not f for p, f in zip(annotations[part], annotations["target_full"]))
                if outside:
                    errors.append({"code": "annotation_not_subset", "annotation": part,
                                   "cells": outside, "message": "Partial target must be inside target_full"})
    return errors


def evaluate_candidates(candidates, annotations=None, annotations_human_verified=None):
    """Return (descriptive report, exit code) for explicit name-to-path mappings."""
    annotations = annotations or {}
    basis = ("annotations_not_supplied" if not annotations else "user_declared_visible_ground_truth"
             if annotations_human_verified else "supplied_unverified_annotation_overlap")
    report = {
        "schema_version": 1,
        "scope": "saved_target_mask_evaluation_only",
        "status": "invalid",
        "errors": [],
        "annotation_semantics": {
            "human_verified_visible_ground_truth": annotations_human_verified,
            "basis": basis,
            "declaration_independently_verified": False,
        },
        "annotations": {},
        "variants": {},
        "limitations": {
            "data_source": "saved_png_not_raw_tensor",
            "model_provenance": "not_verified",
            "scene_alignment": "not_verified_by_matching_dimensions",
            "protected_person_preservation": "not_established",
            "actual_male_leakage": "not_established_by_mask_overlap",
            "visual_quality": "not_assessed",
            "count_unit": "saved_mask_grid_cells_not_necessarily_attention_tokens",
            "unlabelled_cells": "unknown_not_assumed_background",
        },
    }
    if not candidates or any(not isinstance(name, str) or not CANDIDATE_NAME.fullmatch(name) for name in candidates):
        report["errors"].append({"code": "candidate_names_invalid",
                                 "message": "Supply at least one candidate with a safe, non-path name (1-64 characters)"})
    if set(annotations) - set(ANNOTATION_ROLES):
        report["errors"].append({"code": "annotation_role_invalid", "message": "Unknown annotation role"})
    if annotations and type(annotations_human_verified) is not bool:
        report["errors"].append({"code": "annotation_declaration_required",
                                 "message": "Annotations require an explicit boolean human-verified declaration"})
    if report["errors"]:
        return report, 2
    loaded_candidates, loaded_annotations = {}, {}
    for kind, paths, loaded in (("candidate", candidates, loaded_candidates),
                                ("annotation", annotations, loaded_annotations)):
        for name, path in paths.items():
            try:
                path = Path(path)
                content = path.read_bytes()
                size, mask = load_binary_png(BytesIO(content))
                source = {"path": str(path), "source_filename": path.name,
                          "sha256": hashlib.sha256(content).hexdigest(), "grid": list(size)}
                loaded[name] = (size, mask, source)
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                report["errors"].append({"code": "input_invalid", "kind": kind,
                                         "name": name, "message": str(error)})
    if report["errors"]:
        return report, 2
    width, height = next(iter(loaded_candidates.values()))[0]
    report["grid"] = {"width": width, "height": height, "cells": width * height,
                      "unit": "mask_grid_cell", "layout": "row_major_y_x"}
    for kind, loaded in (("candidate", loaded_candidates), ("annotation", loaded_annotations)):
        for name, (size, _, _) in loaded.items():
            if size != (width, height):
                report["errors"].append({"code": "grid_mismatch", "kind": kind, "name": name,
                                         "message": f"Expected grid {width}x{height}; got {size[0]}x{size[1]}"})
    if report["errors"]:
        return report, 2
    annotation_masks = {name: value[1] for name, value in loaded_annotations.items()}
    report["annotations"] = {name: {**value[2], "cells": sum(value[1])}
                             for name, value in loaded_annotations.items()}
    report["errors"].extend(annotation_errors(annotation_masks))
    if report["errors"]:
        return report, 2
    for name, (_, mask, source) in loaded_candidates.items():
        support = summarize(mask, width, height)
        support["coverage_fraction"] = support["cells"] / (width * height)
        report["variants"][name] = {"source": source, "support": support,
                                     "metrics": evaluate_metrics(mask, annotation_masks)}
    report["status"] = "evaluated"
    return report, 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=PNG",
                        help="Explicit saved TARGET mask; repeat for each variant")
    for role in ANNOTATION_ROLES:
        parser.add_argument("--" + role.replace("_", "-"), type=Path,
                            help="Independent visible-region annotation on the exact same grid")
    parser.add_argument("--annotations-human-verified", choices=("true", "false"),
                        help="Required with annotations: manually drawn/human-verified visible ground truth?")
    parser.add_argument("--output", type=Path, help="Create a new JSON report, refusing any existing destination")
    args = parser.parse_args(argv)
    candidates = {}
    for item in args.candidate:
        name, separator, path = item.partition("=")
        if not separator or not path or not CANDIDATE_NAME.fullmatch(name):
            parser.error("--candidate must be NAME=PNG with a 1-64 character non-path name using letters, digits, _, . or -")
        if name in candidates:
            parser.error(f"--candidate has duplicate name: {name}")
        candidates[name] = Path(path)
    annotations = {role: getattr(args, role) for role in ANNOTATION_ROLES if getattr(args, role) is not None}
    if annotations and args.annotations_human_verified is None:
        parser.error("--annotations-human-verified true|false is required when annotations are supplied")
    verified = None if args.annotations_human_verified is None else args.annotations_human_verified == "true"
    report, exit_code = evaluate_candidates(candidates, annotations, verified)
    if args.output is not None:
        try:
            # Exclusive creation refuses existing files, input aliases, and even dangling symlinks.
            with args.output.open("x", encoding="utf-8") as output:
                output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        except OSError as error:
            report["status"] = "invalid"
            report["errors"].append({"code": "output_exists" if isinstance(error, FileExistsError) else "output_write_failed",
                                     "message": str(error)})
            exit_code = 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
