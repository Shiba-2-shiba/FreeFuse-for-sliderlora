"""Measure explicit saved mask PNGs without importing ComfyUI or inference code.

Usage: python tools/audit_mask_support.py TARGET PROTECTED ACTUAL_EFFECTIVE
         [--anatomical-exclusion PNG] [--compare-effective PNG] [--output JSON]

Inputs must share a grid and contain only black/white values. Accepted PNG modes
are 1, L (0/255), and RGB with identical 0/255 channels, including MaskToImage
saves. Alpha, transparency metadata, palette, animated, and 16-bit images are
rejected, never flattened or thresholded. No resizing, filling, dilation, or
semantic inference occurs. Counts and zero-based coordinates describe saved
mask-grid cells, not generated-image pixels. Bounding boxes are half-open.

The supplied effective mask must be the actual saved support, not a predicted
reconstruction. Input provenance, original tensors, generated-image quality,
background identity, and preservation of any person are not established here.
In particular, an initial hand exclusion can regrow in the actual effective
mask if the protected partition misses it: supply a separate anatomical
exclusion to test this. This tool does not change inference behavior.

JSON is written to stdout. --output additionally creates a new file exclusively;
existing files (including inputs and symlinks) are never overwritten. Exit codes:
0 = supplied mask support checks passed; 1 = support violation; 2 = input/output
error. Comparison IoU is descriptive and imposes no pass/fail threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from PIL import Image


def load_binary_png(path):
    """Return (size, 0/1 bytes) after strict validation, without altering a file."""
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError("Expected a PNG file")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Animated PNG masks are unsupported")
        if image.mode not in ("1", "L", "RGB") or "transparency" in image.info:
            raise ValueError("Use PNG mode 1, L, or identical-channel RGB without alpha/transparency")
        size = image.size
        if image.mode == "RGB":
            rgb = image.tobytes()
            values = rgb[::3]
            if values != rgb[1::3] or values != rgb[2::3]:
                raise ValueError("RGB mask channels must be identical grayscale values")
        else:
            values = image.convert("L").tobytes()
        if any(value not in (0, 255) for value in values):
            raise ValueError("Expected strictly binary black/white values (0 or 255)")
        return size, bytearray(value == 255 for value in values)


def summarize(mask, width, height):
    """Count support and 8-connected components without filling interior holes."""
    remaining = bytearray(mask)
    component_count = 0
    largest = None
    for first, active in enumerate(remaining):
        if not active:
            continue
        component_count += 1
        remaining[first] = 0
        stack = [first]
        first_x, first_y = first % width, first // width
        min_x = max_x = first_x
        min_y = max_y = first_y
        count = 0
        while stack:
            cell = stack.pop()
            x, y = cell % width, cell // width
            count += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for neighbor_y in range(max(0, y - 1), min(height, y + 2)):
                for neighbor_x in range(max(0, x - 1), min(width, x + 2)):
                    neighbor = neighbor_y * width + neighbor_x
                    if remaining[neighbor]:
                        remaining[neighbor] = 0
                        stack.append(neighbor)
        # Strict comparison resolves equal-size ties by first row-major cell.
        if largest is None or count > largest["cells"]:
            largest = {"cells": count, "first_cell_xy": [first_x, first_y],
                       "bbox_xyxy_exclusive": [min_x, min_y, max_x + 1, max_y + 1]}
    return {"cells": sum(mask), "components_8": {"count": component_count, "largest": largest}}


def audit(paths):
    """Return a report and exit code for explicitly named saved mask artifacts."""
    report = {
        "schema_version": 1,
        "scope": "saved_mask_support_only",
        "status": "failed",
        "inputs": {name: str(path) for name, path in paths.items()},
        "errors": [],
        "limitations": {
            "data_source": "saved_png_not_raw_tensor",
            "input_provenance": "not_verified",
            "background_classification": "unknown",
            "protected_person_preservation": "not_established",
            "visual_quality": "not_assessed",
        },
    }
    errors = report["errors"]
    loaded = {}
    for name, path in paths.items():
        try:
            loaded[name] = load_binary_png(path)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            errors.append({"code": "input_invalid", "mask": name, "message": str(error)})
    if errors:
        return report, 2

    width, height = loaded["raw_target"][0]
    report["grid"] = {"width": width, "height": height, "unit": "mask_grid_cell"}
    for name, (size, _) in loaded.items():
        if size != (width, height):
            errors.append({"code": "grid_mismatch", "mask": name,
                           "message": f"Expected grid {width}x{height}; got {size[0]}x{size[1]}"})
    if errors:
        return report, 2

    masks = {name: value[1] for name, value in loaded.items()}
    report["masks"] = {name: summarize(mask, width, height) for name, mask in masks.items()}
    for name in ("raw_target", "protected"):
        if not report["masks"][name]["cells"]:
            errors.append({"code": "empty_" + name, "message": f"{name} must be nonempty"})

    target = masks["raw_target"]
    protected = masks["protected"]
    effective = masks["actual_effective"]
    checks = {
        "raw_target_protected_overlap_cells": sum(t and p for t, p in zip(target, protected)),
        "target_missing_cells": sum(t and not e for t, e in zip(target, effective)),
        "protected_intrusion_cells": sum(e and p for e, p in zip(effective, protected)),
    }
    report["checks"] = checks
    violations = [
        ("raw_target_protected_overlap_cells", "partition_overlap", "Raw target and protected masks must be disjoint"),
        ("target_missing_cells", "target_omitted", "Actual effective mask omits raw target support"),
        ("protected_intrusion_cells", "protected_intrusion", "Actual effective mask intersects protected support"),
    ]
    if "anatomical_exclusion" in masks:
        checks["anatomical_exclusion_intrusion_cells"] = sum(
            e and a for e, a in zip(effective, masks["anatomical_exclusion"]))
        violations.append(("anatomical_exclusion_intrusion_cells", "anatomical_exclusion_intrusion",
                           "Actual effective mask intersects the supplied anatomical exclusion"))
    for key, code, message in violations:
        if checks[key]:
            errors.append({"code": code, "cells": checks[key], "message": message})

    added = bytearray(e and not t for e, t in zip(effective, target))
    report["added_margin"] = summarize(added, width, height)
    if "compare_effective" in masks:
        comparison = masks["compare_effective"]
        intersection = sum(e and c for e, c in zip(effective, comparison))
        union = sum(e or c for e, c in zip(effective, comparison))
        report["effective_comparison"] = {"intersection_cells": intersection, "union_cells": union,
                                          "iou": intersection / union if union else 1.0}
    report["status"] = "failed" if errors else "passed"
    return report, 1 if errors else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", type=Path, help="Saved raw target mask PNG")
    parser.add_argument("protected", type=Path, help="Saved protected partition mask PNG")
    parser.add_argument("effective", type=Path, help="Saved actual effective support PNG")
    parser.add_argument("--anatomical-exclusion", type=Path, help="Independent anatomical exclusion on the same grid")
    parser.add_argument("--compare-effective", type=Path, help="Reference effective mask for descriptive IoU")
    parser.add_argument("--output", type=Path, help="Create a JSON report; refuse an existing destination")
    args = parser.parse_args(argv)
    paths = {"raw_target": args.target, "protected": args.protected, "actual_effective": args.effective}
    if args.anatomical_exclusion is not None:
        paths["anatomical_exclusion"] = args.anatomical_exclusion
    if args.compare_effective is not None:
        paths["compare_effective"] = args.compare_effective
    report, exit_code = audit(paths)
    if args.output is not None:
        try:
            # Exclusive creation also rejects symlinks and input/output aliasing.
            with args.output.open("x", encoding="utf-8") as output:
                output.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        except OSError as error:
            report["status"] = "failed"
            report["errors"].append({"code": "output_exists" if isinstance(error, FileExistsError) else "output_write_failed",
                                     "message": str(error)})
            exit_code = 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
