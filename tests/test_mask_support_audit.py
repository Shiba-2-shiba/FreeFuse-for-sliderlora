"""Offline PNG support measurements; run with stdlib unittest, no torch needed."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit_mask_support.py"


class MaskSupportAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def png(self, name, cells=(), size=(7, 6), mode="L"):
        image = Image.new("L", size, 0)
        for xy in cells:
            image.putpixel(xy, 255)
        if mode != "L":
            image = image.convert(mode)
        path = self.directory / (name + ".png")
        image.save(path)
        return path

    def inputs(self):
        return [self.png("target", [(1, 1), (1, 2)]),
                self.png("protected", [(5, 4)]),
                self.png("effective", [(1, 1), (1, 2), (2, 2)])]

    def audit(self, paths, *options):
        # This assertion provides a real RED before the command exists.
        self.assertTrue(TOOL.is_file(), "The saved-mask audit CLI has not been implemented")
        result = subprocess.run([sys.executable, str(TOOL), *map(str, paths),
                                 *map(str, options)], capture_output=True, text=True)
        self.assertTrue(result.stdout.strip(), "Audit must emit a JSON report: " + result.stderr)
        return result, json.loads(result.stdout)

    def codes(self, report):
        return {error["code"] for error in report["errors"]}

    def test_clean_masks_report_only_saved_grid_support(self):
        result, report = self.audit(self.inputs())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["grid"], {"width": 7, "height": 6, "unit": "mask_grid_cell"})
        self.assertEqual(report["masks"]["raw_target"]["cells"], 2)
        self.assertEqual(report["masks"]["actual_effective"]["cells"], 3)
        self.assertEqual(report["added_margin"]["cells"], 1)
        self.assertEqual(report["checks"]["target_missing_cells"], 0)
        self.assertEqual(report["checks"]["protected_intrusion_cells"], 0)
        self.assertEqual(report["limitations"]["data_source"], "saved_png_not_raw_tensor")
        self.assertEqual(report["limitations"]["background_classification"], "unknown")
        self.assertEqual(report["limitations"]["protected_person_preservation"], "not_established")
        self.assertEqual(report["limitations"]["visual_quality"], "not_assessed")

    def test_holes_remain_and_diagonal_cells_are_eight_connected(self):
        ring = [(x, y) for x in range(1, 4) for y in range(1, 4) if (x, y) != (2, 2)]
        paths = [self.png("target", ring, (9, 7)), self.png("protected", [(8, 6)], (9, 7)),
                 self.png("effective", ring + [(5, 0), (6, 1), (0, 6)], (9, 7))]
        result, report = self.audit(paths)
        self.assertEqual(result.returncode, 0)
        effective = report["masks"]["actual_effective"]
        self.assertEqual(effective["cells"], 11)
        self.assertEqual(effective["components_8"]["count"], 3)
        self.assertEqual(effective["components_8"]["largest"]["cells"], 8)
        self.assertEqual(effective["components_8"]["largest"]["bbox_xyxy_exclusive"], [1, 1, 4, 4])
        margin = report["added_margin"]
        self.assertEqual(margin["cells"], 3)
        self.assertEqual(margin["components_8"]["count"], 2)
        self.assertEqual(margin["components_8"]["largest"],
                         {"cells": 2, "first_cell_xy": [5, 0], "bbox_xyxy_exclusive": [5, 0, 7, 2]})

    def test_largest_component_ties_use_first_row_major_cell(self):
        paths = self.inputs()
        paths[2] = self.png("effective", [(1, 1), (1, 2), (5, 0), (5, 1)])
        _, report = self.audit(paths)
        self.assertEqual(report["masks"]["actual_effective"]["components_8"]["largest"]["first_cell_xy"], [5, 0])

    def test_zero_margin_has_no_largest_component(self):
        paths = self.inputs()
        paths[2] = paths[0]
        result, report = self.audit(paths)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(report["added_margin"], {"cells": 0, "components_8": {"count": 0, "largest": None}})

    def test_effective_target_omission_fails(self):
        paths = self.inputs()
        paths[2] = self.png("effective", [(1, 1)])
        result, report = self.audit(paths)
        self.assertEqual(result.returncode, 1)
        self.assertIn("target_omitted", self.codes(report))
        self.assertEqual(report["checks"]["target_missing_cells"], 1)

    def test_effective_protected_intrusion_fails(self):
        paths = self.inputs()
        paths[2] = self.png("effective", [(1, 1), (1, 2), (5, 4)])
        result, report = self.audit(paths)
        self.assertEqual(result.returncode, 1)
        self.assertIn("protected_intrusion", self.codes(report))
        self.assertEqual(report["checks"]["protected_intrusion_cells"], 1)

    def test_raw_partition_overlap_fails(self):
        paths = self.inputs()
        paths[1] = self.png("protected", [(1, 1), (5, 4)])
        result, report = self.audit(paths)
        self.assertEqual(result.returncode, 1)
        self.assertIn("partition_overlap", self.codes(report))
        self.assertEqual(report["checks"]["raw_target_protected_overlap_cells"], 1)

    def test_manual_hand_exclusion_regrowth_fails_even_when_protected_misses_hand(self):
        paths = self.inputs()
        exclusion = self.png("anatomical", [(2, 2)])
        result, report = self.audit(paths, "--anatomical-exclusion", exclusion)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["checks"]["protected_intrusion_cells"], 0)
        self.assertEqual(report["checks"]["anatomical_exclusion_intrusion_cells"], 1)
        self.assertIn("anatomical_exclusion_intrusion", self.codes(report))
        self.assertEqual(report["limitations"]["protected_person_preservation"], "not_established")

    def test_nonintersecting_anatomical_exclusion_passes(self):
        exclusion = self.png("anatomical", [(3, 3)])
        result, report = self.audit(self.inputs(), "--anatomical-exclusion", exclusion)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(report["checks"]["anatomical_exclusion_intrusion_cells"], 0)

    def test_optional_comparison_measures_iou_without_imposing_threshold(self):
        comparison = self.png("comparison", [(1, 1), (1, 2), (3, 3)])
        result, report = self.audit(self.inputs(), "--compare-effective", comparison)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(report["effective_comparison"],
                         {"intersection_cells": 2, "union_cells": 4, "iou": 0.5})

    def test_empty_comparison_is_measurable(self):
        result, report = self.audit(self.inputs(), "--compare-effective", self.png("comparison"))
        self.assertEqual(result.returncode, 0)
        self.assertEqual(report["effective_comparison"]["iou"], 0.0)

    def test_empty_target_and_protected_fail(self):
        for index, code in ((0, "empty_raw_target"), (1, "empty_protected")):
            with self.subTest(code=code):
                paths = self.inputs()
                paths[index] = self.png("empty")
                result, report = self.audit(paths)
                self.assertEqual(result.returncode, 1)
                self.assertIn(code, self.codes(report))

    def test_empty_effective_fails_target_coverage(self):
        paths = self.inputs()
        paths[2] = self.png("empty")
        result, report = self.audit(paths)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(report["checks"]["target_missing_cells"], 2)

    def test_mismatched_primary_or_optional_grid_is_rejected(self):
        for role in ("protected", "effective", "--anatomical-exclusion", "--compare-effective"):
            with self.subTest(role=role):
                paths = self.inputs()
                mismatched = self.png("mismatched", [(0, 0)], (6, 7))
                options = []
                if role.startswith("--"):
                    options = [role, mismatched]
                else:
                    paths[1 if role == "protected" else 2] = mismatched
                result, report = self.audit(paths, *options)
                self.assertEqual(result.returncode, 2)
                self.assertIn("grid_mismatch", self.codes(report))

    def test_nonbinary_grayscale_is_rejected_without_thresholding(self):
        paths = self.inputs()
        with Image.open(paths[0]) as source:
            source.putpixel((0, 0), 128)
            source.save(paths[0])
        result, report = self.audit(paths)
        self.assertEqual(result.returncode, 2)
        self.assertIn("input_invalid", self.codes(report))
        self.assertIn("binary", report["errors"][0]["message"])

    def test_mask_to_image_identical_rgb_channels_are_accepted(self):
        paths = self.inputs()
        for path in paths:
            with Image.open(path) as source:
                source.convert("RGB").save(path)
        result, report = self.audit(paths)
        self.assertEqual(result.returncode, 0, report)
        self.assertEqual(report["masks"]["raw_target"]["cells"], 2)

    def test_one_bit_binary_png_is_accepted(self):
        paths = self.inputs()
        paths[0] = self.png("target1", [(1, 1), (1, 2)], mode="1")
        result, _ = self.audit(paths)
        self.assertEqual(result.returncode, 0)

    def test_color_channels_are_not_silently_luminance_converted(self):
        paths = self.inputs()
        with Image.open(paths[0]) as source:
            colored = source.convert("RGB")
            colored.putpixel((0, 0), (255, 0, 0))
            colored.save(paths[0])
        result, report = self.audit(paths)
        self.assertEqual(result.returncode, 2)
        self.assertIn("input_invalid", self.codes(report))

    def test_alpha_palette_and_transparency_are_rejected(self):
        for mode in ("RGBA", "LA", "P", "L_transparency"):
            with self.subTest(mode=mode):
                paths = self.inputs()
                with Image.open(paths[0]) as source:
                    if mode == "L_transparency":
                        source.save(paths[0], transparency=0)
                    else:
                        source.convert(mode).save(paths[0])
                result, report = self.audit(paths)
                self.assertEqual(result.returncode, 2)
                self.assertIn("input_invalid", self.codes(report))

    def test_non_png_and_unreadable_inputs_return_json(self):
        for kind in ("bmp", "missing", "corrupt"):
            with self.subTest(kind=kind):
                paths = self.inputs()
                if kind == "bmp":
                    with Image.open(paths[0]) as source:
                        source.save(paths[0], format="BMP")
                elif kind == "missing":
                    paths[0].unlink()
                else:
                    paths[0].write_bytes(b"not a PNG")
                result, report = self.audit(paths)
                self.assertEqual(result.returncode, 2)
                self.assertIn("input_invalid", self.codes(report))

    def test_source_files_are_never_rewritten(self):
        paths = self.inputs()
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
        result, _ = self.audit(paths)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths})

    def test_output_is_created_and_matches_stdout(self):
        output = self.directory / "report.json"
        result, report = self.audit(self.inputs(), "--output", output)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(json.loads(output.read_text()), report)

    def test_existing_output_including_source_is_never_overwritten(self):
        for use_source in (False, True):
            with self.subTest(use_source=use_source):
                paths = self.inputs()
                output = paths[0] if use_source else self.directory / "existing.json"
                if not use_source:
                    output.write_text("retain me")
                before = output.read_bytes()
                result, report = self.audit(paths, "--output", output)
                self.assertEqual(result.returncode, 2)
                self.assertIn("output_exists", self.codes(report))
                self.assertEqual(output.read_bytes(), before)

    def test_output_error_returns_json_and_nonzero(self):
        result, report = self.audit(self.inputs(), "--output", self.directory / "missing" / "report.json")
        self.assertEqual(result.returncode, 2)
        self.assertIn("output_write_failed", self.codes(report))


if __name__ == "__main__":
    unittest.main()
