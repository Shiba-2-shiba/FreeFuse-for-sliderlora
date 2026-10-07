"""Run native ComfyUI CPU/schema integration. Missing imports/skips are failure."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import unittest

from comfy_environment import bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfy_root", help="Real ComfyUI source root; use its Python environment")
    args = parser.parse_args()
    if not (Path(args.comfy_root) / "comfy/ldm/krea2/model.py").is_file():
        parser.error("This ComfyUI root does not contain the native Krea2 model")
    report = {"status": "failed", "int8_validated": False, "image_quality_validated": False}
    try:
        report["comfy_commit"] = bootstrap(args.comfy_root, cpu=True)
        from comfy.ldm.krea2.model import SingleStreamDiT
        path = Path(__file__).parents[1] / "tests/native_checks.py"
        spec = importlib.util.spec_from_file_location("slider_native_checks", path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        if suite.countTestCases() != 6:
            raise RuntimeError(f"Expected 6 native tests, collected {suite.countTestCases()}")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        passed = result.wasSuccessful() and result.testsRun == 6 and not result.skipped
        report.update(status="passed" if passed else "failed", tests_run=result.testsRun, skipped=len(result.skipped))
    except Exception as error:
        passed = False
        report.update(error_type=type(error).__name__, error=str(error), tests_run=0)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
