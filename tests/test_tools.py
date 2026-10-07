from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("tool", ["probe_krea2_slider.py", "validate_comfy.py"])
def test_help_does_not_require_comfy_or_gpu(tool):
    result = subprocess.run([sys.executable, "-B", str(ROOT / "tools" / tool), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "comfy" in result.stdout.lower()


def test_validator_missing_root_cannot_be_a_success(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(ROOT / "tools" / "validate_comfy.py"), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Krea2" in result.stderr
