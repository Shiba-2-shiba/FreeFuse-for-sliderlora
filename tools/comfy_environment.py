"""Shared setup for opt-in validation tools; never installs dependencies."""
from pathlib import Path
import subprocess
import sys


def bootstrap(root, *, cpu=True):
    root = Path(root).resolve()
    if not (root / "comfy/ldm/krea2/model.py").is_file():
        raise ValueError("The supplied ComfyUI root has no native Krea2 model")
    sys.path[:0] = [str(root), str(Path(__file__).parents[1])]
    sys.argv = [sys.argv[0]] + (["--cpu"] if cpu else [])
    import comfy.options
    comfy.options.enable_args_parsing()
    try:
        revision = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), "rev-parse", "HEAD"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    return revision
