"""Export a verified run's target/protected partition for manual replay."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def export_reference_masks(report_path, output_dir=None):
    import torch
    from PIL import Image
    from tools.compare_slider_diagnostics import load_report
    from slider_fuse.diagnostics import validate_prefix
    path = Path(report_path)
    _, tensors, _ = load_report(path)
    validate_prefix(path.stem, flat=True)
    masks = {}
    for name in ("target", "protected"):
        value = tensors.get("reference_" + name + "_mask")
        if value is None or value.ndim != 3 or value.shape[0] != 1 or not ((value == 0) | (value == 1)).all():
            raise ValueError("Run has no verified binary reference partition")
        masks[name] = Image.fromarray((value[0].cpu() * 255).to(dtype=torch.uint8).numpy())
    folder = Path(output_dir) if output_dir is not None else path.parent
    folder.mkdir(parents=True, exist_ok=True)
    outputs = {k: folder / (path.stem + "_reference_" + k + ".png") for k in masks}
    created = []
    try:
        for name, image in masks.items():
            with outputs[name].open("xb") as stream:
                created.append(outputs[name])
                image.save(stream, format="PNG")
    except BaseException:
        for p in created:
            p.unlink(missing_ok=True)
        raise
    return {k: str(p) for k,p in outputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="Complete diagnostic JSON, with its original artifact names")
    parser.add_argument("--output-dir", help="Destination directory; existing mask files are never overwritten")
    args = parser.parse_args()
    try:
        result = {"status": "exported", "files": export_reference_masks(args.report, args.output_dir)}
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
