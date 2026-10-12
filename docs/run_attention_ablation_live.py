"""Run the documented cold comparison and preserve live evidence locally."""
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.error

HELPER = Path(r"C:\Users\inott\.codex\skills\comfyui-local-runtime\scripts\comfy_runtime.py")
spec = importlib.util.spec_from_file_location("comfy_runtime", HELPER)
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
BASE = "http://127.0.0.1:8000"
HERE = Path(__file__).resolve().parent
OUTPUT = Path(r"C:\ComfyUI\output").resolve()


def write(path, value):
    if path.exists():
        return
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_file(subfolder, filename):
    path = (OUTPUT / subfolder / filename).resolve()
    if not path.is_relative_to(OUTPUT) or not path.is_file():
        raise ValueError("Invalid output artifact path")
    return path


def main():
    if len(sys.argv) == 2:
        root = Path(sys.argv[1]).resolve()
        if not root.is_relative_to(HERE / "attention-ablation-results") or not root.is_dir():
            raise ValueError("Resume requires an existing local evidence directory")
        stamp = root.name
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = HERE / "attention-ablation-results" / stamp
        root.mkdir(parents=True, exist_ok=False)
    print("RESULT_ROOT", root, flush=True)
    write(root / "system_stats.before.json", rt.request_json(BASE + "/system_stats"))
    write(root / "queue.before.json", rt.request_json(BASE + "/queue"))
    models = rt.request_json(BASE + "/models/loras", timeout=30)
    slider_matches = [name for name in models if Path(name.replace("\\", "/")).name == "krea2_deaging_20261001T075826Z_c06cb035.safetensors"]
    if len(slider_matches) != 1:
        raise ValueError(f"Slider identity path ambiguous: {slider_matches}")
    summary = {"server": BASE, "root": str(root), "cases": [], "slider_runtime_path": slider_matches[0]}
    for case in ("park", "woman_front_strong_overlap"):
        folder = root / case
        folder.mkdir(exist_ok=True)
        original = HERE.parent / "workflows" / f"krea2_female_slider_attention_ablation_{case}_api.json"
        index_path = original.with_name(original.name.replace("_api.json", ".index.json"))
        index = rt.read_json(index_path)
        prompt = rt.load_prompt_dict(str(original))
        for node_id, node in prompt.items():
            inputs = node["inputs"]
            if node["class_type"] == "Krea2SliderFusePredictionMixSampler":
                inputs["lora_name"] = slider_matches[0]
            if "filename_prefix" in inputs:
                prefix = inputs["filename_prefix"]
                inputs["filename_prefix"] = (f"aa_{stamp}_{case}_k{1 if node_id == '10' else 3}"
                    if node["class_type"] == "Krea2SliderFuseExperimentalMaskSave"
                    else f"attention_ablation_live/{stamp}/{case}/{Path(prefix).name}")
        queued = folder / "queued_api.json"
        write(queued, {"prompt": prompt})
        write(folder / "source.index.json", index)
        check = rt.preflight(BASE, str(queued))
        issues = rt.format_preflight_issues(check)
        write(folder / "preflight.json", {"issues": issues, "requirements": check["requirements"], "output_node_count": check["output_node_count"]})
        if issues:
            raise ValueError(issues)
        started = time.perf_counter()
        before = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if (folder / "submission.json").exists():
            response = rt.read_json(folder / "submission.json")
            print("RESUME_EXISTING_JOB", case, response["prompt_id"], flush=True)
        else:
            try:
                response = rt.request_json(BASE + "/prompt", method="POST", payload={"prompt": prompt}, timeout=30)
            except urllib.error.HTTPError as exc:
                write(folder / "submission_error.json", rt.parse_http_error(exc))
                raise
        write(folder / "submission.json", {**response, "started_utc": before, "queued_api_sha256": sha(queued)})
        job_id = response["prompt_id"]
        print("SUBMITTED", case, job_id, flush=True)
        last_notice = 0
        while True:
            history = rt.request_json(BASE + f"/history/{job_id}", timeout=30)
            if job_id in history:
                entry = history[job_id]
                status = entry.get("status", {})
                if status.get("completed") or status.get("status_str") in ("success", "error"):
                    break
            elapsed = time.perf_counter() - started
            if elapsed - last_notice >= 25:
                print("WAITING", case, round(elapsed, 1), flush=True)
                last_notice = elapsed
            time.sleep(3)
        wall = time.perf_counter() - started
        write(folder / "history.json", entry)
        write(folder / "timing.json", {"wall_seconds": wall, "started_utc": before, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "scope": "one queue wall time, including polling delay"})
        wall = rt.read_json(folder / "timing.json")["wall_seconds"]
        bundle_records = []
        for node_id in ("10", "12"):
            prefix = prompt[node_id]["inputs"]["filename_prefix"]
            bundles = list(OUTPUT.glob(prefix + "_*"))
            for bundle in bundles:
                resolved = bundle.resolve()
                if not resolved.is_relative_to(OUTPUT):
                    raise ValueError("Suite bundle escapes output directory")
                dest = folder / "suites" / bundle.name
                dest.parent.mkdir(exist_ok=True)
                if not dest.exists():
                    shutil.copytree(resolved, dest)
                bundle_records.append({"node_id": node_id, "source": str(resolved), "local": str(dest), "manifest_sha256": sha(dest / "manifest.json")})
        download_records = rt.download_outputs(BASE, entry, str(folder / "downloaded")) if not (folder / "downloaded").exists() else []
        diagnostics = []
        for node_id in ("113", "143", "173"):
            for item in entry.get("outputs", {}).get(node_id, {}).get("images", []):
                image = source_file(item["subfolder"], item["filename"])
                report_path = image.with_suffix(".json")
                report = rt.read_json(report_path)
                dest = folder / "edits" / node_id
                dest.mkdir(parents=True, exist_ok=True)
                for name in report["artifacts"].values():
                    source = source_file(item["subfolder"], name)
                    if not (dest / source.name).exists():
                        shutil.copyfile(source, dest / source.name)
                diagnostics.append({"node_id": node_id, "source_report": str(report_path), "local_report": str(dest / report_path.name), "run_id": report["run_id"]})
        record = {"case": case, "prompt_id": job_id, "status": entry.get("status"), "wall_seconds": wall,
                  "suites": bundle_records, "diagnostics": diagnostics, "downloads": download_records}
        write(folder / "artifacts.json", record)
        summary["cases"].append(record)
        write(root / f"summary.after_{case}.json", summary)
        print("FINISHED", case, entry.get("status"), "wall", round(wall, 1), "suites", len(bundle_records), "edits", len(diagnostics), flush=True)
        if entry.get("status", {}).get("status_str") != "success":
            print("CASE_ERROR_RETAINED", case, flush=True)
    write(root / "summary.json", summary)
    print("DONE", root, flush=True)


if __name__ == "__main__":
    main()
