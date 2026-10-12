"""Run a frozen mask-source experiment using existing live ComfyUI nodes."""
import copy
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.error

HERE = Path(__file__).resolve().parent
PRIOR = HERE / "attention-ablation-results" / "20261010T190433Z"
OUTPUT = Path(r"C:\ComfyUI\output").resolve()
BASE = "http://127.0.0.1:8000"
HELPER = Path(r"C:\Users\inott\.codex\skills\comfyui-local-runtime\scripts\comfy_runtime.py")
spec = importlib.util.spec_from_file_location("comfy_runtime", HELPER)
rt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rt)
ROUTES = (
    ("A_off", "20", "21", "22", "23", ["13", 0], ["13", 1]),
    ("B_on", "30", "31", "32", "33", ["15", 0], ["15", 1]),
    ("C_on_base_protected", "41", "42", "43", "44", ["40", 0], ["13", 1]),
    ("D_union_base_protected", "47", "48", "49", "50", ["46", 0], ["13", 1]),
)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(path, value):
    if path.exists():
        return
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def node(prompt, sid, cls, inputs, title):
    prompt[str(sid)] = {"class_type": cls, "inputs": inputs, "_meta": {"title": title}}


def build(case, slider, stamp, trial):
    old = rt.read_json(PRIOR / case / "queued_api.json")["prompt"]
    prompt = {sid: copy.deepcopy(old[sid]) for sid in ("1", "2", "3", "4", "5", "6", "7", "9")}
    node(prompt, "10", "LoraLoaderModelOnly", {
        "model": ["2", 0], "lora_name": slider, "strength_model": 4.0,
    }, "Slider +4 ONLY for ON collection")
    for collector, saver, select, preview, branch, model in (
        ("8", "18", "12", "13", "off", "2"),
        ("11", "19", "14", "15", "on", "10"),
    ):
        inputs = copy.deepcopy(old["11"]["inputs"])
        inputs.update(model=[model, 0], trial_id=trial, audit_tensors=True)
        node(prompt, collector, "Krea2SliderFuseExperimentalMaskCollect", inputs,
             f"{branch}: same k3 algorithm, two forwards, selected model weights")
        node(prompt, saver, "Krea2SliderFuseExperimentalMaskSave", {
            "suite": [collector, 0], "filename_prefix": f"ms_{stamp}_{case}_{branch}",
        }, f"Retain {branch} raw evidence and seven candidate statuses")
        node(prompt, select, "Krea2SliderFuseExperimentalMaskSelect", {
            "suite": [collector, 0], "variant": "multi_proto_bg",
        }, f"Select {branch} k3 background-on mask")
        node(prompt, preview, "Krea2SliderFuseMaskPreview", {"mask_bank": [select, 0]}, f"{branch} partition")
    for sid, destination, source, operation, title in (
        ("40", ["15", 0], ["13", 1], "subtract", "C: ON target minus OFF protected"),
        ("45", ["13", 0], ["15", 0], "add", "D: binary clamped OFF/ON target union"),
        ("46", ["45", 0], ["13", 1], "subtract", "D: target union minus OFF protected"),
        ("51", ["15", 0], ["13", 1], "multiply", "ON target / OFF protected collision"),
        ("52", ["13", 0], ["15", 0], "subtract", "Old OFF-only target region"),
    ):
        node(prompt, sid, "MaskComposite", dict(destination=destination, source=source,
             x=0, y=0, operation=operation), title)
    prefix = f"mask_source_compare/{stamp}/{case}/"
    for label, subject, sampler, decode, saver, target, protected in ROUTES:
        inputs = copy.deepcopy(prompt["6"]["inputs"])
        inputs.update(target_mask=target, protected_mask=protected)
        node(prompt, subject, "Krea2SliderFuseSubjects", inputs, label + " subjects")
        inputs = copy.deepcopy(old["141"]["inputs"])
        inputs.update(model=["2", 0], subjects=[subject, 0], lora_name=slider, strength=4.0,
                      trial_id=trial, mask_mode="manual", selection_dilate_radius=0,
                      fill_holes_max_area=0, mask_dilate_radius=0, diagnostic_level="audit")
        node(prompt, sampler, "Krea2SliderFusePredictionMixSampler", inputs, label + " +4 / all radii zero")
        node(prompt, decode, "VAEDecode", {"samples": [sampler, 0], "vae": ["9", 0]}, label)
        node(prompt, saver, "Krea2SliderFuseDiagnosticSave", {
            "images": [decode, 0], "latent": [sampler, 0], "diagnostics": [sampler, 2],
            "filename_prefix": prefix + label,
        }, label + " full audit bundle")
    masks = (
        ("raw_T_off", ["13", 0]), ("raw_P_off", ["13", 1]),
        ("raw_T_on", ["15", 0]), ("raw_P_on", ["15", 1]),
        ("C_target", ["40", 0]), ("D_target", ["46", 0]),
        ("conflict_T_on_P_off", ["51", 0]), ("old_T_off_only", ["52", 0]),
    )
    for index, (label, reference) in enumerate(masks):
        convert, save = str(60 + index * 2), str(61 + index * 2)
        node(prompt, convert, "MaskToImage", {"mask": reference}, label)
        node(prompt, save, "SaveImage", {"images": [convert, 0], "filename_prefix": prefix + label}, label)
    return prompt


def output_file(subfolder, filename):
    path = (OUTPUT / subfolder / filename).resolve()
    if not path.is_relative_to(OUTPUT) or not path.is_file():
        raise ValueError("Invalid output artifact")
    return path


def retain(folder, prompt, history):
    suites, edits = [], []
    for sid, branch in (("18", "off"), ("19", "on")):
        matches = list(OUTPUT.glob(prompt[sid]["inputs"]["filename_prefix"] + "_*"))
        if len(matches) != 1:
            raise ValueError(f"Expected one {branch} suite, found {len(matches)}")
        source = matches[0].resolve()
        if source.parent != OUTPUT:
            raise ValueError("Suite escapes output root")
        dest = folder / "suites" / branch
        dest.mkdir(parents=True, exist_ok=True)
        for file in source.iterdir():
            resolved = file.resolve()
            if resolved.parent != source or not resolved.is_file():
                raise ValueError("Unsafe suite entry")
            if not (dest / file.name).exists():
                shutil.copyfile(resolved, dest / file.name)
        suites.append({"branch": branch, "collector_node": "8" if branch == "off" else "11",
                       "source": str(source), "local": str(dest), "manifest_sha256": sha(dest / "manifest.json")})
    for label, _, sampler, _, saver, _, _ in ROUTES:
        outputs = history["outputs"][saver]["images"]
        if len(outputs) != 1:
            raise ValueError("Expected one diagnostic image")
        item = outputs[0]
        source = output_file(item["subfolder"], item["filename"])
        report_path = source.with_suffix(".json")
        report = rt.read_json(report_path)
        dest = folder / "edits" / label
        dest.mkdir(parents=True, exist_ok=True)
        for name in {*report["artifacts"].values(), report_path.name}:
            if Path(name).name != name:
                raise ValueError("Diagnostic artifact must be a basename")
            artifact = output_file(item["subfolder"], name)
            if not (dest / name).exists():
                shutil.copyfile(artifact, dest / name)
        edits.append({"label": label, "sampler_node": sampler, "save_node": saver,
                      "local_report": str(dest / report_path.name), "run_id": report["run_id"]})
    if not (folder / "downloaded").exists():
        downloads = rt.download_outputs(BASE, history, str(folder / "downloaded"))
        write(folder / "downloads.json", downloads)
    return suites, edits


def main():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if len(sys.argv) == 2:
        root = Path(sys.argv[1]).resolve()
        if not root.is_relative_to(HERE / "mask-source-comparison-results") or not root.is_dir():
            raise ValueError("Resume requires an existing evidence root")
        stamp = root.name
    else:
        root = HERE / "mask-source-comparison-results" / stamp
        root.mkdir(parents=True, exist_ok=False)
    print("RESULT_ROOT", root, flush=True)
    queue = rt.request_json(BASE + "/queue")
    write(root / "queue.before.json", queue)
    write(root / "system_stats.before.json", rt.request_json(BASE + "/system_stats"))
    choices = rt.request_json(BASE + "/models/loras", timeout=30)
    matches = [name for name in choices if Path(name.replace("\\", "/")).name == "krea2_deaging_20261001T075826Z_c06cb035.safetensors"]
    if len(matches) != 1:
        raise ValueError("Slider runtime path is not unique")
    trial = int(time.time()) % 2147483647
    plan = {"question": "Does collecting with Slider +4 change masks and recover full-body editing?",
            "server": BASE, "trial_id": trial, "slider_runtime_path": matches[0],
            "cases": [{"case": "park", "seed": 444444}, {"case": "woman_front_strong_overlap", "seed": 42}],
            "frozen": "same k3 multi_proto_bg config; same prompt/noise/latent/sigmas; +4 edits; all radii zero",
            "routes": {"A_off": "T_off / P_off", "B_on": "T_on / P_on",
                       "C_on_base_protected": "T_on minus P_off / P_off",
                       "D_union_base_protected": "(T_off union T_on) minus P_off / P_off"},
            "nfe_budget_per_case": 68, "nfe_budget_total": 136,
            "scope": "existing nodes; isolated API copies; no source workflow/backend edits; no completed OFF/all reference added",
            "interpretation_limits": "diagnostic seeds only; assisted visual review is not human approval; masks are not independently labelled ground truth"}
    write(root / "plan.json", plan)
    plan = rt.read_json(root / "plan.json")
    for case in plan["cases"]:
        folder = root / case["case"]
        folder.mkdir(exist_ok=True)
        queued = folder / "queued_api.json"
        write(queued, {"prompt": build(case["case"], plan["slider_runtime_path"], stamp, plan["trial_id"])})
        check = rt.preflight(BASE, str(queued))
        issues = rt.format_preflight_issues(check)
        write(folder / "preflight.json", {"issues": issues, "requirements": check["requirements"], "output_node_count": check["output_node_count"]})
        if issues:
            raise ValueError(issues)
    summary = {"server": BASE, "root": str(root), "cases": []}
    for case in plan["cases"]:
        name = case["case"]
        folder = root / name
        prompt = rt.read_json(folder / "queued_api.json")["prompt"]
        started = time.perf_counter()
        before = datetime.datetime.now(datetime.timezone.utc).isoformat()
        if (folder / "submission.json").exists():
            response = rt.read_json(folder / "submission.json")
        else:
            try:
                response = rt.request_json(BASE + "/prompt", method="POST", payload={"prompt": prompt}, timeout=30)
            except urllib.error.HTTPError as error:
                write(folder / "submission_error.json", rt.parse_http_error(error))
                raise
            write(folder / "submission.json", {**response, "started_utc": before, "queued_api_sha256": sha(folder / "queued_api.json")})
        pid = response["prompt_id"]
        print("SUBMITTED", name, pid, flush=True)
        last_notice = 0
        while True:
            history = rt.request_json(BASE + f"/history/{pid}", timeout=30)
            entry = history.get(pid)
            if entry and (entry.get("status", {}).get("completed") or entry.get("status", {}).get("status_str") in ("success", "error")):
                break
            elapsed = time.perf_counter() - started
            if elapsed - last_notice >= 25:
                print("WAITING", name, round(elapsed, 1), flush=True)
                last_notice = elapsed
            time.sleep(3)
        write(folder / "history.json", entry)
        write(folder / "timing.json", {"wall_seconds": time.perf_counter() - started,
              "started_utc": before, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": "one queue wall time including polling delay; resumed timings preserve first record"})
        if entry.get("status", {}).get("status_str") != "success":
            raise RuntimeError(f"Saved failed job history for {name}: {pid}")
        suites, edits = retain(folder, prompt, entry)
        record = {"case": name, "seed": case["seed"], "prompt_id": pid,
                  "status": entry["status"], "timing": rt.read_json(folder / "timing.json"),
                  "suites": suites, "diagnostics": edits}
        write(folder / "artifacts.json", record)
        summary["cases"].append(record)
        write(root / f"summary.after_{name}.json", summary)
        print("FINISHED", name, "suites", len(suites), "edits", len(edits), flush=True)
    write(root / "summary.json", summary)
    print("DONE", root, flush=True)


if __name__ == "__main__":
    main()
