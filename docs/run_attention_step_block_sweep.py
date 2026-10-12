"""Frozen late-block/step experiment on an existing local ComfyUI runtime."""
import copy
import datetime
import json
from pathlib import Path
import shutil
import sys
import time
import urllib.error
import urllib.request

import torch
from safetensors.torch import load_file

from run_mask_source_comparison_live import BASE, OUTPUT, PRIOR, rt, sha, write, node, output_file


def tap_path(folder, item):
    path = (folder / item["manifest"]).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError("Unsafe tap manifest path")
    return path


def collect_graph(case, slider, stamp, trial, stage="mask"):
    old = rt.read_json(PRIOR / case / "queued_api.json")["prompt"]
    prompt = {sid: copy.deepcopy(old[sid]) for sid in ("1", "2", "3", "4", "5", "6", "7", "9")}
    node(prompt, "10", "LoraLoaderModelOnly", dict(model=["2", 0], lora_name=slider, strength_model=4.), "Slider +4 for ON collection only")
    for sid, saver, branch, model in (("8", "18", "off", "2"), ("11", "19", "on", "10")):
        inputs = copy.deepcopy(old["11"]["inputs"])
        config = json.loads(inputs["config_json"])
        config.update(collect_step=3, selected_steps=[1, 2, 3], selected_blocks=list(range(14, 28)))
        inputs.update(model=[model, 0], trial_id=trial, config_json=json.dumps(config), audit_tensors=True)
        node(prompt, sid, "Krea2SliderFuseAttentionSweepCollect", inputs, branch + ": shared three-forward sweep")
        node(prompt, saver, "Krea2SliderFuseAttentionSweepSave", {
            "sweep": [sid, 0], "filename_prefix": f"sw_{stamp}_{case}_{stage}_{branch}",
        }, branch + ": all 42 masks and head evidence")
    return prompt


def retain_sweeps(folder, prompt, stage):
    records = []
    for sid, branch in (("18", "off"), ("19", "on")):
        matches = list(OUTPUT.glob(prompt[sid]["inputs"]["filename_prefix"] + "_*"))
        if len(matches) != 1:
            raise ValueError(f"Expected one exact {branch} sweep bundle")
        source = matches[0].resolve()
        if source.parent != OUTPUT:
            raise ValueError("Sweep bundle escapes output root")
        dest = folder / "sweeps" / stage / branch
        dest.mkdir(parents=True, exist_ok=True)
        for file in source.rglob("*"):
            resolved = file.resolve()
            if not resolved.is_relative_to(source):
                raise ValueError("Sweep entry escapes source bundle")
            target = dest / file.relative_to(source)
            if resolved.is_dir():
                target.mkdir(exist_ok=True)
            elif resolved.is_file() and not target.exists():
                shutil.copyfile(resolved, target)
        manifest = rt.read_json(dest / "manifest.json")
        if not manifest["complete"] or len(manifest["taps"]) != 42:
            raise ValueError("Incomplete sweep bundle")
        records.append({"branch": branch, "collector_node": "8" if branch == "off" else "11",
                        "source": str(source), "local": str(dest), "manifest_sha256": sha(dest / "manifest.json")})
    return records


def edit_graph(case, prompt, folder, stamp, slider, trial):
    off_folder = folder / "sweeps" / "mask" / "off"
    on_folder = folder / "sweeps" / "mask" / "on"
    off, on = rt.read_json(off_folder / "manifest.json"), rt.read_json(on_folder / "manifest.json")
    fixed = tap_path(off_folder, off["taps"]["step_2.block_18"])
    fixed_report = rt.read_json(fixed)
    if fixed_report["report"]["algorithm"]["variants"]["multi_proto_bg"]["status"] != "ok":
        raise ValueError("Fixed protection control failed; cannot make a comparable edit")
    protected = load_file(str(fixed.parent / "maps.safetensors"))["multi_proto_bg.masks.protected"]
    node(prompt, "25", "Krea2SliderFuseAttentionSweepSelect", {"sweep": ["8", 0], "step": 2, "block": 18}, "Fixed OFF step2/block18 protection")
    node(prompt, "26", "Krea2SliderFuseMaskPreview", {"mask_bank": ["25", 0]}, "Fixed protection")
    old = rt.read_json(PRIOR / case / "queued_api.json")["prompt"]
    candidates = []
    for index, (block, step) in enumerate((b, s) for b in (18, 22, 27) for s in (1, 2, 3)):
        key = f"step_{step}.block_{block}"
        item = on["taps"][key]
        record = {"step": step, "block": block, "label": f"on_s{step}_b{block}", "status": "planned"}
        if item["status"]["status"] != "ok":
            record.update(status="not_generated", reason="source_candidate_failed", candidate=item["status"])
            candidates.append(record)
            continue
        path = tap_path(on_folder, item)
        tensors = load_file(str(path.parent / "maps.safetensors"))
        target = tensors["multi_proto_bg.masks.target"]
        if not (target * (1 - protected)).any():
            record.update(status="not_generated", reason="empty_target_after_fixed_protection")
            candidates.append(record)
            continue
        base = 100 + index * 10
        select, preview, subtract, subject, sampler, decode, save = map(str, range(base, base + 7))
        node(prompt, select, "Krea2SliderFuseAttentionSweepSelect", {"sweep": ["11", 0], "step": step, "block": block}, key)
        node(prompt, preview, "Krea2SliderFuseMaskPreview", {"mask_bank": [select, 0]}, key + " raw mask")
        node(prompt, subtract, "MaskComposite", {"destination": [preview, 0], "source": ["26", 1],
             "x": 0, "y": 0, "operation": "subtract"}, "Keep fixed protected region")
        inputs = copy.deepcopy(prompt["6"]["inputs"])
        inputs.update(target_mask=[subtract, 0], protected_mask=["26", 1])
        node(prompt, subject, "Krea2SliderFuseSubjects", inputs, key + " subjects")
        inputs = copy.deepcopy(old["141"]["inputs"])
        inputs.update(model=["2", 0], subjects=[subject, 0], lora_name=slider,
                      strength=4., trial_id=trial, selection_dilate_radius=0, diagnostic_level="audit")
        node(prompt, sampler, "Krea2SliderFusePredictionMixSampler", inputs, key + " +4")
        node(prompt, decode, "VAEDecode", {"samples": [sampler, 0], "vae": ["9", 0]}, key)
        node(prompt, save, "Krea2SliderFuseDiagnosticSave", {
            "images": [decode, 0], "latent": [sampler, 0], "diagnostics": [sampler, 2],
            "filename_prefix": f"attention_step_block_sweep/{stamp}/{case}/{record['label']}",
        }, key + " full audit bundle")
        record.update(status="queued", sampler_node=sampler, save_node=save)
        candidates.append(record)
    return prompt, candidates


def run_stage(folder, stage, prompt):
    queued = folder / f"{stage}.queued_api.json"
    write(queued, {"prompt": prompt})
    prompt = rt.read_json(queued)["prompt"]
    check = rt.preflight(BASE, str(queued))
    issues = rt.format_preflight_issues(check)
    write(folder / f"{stage}.preflight.json", {"issues": issues, "requirements": check["requirements"], "output_node_count": check["output_node_count"]})
    if issues:
        raise ValueError(issues)
    started = time.perf_counter()
    before = datetime.datetime.now(datetime.timezone.utc).isoformat()
    submission = folder / f"{stage}.submission.json"
    if submission.exists():
        response = rt.read_json(submission)
    else:
        try:
            response = rt.request_json(BASE + "/prompt", method="POST", payload={"prompt": prompt}, timeout=30)
        except urllib.error.HTTPError as error:
            write(folder / f"{stage}.submission_error.json", rt.parse_http_error(error))
            raise
        write(submission, {**response, "started_utc": before, "queued_api_sha256": sha(queued)})
    pid = response["prompt_id"]
    print("SUBMITTED", folder.name, stage, pid, flush=True)
    notice = 0
    while True:
        history = rt.request_json(BASE + f"/history/{pid}", timeout=30)
        entry = history.get(pid)
        if entry and (entry.get("status", {}).get("completed") or entry.get("status", {}).get("status_str") in ("success", "error")):
            break
        elapsed = time.perf_counter() - started
        if elapsed - notice >= 25:
            print("WAITING", folder.name, stage, round(elapsed, 1), flush=True)
            notice = elapsed
        time.sleep(3)
    write(folder / f"{stage}.history.json", entry)
    write(folder / f"{stage}.timing.json", {"wall_seconds": time.perf_counter() - started,
          "started_utc": before, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "scope": "one queue wall including polling delay; overlapping reports are not added"})
    if entry["status"]["status_str"] != "success":
        raise RuntimeError(f"Retained failed {stage} history for {folder.name}")
    print("FINISHED", folder.name, stage, flush=True)
    return prompt, entry


def retain_edits(folder, candidates, history):
    result = []
    for candidate in candidates:
        if candidate["status"] != "queued":
            continue
        images = history["outputs"][candidate["save_node"]]["images"]
        if len(images) != 1:
            raise ValueError("Expected one saved diagnostic image")
        item = images[0]
        source = output_file(item["subfolder"], item["filename"])
        report_path = source.with_suffix(".json")
        report = rt.read_json(report_path)
        dest = folder / "edits" / candidate["label"]
        dest.mkdir(parents=True, exist_ok=True)
        for name in {*report["artifacts"].values(), report_path.name}:
            if Path(name).name != name:
                raise ValueError("Unsafe diagnostic artifact name")
            source = output_file(item["subfolder"], name)
            if not (dest / name).exists():
                shutil.copyfile(source, dest / name)
        result.append({**candidate, "status": "generated", "local_report": str(dest / report_path.name)})
    return result


def main():
    torch.set_num_threads(4)
    root = Path(sys.argv[1]).resolve()
    expected_root = Path(__file__).resolve().parent / "attention-step-block-sweep-results"
    if not root.is_relative_to(expected_root) or not (root / "plan.json").is_file():
        raise ValueError("Use an existing frozen experiment root")
    plan = rt.read_json(root / "plan.json")
    choices = rt.request_json(BASE + "/models/loras", timeout=30)
    sliders = [name for name in choices if Path(name.replace("\\", "/")).name == "krea2_deaging_20261001T075826Z_c06cb035.safetensors"]
    if len(sliders) != 1:
        raise ValueError("Slider path is not unique")
    write(root / "runtime.json", {"server": BASE, "slider": sliders[0], "trial_id": int(time.time()) % 2147483647,
          "cached_capture_budget": 300, "cold_two_stage_budget": 312,
          "scope": "mask stage and predeclared edit stage; second save preserves actual capture evidence even on cache hits"})
    runtime = rt.read_json(root / "runtime.json")
    write(root / "system_stats.before.json", rt.request_json(BASE + "/system_stats"))
    write(root / "queue.before.json", rt.request_json(BASE + "/queue"))
    summary = {"root": str(root), "server": BASE, "cases": []}
    for case in plan["cases"]:
        folder = root / case["case"]
        folder.mkdir(exist_ok=True)
        prompt = collect_graph(case["case"], runtime["slider"], root.name, runtime["trial_id"])
        prompt, mask_history = run_stage(folder, "mask", prompt)
        mask_sweeps = retain_sweeps(folder, prompt, "mask")
        write(folder / "mask.artifacts.json", mask_sweeps)
        prompt = collect_graph(case["case"], runtime["slider"], root.name, runtime["trial_id"], "edit")
        prompt, candidates = edit_graph(case["case"], prompt, folder, root.name, runtime["slider"], runtime["trial_id"])
        write(folder / "planned-edits.json", candidates)
        prompt, edit_history = run_stage(folder, "edit", prompt)
        edit_sweeps = retain_sweeps(folder, prompt, "edit")
        edits = retain_edits(folder, candidates, edit_history)
        cached = {sid for kind, message in edit_history["status"]["messages"] if kind == "execution_cached" for sid in message.get("nodes", [])}
        extra_capture_nfe = sum(3 for sid in ("8", "11") if sid not in cached)
        record = {**case, "mask_sweeps": mask_sweeps, "edit_sweeps": edit_sweeps, "diagnostics": edits,
                  "not_generated": [row for row in candidates if row["status"] != "queued"],
                  "mask_capture_nfe": 6, "edit_capture_nfe": extra_capture_nfe,
                  "edit_model_nfe": sum(rt.read_json(row["local_report"])["total_model_nfe"] for row in edits),
                  "mask_prompt_id": rt.read_json(folder / "mask.submission.json")["prompt_id"],
                  "edit_prompt_id": rt.read_json(folder / "edit.submission.json")["prompt_id"],
                  "mask_timing": rt.read_json(folder / "mask.timing.json"), "edit_timing": rt.read_json(folder / "edit.timing.json")}
        write(folder / "artifacts.json", record)
        summary["cases"].append(record)
        write(root / f"summary.after_{case['case']}.json", summary)
        print("CASE_DONE", case["case"], "images", len(edits), "extra_capture_nfe", extra_capture_nfe, flush=True)
    write(root / "summary.json", summary)
    print("DONE", root, flush=True)


if __name__ == "__main__":
    main()
