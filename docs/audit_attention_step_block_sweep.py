"""Verify retained sweep controls and artifacts, and measure a fixed old hole ROI."""
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit_attention_ablation_live import read, sha, verify_edit, verify_suite
from slider_fuse.diagnostics import diagnostic_provenance
from slider_fuse.experimental_masks import _labels
from slider_fuse.sampling import tensor_hash

ROLES = ("target", "protected", "background")
COMMON = ("normalized_config", "seed", "steps", "trial_id", "sampler", "scheduler", "cfg",
          "prompt", "background_phrase", "background_occurrence", "token_positions", "conditioning",
          "initial_noise_sha256", "initial_latent_sha256", "full_sigmas_sha256", "used_sigmas_sha256", "implementation")


def safe_tap(folder, item):
    path = (folder / item["manifest"]).resolve()
    assert path.is_relative_to(folder.resolve()) and path.is_file()
    assert sha(path) == item["manifest_sha256"]
    return path


def verify_sweep(folder, expected):
    outer = read(folder / "manifest.json")
    assert outer["complete"] and set(outer["taps"]) == expected
    report = outer["report"]
    assert report["phase1_nfe"] == 3 and report["phase2_nfe"] == 0
    assert report["owned_hooks_removed"] and report["tensor_audit_enabled"]
    assert report["normalized_config"]["prototypes"] == 3
    config = report["normalized_config"]
    assert {f"step_{s}.block_{b}" for s in config["selected_steps"] for b in config["selected_blocks"]} == expected
    expected_pairs = {tuple(int(part.split("_")[1]) for part in key.split(".")) for key in expected}
    assert len(report["tap_manifest"]) == len(expected)
    assert {(t["step"], t["block"]) for t in report["tap_manifest"]} == expected_pairs
    assert all(t["input_before_sha256"] == t["input_after_sha256"] and
               t["output_before_sha256"] == t["output_after_sha256"] for t in report["forward_integrity"])
    assert all(t["tensor_sha256_before"] == t["tensor_sha256_after"] and
               t["block_input_before_sha256"] == t["block_input_after_sha256"] for t in report["tap_manifest"])
    taps = {}
    for key, item in outer["taps"].items():
        manifest, tensors = verify_suite(safe_tap(folder, item))
        r = manifest["report"]
        assert r["shared_collection_id"] == report["run_id"]
        step, block = (int(part.split("_")[1]) for part in key.split("."))
        assert r["primary_tap"] == {"step": step, "block": block}
        assert r["algorithm"]["variants"]["multi_proto_bg"] == item["status"]
        if item["status"]["status"] == "ok":
            scores = torch.stack([tensors[f"multi_proto_bg.raw_maps.{role}"].flatten() for role in ROLES], dim=-1)
            labels = _labels(scores, r["normalized_config"])
            for index, role in enumerate(ROLES[:2]):
                assert torch.equal((labels == index).reshape(1, *r["grid"]).float(), tensors[f"multi_proto_bg.masks.{role}"])
            assert torch.equal((labels == -1).reshape(1, *r["grid"]).float(), tensors["multi_proto_bg.uncertain_mask"])
            assert torch.equal((labels == 2).reshape(1, *r["grid"]).float(), tensors["multi_proto_bg.real_background_mask"])
            assert torch.equal(sum(tensors[f"multi_proto_bg.masks.{role}"] for role in ROLES), torch.ones(1,*r["grid"]))
        taps[key] = (manifest, tensors)
    return outer, taps


def hole_statistics(taps, indices, protected):
    rows = []
    for key, (manifest, tensors) in taps.items():
        r = manifest["report"]
        step, block = r["primary_tap"]["step"], r["primary_tap"]["block"]
        prefix = f"evidence.step_{step}.block_{block}."
        assert all(torch.all(tensors[prefix+role+".token_allowed_fraction"] == 1) for role in ROLES)
        heads = torch.stack([tensors[prefix+role+".head_scores"] for role in ROLES], dim=-1)
        logits = torch.stack([tensors[prefix+role+".token_scores"].mean(dim=0) for role in ROLES], dim=-1)
        context = (logits / r["normalized_config"]["context_temperature"]).softmax(dim=-1)
        head_context = (heads / r["normalized_config"]["context_temperature"]).softmax(dim=-1)
        row = {"step":step,"block":block,"status":r["algorithm"]["variants"]["multi_proto_bg"],
               "qk_role_mean": dict(zip(ROLES,context[indices].mean(dim=0).tolist())),
               "heads_target_wins_on_mean_roi": int((head_context[:,indices].mean(dim=1).argmax(dim=-1)==0).sum()),
               "head_count":heads.shape[0]}
        if row["status"]["status"] == "ok":
            masks = {role:tensors[f"multi_proto_bg.masks.{role}"] for role in ROLES}
            target = masks["target"]
            effective = (target-protected).clamp(0,1)
            row.update(target_cells=int(target.sum()), protected_cells=int(masks["protected"].sum()),
                       effective_target_cells=int(effective.sum()),
                       roi_routing={role:int(mask.flatten()[indices].sum()) for role,mask in masks.items()},
                       roi_uncertain=int(tensors["multi_proto_bg.uncertain_mask"].flatten()[indices].sum()),
                       roi_effective_target_cells=int(effective.flatten()[indices].sum()),
                       prototype_role_mean={role:float(tensors[f"multi_proto_bg.raw_maps.{role}"].flatten()[indices].mean()) for role in ROLES})
        rows.append(row)
    return sorted(rows,key=lambda row:(row["block"],row["step"]))


def main():
    torch.set_num_threads(4)
    root = Path(sys.argv[1]).resolve()
    summary, plan = read(root/"summary.json"), read(root/"plan.json")
    expected = {f"step_{s}.block_{b}" for s in plan["steps"] for b in plan["blocks"]}
    prior = root.parent.parent / "mask-source-comparison-results" / "20261011T104616Z"
    hole = read(prior/"park"/"head-hole-probe.json")
    result = {"cases":[],"verified_mask_taps":0,"verified_edit_bundles":0,"total_model_nfe":0,
              "roi_source":str(prior/"park"/"head-hole-probe.json"),"roi_cells":len(hole["grid_indices"]),
              "roi_scope":"Fixed 70 grid cells of old park ON s2/b18 closed hole; not independent face labels. Used only for park.",
              "limits":"Heuristic Q/K role probabilities and prototype scores; no calibrated confidence, independent segmentation, completed OFF/all reference or holdout. Assistant visual observations are separate from human approval."}
    for case in summary["cases"]:
        folder = root/case["case"]
        graphs = {stage:read(folder/f"{stage}.queued_api.json")["prompt"] for stage in ("mask","edit")}
        histories = {stage:read(folder/f"{stage}.history.json") for stage in graphs}
        for stage, graph in graphs.items():
            history = histories[stage]
            assert history["status"]["completed"] and history["status"]["status_str"] == "success"
            assert history["prompt"][2] == graph
            assert sha(folder/f"{stage}.queued_api.json") == read(folder/f"{stage}.submission.json")["queued_api_sha256"]
            assert graph["10"]["class_type"] == "LoraLoaderModelOnly"
            assert graph["10"]["inputs"]["model"] == ["2",0] and graph["10"]["inputs"]["strength_model"] == 4
            for sid, model in (("8","2"),("11","10")):
                inputs = graph[sid]["inputs"]
                assert inputs["model"] == [model,0] and inputs["seed"] == case["seed"] and inputs["steps"] == 8
                assert inputs["audit_tensors"] is True
            assert graphs["mask"]["8"] == graph["8"] and graphs["mask"]["11"] == graph["11"]
        collected, stages = {}, {}
        for stage in graphs:
            stages[stage] = {}
            for record in case[f"{stage}_sweeps"]:
                path = Path(record["local"])
                assert sha(path/"manifest.json") == record["manifest_sha256"]
                outer, taps = verify_sweep(path,expected)
                stages[stage][record["branch"]] = outer
                if stage == "mask": collected[record["branch"]] = taps
                else:
                    for key,(manifest,tensors) in taps.items():
                        oldm,oldt = collected[record["branch"]][key]
                        assert manifest["tensor_sha256"] == oldm["tensor_sha256"]
            off,on = (stages[stage][branch]["report"] for branch in ("off","on"))
            assert all(off[key] == on[key] for key in COMMON)
        protected = collected["off"]["step_2.block_18"][1]["multi_proto_bg.masks.protected"]
        graph = graphs["edit"]
        assert graph["25"]["inputs"] == {"sweep":["8",0],"step":2,"block":18}
        cache = {sid for kind,msg in histories["edit"]["status"]["messages"] if kind=="execution_cached" for sid in msg.get("nodes",[])}
        extra = sum(3 for sid in ("8","11") if sid not in cache)
        assert case["edit_capture_nfe"] == extra
        rows, base_hashes, slider_hashes = [], [], []
        for edit in case["diagnostics"]:
            path = Path(edit["local_report"])
            r = verify_edit(path)
            tensors = load_file(str(path.parent/r["artifacts"]["tensors"]))
            target = collected["on"][f"step_{edit['step']}.block_{edit['block']}"][1]["multi_proto_bg.masks.target"]
            target = (target-protected).clamp(0,1)
            assert torch.equal(tensors["reference_target_mask"],target)
            assert torch.equal(tensors["reference_protected_mask"],protected)
            assert torch.equal(tensors["reference_background_mask"],1-target-protected)
            inputs = graph[edit["sampler_node"]]["inputs"]
            assert inputs["model"] == ["2",0] and inputs["strength"] == 4 and inputs["mask_mode"] == "manual"
            assert inputs["seed"] == case["seed"] and inputs["steps"] == 8
            assert all(inputs[key]==0 for key in ("fill_holes_max_area","mask_dilate_radius","selection_dilate_radius"))
            assert r["generation"] == diagnostic_provenance(graph,edit["save_node"],r)
            assert r["phase1_nfe"] == 0 and r["branch_nfe"] == {"base":8,"slider":8}
            assert r["effective_slider_strength"] == 4 and r["owned_hooks_removed"]
            off = stages["mask"]["off"]["report"]
            assert all(r[key]==off[key] for key in ("initial_noise_sha256","initial_latent_sha256","full_sigmas_sha256"))
            base_hashes.append(tensor_hash(tensors["trace_base_predictions"][0]))
            slider_hashes.append(tensor_hash(tensors["trace_slider_predictions"][0]))
            rows.append({**edit,"image":str(path.parent/r["artifacts"]["image"]),
                         "effective_mask":str(path.parent/r["artifacts"]["effective_mask"]),
                         "target_cells":int(target.sum()),"total_model_nfe":r["total_model_nfe"],
                         "exact_routing_all_steps":True,"final_latent_sha256":r["final_latent_sha256"]})
        assert rows and len(set(base_hashes))==len(set(slider_hashes))==1
        planned = read(folder/"planned-edits.json")
        assert len(planned)==9 and len(rows)+len(case["not_generated"])==9
        for omitted in case["not_generated"]:
            status = collected["on"][f"step_{omitted['step']}.block_{omitted['block']}"][0]["report"]["algorithm"]["variants"]["multi_proto_bg"]
            if omitted["reason"] == "source_candidate_failed": assert status["status"] == "failed"
            else:
                target=collected["on"][f"step_{omitted['step']}.block_{omitted['block']}"][1]["multi_proto_bg.masks.target"]
                assert not (target*(1-protected)).any()
        nfe = 6+extra+sum(row["total_model_nfe"] for row in rows)
        assert case["mask_capture_nfe"]==6 and case["edit_model_nfe"]==sum(row["total_model_nfe"] for row in rows)
        failures = {branch:{key:outer["taps"][key]["status"] for key in expected if outer["taps"][key]["status"]["status"]!="ok"}
                    for branch,outer in stages["mask"].items()}
        oldm=read(prior/case["case"]/"suites"/"on"/"manifest.json")
        newm=collected["on"]["step_2.block_18"][0]
        parity = {role:oldm["tensor_sha256"][f"multi_proto_bg.masks.{role}"] == newm["tensor_sha256"][f"multi_proto_bg.masks.{role}"] for role in ROLES}
        old_verification=read(prior/"verification.json")
        old_case=next(row for row in old_verification["cases"] if row["case"]==case["case"])
        old_route=next(row for row in old_case["routes"] if row["label"]=="C_on_base_protected")
        old_edit=read(Path(old_route["diagnostic"]))
        current_control=read(Path(next(row for row in rows if row["label"]=="on_s2_b18")["local_report"]))
        historical_control={key:old_edit[key]==current_control[key] for key in
                            ("final_latent_sha256","reference_partition_sha256","lora_sha256",
                             "initial_noise_sha256","conditioning_sha256","implementation_source_sha256")}
        record={"case":case["case"],"seed":case["seed"],"nfe":nfe,"extra_capture_nfe":extra,
                "failed_taps":failures,"not_generated":case["not_generated"],"edits":rows,
                "prior_s2_b18_on_mask_equal":parity,"prior_C_control_comparison":historical_control,
                "first_prediction_parity":True,
                "wall_seconds":case["mask_timing"]["wall_seconds"]+case["edit_timing"]["wall_seconds"]}
        if case["case"]=="park":
            record["hole_probe"]={branch:hole_statistics(taps,hole["grid_indices"],protected) for branch,taps in collected.items()}
        result["cases"].append(record)
        result["verified_mask_taps"] += len(expected)*2
        result["verified_edit_bundles"] += len(rows)
        result["total_model_nfe"] += nfe
        print("VERIFIED",case["case"],len(expected)*2,"taps",len(rows),"edits",nfe,"NFE",flush=True)
    assert result["verified_mask_taps"] == plan["planned_mask_candidates"]
    with (root/"verification.json").open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(root/"verification.json")


if __name__ == "__main__":
    main()
