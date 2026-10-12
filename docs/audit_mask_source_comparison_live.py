"""Verify retained mask-source experiment evidence; no model evaluation."""
import hashlib
import html
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit_attention_ablation_live import verify_edit, verify_suite
from slider_fuse.diagnostics import diagnostic_provenance
from slider_fuse.sampling import tensor_hash


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def mask_stats(a, b):
    intersection, union = int((a.bool() & b.bool()).sum()), int((a.bool() | b.bool()).sum())
    return {"off_cells": int(a.sum()), "on_cells": int(b.sum()),
            "intersection_cells": intersection, "union_cells": union,
            "iou": intersection / union if union else None,
            "off_only_cells": int((a.bool() & ~b.bool()).sum()),
            "on_only_cells": int((b.bool() & ~a.bool()).sum())}


def main():
    torch.set_num_threads(4)
    root = Path(sys.argv[1]).resolve()
    summary, plan = read(root / "summary.json"), read(root / "plan.json")
    result = {"cases": [], "total_model_nfe": 0, "human_quality_approval": "not_evaluated",
              "scope": "retained files/tensors, upstream graph, mask reconstruction, exact routing and first-branch equality",
              "limits": "diagnostic seeds only; no independent mask labels or independent no-hook parity; no new completed OFF/all reference"}
    for case in summary["cases"]:
        folder = root / case["case"]
        prompt = read(folder / "queued_api.json")["prompt"]
        history = read(folder / "history.json")
        assert history["status"]["status_str"] == "success" and history["status"]["completed"]
        assert history["prompt"][2] == prompt
        for sid, branch in (("8", "off"), ("11", "on")):
            inputs = prompt[sid]["inputs"]
            assert inputs["model"] == ["2" if branch == "off" else "10", 0]
            assert inputs["seed"] == case["seed"] and inputs["trial_id"] == plan["trial_id"]
        assert prompt["10"]["class_type"] == "LoraLoaderModelOnly"
        assert prompt["10"]["inputs"] == {"model": ["2", 0], "lora_name": plan["slider_runtime_path"], "strength_model": 4.0}
        manifests, maps = {}, {}
        for suite in case["suites"]:
            branch = suite["branch"]
            path = Path(suite["local"]) / "manifest.json"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == suite["manifest_sha256"]
            manifests[branch], maps[branch] = verify_suite(path)
            r = manifests[branch]["report"]
            assert r["phase1_nfe"] == 2 and r["phase2_nfe"] == 0 and r["owned_hooks_removed"]
            assert r["normalized_config"]["prototypes"] == 3
            assert r["algorithm"]["variants"]["multi_proto_bg"]["status"] == "ok"
            assert all(row["input_before_sha256"] == row["input_after_sha256"] and
                       row["output_before_sha256"] == row["output_after_sha256"] for row in r["forward_integrity"])
            assert all(row["tensor_sha256_before"] == row["tensor_sha256_after"] for row in r["tap_manifest"])
        off, on = manifests["off"]["report"], manifests["on"]["report"]
        common = ("normalized_config", "seed", "steps", "trial_id", "sampler", "scheduler", "cfg", "prompt",
                  "background_phrase", "background_occurrence", "token_positions", "conditioning",
                  "initial_noise_sha256", "initial_latent_sha256", "full_sigmas_sha256", "used_sigmas_sha256", "implementation")
        assert all(off[key] == on[key] for key in common)
        prefix = "multi_proto_bg.masks."
        toff, poff = maps["off"][prefix + "target"], maps["off"][prefix + "protected"]
        ton, pon = maps["on"][prefix + "target"], maps["on"][prefix + "protected"]
        expected = {"A_off": (toff, poff), "B_on": (ton, pon),
                    "C_on_base_protected": ((ton - poff).clamp(0, 1), poff),
                    "D_union_base_protected": ((toff + ton).clamp(0, 1).sub(poff).clamp(0, 1), poff)}
        base_hashes, slider_hashes, reports, rows = [], [], [], []
        for bundle in case["diagnostics"]:
            path = Path(bundle["local_report"])
            r = verify_edit(path)
            assert r["generation"] == diagnostic_provenance(prompt, bundle["save_node"], r)
            assert prompt[bundle["sampler_node"]]["inputs"]["model"] == ["2", 0]
            assert r["effective_slider_strength"] == 4.0 and r["branch_nfe"] == {"base": 8, "slider": 8}
            assert r["prediction_selection"]["dilate_radius"] == 0 and r["phase1_nfe"] == 0
            assert r["mask_postprocess"]["fill_holes_max_area"] == 0 and r["mask_postprocess"]["mask_dilate_radius"] == 0
            t = load_file(str(path.parent / r["artifacts"]["tensors"]), device="cpu")
            target, protected = expected[bundle["label"]]
            assert torch.equal(t["reference_target_mask"], target)
            assert torch.equal(t["reference_protected_mask"], protected)
            assert torch.equal(t["reference_background_mask"], 1 - target - protected)
            base_hashes.append(tensor_hash(t["trace_base_predictions"][0]))
            slider_hashes.append(tensor_hash(t["trace_slider_predictions"][0]))
            assert r["initial_noise_sha256"] == off["initial_noise_sha256"]
            assert r["initial_latent_sha256"] == off["initial_latent_sha256"]
            assert r["full_sigmas_sha256"] == off["full_sigmas_sha256"]
            reports.append(r)
            rows.append({"label": bundle["label"], "target_cells": int(target.sum()),
                         "protected_cells": int(protected.sum()), "coverage": r["effective_image_mask_coverage"],
                         "exact_routing_all_steps": True, "total_model_nfe": r["total_model_nfe"],
                         "first_base_prediction_sha256": base_hashes[-1], "first_slider_prediction_sha256": slider_hashes[-1],
                         "image": str(path.parent / r["artifacts"]["image"]),
                         "effective_mask": str(path.parent / r["artifacts"]["effective_mask"]),
                         "diagnostic": str(path), "final_latent_sha256": r["final_latent_sha256"]})
        assert len(set(base_hashes)) == len(set(slider_hashes)) == 1
        edit_common = ("strength", "seed", "steps", "cfg", "trial_id", "sampler", "scheduler", "lora_sha256",
                       "initial_noise_sha256", "initial_latent_sha256", "full_sigmas_sha256", "conditioning_sha256",
                       "first_input_sha256", "first_timestep_sha256", "generation", "implementation_source_sha256")
        assert all(all(r[key] == reports[0][key] for r in reports) for key in edit_common)
        nfe = off["phase1_nfe"] + on["phase1_nfe"] + sum(r["total_model_nfe"] for r in reports)
        assert nfe == plan["nfe_budget_per_case"]
        prior_path = Path(__file__).resolve().parent / "attention-ablation-results" / "20261010T190433Z" / case["case"] / "edits" / "143"
        prior = read(next(prior_path.glob("*.json")))
        previous = {"partition_equal": reports[0]["reference_partition_sha256"] == prior["reference_partition_sha256"],
                    "final_latent_equal": reports[0]["final_latent_sha256"] == prior["final_latent_sha256"],
                    "first_prediction_equal": reports[0]["first_prediction_sha256"] == prior["first_prediction_sha256"],
                    "slider_hash_equal": reports[0]["lora_sha256"] == prior["lora_sha256"]}
        row = {"case": case["case"], "seed": case["seed"], "prompt_id": case["prompt_id"], "nfe": nfe,
               "wall_seconds": case["timing"]["wall_seconds"], "collection_controls_equal": True,
               "edit_controls_equal": True, "first_base_and_slider_predictions_equal_across_routes": True,
               "target_mask_comparison": mask_stats(toff, ton), "protected_mask_comparison": mask_stats(poff, pon),
               "on_target_off_protected_collision_cells": int((ton * poff).sum()),
               "union_added_vs_off_cells": int((expected["D_union_base_protected"][0] - toff).clamp_min(0).sum()),
               "union_added_vs_on_hybrid_cells": int((expected["D_union_base_protected"][0] - expected["C_on_base_protected"][0]).clamp_min(0).sum()),
               "previous_k3_s4_parity": previous, "routes": rows}
        result["cases"].append(row)
        result["total_model_nfe"] += nfe
        print("VERIFIED", case["case"], "target", row["target_mask_comparison"], "previous", previous, flush=True)
    assert result["total_model_nfe"] == plan["nfe_budget_total"]
    write(root / "verification.json", result)
    print("VERIFIED_TOTAL", result["total_model_nfe"], flush=True)


if __name__ == "__main__":
    main()
