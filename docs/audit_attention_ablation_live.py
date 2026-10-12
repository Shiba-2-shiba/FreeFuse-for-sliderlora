"""Check saved cold-run evidence without further model execution."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slider_fuse.diagnostics import validate_audit_coverage, validate_mask_artifacts
from slider_fuse.sampling import tensor_hash


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    with path.open("x", encoding="utf-8") as out:
        json.dump(obj, out, ensure_ascii=False, indent=2, allow_nan=False)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checked_file(folder, name):
    path = (folder / name).resolve()
    if path.parent != folder.resolve() or not path.is_file():
        raise ValueError(f"Invalid artifact: {name}")
    return path


def compare(a, b, fields, allow_null=()):
    rows = []
    for key in fields:
        present = key in a and key in b and (key in allow_null or (a[key] is not None and b[key] is not None))
        rows.append({"field": key, "status": "unverified" if not present else "equal" if a[key] == b[key] else "mismatch", "k1_or_s4": a.get(key), "k3_or_s2": b.get(key)})
    return rows


def verdict(rows):
    return "mismatch" if any(r["status"] == "mismatch" for r in rows) else "unverified" if any(r["status"] == "unverified" for r in rows) else "verified_equal"


def verify_suite(path):
    manifest = read(path)
    assert manifest["complete"] is True
    for name, expected in manifest["file_sha256"].items():
        assert sha(checked_file(path.parent, name)) == expected, name
    tensors = load_file(str(path.parent / "maps.safetensors"), device="cpu")
    assert set(tensors) == set(manifest["tensor_sha256"])
    for key, tensor in tensors.items():
        assert tensor_hash(tensor) == manifest["tensor_sha256"][key], key
        assert list(tensor.shape) == manifest["tensor_shapes"][key]
    for key, reference in manifest["artifacts"]["mask_pngs"].items():
        with Image.open(checked_file(path.parent, Path(reference).name)) as image:
            assert image.mode == "L" and set(image.getdata()) <= {0,255}
            decoded = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(1,image.height,image.width) / 255
            assert torch.equal(decoded, tensors[key]), key
    return manifest, tensors


def verify_edit(path):
    report = read(path)
    assert report["complete"] is True
    for kind, expected in report["artifact_sha256"].items():
        assert sha(checked_file(path.parent, report["artifacts"][kind])) == expected, kind
    tensors = load_file(str(path.parent / report["artifacts"]["tensors"]), device="cpu")
    assert set(tensors) == set(report["tensor_manifest"])
    for key, tensor in tensors.items():
        row = report["tensor_manifest"][key]
        assert tensor_hash(tensor) == row["sha256"]
        assert list(tensor.shape) == row["shape"] and str(tensor.dtype) == row["dtype"]
    with Image.open(path.parent / report["artifacts"]["effective_mask"]) as image:
        assert image.mode == "L" and set(image.getdata()) <= {0, 255}
        mask = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(1, image.height, image.width) / 255
    validate_audit_coverage(report)
    validate_mask_artifacts(report, tensors, mask)
    assert tensor_hash(tensors["final_latent"]) == report["final_latent_sha256"]
    assert tensor_hash(tensors["first_prediction"]) == report["first_prediction_sha256"]
    for i, row in enumerate(report["step_trace"]):
        assert tensor_hash(tensors["trace_inputs"][i]) == row["input_sha256"]
        assert tensor_hash(tensors["trace_predictions"][i]) == row["prediction_sha256"]
        assert float(tensors["trace_sigmas"][i]) == row["sigma"]
    assert all(row[region]["changed_elements"] == 0 and row[region]["nonfinite_count"] == 0
               for row in report["prediction_mix_audit"] for region in ("base_excluded", "slider_selected"))
    return report


def main():
    root = Path(sys.argv[1]).resolve()
    summary = read(root / "summary.json")
    result = {"cases": [], "total_model_nfe": 0, "human_quality_approval": "not_evaluated", "holdout": "not_run; documented diagnostic seeds only"}
    for case in summary["cases"]:
        folder = root / case["case"]
        index = read(folder / "source.index.json")
        evaluation = copy.deepcopy(index["evaluation_record_template"])
        prompt = read(folder / "queued_api.json")["prompt"]
        assert read(folder / "history.json")["prompt"][2] == prompt
        evaluation["queued_api_path"] = str(folder / "queued_api.json")
        evaluation["queued_api_sha256"] = sha(folder / "queued_api.json")
        suites = {}; maps = {}
        for bundle in case["suites"]:
            k = "k1" if bundle["node_id"] == "10" else "k3"
            path = Path(bundle["local"]) / "manifest.json"
            suites[k], maps[k] = verify_suite(path)
            r = suites[k]["report"]
            assert r["phase1_nfe"] == 2 and r["phase2_nfe"] == 0 and r["owned_hooks_removed"]
            evaluation["collection_run_ids"][k] = r["run_id"]
            evaluation["suite_manifest_paths"][k] = str(path)
            evidence = evaluation["collection_evidence"][k]
            for key in evidence:
                if key in r:
                    evidence[key] = r[key]
            evidence["primary_tap"] = next(t for t in r["tap_manifest"] if (t["step"], t["block"]) == (2, 18))
            evidence["source_partition_sha256"] = {role: suites[k]["tensor_sha256"]["multi_proto_bg.masks." + role] for role in ("target", "protected", "background")}
        a, b = suites["k1"]["report"], suites["k3"]["report"]
        rows = compare(a, b, ["seed", "steps", "trial_id", "sampler", "scheduler", "cfg", "prompt", "background_phrase", "background_occurrence", "token_positions", "conditioning", "initial_noise_sha256", "initial_latent_sha256", "full_sigmas_sha256", "used_sigmas_sha256", "implementation"])
        configs = [{k:v for k,v in s["normalized_config"].items() if k != "prototypes"} for s in (a,b)]
        rows += compare({"config_except_prototypes":configs[0]}, {"config_except_prototypes":configs[1]}, ["config_except_prototypes"])
        assert a["normalized_config"]["prototypes"] == 1 and b["normalized_config"]["prototypes"] == 3
        taps = []
        for ta in a["tap_manifest"]:
            tb = next(t for t in b["tap_manifest"] if (t["step"],t["block"]) == (ta["step"],ta["block"]))
            for r in (ta,tb):
                assert r["tensor_audit_enabled"] is True
                assert r["tensor_sha256_before"] == r["tensor_sha256_after"]
                assert r["block_input_before_sha256"] == r["block_input_after_sha256"]
            tap_rows = compare(ta,tb,["tensor_sha256_before","tensor_sha256_after","block_input_before_sha256","block_input_after_sha256","attention_mask_sha256","attention_mask_shape"],allow_null=("attention_mask_sha256","attention_mask_shape"))
            taps.append({"step":ta["step"],"block":ta["block"],"status":verdict(tap_rows),"checks":tap_rows})
            rows += tap_rows
        evaluation["observation_parity"].update(status=verdict(rows), evidence=rows, all_taps=taps)
        edits = {d["node_id"]: (Path(d["local_report"]), verify_edit(Path(d["local_report"]))) for d in case["diagnostics"]}
        assets = read(root / "current-assets.json")
        for _, r in edits.values():
            assert r["lora_sha256"] == assets["assets"]["slider"]["sha256"]
            assert r["owned_hooks_removed"] is True
        s4, s2 = edits["143"][1], edits["173"][1]
        pair_rows = compare(s4,s2,["reference_partition_sha256","effective_image_mask_sha256","initial_noise_sha256","initial_latent_sha256","full_sigmas_sha256","conditioning_sha256","lora_sha256","mask_generation","prediction_selection","prompt","seed","steps","cfg","trial_id","token_positions"])
        assert s4["strength"] == 4 and s2["strength"] == 2
        for r in (s4,s2):
            assert r["mask_generation"]["fill_holes_max_area"] == r["mask_generation"]["mask_dilate_radius"] == r["prediction_selection"]["dilate_radius"] == 0
        pair_rows += compare({"generation_except_strength":{k:v for k,v in s4["generation"].items() if k != "strength"}}, {"generation_except_strength":{k:v for k,v in s2["generation"].items() if k != "strength"}}, ["generation_except_strength"])
        evaluation["strength_pair_mask_parity"].update(status=verdict(pair_rows), evidence=pair_rows)
        evaluation["asset_identity"].update(status="verified_shared_same_queue_graph_sources", evidence=[{"prompt_id":case["prompt_id"],"model_style_source_node":"2","checkpoint_node":"1","clip_node":"3","conditioning_node":"4","vae_node":"9","slider_sha256":s4["lora_sha256"],"historical_asset_provenance":"not_claimed"}])
        assert all(prompt[n]["inputs"]["model"] == ["2",0] for n in ("8","11","111","141","171"))
        assert all(prompt[n]["inputs"]["positive"] == ["4",0] for n in ("8","11","111","141","171"))
        assert prompt["141"]["inputs"]["subjects"] == prompt["171"]["inputs"]["subjects"]
        for item, config in zip(evaluation["variant_results"], index["edits"] if "edits" in index else index["variants"]):
            node_id = config["save_node"]
            path, r = edits[node_id]
            k = "k1" if item["requested_prototypes"] == 1 else "k3"
            item.update(image_path=str(path.parent/r["artifacts"]["image"]), edit_diagnostic_json_path=str(path), final_latent_sha256=r["final_latent_sha256"], reference_partition_sha256=r["reference_partition_sha256"],effective_image_mask_sha256=r["effective_image_mask_sha256"])
            item["achieved_prototypes_by_role"] = {role:len(ids) for role,ids in suites[k]["report"]["algorithm"]["variants"]["multi_proto_bg"]["prototype_indices"].items()}
            assert item["reference_partition_sha256"] == evaluation["collection_evidence"][k]["source_partition_sha256"]
            assert r["initial_noise_sha256"] == suites[k]["report"]["initial_noise_sha256"]
            assert r["initial_latent_sha256"] == suites[k]["report"]["initial_latent_sha256"]
            assert r["full_sigmas_sha256"] == suites[k]["report"]["full_sigmas_sha256"]
            assert r["conditioning_sha256"] == suites[k]["report"]["conditioning"]["positive_sha256"]
            item["runtime"] = {key:r.get(key) for key in ("run_id","total_model_nfe","elapsed_seconds","base_quant_formats","owned_hooks_removed")}
        mask_stats = {}
        for role in ("target","protected","background"):
            ma,mb = (maps[k]["multi_proto_bg.masks."+role].bool() for k in ("k1","k3"))
            union = int((ma|mb).sum()); intersection = int((ma&mb).sum())
            mask_stats[role] = {"k1_cells":int(ma.sum()),"k3_cells":int(mb.sum()),"intersection_cells":intersection,"union_cells":union,"iou":intersection/union if union else None,"differing_cells":int((ma!=mb).sum())}
        evaluation["mask_descriptive_statistics"] = mask_stats
        evaluation["actual_model_nfe"] = sum(s["report"]["phase1_nfe"] for s in suites.values()) + sum(r["total_model_nfe"] for _,r in edits.values())
        evaluation["wall_seconds"] = case["wall_seconds"]
        evaluation["cache_status"] = [message[1] for message in case["status"]["messages"] if message[0] == "execution_cached"]
        evaluation["memory"] = {"collections":{k:s["report"]["memory"] for k,s in suites.items()},"queue_peak_bytes":None,"queue_peak_unavailable_reason":"No measurement covers the full queue; collection peaks are kept in their reported scopes and never summed."}
        evaluation["artifact_integrity"] = {"status":"verified","suite_count":len(suites),"edit_count":len(edits),"checks":"All file hashes, tensor hashes/shapes/dtypes, binary mask partitions, per-step prediction routing and trace hashes."}
        evaluation["asset_identity"]["current_asset_record"] = str(root / "current-assets.json")
        write(folder / "evaluation.json", evaluation)
        row = {"case":case["case"],"observation_parity":evaluation["observation_parity"]["status"],"strength_pair_mask_parity":evaluation["strength_pair_mask_parity"]["status"],"nfe":evaluation["actual_model_nfe"],"wall_seconds":evaluation["wall_seconds"],"mask_statistics":mask_stats,"prototype_counts":{i["label"]:i["achieved_prototypes_by_role"] for i in evaluation["variant_results"]}}
        result["cases"].append(row)
        result["total_model_nfe"] += row["nfe"]
        print(json.dumps(row,ensure_ascii=False),flush=True)
    write(root / "verification.json", result)


if __name__ == "__main__":
    main()
