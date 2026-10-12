"""Check observed trajectories, cross controls and candidate routing artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from audit_attention_ablation_live import read,sha,verify_edit,verify_suite
from audit_attention_step_block_sweep import safe_tap
from slider_fuse.paired_observation import checked_diagnostic
from slider_fuse.diagnostics import diagnostic_provenance


def main():
    ap=argparse.ArgumentParser();ap.add_argument("root");ap.add_argument("--probe-only",action="store_true");args=ap.parse_args()
    torch.set_num_threads(4);root=Path(args.root).resolve()
    result={"stage":"probe" if args.probe_only else "fixed_replay","cases":[],"diffusion_model_nfe":0,"verified_taps":0,
            "quality_approval":"not_evaluated by numerical audit; see separate visual observations"}
    for entry in read(root/"traces.summary.json"):
        for record in entry["bundles"].values():
            folder=Path(record["local"]);manifest=read(folder/"manifest.json")
            assert sha(folder/"manifest.json")==record["manifest_sha256"]
            assert manifest["final_decoded_rgb_equal"] and manifest["diffusion_model_nfe"]==0
            assert all(sha(folder/name)==expected for name,expected in manifest["file_sha256"].items())
    for case in read(root/"probe.summary.json"):
        cid=case["case"];folder=root/cid;graph=read(folder/"probe.queued_api.json")["prompt"];history=read(folder/"probe.history.json")
        assert history["status"]["status_str"]=="success" and history["prompt"][2]==graph
        expected_inputs={};probes={};tap_maps={}
        for label,record in case["probes"].items():
            path=Path(record["local"]);outer=read(path/"manifest.json");assert outer["complete"]
            assert sha(path/"manifest.json")==record["manifest_sha256"]
            frames=outer["frames"]
            for frame in frames:
                assert all(t["block_input_before_sha256"]==t["block_input_after_sha256"] and t["tensor_sha256_before"]==t["tensor_sha256_after"] for t in frame["taps"])
                assert len(frame["taps"])==5
                if label=="mix":
                    key=frame["step"]
                    if key in expected_inputs:assert expected_inputs[key]==frame["input_sha256"]
                    expected_inputs[key]=frame["input_sha256"]
            taps={}
            for key,item in outer["taps"].items():
                manifest,tensors=verify_suite(safe_tap(path,item));taps[key]=manifest
                result["verified_taps"]+=1
            previews=read(path/"previews.json")
            assert all(sha(path/r["image"])==r["image_sha256"] for r in previews)
            probes[label]=outer;tap_maps[label]=taps
        for source in ("off","on"):
            cross=probes["cross_"+source]
            assert len(cross["report"]["source_prediction_parity"])==2
            assert all(row["exact_equal"] for row in cross["report"]["source_prediction_parity"])
            branch="base" if source=="off" else "slider"
            for step in (2,3):
                for block in (8,10,18,23,26):
                    key=f"{branch}.step_{step}.block_{block}"
                    assert tap_maps[source][key]["tensor_sha256"]==tap_maps["cross_"+source][key]["tensor_sha256"]
        diagnostics={}
        for record in case["diagnostics"]:
            path=Path(record["local_report"]);r=verify_edit(path)
            assert r["generation"]==diagnostic_provenance(graph,record["save_node"],r)
            assert r["owned_hooks_removed"] and r["phase1_nfe"]==0
            diagnostics[record["label"]]=(r,checked_diagnostic(path)[1])
        pilot,t=diagnostics["probe_mix"]
        old=next((root.parent.parent/"mask-source-comparison-results"/"20261011T104616Z"/cid/"edits"/"D_union_base_protected").glob("*.json"))
        old_report,old_t=checked_diagnostic(old)
        compare=("trace_inputs","trace_predictions","trace_base_predictions","trace_slider_predictions","final_latent")
        assert all(torch.equal(t[key],old_t[key]) for key in compare)
        nfe=case["generation_nfe"]+case["cross_nfe"]
        assert nfe==40
        row={"case":cid,"observed_D_exact_parity":True,"cross_prediction_and_tap_parity":True,"same_input_mix_branch_hashes":expected_inputs,"probe_nfe":nfe,"replays":[]}
        if not args.probe_only:
            replay=read(folder/"replay.artifacts.json");replaygraph=read(folder/"replay.queued_api.json")["prompt"]
            assert read(folder/"replay.history.json")["prompt"][2]==replaygraph
            for record in replay["diagnostics"]:
                path=Path(record["local_report"]);r=verify_edit(path);_,saved=checked_diagnostic(path)
                assert r["generation"]==diagnostic_provenance(replaygraph,record["save_node"],r)
                label=record["label"].removeprefix("replay_")
                candidate=next(x for x in read(folder/"candidate-plan.json") if x["label"]==label)
                manifest_path=Path(candidate["manifest"]);meta=read(manifest_path)
                assert sha(manifest_path.parent/meta["tensors"])==meta["tensors_sha256"]
                masks=load_file(str(manifest_path.parent/meta["tensors"]))
                assert all(torch.equal(saved["reference_"+role+"_mask"],masks[role]) for role in ("target","protected","background"))
                assert torch.equal(saved["reference_protected_mask"],t["reference_protected_mask"])
                assert torch.equal(saved["trace_base_predictions"][0],t["trace_base_predictions"][0])
                assert torch.equal(saved["trace_slider_predictions"][0],t["trace_slider_predictions"][0])
                assert r["total_model_nfe"]==16 and r["effective_slider_strength"]==4
                row["replays"].append({"label":label,"exact_routing_all_steps":True,"protected_control_equal":True,
                                       "image":str(path.parent/r["artifacts"]["image"]),"diagnostic":str(path),"target_cells":int(masks["target"].sum())})
            nfe+=replay["generation_nfe"]
        row["nfe"]=nfe;result["cases"].append(row);result["diffusion_model_nfe"]+=nfe
        print("VERIFIED",cid,nfe,"NFE",len(row["replays"]),"replays",flush=True)
    output=root/("verification.probe.json" if args.probe_only else "verification.json")
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(output)


if __name__=="__main__":main()
