"""Add actual block indices 1..13, with repeated block18 for numerical control."""
import copy
import json
from pathlib import Path
import sys

import torch

from run_attention_step_block_sweep import run_stage, retain_sweeps
from run_mask_source_comparison_live import BASE, rt, sha, write
from audit_attention_step_block_sweep import verify_sweep, hole_statistics, safe_tap, COMMON
from audit_attention_ablation_live import verify_suite


def main():
    torch.set_num_threads(4)
    root = Path(sys.argv[1]).resolve()
    assert root.is_relative_to(Path(__file__).resolve().parent/"attention-step-block-sweep-results")
    plan = rt.read_json(root/"plan.json")
    prior = Path(plan["prior_root"]).resolve()
    runtime = rt.read_json(prior/"runtime.json")
    write(root/"runtime.json", {**runtime,"collection_model_nfe_budget":12,"scope":"mask-only supplement; block18 repeated as comparison control; no new edited images"})
    write(root/"system_stats.before.json",rt.request_json(BASE+"/system_stats"))
    queue = rt.request_json(BASE+"/queue")
    write(root/"queue.before.json",queue)
    if queue["queue_running"] or queue["queue_pending"]:
        raise RuntimeError("Existing queue is busy; do not overlap the supplemental experiment")
    expected = {f"step_{s}.block_{b}" for s in plan["steps"] for b in plan["capture_blocks"]}
    hole = rt.read_json(prior.parent.parent/"mask-source-comparison-results"/"20261011T104616Z"/"park"/"head-hole-probe.json")
    summary = {"root":str(root),"server":BASE,"cases":[]}
    verification = {"cases":[],"new_mask_conditions":0,"repeated_control_taps":0,"total_model_nfe":0,
                    "scope":"file/tensor hashes, mask reconstruction, capture integrity, old block18 tensor parity and fixed park 70-cell ROI",
                    "limits":"actual zero-based block labels 1..27; block0 is outside this request; no new edited images, independent face labels or human approval"}
    for case in plan["cases"]:
        cid=case["case"];folder=root/cid;folder.mkdir(exist_ok=True)
        original=rt.read_json(prior/cid/"mask.queued_api.json")["prompt"]
        prompt=copy.deepcopy(original)
        for sid, saver, branch in (("8","18","off"),("11","19","on")):
            config=json.loads(prompt[sid]["inputs"]["config_json"])
            config["selected_blocks"]=plan["capture_blocks"]
            prompt[sid]["inputs"]["config_json"]=json.dumps(config)
            prompt[saver]["inputs"]["filename_prefix"]=f"sw_{root.name}_{cid}_mask_{branch}"
        prompt,history=run_stage(folder,"mask",prompt)
        assert history["prompt"][2]==prompt
        assert sha(folder/"mask.queued_api.json")==rt.read_json(folder/"mask.submission.json")["queued_api_sha256"]
        records=retain_sweeps(folder,prompt,"mask")
        collections={};controls={};failed={};successful={}
        for record in records:
            branch=record["branch"];outer,taps=verify_sweep(Path(record["local"]),expected)
            old_folder=prior/cid/"sweeps"/"mask"/branch
            old_outer=rt.read_json(old_folder/"manifest.json")
            old_report=old_outer["report"];report=outer["report"]
            assert all(report[key]==old_report[key] for key in COMMON if key!="normalized_config")
            assert {k:v for k,v in report["normalized_config"].items() if k!="selected_blocks"} == {k:v for k,v in old_report["normalized_config"].items() if k!="selected_blocks"}
            control=[]
            for step in plan["steps"]:
                key=f"step_{step}.block_18"
                old_manifest,old_tensors=verify_suite(safe_tap(old_folder,old_outer["taps"][key]))
                new_manifest,new_tensors=taps[key]
                assert new_manifest["tensor_sha256"]==old_manifest["tensor_sha256"],(cid,branch,key)
                assert new_manifest["report"]["algorithm"]["variants"]==old_manifest["report"]["algorithm"]["variants"]
                control.append({"step":step,"block":18,"all_tensor_hashes_equal":True,"candidate_status_equal":True})
            controls[branch]=control;collections[branch]=taps
            failed[branch]={key:item["status"] for key,item in outer["taps"].items() if int(key.split(".")[1].split("_")[1]) in plan["new_blocks"] and item["status"]["status"]!="ok"}
            successful[branch]=39-len(failed[branch])
            verification["new_mask_conditions"]+=39
            verification["repeated_control_taps"]+=3
        off,on=(rt.read_json(Path(record["local"])/"manifest.json")["report"] for record in records)
        assert all(off[key]==on[key] for key in COMMON)
        assert off["phase1_nfe"]+on["phase1_nfe"]==6
        row={**case,"sweeps":records,"mask_prompt_id":rt.read_json(folder/"mask.submission.json")["prompt_id"],"mask_timing":rt.read_json(folder/"mask.timing.json"),"nfe":6}
        write(folder/"artifacts.json",row);summary["cases"].append(row)
        checked={"case":cid,"seed":case["seed"],"successful_new_masks":successful,"failed_new_taps":failed,"control_parity":controls,"nfe":6}
        if cid=="park":
            protected=collections["off"]["step_2.block_18"][1]["multi_proto_bg.masks.protected"]
            checked["hole_probe"]={branch:[row for row in hole_statistics(taps,hole["grid_indices"],protected) if row["block"] in plan["new_blocks"]] for branch,taps in collections.items()}
        verification["cases"].append(checked);verification["total_model_nfe"]+=6
        write(root/f"summary.after_{cid}.json",summary)
        print("VERIFIED",cid,successful,"block18 controls equal",flush=True)
    assert verification["new_mask_conditions"]==156 and verification["repeated_control_taps"]==12
    write(root/"summary.json",summary);write(root/"verification.json",verification)
    write(root/"queue.after.json",rt.request_json(BASE+"/queue"))
    print("DONE",root,"156 new conditions, 12 control taps, 12 NFE",flush=True)


if __name__=="__main__":main()
