"""Frozen trace decode, observed trajectories, and fixed candidate replay."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from run_attention_step_block_sweep import run_stage,retain_edits
from run_mask_source_comparison_live import rt,write,node,OUTPUT,sha

PRIOR=Path(__file__).resolve().parent/"mask-source-comparison-results"/"20261011T104616Z"


def diagnostic(case,label):
    return next((PRIOR/case/"edits"/label).glob("*.json"))


def retain_bundle(folder,prefix):
    matches=list(OUTPUT.glob(prefix+"_*"))
    if len(matches)!=1:raise ValueError(f"Expected one complete bundle for {prefix}")
    source=matches[0].resolve();dest=folder
    if source.parent!=OUTPUT:raise ValueError("Unsafe output bundle")
    manifest=rt.read_json(source/"manifest.json")
    if manifest.get("complete") is not True:raise ValueError("Incomplete result")
    for path in source.rglob("*"):
        if not path.resolve().is_relative_to(source):raise ValueError("Unsafe output entry")
        target=dest/path.relative_to(source)
        if path.is_dir():target.mkdir(parents=True,exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(path,target)
    return {"source":str(source),"local":str(dest.resolve()),"manifest_sha256":sha(dest/"manifest.json")}


def common(case):
    old=rt.read_json(PRIOR/case/"queued_api.json")["prompt"]
    prompt={sid:copy.deepcopy(old[sid]) for sid in ("1","2","3","4","5","7","9")}
    node(prompt,"60","Krea2SliderFuseRetainedMask",{"report_path":str(diagnostic(case,"D_union_base_protected"))},"Verified D control masks")
    subject=copy.deepcopy(old["6"]["inputs"]);subject.update(target_mask=["60",0],protected_mask=["60",1])
    node(prompt,"6","Krea2SliderFuseSubjects",subject,"Fixed D pilot subjects")
    return old,prompt


def main():
    ap=argparse.ArgumentParser();ap.add_argument("root");ap.add_argument("--stage",choices=("traces","probe","replay"),required=True);ap.add_argument("--attempt",default="")
    args=ap.parse_args();root=Path(args.root).resolve()
    if not root.is_relative_to(Path(__file__).resolve().parent/"attention-step-block-sweep-results"):raise ValueError("Unexpected experiment root")
    plan=rt.read_json(root/"plan.json");records=[]
    for case in plan["cases"]:
        cid=case["case"];folder=root/cid;folder.mkdir(exist_ok=True);old,prompt=common(cid)
        stamp=root.name.replace("-", "_");stamp+=args.attempt
        if args.stage=="traces":
            for index,label in enumerate(("A_off","C_on_base_protected","D_union_base_protected")):
                node(prompt,str(100+index),"Krea2SliderFuseTraceDecode",{"model":["2",0],"vae":["9",0],"report_path":str(diagnostic(cid,label)),"filename_prefix":f"td_{stamp}_{cid}_{label}"},label+" trace")
            prompt,history=run_stage(folder,"traces"+args.attempt,prompt)
            bundles={label:retain_bundle(folder/"traces"/label,prompt[str(100+i)]["inputs"]["filename_prefix"]) for i,label in enumerate(("A_off","C_on_base_protected","D_union_base_protected"))}
            records.append({"case":cid,"bundles":bundles,"diffusion_model_nfe":0})
        else:
            candidates=[];bundles={}
            scopes=(("off","none"),("on","all"),("mix","target_mask")) if args.stage=="probe" else [(row["label"],"target_mask") for row in rt.read_json(folder/"candidate-plan.json") if row["status"]=="ok"]
            for index,(label,scope) in enumerate(scopes):
                base=100+index*100;sampler,decode,save=str(base),str(base+1),str(base+2)
                subject="6"
                if args.stage=="replay":
                    entry=next(r for r in rt.read_json(folder/"candidate-plan.json") if r["label"]==label)
                    readmask=str(base+3);subject=str(base+4)
                    node(prompt,readmask,"Krea2SliderFuseRetainedMask",{"report_path":entry["manifest"]},label)
                    sub=copy.deepcopy(prompt["6"]["inputs"]);sub.update(target_mask=[readmask,0],protected_mask=[readmask,1])
                    node(prompt,subject,"Krea2SliderFuseSubjects",sub,label)
                inputs=copy.deepcopy(old["21"]["inputs"]);inputs.update(model=["2",0],subjects=[subject,0],mix_scope=scope,trial_id=case["seed"],diagnostic_level="audit")
                if args.stage=="probe":inputs["background_phrase"]=old["8"]["inputs"]["background_phrase"]
                node(prompt,sampler,"Krea2SliderFuseObservedPredictionMix" if args.stage=="probe" else "Krea2SliderFusePredictionMixSampler",inputs,label)
                node(prompt,decode,"VAEDecode",{"samples":[sampler,0],"vae":["9",0]},label)
                node(prompt,save,"Krea2SliderFuseDiagnosticSave",{"images":[decode,0],"latent":[sampler,0],"diagnostics":[sampler,2],"filename_prefix":f"paired_mix/{root.name}/{cid}/{args.stage}{args.attempt}_{label}"},label)
                candidates.append({"label":args.stage+args.attempt+"_"+label,"status":"queued","sampler_node":sampler,"save_node":save})
                if args.stage=="probe":
                    saver=str(base+5);prefix=f"pp_{stamp}_{cid}_{label}"
                    node(prompt,saver,"Krea2SliderFusePairedProbeSave",{"probe":[sampler,3],"model":["2",0],"vae":["9",0],"filename_prefix":prefix},label+" paired evidence")
                    bundles[label]=prefix
                    if scope in ("none","all"):
                        cross,savecross=str(base+6),str(base+7);crossprefix=f"pp_{stamp}_{cid}_cross_{label}"
                        crossinputs={k:inputs[k] for k in ("model","positive","negative","prompt_info","subjects","lora_name","background_phrase")};crossinputs["diagnostics"]=[sampler,2]
                        node(prompt,cross,"Krea2SliderFuseCrossReplay",crossinputs,"Both weights on "+label+" inputs")
                        node(prompt,savecross,"Krea2SliderFusePairedProbeSave",{"probe":[cross,0],"model":["2",0],"vae":["9",0],"filename_prefix":crossprefix},"Cross evidence")
                        bundles["cross_"+label]=crossprefix
            if not candidates:raise ValueError("No successful replay candidates")
            write(folder/(args.stage+args.attempt+".planned.json"),candidates)
            prompt,history=run_stage(folder,args.stage+args.attempt,prompt)
            kept=retain_edits(folder,candidates,history)
            bundle_records={label:retain_bundle(folder/"probes"/(label+args.attempt),prefix) for label,prefix in bundles.items()}
            record={"case":cid,"diagnostics":kept,"probes":bundle_records,"generation_nfe":sum(rt.read_json(r["local_report"])["total_model_nfe"] for r in kept)}
            if args.stage=="probe":record["cross_nfe"]=sum(rt.read_json(Path(r["local"])/"manifest.json")["report"]["total_model_nfe"] for label,r in bundle_records.items() if label.startswith("cross_"))
            records.append(record)
        write(folder/(args.stage+args.attempt+".artifacts.json"),records[-1])
        write(root/(args.stage+args.attempt+f".summary.after_{cid}.json"),records)
        print("CASE_DONE",cid,args.stage,flush=True)
    write(root/(args.stage+args.attempt+".summary.json"),records)
    print("DONE",args.stage,root,flush=True)


if __name__=="__main__":main()
