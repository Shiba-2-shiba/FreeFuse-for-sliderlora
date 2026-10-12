"""Owned passive branch observation and checked diagnostic replay helpers."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import uuid

import torch
from safetensors.torch import load_file

from .attention_sweep import SweepAccumulator, validate_sweep_config
from .experimental_collection import ExperimentalAttentionCollector
from .experimental_artifacts import save_experimental_artifacts, validate_experimental_prefix
from .sampling import tensor_hash


def decode_prediction(model,vae,value,*,processed=False):
    with torch.inference_mode():
        latent=value.clone() if processed else model.model.process_latent_out(value.clone().float())
        pixels=vae.decode(latent).detach().cpu()
    if pixels.ndim==5:pixels=pixels.reshape(-1,*pixels.shape[-3:])
    if pixels.ndim!=4 or len(pixels)!=1:raise ValueError("Trace decoder requires exactly one still frame")
    return pixels[0]


def checked_diagnostic(path, root=None):
    path=Path(path).resolve()
    root=Path(root).resolve() if root is not None else Path(__file__).resolve().parent.parent/"docs"
    if not path.is_relative_to(root) or path.suffix!=".json":raise ValueError("Diagnostic must be a retained repository JSON")
    report=json.loads(path.read_text(encoding="utf-8"))
    if report.get("complete") is not True:raise ValueError("Incomplete diagnostic")
    for kind,expected in report.get("artifact_sha256",{}).items():
        name=report["artifacts"][kind]
        item=(path.parent/name).resolve()
        if Path(name).name!=name or item.parent!=path.parent or not item.is_file():raise ValueError("Unsafe diagnostic artifact")
        if hashlib.sha256(item.read_bytes()).hexdigest()!=expected:raise ValueError("Changed diagnostic artifact")
    if "tensors" not in report.get("artifact_sha256",{}):raise ValueError("Missing diagnostic tensor hash")
    tensors=load_file(str(path.parent/report["artifacts"]["tensors"]),device="cpu")
    for name,row in report["tensor_manifest"].items():
        if name not in tensors or tensor_hash(tensors[name])!=row["sha256"]:raise ValueError("Changed diagnostic tensor")
    return report,tensors


def read_retained_masks(path,root=None):
    from .masks import _validate_partition
    path=Path(path).resolve()
    root=Path(root).resolve() if root is not None else Path(__file__).resolve().parent.parent/"docs"
    if not path.is_relative_to(root) or path.suffix!=".json":raise ValueError("Mask must be retained in repository docs")
    report=json.loads(path.read_text(encoding="utf-8"))
    if report.get("complete") is not True:raise ValueError("Incomplete retained mask")
    if report.get("anchored_mask_schema")==1:
        name=report["tensors"];target=(path.parent/name).resolve()
        if Path(name).name!=name or target.parent!=path.parent:raise ValueError("Unsafe candidate mask tensor")
        if hashlib.sha256(target.read_bytes()).hexdigest()!=report["tensors_sha256"]:raise ValueError("Changed candidate mask")
        tensors=load_file(str(target));masks={role:tensors[role] for role in ("target","protected","background")}
    else:
        _,tensors=checked_diagnostic(path,root)
        masks={role:tensors["reference_"+role+"_mask"] for role in ("target","protected","background")}
    _validate_partition({"masks":masks,"grid":tuple(masks["target"].shape[-2:])})
    return {role:value.detach().clone() for role,value in masks.items()}


class RawSweep(SweepAccumulator):
    def __init__(self,config,grid,positions,keep_raw):
        super().__init__(config,grid,positions);self.keep_raw=keep_raw

    def observe(self,*,step,block,**data):
        super().observe(step=step,block=block,**data)
        if self.keep_raw and step==3 and block in (18,23):
            evidence=self.taps[f"step_{step}.block_{block}"]["evidence_maps"]
            for name in ("attention_output","img_q","img_k"):
                evidence["capture."+name]=data[name][0].detach().float().cpu().clone()


class PairedObserver:
    def __init__(self,positions,*,blocks=(8,10,18,23,26),steps=(1,2,3),keep_raw=False):
        self.config=validate_sweep_config({"selected_blocks":list(blocks),"collect_block":18})
        self.positions=copy.deepcopy(positions);self.steps=tuple(steps);self.keep_raw=keep_raw
        self.accumulators={};self.frames=[];self.step_override=None

    def bind(self,core,grid,text_len,sigmas):
        self.grid=tuple(grid);self.text_len=text_len;self.sigmas=sigmas.detach().cpu().clone()
        if any(b>=len(core.blocks) for b in self.config["selected_blocks"]):raise ValueError("Probe block outside model")

    @contextmanager
    def capture(self,core,branch,step,x,sigma):
        step=self.step_override or step
        if step not in self.steps:
            yield
            return
        if branch not in self.accumulators:
            self.accumulators[branch]=RawSweep(self.config,self.grid,self.positions,self.keep_raw)
        accumulator=self.accumulators[branch]
        collector=ExperimentalAttentionCollector(accumulator,self.config,self.grid,self.text_len,self.sigmas,audit_tensors=True)
        before=tensor_hash(x)
        try:
            collector.begin_forward(sigma,step);collector.install(core)
            yield
            self.frames.append({"branch":branch,"step":step,"input_sha256":before,"sigma":float(sigma.flatten()[0]),
                                "taps":collector.manifest})
            if before!=tensor_hash(x):raise RuntimeError("Probe mutated model input")
        finally:collector.remove()

    def result(self,report,predictions=None):
        expected={f"step_{s}.block_{b}" for s in self.steps for b in self.config["selected_blocks"]}
        for branch,acc in self.accumulators.items():
            if set(acc.taps)!=expected:raise RuntimeError("Incomplete paired observation")
        if not self.accumulators:raise RuntimeError("No observed branch")
        return {"probe_schema_version":1,"report":copy.deepcopy(report),"frames":self.frames,
                "branches":{branch:acc.taps for branch,acc in self.accumulators.items()},"predictions":predictions or {}}


def save_probe(directory,prefix,probe):
    validate_experimental_prefix(prefix)
    root=Path(directory).resolve()/f"{prefix}_{uuid.uuid4().hex}";root.mkdir(exist_ok=False)
    records={}
    for branch,taps in probe["branches"].items():
        for key,tap in taps.items():
            report=copy.deepcopy(probe["report"]);report.update(algorithm=tap["algorithm"],grid=list(tap["algorithm"]["grid"]),probe_branch=branch)
            suite={"experiment_schema_version":1,"banks":tap["banks"],"report":report,"evidence_maps":tap["evidence_maps"]}
            saved=save_experimental_artifacts(root,branch+"_"+key.replace(".","_"),suite)
            records[branch+"."+key]={"manifest":saved["manifest"],"manifest_sha256":hashlib.sha256((root/saved["manifest"]).read_bytes()).hexdigest()}
    manifest={"probe_schema_version":1,"complete":True,"report":probe["report"],"frames":probe["frames"],"taps":records}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding="utf-8")
    return root


def replay_cross(model,positive,negative,prompt_info,subjects,lora_path,payload,background_phrase):
    import comfy.samplers
    import comfy.sample
    import comfy.sampler_helpers
    import comfy.model_management
    from .conditioning import resolve_phrase
    from .native_pair import NativePairRunner,validate_mix_options,_cleanup
    from .lora import core_guard,load_adapters
    from .sampling import apply_native_slider
    report=payload.report
    if report["mix_scope"] not in ("none","all") or report["diagnostic_level"]!="audit":
        raise ValueError("Cross replay requires an audited single-branch trajectory")
    validate_mix_options(model)
    source=load_file(str(lora_path),device="cpu")
    if hashlib.sha256(Path(lora_path).read_bytes()).hexdigest()!=report["lora_sha256"]:raise ValueError("Cross replay Slider differs")
    core=model.model.diffusion_model;adapters=load_adapters(source,core)
    positions={s.id:list(s.positions) for s in subjects};positions["background"]=list(resolve_phrase(prompt_info,background_phrase,0))
    observer=PairedObserver(positions,steps=(2,3))
    tensors=payload.extra_tensors;sigmas=torch.cat((tensors["trace_sigmas"],torch.zeros(1)))
    if tensor_hash(sigmas)!=report["full_sigmas_sha256"]:raise ValueError("Cross replay sigma schedule differs")
    device=model.load_device;predictions={"base":[],"slider":[]};parity=[]
    expected_branch="base" if report["mix_scope"]=="none" else "slider"
    with core_guard(core):
        base=model.clone();slider=apply_native_slider(base,source,report["strength"],{name:core.get_submodule(name) for name in adapters})
        runner=NativePairRunner(base,slider,observer=observer);prepared=False
        try:
            guider=comfy.samplers.CFGGuider(base);guider.set_conds(positive,negative)
            cond_input=getattr(guider,"original_conds",None)
            if cond_input is None:raise RuntimeError("Unsupported replay conditioning contract")
            shape=tensors["trace_inputs"][0].shape
            _,conds,loaded=comfy.sampler_helpers.prepare_sampling(base,shape,cond_input,base.model_options);prepared=True
            if loaded:raise ValueError("Additional replay conditioning models unsupported")
            zero=torch.zeros(shape,device=device,dtype=torch.float32)
            noise=comfy.sample.prepare_noise(zero.cpu(),report["seed"],None).to(device)
            if tensor_hash(noise.cpu())!=report["initial_noise_sha256"]:raise ValueError("Replay noise differs")
            conds=comfy.samplers.process_conds(base.model,noise,conds,device,latent_image=zero,seed=report["seed"])
            options=copy.deepcopy(base.model_options);options.setdefault("transformer_options",{})["sample_sigmas"]=sigmas.to(device)
            comfy.samplers.cast_to_load_options(options,device=device,dtype=base.model_dtype())
            observer.bind(core,report["grid"],len(prompt_info.token_ids),sigmas)
            with torch.inference_mode():
                for step in (2,3):
                    observer.step_override=step;x=tensors["trace_inputs"][step-1].to(device);sigma=sigmas[step-1].reshape(1).to(device)
                    for branch in ("base","slider"):
                        p=runner.predict(branch,x,sigma,positive=conds["positive"],negative=conds.get("negative"),model_options=options,seed=report["seed"])
                        predictions[branch].append(p.detach().cpu())
                        if branch==expected_branch:
                            same=torch.equal(p.cpu(),tensors["trace_predictions"][step-1])
                            parity.append({"step":step,"branch":branch,"exact_equal":same})
                            if not same:raise RuntimeError("Cross replay original prediction does not match saved trajectory")
        finally:
            _cleanup(runner.close,*( [lambda:comfy.sampler_helpers.cleanup_models(conds,loaded)] if prepared else []),
                     lambda:comfy.model_management.unload_model_and_clones(base,unload_additional_models=False),*(a.clear for a in adapters.values()))
    replay_report={"seed":report["seed"],"steps":8,"replay_steps":[2,3],"source_scope":report["mix_scope"],
                   "branch_nfe":runner.calls,"total_model_nfe":sum(runner.calls.values()),"source_prediction_parity":parity,
                   "grid":report["grid"],"prediction_space":"comfy_cfg1_denoised","owned_hooks_removed":True}
    return observer.result(replay_report,{name:torch.stack(values) for name,values in predictions.items()})
