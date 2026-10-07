"""Probe real ComfyUI Krea2/Slider keys and representative native linear routing."""
import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path

from comfy_environment import bootstrap


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root",required=True,help="Real ComfyUI root; use its Python interpreter")
    parser.add_argument("--model",required=True,help="Actual Krea2 checkpoint path")
    parser.add_argument("--lora",required=True,help="Trained attention-target Slider safetensors path")
    parser.add_argument("--cpu",action="store_true",help="CPU probe for a floating point reference; INT8 may require CUDA")
    args=parser.parse_args()
    for name in ("model","lora"):
        if not Path(getattr(args,name)).is_file(): parser.error(f"Missing {name} file")
    report={"status":"failed","image_quality_validated":False,"int8_image_generation_validated":False}
    model=None
    try:
        report["comfy_commit"]=bootstrap(args.comfy_root,cpu=args.cpu)
        import torch
        import comfy.sd
        import comfy.model_management
        from comfy.ldm.krea2.model import SingleStreamDiT
        from safetensors.torch import load_file
        from slider_fuse.lora import load_adapters,SliderHook,RoutingState,core_guard,compare_probe_outputs
        from slider_fuse.sampling import tensor_hash

        model=comfy.sd.load_diffusion_model(args.model)
        if model is None or not isinstance(model.model.diffusion_model,SingleStreamDiT):
            raise ValueError("The checkpoint is not a native Krea2 model")
        core=model.model.diffusion_model
        adapters=load_adapters(load_file(args.lora,device="cpu"),core)
        report["model_class"]=type(core).__name__
        report["matched_modules"]=len(adapters)
        report["lora_keys"]={name:{"rank":adapter.down.shape[0],"alpha":adapter.alpha,
                                  "down_shape":list(adapter.down.shape),"up_shape":list(adapter.up.shape)} for name,adapter in adapters.items()}
        report["probes"]=[]
        comfy.model_management.load_models_gpu([model])
        def fingerprint(weight):
            data=getattr(weight,"_qdata",weight)
            result={"data":tensor_hash(data)}
            params=getattr(weight,"_params",None)
            if params is not None:
                attributes=[f.name for f in fields(params)] if is_dataclass(params) else list(vars(params))
                for key in attributes:
                    value=getattr(params,key)
                    result[key]=tensor_hash(value) if isinstance(value,torch.Tensor) else str(value)
            return result
        with core_guard(core),torch.inference_mode():
            # One representative layer for each projection family, when present.
            selected={}
            for name in adapters: selected.setdefault(name.rsplit(".",1)[-1],name)
            for name in selected.values():
                adapter=adapters[name]; module=core.get_submodule(name)
                quant=getattr(module,"quant_format",None)
                if quant not in (None,"int8_tensorwise"): raise ValueError(f"Unsupported base quantization {quant} at {name}")
                dtype=model.model.get_dtype()
                if not dtype.is_floating_point: dtype=torch.float32
                x=torch.randn(1,6,adapter.down.shape[1],device=model.load_device,dtype=dtype)
                before=fingerprint(module.weight)
                reference=module(x); repeat=module(x)
                state=RoutingState(phase="route",cap_len=2,mask=torch.tensor([[[1.,0.],[0.,1.]]]))
                original=module.forward; hook=SliderHook(module,adapter,1.,state,name)
                try:
                    hook.inject(); actual=module(x)
                finally: hook.eject()
                comparison=compare_probe_outputs(reference,repeat,actual,adapter.delta(x[:,2:]),state.mask.reshape(1,4,1),2)
                unchanged=before==fingerprint(module.weight)
                passed=comparison["passed"] and unchanged and module.forward==original
                report["probes"].append({"module":name,"base_quant_format":quant,**comparison,
                    "packed_weight_unchanged":unchanged,"passed":passed})
                adapter.clear()
                if not passed: raise RuntimeError(f"Native linear routing/restoration failed at {name}")
        report["status"]="passed"
        report["note"]="Representative native linear checks only; actual sampling, masks and image quality remain unvalidated."
    except Exception as error:
        report.update(error_type=type(error).__name__,error=str(error))
    finally:
        if model is not None:
            import comfy.model_management
            comfy.model_management.unload_model_and_clones(model,unload_additional_models=False)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report["status"]=="passed" else 1


if __name__=="__main__":
    raise SystemExit(main())
