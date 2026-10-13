"""Experimental V3 branch observation, checked trace decoding and cross replay."""
import hashlib
import json
from pathlib import Path
import uuid

import folder_paths
import numpy as np
from PIL import Image
import torch
from comfy_api.latest import io

from .nodes import (CATEGORY,DiagnosticsType,PromptType,SubjectsType,MasksType,
                    Krea2SliderFusePredictionMixSampler)
from .slider_fuse.conditioning import resolve_phrase
from .slider_fuse.native_pair import sample_krea2_prediction_mix
from .slider_fuse.paired_observation import PairedObserver,checked_diagnostic,save_probe,replay_cross,decode_prediction,read_retained_masks
from .slider_fuse.experimental_artifacts import validate_experimental_prefix
from .slider_fuse.sampling import tensor_hash
from .slider_fuse.paired_prefix import sample_paired_pilot

ProbeType=io.Custom("KREA2_SLIDER_FUSE_PAIRED_PROBE")


class Krea2SliderFuseRetainedMask(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name="Load verified retained mask",category=CATEGORY+"/Experimental",
            inputs=[io.String.Input("report_path")],outputs=[io.Mask.Output("target"),io.Mask.Output("protected")])

    @classmethod
    def execute(cls,report_path):
        masks=read_retained_masks(report_path)
        return io.NodeOutput(masks["target"],masks["protected"])

    @classmethod
    def fingerprint_inputs(cls,report_path,**kwargs):
        masks=read_retained_masks(report_path)
        return hashlib.sha256(Path(report_path).read_bytes()).hexdigest()+"".join(tensor_hash(masks[r]) for r in ("target","protected","background"))


def decoded_image(model,vae,value,*,processed=False):
    pixels=decode_prediction(model,vae,value,processed=processed)
    return Image.fromarray((pixels.clamp(0,1).numpy()*255).astype(np.uint8))


class Krea2SliderFuseTraceDecode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name="Decode retained Prediction Mix trace",category=CATEGORY+"/Experimental",is_output_node=True,
            description="Decode verified retained denoised predictions with the model latent transform; does not run diffusion inference.",
            inputs=[io.Model.Input("model"),io.Vae.Input("vae"),io.String.Input("report_path"),io.String.Input("filename_prefix")],outputs=[])

    @classmethod
    def execute(cls,model,vae,report_path,filename_prefix):
        validate_experimental_prefix(filename_prefix);report,tensors=checked_diagnostic(report_path)
        if report.get("prediction_space")!="comfy_cfg1_denoised":raise ValueError("Unsupported trace prediction space")
        root=Path(folder_paths.get_output_directory())/(filename_prefix+"_"+uuid.uuid4().hex);root.mkdir()
        rows=[];images=[]
        for name in ("trace_base_predictions","trace_slider_predictions","trace_predictions"):
            if name not in tensors:continue
            for index in (0,1,2,report["steps"]-1):
                value=tensors[name][index];filename=f"{name}_s{index+1}.png"
                decoded_image(model,vae,value).save(root/filename)
                rows.append({"branch":name,"step":index+1,"sigma":float(tensors["trace_sigmas"][index]),"prediction_sha256":tensor_hash(value),"image":filename})
                images.append({"filename":filename,"subfolder":root.name,"type":"output"})
        decoded_image(model,vae,tensors["final_latent"],processed=True).save(root/"final.png")
        final_source=Path(report_path).resolve().parent/report["artifacts"]["image"]
        final_equal=np.array_equal(np.asarray(Image.open(root/"final.png")),np.asarray(Image.open(final_source)))
        manifest={"complete":True,"source_report":str(Path(report_path).resolve()),"source_report_sha256":hashlib.sha256(Path(report_path).read_bytes()).hexdigest(),
                  "rows":rows,"final_decoded_rgb_equal":final_equal,"diffusion_model_nfe":0,
                  "file_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob("*.png")}}
        if not final_equal:raise RuntimeError("Trace decoder final RGB does not match retained diagnostic")
        (root/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
        return io.NodeOutput(ui={"images":images,"text":[str(root/"manifest.json")]})

    @classmethod
    def fingerprint_inputs(cls,report_path,**kwargs):
        report,tensors=checked_diagnostic(report_path)
        return hashlib.sha256(Path(report_path).read_bytes()).hexdigest()+report["artifact_sha256"]["tensors"]


class Krea2SliderFuseObservedPredictionMix(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema=Krea2SliderFusePredictionMixSampler.define_schema()
        schema.node_id=cls.__name__;schema.display_name="Prediction Mix with passive paired observation";schema.category=CATEGORY+"/Experimental"
        schema.inputs.append(io.String.Input("background_phrase"))
        for field in schema.inputs:
            if field.id=="mask_mode":field.options=["manual"]
            if field.id=="diagnostic_level":field.options=["audit"]
        schema.description="Observe native base / Slider branches on their actual inputs in manual audit mode; production defaults are unchanged."
        schema.outputs.append(ProbeType.Output("probe"))
        return schema

    @classmethod
    def fingerprint_inputs(cls,**kwargs):return Krea2SliderFusePredictionMixSampler.fingerprint_inputs(**kwargs)

    @classmethod
    def execute(cls,background_phrase,**kwargs):
        positions={s.id:list(s.positions) for s in kwargs["subjects"]}
        positions["background"]=list(resolve_phrase(kwargs["prompt_info"],background_phrase,0))
        observer=PairedObserver(positions,keep_raw=kwargs["mix_scope"]=="target_mask")
        path=folder_paths.get_full_path_or_raise("loras",kwargs.pop("lora_name"))
        latent,bank,payload=sample_krea2_prediction_mix(lora_path=path,_observer=observer,**kwargs)
        extra=payload.extra_tensors
        if kwargs["mix_scope"]=="target_mask":
            predictions={"base":extra["trace_base_predictions"],"slider":extra["trace_slider_predictions"],"mix":extra["trace_predictions"]}
        else:predictions={"base" if kwargs["mix_scope"]=="none" else "slider":extra["trace_predictions"]}
        return io.NodeOutput(latent,bank,payload,observer.result(payload.report,predictions))


class Krea2SliderFuseCrossReplay(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name="Cross replay base / Slider on retained trajectory",category=CATEGORY+"/Experimental",
            description="Replay steps 2/3 under both weights; exact original prediction equality is required.",
            inputs=[io.Model.Input("model"),io.Conditioning.Input("positive"),io.Conditioning.Input("negative"),PromptType.Input("prompt_info"),SubjectsType.Input("subjects"),
                    DiagnosticsType.Input("diagnostics"),io.Combo.Input("lora_name",options=folder_paths.get_filename_list("loras")),io.String.Input("background_phrase")],outputs=[ProbeType.Output("probe")])

    @classmethod
    def execute(cls,model,positive,negative,prompt_info,subjects,diagnostics,lora_name,background_phrase):
        path=folder_paths.get_full_path_or_raise("loras",lora_name)
        return io.NodeOutput(replay_cross(model,positive,negative,prompt_info,subjects,path,diagnostics,background_phrase))


class Krea2SliderFusePairedProbeSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__,display_name="Save paired probe masks and branch previews",category=CATEGORY+"/Experimental",is_output_node=True,
            description="Save per-branch taps, optional raw features and correctly transformed denoised previews.",
            inputs=[ProbeType.Input("probe"),io.Model.Input("model"),io.Vae.Input("vae"),io.String.Input("filename_prefix")],outputs=[])

    @classmethod
    def execute(cls,probe,model,vae,filename_prefix):
        root=save_probe(folder_paths.get_output_directory(),filename_prefix,probe);images=[];rows=[]
        for branch,values in probe["predictions"].items():
            steps=probe["report"].get("replay_steps",list(range(1,len(values)+1)))
            indices=range(len(values)) if len(values)<=3 else (0,1,2,len(values)-1)
            for index in indices:
                filename=f"{branch}_s{steps[index]}.png";decoded_image(model,vae,values[index]).save(root/filename)
                rows.append({"branch":branch,"step":steps[index],"image":filename,"prediction_sha256":tensor_hash(values[index]),"image_sha256":hashlib.sha256((root/filename).read_bytes()).hexdigest()})
                images.append({"filename":filename,"subfolder":root.name,"type":"output"})
        (root/"previews.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        return io.NodeOutput(ui={"images":images,"text":[str(root/"manifest.json")]})


class Krea2SliderFusePairedPrefixPilot(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = Krea2SliderFuseObservedPredictionMix.define_schema()
        schema.node_id = cls.__name__
        schema.display_name = "Experimental paired prefix mask pilot (3 of 8 steps)"
        schema.description = ("Build A-R0 from a paired pilot on the original eight-step schedule. "
            "Prefix stops after three updates; full_reference retains eight for exact comparison. "
            "Only masks and validation evidence are returned. Replay from the original latent with eight steps.")
        for field in schema.inputs:
            if field.id == "steps":
                field.default = field.min = field.max = 8
            if field.id == "mix_scope":
                field.options = ["target_mask"]
            if field.id in ("fill_holes_max_area", "mask_dilate_radius", "selection_dilate_radius"):
                field.default = field.min = field.max = 0
        schema.inputs.append(io.Combo.Input("pilot_mode", options=["prefix", "full_reference"], default="prefix"))
        schema.inputs.append(io.String.Input("source_identity_json", optional=True, default="{}",
            tooltip="Frozen workflow/model provenance for comparison; supplied hashes are not runtime asset verification."))
        schema.outputs = [MasksType.Output("mask_bank"), ProbeType.Output("probe")]
        return schema

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return Krea2SliderFusePredictionMixSampler.fingerprint_inputs(**kwargs)

    @classmethod
    def execute(cls, lora_name, background_phrase, pilot_mode="prefix", **kwargs):
        if not lora_name:
            raise ValueError("Select a Slider for the paired pilot")
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        bank, probe = sample_paired_pilot(lora_path=path, background_phrase=background_phrase,
                                         pilot_mode=pilot_mode, **kwargs)
        return io.NodeOutput(bank, probe)


class Krea2SliderFusePrefixPilotSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Save paired prefix parity evidence",
            category=CATEGORY+"/Experimental", is_output_node=True,
            description="Retain raw taps, actual input/prediction traces, original schedule and R0 masks. No VAE or partial image.",
            inputs=[ProbeType.Input("probe"), MasksType.Input("mask_bank"), io.String.Input("filename_prefix")], outputs=[])

    @classmethod
    def execute(cls, probe, mask_bank, filename_prefix):
        from .slider_fuse.prefix_artifacts import save_prefix_artifacts
        output = folder_paths.get_output_directory()
        saved = save_prefix_artifacts(output, filename_prefix, probe, mask_bank)
        return io.NodeOutput(ui={"text": [str(Path(output)/saved["manifest"])]})
