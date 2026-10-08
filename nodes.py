"""ComfyUI V3 nodes for Krea2 target-only Slider LoRA and diagnostics.

SPDX-License-Identifier: Apache-2.0
V3 node layout is adapted from FreeFuse-for-anima nodes.py at
1b924b5dd1f7266fa6e5869331e67a2033e7ec2f.
Modified for FreeFuse-for-sliderlora: Krea2 encoding, two subject roles,
one target Slider, mask diagnostics and optional target-text routing.
See LICENSE, NOTICE and THIRD_PARTY_NOTICES.md for licensing and attribution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .slider_fuse.conditioning import encode_prompt, make_subjects
from .slider_fuse.masks import similarity_preview
from .slider_fuse.sampling import sample_krea2, sample_krea2_diagnostic
from .slider_fuse.diagnostics import save_diagnostic_artifacts, validate_prefix
from .slider_fuse.native_pair import sample_krea2_prediction_mix

PromptType = io.Custom("KREA2_SLIDER_FUSE_PROMPT")
SubjectsType = io.Custom("KREA2_SLIDER_FUSE_SUBJECTS")
MasksType = io.Custom("KREA2_SLIDER_FUSE_MASKS")
DiagnosticsType = io.Custom("KREA2_SLIDER_FUSE_DIAGNOSTICS")
CATEGORY = "Krea2/Slider FreeFuse"


class Krea2SliderFuseEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Krea2SliderFuseEncode", display_name="Krea2 Slider Fuse Encode", category=CATEGORY,
            description="Encode one complete scene and retain the exact Krea2 token positions. Use CLIPLoader type krea2.",
            inputs=[io.Clip.Input("clip"), io.String.Input("prompt", multiline=True)],
            outputs=[io.Conditioning.Output("positive"), PromptType.Output("prompt_info")])

    @classmethod
    def execute(cls, clip, prompt):
        positive, info = encode_prompt(clip, prompt)
        return io.NodeOutput(positive, info)


class Krea2SliderFuseSubjects(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Krea2SliderFuseSubjects", display_name="Krea2 Slider Fuse Subjects", category=CATEGORY,
            description="Identify the target woman and protected man by phrases in the full prompt. Neither phrase is a slider trigger. Manual mode needs both MASK inputs.",
            inputs=[PromptType.Input("prompt_info"),
                    io.String.Input("target_phrase", default="adult woman in a sage-green top"),
                    io.String.Input("protected_phrase", default="adult man in a blue T-shirt"),
                    io.Int.Input("target_occurrence", default=0, min=0, max=999),
                    io.Int.Input("protected_occurrence", default=0, min=0, max=999),
                    io.Mask.Input("target_mask", optional=True), io.Mask.Input("protected_mask", optional=True)],
            outputs=[SubjectsType.Output("subjects")])

    @classmethod
    def execute(cls, prompt_info, target_phrase, protected_phrase, target_occurrence, protected_occurrence,
                target_mask=None, protected_mask=None):
        return io.NodeOutput(make_subjects(prompt_info, target_phrase, protected_phrase, target_occurrence,
                                           protected_occurrence, target_mask, protected_mask))


class Krea2SliderFuseSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Krea2SliderFuseSampler", display_name="Krea2 Slider Fuse Sampler", category=CATEGORY,
            description="Generate target image slider deltas and optional target phrase deltas, with shared scene attention. Euler/simple, CFG1, empty latent, batch1. Connect an optional GLOBAL style LoRA to MODEL; select the LOCAL slider here only. Real INT8/quality validation is pending.",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
                    PromptType.Input("prompt_info"), SubjectsType.Input("subjects"), io.Latent.Input("latent"),
                    io.Combo.Input("lora_name", options=[""] + folder_paths.get_filename_list("loras")),
                    io.Float.Input("strength", default=1., min=-10., max=10., step=.05),
                    io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                    io.Int.Input("steps", default=8, min=1, max=100),
                    io.Float.Input("cfg", default=1., min=1., max=1.),
                    io.Combo.Input("mask_mode", options=["auto", "manual"]),
                    io.Int.Input("collect_step", default=2, min=1, max=100),
                    io.Int.Input("collect_block", default=18, min=0, max=100),
                    io.Float.Input("top_k_ratio", default=.3, min=.001, max=1.),
                    io.Float.Input("temperature", default=4000., min=.001, max=100000.),
                    io.Int.Input("fill_holes_max_area", optional=True, default=0, min=0, max=64,
                                 tooltip="Auto only. Fill enclosed background-only holes up to this token area; 0 disables. Try 8 before dilation."),
                    io.Int.Input("mask_dilate_radius", optional=True, default=0, min=0, max=1,
                                 tooltip="Auto only. Expand the largest target component into background only; 1 token is about 16 image pixels for standard Krea2."),
                    io.Float.Input("target_text_scale", optional=True, default=0., min=0., max=1., step=.05,
                                   tooltip="Experimental. Target phrase rows inside Krea2 get strength * this scale; 0 preserves image-only routing. Other text gets no direct delta, but shared attention can affect the protected person.")],
            outputs=[io.Latent.Output("latent"), MasksType.Output("mask_bank"), io.String.Output("diagnostics")])

    @classmethod
    def fingerprint_inputs(cls, lora_name, **kwargs):
        if not lora_name:
            return "no-slider-selected"
        path = Path(folder_paths.get_full_path_or_raise("loras", lora_name))
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def execute(cls, model, positive, negative, prompt_info, subjects, latent, lora_name, strength, seed, steps,
                cfg, mask_mode, collect_step, collect_block, top_k_ratio, temperature,
                fill_holes_max_area=0, mask_dilate_radius=0, target_text_scale=0.):
        if not lora_name:
            raise ValueError("Select a trained Krea2 attention-target Slider LoRA in this Sampler")
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        output, bank, report = sample_krea2(model, positive, negative, prompt_info, subjects, latent, path,
            strength=strength, seed=seed, steps=steps, cfg=cfg, mask_mode=mask_mode, collect_step=collect_step,
            collect_block=collect_block, top_k_ratio=top_k_ratio, temperature=temperature,
            fill_holes_max_area=fill_holes_max_area, mask_dilate_radius=mask_dilate_radius,
            target_text_scale=target_text_scale)
        return io.NodeOutput(output, bank, json.dumps(report, ensure_ascii=False, indent=2))


class Krea2SliderFuseMaskPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Krea2SliderFuseMaskPreview", display_name="Krea2 Slider Fuse Mask Preview", category=CATEGORY,
            description="Outputs 1-3: routing masks; 4-5: independently normalized raw similarity, not calibrated confidence; 6-7: original target and added area. Manual mode has black similarity maps. Old/no-op banks have original=current target and added=black.",
            inputs=[MasksType.Input("mask_bank")],
            outputs=[io.Mask.Output("target_mask"), io.Mask.Output("protected_mask"), io.Mask.Output("background_mask"),
                     io.Mask.Output("target_similarity"), io.Mask.Output("protected_similarity"),
                     io.Mask.Output("original_target_mask"), io.Mask.Output("added_target_mask")])

    @classmethod
    def execute(cls, mask_bank):
        masks = mask_bank["masks"]
        return io.NodeOutput(masks["target"], masks["protected"], masks["background"],
                             similarity_preview(mask_bank, "target"), similarity_preview(mask_bank, "protected"),
                             mask_bank.get("original_masks", {}).get("target", masks["target"]),
                             mask_bank.get("added_target_mask", masks["target"].new_zeros(masks["target"].shape)))


class Krea2SliderFuseDiagnosticSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Krea2SliderFuseDiagnosticSampler", display_name="Krea2 Slider Diagnostic Sampler",
            category=CATEGORY + "/Diagnostics",
            description="Diagnostic only. All-image/all-text scopes directly affect protected people. Manual reference masks, Euler/simple, CFG1. Change trial_id to remeasure with the same seed.",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
                PromptType.Input("prompt_info"), SubjectsType.Input("subjects"), io.Latent.Input("latent"),
                io.Combo.Input("lora_name", options=[""] + folder_paths.get_filename_list("loras")),
                io.Float.Input("strength", default=4., min=-10., max=10., step=.05),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("steps", default=8, min=1, max=100), io.Float.Input("cfg", default=1., min=1., max=1.),
                io.Combo.Input("backend", options=["hook", "native"]),
                io.Combo.Input("image_scope", options=["target_mask", "all"]),
                io.Combo.Input("text_scope", options=["none", "target_phrase", "all"]),
                io.Int.Input("trial_id", default=0, min=0, max=0x7FFFFFFF),
                io.Combo.Input("diagnostic_level", options=["summary", "audit"], optional=True, default="summary")],
            outputs=[io.Latent.Output("latent"), MasksType.Output("mask_bank"), DiagnosticsType.Output("diagnostics")])

    @classmethod
    def fingerprint_inputs(cls, lora_name, **kwargs):
        return Krea2SliderFuseSampler.fingerprint_inputs(lora_name=lora_name, **kwargs)

    @classmethod
    def execute(cls, model, positive, negative, prompt_info, subjects, latent, lora_name, strength, seed, steps,
                cfg, backend, image_scope, text_scope, trial_id=0, diagnostic_level="summary"):
        if not lora_name:
            raise ValueError("Select a Slider for diagnostic evaluation")
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return io.NodeOutput(*sample_krea2_diagnostic(model, positive, negative, prompt_info, subjects, latent, path,
            strength=strength, seed=seed, steps=steps, cfg=cfg, backend=backend,
            image_scope=image_scope, text_scope=text_scope, trial_id=trial_id, diagnostic_level=diagnostic_level))


class Krea2SliderFusePredictionMixSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Krea2SliderFusePredictionMixSampler",
            display_name="Krea2 Slider Prediction Mix Sampler", category=CATEGORY + "/Diagnostics",
            description="Experimental. Select native Slider predictions inside a manual mask and base predictions outside. Partial masks require two serial model evaluations per step and model switching. Protected pixels are not frozen. Change trial_id to remeasure.",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
                PromptType.Input("prompt_info"), SubjectsType.Input("subjects"), io.Latent.Input("latent"),
                io.Combo.Input("lora_name", options=[""] + folder_paths.get_filename_list("loras")),
                io.Float.Input("strength", default=4., min=-10., max=10., step=.05),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("steps", default=8, min=1, max=100), io.Float.Input("cfg", default=1., min=1., max=1.),
                io.Combo.Input("mix_scope", options=["target_mask", "none", "all"]),
                io.Int.Input("trial_id", default=0, min=0, max=0x7FFFFFFF),
                io.Combo.Input("diagnostic_level", options=["audit", "summary"], default="audit")],
            outputs=[io.Latent.Output("latent"), MasksType.Output("mask_bank"), DiagnosticsType.Output("diagnostics")])

    @classmethod
    def fingerprint_inputs(cls, lora_name, **kwargs):
        return Krea2SliderFuseSampler.fingerprint_inputs(lora_name=lora_name, **kwargs)

    @classmethod
    def execute(cls, model, positive, negative, prompt_info, subjects, latent, lora_name, strength, seed,
                steps, cfg, mix_scope, trial_id=0, diagnostic_level="audit"):
        if not lora_name:
            raise ValueError("Select a Slider for prediction mixing")
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        return io.NodeOutput(*sample_krea2_prediction_mix(model, positive, negative, prompt_info, subjects,
            latent, path, strength=strength, seed=seed, steps=steps, cfg=cfg, mix_scope=mix_scope,
            trial_id=trial_id, diagnostic_level=diagnostic_level))


class Krea2SliderFuseDiagnosticSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Krea2SliderFuseDiagnosticSave", display_name="Save Krea2 Slider Diagnostic Result",
            category=CATEGORY + "/Diagnostics", is_output_node=True,
            description="Save correlated image, effective mask, first prediction, final latent and JSON report. Connect one diagnostic Sampler and its VAEDecode.",
            inputs=[io.Image.Input("images"), io.Latent.Input("latent"), DiagnosticsType.Input("diagnostics"),
                    io.String.Input("filename_prefix", default="krea2_slider_diag")], outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id])

    @classmethod
    def execute(cls, images, latent, diagnostics, filename_prefix):
        validate_prefix(filename_prefix)
        output_dir = Path(folder_paths.get_output_directory()).resolve()
        folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, str(output_dir), images.shape[2], images.shape[1])
        if not Path(folder).resolve().is_relative_to(output_dir):
            raise ValueError("Diagnostic output must stay within ComfyUI output directory")
        basename = f"{filename}_s{diagnostics.report['seed']}_{counter:05}_"
        report = save_diagnostic_artifacts(folder, basename, latent["samples"], images, diagnostics,
            cls.hidden.prompt, cls.hidden.extra_pnginfo, cls.hidden.unique_id)
        return io.NodeOutput(ui={"images": [{"filename": report["artifacts"]["image"],
                                            "subfolder": subfolder, "type": "output"}]})
