"""Two phases share initial noise, latent and the original sigma schedule.

SPDX-License-Identifier: Apache-2.0
Adapted from FreeFuse-for-anima anima_freefuse/sampling.py at
1b924b5dd1f7266fa6e5869331e67a2033e7ec2f.
Modified for FreeFuse-for-sliderlora: Krea2-only sampling, sigma-based collection,
one target adapter, shared-core guards, diagnostics and stricter cleanup.
See LICENSE, NOTICE and THIRD_PARTY_NOTICES.md for licensing and attribution.
"""
from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
import time
import uuid

import torch

from .attention import AttentionCollector
from .lora import RoutingState, SliderHook, core_guard, load_adapters, validate_target_text_scale
from .masks import generate_masks, manual_masks, patch_grid, postprocess_masks, validate_postprocess_settings

INJECTION_KEY = "krea2_slider_freefuse"


def apply_native_slider(model, source, strength, modules):
    """The exact model-only API called by ComfyUI LoraLoaderModelOnly."""
    import comfy.sd
    original_counts = {key: len(value) for key, value in model.patches.items()}
    patched, _ = comfy.sd.load_lora_for_models(model, None, source, strength, 0.)
    missing = [name for name in modules if len(patched.patches.get("diffusion_model." + name + ".weight", ()))
               != original_counts.get("diffusion_model." + name + ".weight", 0) + 1]
    if missing:
        raise ValueError(f"Standard Loader did not map all Slider modules: {missing}")
    return patched


def sample_krea2_diagnostic(model, positive, negative, prompt_info, subjects, latent, lora_path, *,
                           backend, image_scope, text_scope, strength, seed, trial_id=0, steps=8, cfg=1.,
                           diagnostic_level="summary"):
    from .diagnostics import DiagnosticRecorder
    RoutingState(image_scope=image_scope, text_scope=text_scope)
    if backend not in ("native", "hook"):
        raise ValueError("Diagnostic backend must be native or hook")
    if backend == "native" and (image_scope != "all" or text_scope != "all"):
        raise ValueError("native backend requires all image and all text scope")
    if isinstance(trial_id, bool) or not isinstance(trial_id, int) or trial_id < 0:
        raise ValueError("trial_id must be a nonnegative integer")
    diagnostic = {"backend": backend, "image_scope": image_scope, "text_scope": text_scope,
                  "trial_id": trial_id, "recorder": DiagnosticRecorder(set(), level=diagnostic_level)}
    return sample_krea2(model, positive, negative, prompt_info, subjects, latent, lora_path,
        strength=strength, seed=seed, steps=steps, cfg=cfg, mask_mode="manual", collect_step=1,
        collect_block=0, top_k_ratio=.3, temperature=4000.,
        target_text_scale=0. if text_scope == "none" else 1., _diagnostic=diagnostic)


def validate_settings(*, steps, cfg, strength, mask_mode, collect_step, top_k_ratio, temperature,
                      fill_holes_max_area=0, mask_dilate_radius=0, target_text_scale=0.):
    validate_postprocess_settings(fill_holes_max_area, mask_dilate_radius)
    validate_target_text_scale(target_text_scale)
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be positive")
    if cfg != 1.:
        raise ValueError("Initial Krea2 Slider FreeFuse supports CFG=1 only")
    if not math.isfinite(strength) or not -10 <= strength <= 10:
        raise ValueError("Slider strength must be finite and in [-10,10]")
    if mask_mode not in ("auto", "manual"):
        raise ValueError("mask_mode must be auto or manual")
    if mask_mode == "manual" and (fill_holes_max_area or mask_dilate_radius):
        raise ValueError("Mask postprocessing is auto-only; use both settings 0 for manual masks")
    if mask_mode == "auto" and (not isinstance(collect_step, int) or not 1 <= collect_step <= steps):
        raise ValueError("collect_step must be in 1..total steps")
    if not math.isfinite(top_k_ratio) or not 0 < top_k_ratio <= 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid top_k_ratio or temperature")


def validate_model_options(options):
    if options.get("model_function_wrapper"):
        raise ValueError("Existing model_function_wrapper is unsupported; use a loader plus optional global weight LoRA")
    transformer = options.get("transformer_options", {})
    if transformer.get("optimized_attention_override") or any(transformer.get("patches", {}).values()) or transformer.get("patches_replace"):
        raise ValueError("Custom attention/token patches are unsupported in the initial Slider FreeFuse")
    if options.get("multigpu_clones"):
        raise ValueError("Multi-GPU model cloning is unsupported")


def tensor_hash(value: torch.Tensor) -> str:
    cpu = value.detach().contiguous().cpu()
    digest = hashlib.sha256(str((tuple(cpu.shape), cpu.dtype)).encode())
    digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_phases(sample, noise, latent, sigmas, collect_step, collector, state, make_masks):
    if not 1 <= collect_step < len(sigmas):
        raise ValueError("collect_step must be in 1..total steps")
    collector.active = True; state.phase = "collect"
    try:
        sample(noise.clone(), latent.clone(), sigmas[:collect_step + 1])
        if not collector.maps:
            raise RuntimeError("Krea2 observation collected no concept maps; verify sigma/token/block alignment")
        bank = make_masks(collector.maps)
        collector.active = False
        state.mask = bank["masks"]["target"]
        state.phase = "route"
        result = sample(noise.clone(), latent.clone(), sigmas)
        return result, bank
    finally:
        state.clear(); collector.reset()


def validate_run_inputs(model, positive, prompt_info, subjects, latent, *, strength, seed, steps, cfg,
                        mask_mode, collect_step, collect_block, top_k_ratio, temperature,
                        fill_holes_max_area=0, mask_dilate_radius=0, target_text_scale=0.):
    from comfy.ldm.krea2.model import SingleStreamDiT
    validate_settings(steps=steps, cfg=cfg, strength=strength, mask_mode=mask_mode, collect_step=collect_step,
                      top_k_ratio=top_k_ratio, temperature=temperature,
                      fill_holes_max_area=fill_holes_max_area, mask_dilate_radius=mask_dilate_radius,
                      target_text_scale=target_text_scale)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not prompt_info.matches(positive):
        raise ValueError("positive must connect directly to its matching Krea2 Slider Fuse Encode")
    if (len(subjects) != 2 or [s.id for s in subjects] != ["target", "protected"]
            or any(s.prompt_info is not prompt_info for s in subjects)):
        raise ValueError("Both target and protected subjects must use the matching prompt_info")
    core = model.model.diffusion_model
    if not isinstance(core, SingleStreamDiT):
        raise ValueError("Krea2 Slider FreeFuse requires a native Krea2 MODEL")
    validate_model_options(model.model_options)
    if getattr(model, "forced_hooks", None) or getattr(model, "hook_patches", None) or any(getattr(model, "wrappers", {}).values()):
        raise ValueError("Existing native hooks/diffusion wrappers are unsupported")
    if (any(getattr(model, "injections", {}).values()) or getattr(model, "is_injected", False)
            or getattr(model, "skip_injection", False) or getattr(model, "object_patches", None)
            or getattr(model, "weight_wrapper_patches", None)):
        raise ValueError("Existing model injections/object/forward patches are unsupported; connect a clean loader plus optional standard weight LoRA")
    for name, module in core.named_modules():
        if (module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
                or "forward" in module.__dict__):
            raise ValueError(f"Foreign raw hooks or per-instance forward at {name or '<core>'} are unsupported")
    if latent.get("noise_mask") is not None or set(latent) - {"samples", "batch_index", "downscale_ratio_spacial", "downscale_ratio_temporal"}:
        raise ValueError("Use an empty txt2img latent without noise masks or reference data")
    if torch.count_nonzero(latent["samples"]):
        raise ValueError("Initial Krea2 Slider FreeFuse supports empty txt2img latents only")
    if mask_mode == "manual" and any(s.manual_mask is None for s in subjects):
        raise ValueError("Manual mode requires both target and protected masks")
    if mask_mode == "auto" and any(s.manual_mask is not None for s in subjects):
        raise ValueError("Auto mode does not accept manual masks; disconnect both MASK inputs")
    if mask_mode == "auto" and (not isinstance(collect_block, int) or not 0 <= collect_block < len(core.blocks)):
        raise ValueError(f"collect_block must be in 0..{len(core.blocks)-1}")

    return core


def sample_krea2(model, positive, negative, prompt_info, subjects, latent, lora_path, *, strength, seed,
                 steps, cfg, mask_mode, collect_step, collect_block, top_k_ratio, temperature,
                 fill_holes_max_area=0, mask_dilate_radius=0, target_text_scale=0., _diagnostic=None):
    """Own the clone and reversible native injections for one complete run."""
    import comfy.model_management
    import comfy.patcher_extension
    import comfy.sample
    import comfy.samplers
    from comfy.ldm.krea2.model import SingleStreamDiT
    from safetensors.torch import load_file

    core = validate_run_inputs(model, positive, prompt_info, subjects, latent, strength=strength, seed=seed,
        steps=steps, cfg=cfg, mask_mode=mask_mode, collect_step=collect_step, collect_block=collect_block,
        top_k_ratio=top_k_ratio, temperature=temperature, fill_holes_max_area=fill_holes_max_area,
        mask_dilate_radius=mask_dilate_radius, target_text_scale=target_text_scale)

    state = RoutingState()
    state.configure_target_text(target_text_scale, subjects[0].positions, subjects[1].positions,
                                len(prompt_info.token_ids))
    recorder = _diagnostic["recorder"] if _diagnostic is not None else None
    backend = _diagnostic["backend"] if _diagnostic is not None else "hook"
    if _diagnostic is not None:
        state.image_scope = _diagnostic["image_scope"]; state.text_scope = _diagnostic["text_scope"]
    started = time.perf_counter()
    run_id = uuid.uuid4().hex[:12]
    report = {"run_id": run_id, "extension_version": "0.1.4", "lora_file_name": Path(lora_path).name,
              "mask_mode": mask_mode, "seed": seed, "steps": steps,
              "cfg": cfg, "sampler": "euler", "scheduler": "simple", "strength": strength,
              "adapter_groups": {"target": 1, "protected": 0, "background": 0},
              "text_delta_policy": "target_phrase_only" if target_text_scale else "zero",
              "target_text_scale": target_text_scale,
              "target_text_effective_strength": strength * target_text_scale,
              "target_text_positions": list(state.target_text_positions),
              "protected_text_positions": list(state.protected_text_positions),
              "other_text_direct_delta_policy": "zero", "protected_text_direct_delta_policy": "zero",
              "outside_target_direct_delta_policy": "zero",
              "int8_real_machine_validated": False, "image_quality_validated": False,
              "phase1_nfe": 0, "phase2_nfe": 0}
    with core_guard(core):
        patcher = model.clone()
        hooks = []; collector = None; adapters = {}
        try:
            adapters = load_adapters(load_file(lora_path, device="cpu"), core)
            modules = {name: core.get_submodule(name) for name in adapters}
            for name, module in modules.items():
                quant_format = getattr(module, "quant_format", None)
                if quant_format not in (None, "int8_tensorwise"):
                    raise ValueError(f"Unsupported base quantization {quant_format!r} at {name}; initial support is floating point or ConvRot/int8_tensorwise")
            report["matched_modules"] = len(adapters)
            report["base_quant_formats"] = sorted({str(getattr(m, "quant_format", None)) for m in modules.values()})
            if backend == "native":
                if strength:
                    patcher = apply_native_slider(patcher, load_file(lora_path, device="cpu"), strength, modules)
            else:
                hooks = [SliderHook(modules[name], adapter, strength, state, name) for name, adapter in adapters.items()]
            if recorder is not None and backend == "hook":
                first_block = min(int(name.split(".")[1]) for name in modules)
                recorder.selected_modules = {name for name in modules if int(name.split(".")[1]) == first_block}
                state.recorder = recorder
            image = comfy.sample.fix_empty_latent_channels(patcher, latent["samples"],
                latent.get("downscale_ratio_spacial"), latent.get("downscale_ratio_temporal"))
            grid = patch_grid(image, core.patch)
            noise = comfy.sample.prepare_noise(image, seed, latent.get("batch_index"))
            sigmas = comfy.samplers.KSampler(patcher, steps, patcher.load_device, sampler="euler", scheduler="simple",
                                             model_options=patcher.model_options).sigmas
            collector = AttentionCollector({s.id: s.positions for s in subjects}, grid,
                collect_step if mask_mode == "auto" else 1, collect_block if mask_mode == "auto" else 0,
                top_k_ratio, temperature, sigmas)
            text_len = len(prompt_info.token_ids)
            collector.cap_len = text_len
            report.update({"initial_noise_sha256": tensor_hash(noise), "initial_latent_sha256": tensor_hash(image),
                           "full_sigmas_sha256": tensor_hash(sigmas), "grid": list(grid), "text_token_count": text_len,
                           "token_positions": {s.id: list(s.positions) for s in subjects}})

            def inject(_patcher):
                if not collector._handles:
                    collector.install(core, state)
                for hook in hooks: hook.inject()

            def eject(_patcher):
                for hook in reversed(hooks): hook.eject()
                collector.remove()

            patcher.set_injections(INJECTION_KEY, [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)])

            def forward(apply_model, args):
                if args.get("cond_or_uncond", [0]) != [0]:
                    raise ValueError("Only the positive single-image CFG=1 branch is supported")
                context = args["c"].get("c_crossattn")
                if context is None or context.shape[0] != 1 or context.shape[1] != text_len:
                    raise ValueError("Runtime conditioning token count differs from prompt_info")
                if patch_grid(args["input"], core.patch) != grid:
                    raise ValueError("Runtime latent grid differs from the mask grid")
                if state.phase == "collect":
                    collector.begin_forward(args["timestep"], text_len)
                    report["phase1_nfe"] += 1
                elif state.phase == "route":
                    report["phase2_nfe"] += 1
                if recorder is not None:
                    recorder.begin_step(report["phase2_nfe"] - 1, args["timestep"])
                    recorder.record_inputs(args["input"], context, args["timestep"])
                prediction = apply_model(args["input"], args["timestep"], **args["c"])
                if recorder is not None:
                    recorder.record_prediction(prediction)
                    recorder.record_step(prediction, args["input"], args["timestep"])
                return prediction

            patcher.model_options["model_function_wrapper"] = forward

            def sample(n, x, schedule):
                return comfy.sample.sample(patcher, n, steps, cfg, "euler", "simple", positive, negative, x,
                                           sigmas=schedule, seed=seed)

            def make_masks(maps):
                return postprocess_masks(generate_masks(maps, grid), max_hole_area=fill_holes_max_area,
                                         dilate_radius=mask_dilate_radius)

            with torch.inference_mode():
                if mask_mode == "auto":
                    samples, bank = run_phases(sample, noise, image, sigmas, collect_step, collector, state,
                                               make_masks)
                else:
                    bank = postprocess_masks(manual_masks(subjects[0].manual_mask, subjects[1].manual_mask, grid))
                    state.mask = bank["masks"]["target"]; state.phase = "route"
                    state.audit_partition = bank["masks"]
                    samples = sample(noise.clone(), image.clone(), sigmas)
            report["reached_modules"] = len(state.reached)
            if backend == "hook" and set(adapters) != state.reached:
                raise RuntimeError("Not all matched Slider modules were reached; refusing to report a valid routed run")
            report["lora_linear_calls"] = state.calls
            report["target_text_linear_calls"] = state.target_text_calls
            report["observation"] = collector.observation if mask_mode == "auto" else None
            bank["masks"] = {k: v.detach().cpu() for k, v in bank["masks"].items()}
            bank["original_masks"] = {k: v.detach().cpu() for k, v in bank["original_masks"].items()}
            bank["added_target_mask"] = bank["added_target_mask"].detach().cpu()
            report["masks"] = bank["diagnostics"]
            report["raw_maps_available"] = bool(bank.get("raw_maps"))
            report["map_diagnostics"] = bank.get("map_diagnostics")
            report["mask_postprocess"] = bank["postprocess_diagnostics"]
            result = latent.copy()
            result.pop("downscale_ratio_spacial", None); result.pop("downscale_ratio_temporal", None)
            result["samples"] = samples
            if _diagnostic is not None:
                from .diagnostics import file_hash, environment_info, implementation_info
                audit_report, extra_tensors = recorder.finalize()
                extra_tensors.update({"reference_" + k + "_mask": v.detach().cpu().clone()
                                      for k, v in bank["masks"].items()})
                report.update(audit_report)
                report["matched_module_names"] = sorted(adapters)
                if backend == "native":
                    report["linear_audit"] = None
                    report["linear_audit_unavailable_reason"] = "native_branch_not_instrumented"
                report.update(implementation_info())
                report.update(diagnostic_schema_version=2, prediction_space="comfy_cfg1_denoised", patch_size=core.patch)
                effective_mask = (torch.ones_like(bank["masks"]["target"]) if state.image_scope == "all"
                                  else bank["masks"]["target"].clone())
                bank["effective_image_mask"] = effective_mask
                report.update({key: _diagnostic[key] for key in ("backend", "image_scope", "text_scope", "trial_id")})
                report.update(recorder.first_inputs)
                report.update({"case_id": f"{backend}_{state.image_scope}_{state.text_scope}_s{strength:g}",
                    "prompt": prompt_info.prompt, "lora_sha256": file_hash(lora_path),
                    "environment": environment_info(), "text_linear_calls": state.text_linear_calls,
                    "selected_image_row_count": int(effective_mask.sum()),
                    "selected_text_row_count": text_len if state.text_scope == "all" else len(subjects[0].positions) if state.text_scope == "target_phrase" else 0,
                    "protected_text_direct_delta_policy": "enabled" if state.text_scope == "all" and strength else "zero",
                    "other_text_direct_delta_policy": "enabled" if state.text_scope == "all" and strength else "zero",
                    "outside_target_direct_delta_policy": "enabled" if state.image_scope == "all" and strength else "zero",
                    "text_delta_policy": state.text_scope if strength else "zero",
                    "adapter_stats": recorder.adapter_stats if backend == "hook" else None,
                    "effective_image_mask_sha256": tensor_hash(effective_mask),
                    "effective_image_mask_coverage": float(effective_mask.mean()),
                    "reference_partition_sha256": {name: tensor_hash(mask) for name, mask in bank["masks"].items()},
                    "first_prediction_sha256": tensor_hash(recorder.first_prediction),
                    "final_latent_sha256": tensor_hash(samples),
                    "final_latent_shape": list(samples.shape), "final_latent_dtype": str(samples.dtype)})
        finally:
            try:
                comfy.model_management.unload_model_and_clones(patcher, unload_additional_models=False)
            finally:
                for hook in reversed(hooks): hook.eject()
                if collector is not None:
                    collector.remove(); collector.reset()
                state.clear()
                if backend == "native":
                    for adapter in adapters.values(): adapter.clear()
    report["owned_hooks_removed"] = all(h.original_forward is None for h in hooks) and not collector._handles
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    if torch.cuda.is_available():
        report["process_cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["process_cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    logging.info("[Krea2SliderFuse] %s", {k: v for k, v in report.items()
                 if k not in ("linear_audit", "step_trace")})
    if _diagnostic is not None:
        from .diagnostics import DiagnosticPayload
        return result, bank, DiagnosticPayload(report, recorder.first_prediction,
            effective_mask.detach().cpu().clone(), extra_tensors)
    return result, bank, report
