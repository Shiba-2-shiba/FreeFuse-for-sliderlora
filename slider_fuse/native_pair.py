"""Diagnostic native/base prediction selection in one ComfyUI sampler loop.

SPDX-License-Identifier: Apache-2.0
ComfyUI imports are lazy; ordinary ModelPatcher clones share their weight core.
"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path
import sys
import time
import uuid

import torch

from .diagnostics import (DiagnosticPayload, DiagnosticRecorder, environment_info, file_hash,
                          implementation_info, region_measurement, serialize_measurement)
from .lora import core_guard, load_adapters, RoutingState
from .masks import manual_masks, patch_grid, postprocess_masks
from .prediction_mixing import mix_predictions, prediction_mask
from .sampling import apply_native_slider, tensor_hash, validate_run_inputs


def _cleanup(*callbacks):
    """Try every owned cleanup without hiding an active generation exception."""
    primary = sys.exc_info()[1]
    errors = []
    for callback in callbacks:
        try:
            callback()
        except Exception as error:
            errors.append(error)
            logging.exception("Prediction mix cleanup failed")
    if errors and primary is None:
        raise errors[0]


class NativePairRunner:
    def __init__(self, base_patcher, slider_patcher):
        if slider_patcher is not None and (base_patcher.model is not slider_patcher.model
                or base_patcher.load_device != slider_patcher.load_device):
            raise ValueError("Native pair requires same-core, same-device patcher clones")
        self.patchers = {"base": base_patcher, "slider": slider_patcher}
        self.active = None
        self.calls = {"base": 0, "slider": 0}

    def activate(self, branch, x, conds):
        import comfy.model_management
        import comfy.sampler_helpers
        if branch not in self.patchers or self.patchers[branch] is None:
            raise ValueError("Requested native branch is unavailable")
        selected = self.patchers[branch]
        if selected is not self.active:
            self.close()
            self.active = selected  # cleanup also owns partially failed activation
            memory, minimum = comfy.sampler_helpers.estimate_memory(selected, x.shape, conds)
            comfy.model_management.load_models_gpu([selected], memory_required=memory,
                                                    minimum_memory_required=minimum)
            selected.pre_run()
        if getattr(selected.model, "current_patcher", None) is not selected:
            raise RuntimeError("ComfyUI current_patcher does not identify the selected branch")
        return selected.model

    def predict(self, branch, x, sigma, *, positive, negative, model_options, seed):
        import comfy.samplers
        model = self.activate(branch, x, {"positive": positive, "negative": negative})
        # Isolate input and output storage across shared-core load/offload operations.
        output = comfy.samplers.sampling_function(model, x.clone(), sigma.clone(), negative,
            positive, 1., model_options=model_options, seed=seed)
        self.calls[branch] += 1
        if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
            raise ValueError("Non-finite or unsupported branch prediction")
        return output.clone()

    def close(self):
        active, self.active = self.active, None
        if active is not None:
            active.cleanup()


def _check_runtime_api():
    import comfy.samplers
    import comfy.sampler_helpers
    required = {
        comfy.samplers.CFGGuider: ("sample", "inner_sample", "outer_sample", "set_conds", "set_cfg"),
        comfy.sampler_helpers: ("prepare_sampling", "cleanup_models", "estimate_memory"),
        comfy.samplers: ("sampling_function", "cast_to_load_options", "sampler_object"),
    }
    for owner, names in required.items():
        if any(not callable(getattr(owner, name, None)) for name in names):
            raise RuntimeError("Unsupported ComfyUI prediction-mix lifecycle API")
    if "latent_shapes" not in inspect.signature(comfy.samplers.CFGGuider.outer_sample).parameters:
        raise RuntimeError("Unsupported ComfyUI CFGGuider outer_sample signature")


def make_prediction_mix_guider(base, slider, token_mask, *, patch, recorder, report, mode):
    """Define the native subclass lazily so package imports never require ComfyUI."""
    import comfy.samplers
    import comfy.sampler_helpers
    import comfy.model_management
    _check_runtime_api()

    class NativePredictionMixGuider(comfy.samplers.CFGGuider):
        def __init__(self):
            super().__init__(base)
            self.runner = NativePairRunner(base, slider)
            self.pair_trace = []; self.base_trace = []; self.slider_trace = []

        def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
                         callback=None, disable_pbar=False, seed=None, latent_shapes=None):
            prepared = False
            try:
                self.inner_model, self.conds, self.loaded_models = comfy.sampler_helpers.prepare_sampling(
                    base, noise.shape, self.conds, self.model_options)
                prepared = True
                if self.loaded_models:
                    raise ValueError("Additional conditioning/models are unsupported for prediction mixing")
                device = base.load_device
                with comfy.model_management.cuda_device_context(device):
                    noise = noise.to(device=device, dtype=torch.float32)
                    latent_image = latent_image.to(device=device, dtype=torch.float32)
                    sigmas = sigmas.to(device)
                    comfy.samplers.cast_to_load_options(self.model_options, device=device, dtype=base.model_dtype())
                    self.runner.activate("slider" if mode == "slider" else "base", noise, self.conds)
                    return self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask,
                                             callback, disable_pbar, seed, latent_shapes=latent_shapes)
            finally:
                callbacks = [self.runner.close]
                if prepared:
                    callbacks.append(lambda: comfy.sampler_helpers.cleanup_models(self.conds, self.loaded_models))
                _cleanup(*callbacks)

        def predict_noise(self, x, timestep, model_options=None, seed=None):
            index = report["sampler_nfe"]
            recorder.begin_step(index, timestep)
            positive = self.conds.get("positive")
            negative = self.conds.get("negative")
            if not positive or len(positive) != 1:
                raise ValueError("Prediction mixing requires one positive conditioning entry")
            context = positive[0]["model_conds"]["c_crossattn"].cond
            if context.shape[0] != 1 or context.shape[1] != report["text_token_count"]:
                raise ValueError("Runtime conditioning text boundary differs from prompt_info")
            if patch_grid(x, patch) != tuple(report["grid"]):
                raise ValueError("Runtime prediction grid differs from mask grid")
            # Base/native patcher options are identical: only standard weight patches differ.
            kwargs = dict(positive=positive, negative=negative, model_options=model_options or {}, seed=seed)
            p_base = self.runner.predict("base", x, timestep, **kwargs) if mode != "slider" else None
            p_slider = self.runner.predict("slider", x, timestep, **kwargs) if mode != "base" else None
            if mode == "both":
                mask = prediction_mask(token_mask, p_base, patch=patch)
                prediction = mix_predictions(p_base, p_slider, mask)
                if recorder.level == "audit":
                    exclusion = (~mask).expand_as(prediction)
                    inclusion = mask.expand_as(prediction)
                    self.pair_trace.append({"eval_index": index,
                        "base_excluded": region_measurement(p_base, prediction, exclusion),
                        "slider_selected": region_measurement(p_slider, prediction, inclusion)})
                    self.base_trace.append(p_base.detach().cpu().clone())
                    self.slider_trace.append(p_slider.detach().cpu().clone())
            else:
                prediction = p_base if mode == "base" else p_slider
            recorder.record_inputs(x, context, timestep)
            recorder.record_prediction(prediction)
            recorder.record_step(prediction, x, timestep)
            report["sampler_nfe"] += 1
            return prediction

        def finish_report(self):
            report["branch_nfe"] = dict(self.runner.calls)
            report["phase2_nfe"] = report["sampler_nfe"]
            audit, tensors = recorder.finalize()
            audit["linear_audit"] = None
            report.update(audit)
            report["linear_audit_unavailable_reason"] = "native_branch_not_instrumented"
            report["prediction_mix_audit"] = []
            if self.pair_trace:
                values = torch.stack([r[k] for r in self.pair_trace
                                      for k in ("base_excluded", "slider_selected")]).cpu().double().tolist()
                for i, row in enumerate(self.pair_trace):
                    report["prediction_mix_audit"].append({"eval_index": row["eval_index"],
                        "base_excluded": serialize_measurement(values[2*i]),
                        "slider_selected": serialize_measurement(values[2*i+1])})
                tensors["trace_base_predictions"] = torch.stack(self.base_trace)
                tensors["trace_slider_predictions"] = torch.stack(self.slider_trace)
            else:
                report["prediction_mix_audit_unavailable_reason"] = (
                    "summary_level" if recorder.level == "summary" else "single_branch_endpoint")
            return tensors

    return NativePredictionMixGuider()


def sample_krea2_prediction_mix(model, positive, negative, prompt_info, subjects, latent, lora_path, *,
                               strength, seed, steps=8, cfg=1., mix_scope="target_mask", trial_id=0,
                               diagnostic_level="audit"):
    import comfy.sample
    import comfy.samplers
    import comfy.model_management
    from safetensors.torch import load_file
    if mix_scope not in ("none", "target_mask", "all"):
        raise ValueError("mix_scope must be none, target_mask or all")
    if isinstance(trial_id, bool) or not isinstance(trial_id, int) or trial_id < 0:
        raise ValueError("trial_id must be a nonnegative integer")
    recorder = DiagnosticRecorder(set(), level=diagnostic_level)
    core = validate_run_inputs(model, positive, prompt_info, subjects, latent, strength=strength, seed=seed,
        steps=steps, cfg=cfg, mask_mode="manual", collect_step=1, collect_block=0,
        top_k_ratio=.3, temperature=4000.)
    unsupported = ("sampler_cfg_function", "sampler_post_cfg_function", "sampler_pre_cfg_function",
                   "disable_cfg1_optimization")
    if any(model.model_options.get(k) for k in unsupported):
        raise ValueError("Custom CFG processing is unsupported for prediction mixing")
    if any(callbacks for group in getattr(model, "callbacks", {}).values() for callbacks in group.values()):
        raise ValueError("Foreign model callbacks are unsupported for prediction mixing")
    if any(metadata.get("hooks") or metadata.get("control") for _, metadata in positive + negative):
        raise ValueError("Conditioning hooks/ControlNet are unsupported for prediction mixing")
    state = RoutingState()
    state.configure_target_text(0., subjects[0].positions, subjects[1].positions, len(prompt_info.token_ids))
    mode = "base" if not strength or mix_scope == "none" else "slider" if mix_scope == "all" else "both"
    started = time.perf_counter()
    with core_guard(core):
        base = model.clone(); slider = None
        adapters = {}
        try:
            source = load_file(lora_path, device="cpu")
            adapters = load_adapters(source, core)
            modules = {name: core.get_submodule(name) for name in adapters}
            formats = {getattr(m, "quant_format", None) for m in modules.values()}
            if not formats <= {None, "int8_tensorwise"}:
                raise ValueError("Unsupported base quantization for prediction mixing")
            if mode != "base":
                slider = apply_native_slider(base, source, strength, modules)
            image = comfy.sample.fix_empty_latent_channels(base, latent["samples"],
                latent.get("downscale_ratio_spacial"), latent.get("downscale_ratio_temporal"))
            grid = patch_grid(image, core.patch)
            bank = postprocess_masks(manual_masks(subjects[0].manual_mask, subjects[1].manual_mask, grid))
            effective = (torch.zeros_like(bank["masks"]["target"]) if mix_scope == "none" else
                         torch.ones_like(bank["masks"]["target"]) if mix_scope == "all" else
                         bank["masks"]["target"].clone())
            bank["effective_image_mask"] = effective
            noise = comfy.sample.prepare_noise(image, seed, latent.get("batch_index"))
            sigmas = comfy.samplers.KSampler(base, steps, base.load_device, sampler="euler",
                scheduler="simple", model_options=base.model_options).sigmas
            report = {"run_id": uuid.uuid4().hex[:12], "extension_version": "0.1.4",
                "backend": "prediction_mix", "mix_scope": mix_scope, "model_text_scope": "all",
                "image_scope": mix_scope, "text_scope": "all", "mask_mode": "manual",
                "strength": strength, "effective_slider_strength": strength if mode != "base" else 0.,
                "seed": seed, "steps": steps, "cfg": cfg, "trial_id": trial_id,
                "sampler": "euler", "scheduler": "simple", "phase1_nfe": 0, "phase2_nfe": 0,
                "sampler_nfe": 0, "diagnostic_schema_version": 2, "prediction_space": "comfy_cfg1_denoised",
                "lora_file_name": Path(lora_path).name, "lora_sha256": file_hash(lora_path),
                "matched_modules": len(adapters), "reached_modules": None, "adapter_stats": None,
                "base_quant_formats": sorted(map(str, formats)), "prompt": prompt_info.prompt,
                "grid": list(grid), "patch_size": core.patch, "text_token_count": len(prompt_info.token_ids),
                "token_positions": {s.id: list(s.positions) for s in subjects},
                "reference_partition_sha256": {k: tensor_hash(v) for k, v in bank["masks"].items()},
                "effective_image_mask_sha256": tensor_hash(effective),
                "effective_image_mask_coverage": float(effective.mean()),
                "initial_noise_sha256": tensor_hash(noise), "initial_latent_sha256": tensor_hash(image),
                "full_sigmas_sha256": tensor_hash(sigmas), "masks": bank["diagnostics"],
                "output_routing_policy": "native_inside_base_outside_same_input",
                "skipped_branch_reason": None if mode == "both" else
                    "zero_strength" if not strength else "single_branch_endpoint",
                "environment": environment_info(), "int8_real_machine_validated": False,
                "image_quality_validated": False, **implementation_info()}
            guider = make_prediction_mix_guider(base, slider, effective, patch=core.patch,
                                                recorder=recorder, report=report, mode=mode)
            guider.set_conds(positive, negative); guider.set_cfg(1.)
            with torch.inference_mode():
                samples = guider.sample(noise, image, comfy.samplers.sampler_object("euler"), sigmas, seed=seed)
            extra = guider.finish_report()
            if report["sampler_nfe"] != steps:
                raise RuntimeError("Prediction mix did not execute the expected Euler steps")
            report.update(recorder.first_inputs)
            report.update(first_prediction_sha256=tensor_hash(recorder.first_prediction),
                          final_latent_sha256=tensor_hash(samples), final_latent_shape=list(samples.shape),
                          final_latent_dtype=str(samples.dtype))
            result = dict(latent, samples=samples)
            result.pop("downscale_ratio_spacial", None); result.pop("downscale_ratio_temporal", None)
        finally:
            _cleanup(lambda: comfy.model_management.unload_model_and_clones(base, unload_additional_models=False),
                     *(adapter.clear for adapter in adapters.values()))
    report.update(owned_hooks_removed=True, elapsed_seconds=round(time.perf_counter() - started, 3))
    logging.info("[Krea2SliderPredictionMix] %s", report)
    return result, bank, DiagnosticPayload(report, recorder.first_prediction, effective.detach().cpu().clone(), extra)
