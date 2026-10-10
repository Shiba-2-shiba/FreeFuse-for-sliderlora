"""Bounded, observation-only Krea2 mask experiments; no completed reference image.

This module deliberately does not alter the existing sampler/diagnostic contract.
CPU tests cover the lifecycle boundary, not native Krea2 or image-quality validation.
"""
from __future__ import annotations

import copy
import json
import time
import uuid

import torch

from .conditioning import resolve_phrase
from .diagnostics import implementation_info
from .experimental_masks import ExperimentalMaskAccumulator, VARIANTS, validate_experimental_config
from .lora import core_guard
from .masks import patch_grid, _validate_partition
from .native_pair import _cleanup, validate_mix_options
from .sampling import tensor_hash, validate_run_inputs
from .version import __version__

EXPERIMENT_SCHEMA_VERSION = 1
COLLECTION_KEY = "krea2_slider_experimental_mask_collection"


def _config_from_json(value):
    if not isinstance(value, str):
        raise ValueError("config_json must be a JSON object string")
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key}")
            result[key] = item
        return result
    try:
        parsed = json.loads(value, object_pairs_hook=pairs)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid experiment config JSON: {error}") from error
    return validate_experimental_config(parsed)


class ExperimentalAttentionCollector:
    """Observe selected block inputs without replacing any model operation."""
    def __init__(self, accumulator, config, grid, text_len, sigmas, audit_tensors=False):
        self.accumulator = accumulator
        self.audit_tensors = audit_tensors
        self.config, self.grid, self.text_len = config, tuple(grid), text_len
        self.sigmas = sigmas.detach().float().cpu()
        self.step = None
        self._handles = []
        self.manifest = []
        self.algorithm_seconds = 0.
        self.observed = set()

    def begin_forward(self, sigma, step):
        values = sigma.detach().float().cpu().reshape(-1)
        if values.numel() != 1 or not torch.isclose(values[0], self.sigmas[step - 1], rtol=1e-5, atol=1e-6):
            raise ValueError("Experimental observation sigma does not match the schedule prefix")
        self.step = step

    def install(self, core):
        if self._handles:
            raise RuntimeError("Experimental collector is already installed")
        def check_text(_module, _args, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 3 or output.shape[:2] != (1, self.text_len):
                raise ValueError("Experimental txtfusion token count differs from the encoded prompt")
        self._handles.append(core.txtfusion.register_forward_hook(check_text))
        for index in self.config["selected_blocks"]:
            def observe(block, args, kwargs, index=index):
                return self._observe(index, block, args, kwargs)
            self._handles.append(core.blocks[index].register_forward_pre_hook(observe, with_kwargs=True))

    def _observe(self, index, block, args, kwargs):
        if self.step not in self.config["selected_steps"]:
            return None
        key = (self.step, index)
        if key in self.observed:
            raise ValueError("Experimental collection saw a repeated step/block tap")
        if len(args) < 3:
            raise ValueError("Unsupported native Krea2 block signature")
        from comfy.ldm.flux.math import apply_rope
        from comfy.ldm.modules.attention import optimized_attention_masked
        x, vec, freqs = args[:3]
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[:2] != (1, self.text_len + self.grid[0] * self.grid[1]):
            raise ValueError("Experimental image tokens differ from the patch grid")
        mask = kwargs.get("mask", args[3] if len(args) > 3 else None)
        options = kwargs.get("transformer_options", args[4] if len(args) > 4 else {}) or {}
        before = tensor_hash(x) if self.audit_tensors else None
        scale, shift, _gate, *_rest = block.mod(vec)
        value = (1 + scale) * block.prenorm(x) + shift
        attn = block.attn
        if attn.heads < 1 or attn.kvheads < 1 or attn.heads % attn.kvheads:
            raise ValueError("Unsupported Krea2 grouped-query attention head ratio")
        q = attn.wq(value).reshape(1, value.shape[1], attn.heads, -1).transpose(1, 2)
        k = attn.wk(value).reshape(1, value.shape[1], attn.kvheads, -1).transpose(1, 2)
        v = attn.wv(value).reshape(1, value.shape[1], attn.kvheads, -1).transpose(1, 2)
        q, k = attn.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)
        ratio = attn.heads // attn.kvheads
        if ratio != 1:
            k = k.repeat_interleave(ratio, dim=1)
            v = v.repeat_interleave(ratio, dim=1)
        output = optimized_attention_masked(q, k, v, attn.heads, mask=mask, skip_reshape=True,
            preferred_attention=getattr(attn, "comfy_attention", None), transformer_options=options)
        observation = {"img_q": q.transpose(1, 2)[:, self.text_len:],
            "txt_k": k.transpose(1, 2)[:, :self.text_len],
            "img_k": k.transpose(1, 2)[:, self.text_len:],
            "attention_output": output[:, self.text_len:], "attention_input": value[:, self.text_len:]}
        hashes_before = {name: tensor_hash(tensor) for name, tensor in observation.items()} if self.audit_tensors else None
        mask_hash = tensor_hash(mask) if self.audit_tensors and isinstance(mask, torch.Tensor) else None
        started = time.perf_counter()
        try:
            self.accumulator.observe(step=self.step, block=index, **observation, attention_mask=mask)
        finally:
            self.algorithm_seconds += time.perf_counter() - started
        hashes_after = {name: tensor_hash(tensor) for name, tensor in observation.items()} if self.audit_tensors else None
        after = tensor_hash(x) if self.audit_tensors else None
        if before != after or hashes_before != hashes_after or (mask_hash is not None and mask_hash != tensor_hash(mask)):
            raise RuntimeError("Experimental observation mutated an input or attention tensor")
        self.observed.add(key)
        self.manifest.append({"step": self.step, "block": index,
            "sigma": float(self.sigmas[self.step - 1]), "text_tokens": self.text_len,
            "image_tokens": self.grid[0] * self.grid[1], "query_heads": attn.heads,
            "key_value_heads": attn.kvheads, "gqa_expanded_heads": int(k.shape[1]),
            "tensor_audit_enabled": self.audit_tensors,
            "attention_mask_shape": list(mask.shape) if isinstance(mask, torch.Tensor) else None})
        if self.audit_tensors:
            self.manifest[-1].update({"block_input_before_sha256": before, "block_input_after_sha256": after,
                "tensor_sha256_before": hashes_before, "tensor_sha256_after": hashes_after,
                "attention_mask_sha256": mask_hash})
        return None

    def remove(self):
        handles, self._handles = self._handles, []
        errors = []
        for handle in reversed(handles):
            try:
                handle.remove()
            except Exception as error:
                self._handles.append(handle)
                errors.append(error)
        self.step = None
        if errors:
            raise errors[0]


def _memory_start(device):
    device = torch.device(device)
    report = {"device": str(device), "scope": "shared seven-variant collection including resident model allocations, model loading and observation; not per-variant cost or baseline overhead",
              "peak_allocated_bytes": None, "peak_reserved_bytes": None, "unavailable_reason": None}
    if device.type != "cuda" or not torch.cuda.is_available():
        report["unavailable_reason"] = "CUDA memory metrics unavailable on this execution device"
        return report
    try:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    except Exception as error:
        report["unavailable_reason"] = f"CUDA peak reset unavailable: {type(error).__name__}: {error}"
    return report


def _memory_finish(report):
    if report["unavailable_reason"] is None:
        try:
            device = torch.device(report["device"])
            torch.cuda.synchronize(device)
            report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            report["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        except Exception as error:
            report["peak_allocated_bytes"] = report["peak_reserved_bytes"] = None
            report["unavailable_reason"] = f"CUDA peak measurement unavailable: {type(error).__name__}: {error}"


def collect_experimental_masks(model, positive, negative, prompt_info, subjects, latent, *, seed=42, steps=8,
                               trial_id=0, config_json="{}", background_phrase="", background_occurrence=0, audit_tensors=False):
    """Run exactly the first two evaluations of the original eight-step schedule."""
    import comfy.model_management
    import comfy.patcher_extension
    import comfy.sample
    import comfy.samplers

    if not isinstance(audit_tensors, bool):
        raise ValueError("audit_tensors must be a boolean")
    config = _config_from_json(config_json)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps != 8 or config["collect_step"] != 2:
        raise ValueError("This bounded experiment requires steps=8 and collect_step=2; it uses only sigmas[:3]")
    if isinstance(trial_id, bool) or not isinstance(trial_id, int) or trial_id < 0:
        raise ValueError("trial_id must be a nonnegative integer")
    core = validate_run_inputs(model, positive, prompt_info, subjects, latent, strength=0., seed=seed,
        steps=steps, cfg=1., mask_mode="auto", collect_step=2, collect_block=config["collect_block"],
        top_k_ratio=config["top_k_ratio"], temperature=config["temperature"])
    validate_mix_options(model)
    if any(metadata.get("hooks") or metadata.get("control") for _, metadata in positive + negative):
        raise ValueError("Conditioning hooks/ControlNet are unsupported for experimental collection")
    if any(block >= len(core.blocks) for block in config["selected_blocks"]):
        raise ValueError("selected_blocks contains a block outside the Krea2 model")
    if not isinstance(background_phrase, str) or not background_phrase.strip():
        raise ValueError("Provide a nonempty background_phrase already present in the prompt")
    background = resolve_phrase(prompt_info, background_phrase, background_occurrence)
    positions = {s.id: list(s.positions) for s in subjects}
    if set(background) & set(positions["target"] + positions["protected"]):
        raise ValueError("Background phrase token positions overlap a subject phrase")
    positions["background"] = list(background)
    if len(positive) != 1 or not isinstance(positive[0][0], torch.Tensor):
        raise ValueError("Experimental collection requires a single matching positive conditioning tensor")
    metadata = positive[0][1]
    if set(metadata) - {"pooled_output", "attention_mask"}:
        raise ValueError("Unsupported positive conditioning metadata in experimental collection")
    metadata_summary = {}
    for key, value in metadata.items():
        if value is None:
            metadata_summary[key] = None
        elif isinstance(value, torch.Tensor):
            metadata_summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": tensor_hash(value)}
        else:
            raise ValueError("Unsupported positive conditioning metadata value")
    conditioning = {"positive_sha256": tensor_hash(positive[0][0]), "positive_metadata": metadata_summary,
                    "negative_used": False, "negative_sha256": None, "negative_reason": "CFG1 positive-only collection"}
    text_len = len(prompt_info.token_ids)
    started = time.perf_counter()
    report = {"experiment_schema_version": EXPERIMENT_SCHEMA_VERSION, "run_id": uuid.uuid4().hex,
        "extension_version": __version__, "implementation": implementation_info(),
        "tensor_audit_enabled": audit_tensors, "tensor_audit_scope": "optional full-tensor transfer/hash audit; disabled by default to avoid large GPU-to-CPU copies",
        "seed": seed, "steps": steps, "trial_id": trial_id,
        "sampler": "euler", "scheduler": "simple", "cfg": 1., "normalized_config": config,
        "phase1_nfe": 0, "phase2_nfe": 0,
        "conditioning": conditioning, "model": {"diffusion_class": f"{type(core).__module__}.{type(core).__qualname__}",
            "weights_identity_verified": False, "weights_identity_reason": "Full model weights are not hashed; map hashes do not identify the checkpoint or style LoRAs",
            "quant_formats": sorted({str(getattr(module, "quant_format", None)) for module in core.modules()})}, "completed_reference_image_generated": False,
        "slider_loaded": False, "image_quality_validated": False, "native_runtime_validated": False,
        "prompt": prompt_info.prompt, "background_phrase": background_phrase,
        "background_occurrence": background_occurrence, "token_positions": positions,
        "forward_integrity": [], "collection_seconds": None, "algorithm_seconds": None,
        "save_seconds": None, "elapsed_seconds": None,
        "timing_scope": "shared seven-variant wall time: collection_seconds includes synchronous observation; algorithm_seconds is its measured subset plus finalization. Neither is per-variant end-to-end latency or baseline overhead"}
    with core_guard(core):
        probe = model.clone()
        collector = None
        try:
            image = comfy.sample.fix_empty_latent_channels(probe, latent["samples"],
                latent.get("downscale_ratio_spacial"), latent.get("downscale_ratio_temporal"))
            grid = patch_grid(image, core.patch)
            noise = comfy.sample.prepare_noise(image, seed, latent.get("batch_index"))
            sigmas = comfy.samplers.KSampler(probe, steps, probe.load_device, sampler="euler", scheduler="simple",
                model_options=probe.model_options).sigmas
            if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) != steps + 1
                    or not torch.isfinite(sigmas).all() or not torch.all(sigmas[:-1] > sigmas[1:])
                    or float(sigmas[-1]) != 0. or float(sigmas[2]) <= 0.):
                raise ValueError("Expected a finite descending eight-step sigma schedule with a nonterminal two-step prefix")
            used_sigmas = sigmas[:3]
            accumulator = ExperimentalMaskAccumulator(config, grid, positions)
            collector = ExperimentalAttentionCollector(accumulator, config, grid, text_len, sigmas, audit_tensors=audit_tensors)
            report.update({"grid": list(grid), "text_token_count": text_len,
                "initial_noise_sha256": tensor_hash(noise), "initial_latent_sha256": tensor_hash(image),
                "full_sigmas_sha256": tensor_hash(sigmas), "used_sigmas_sha256": tensor_hash(used_sigmas),
                "full_sigmas": sigmas.detach().float().cpu().tolist(),
                "used_sigmas": used_sigmas.detach().float().cpu().tolist()})
            report["memory"] = _memory_start(probe.load_device)

            def inject(_patcher):
                if not collector._handles:
                    collector.install(core)
            def eject(_patcher):
                collector.remove()
            def forward(apply_model, args):
                context = args["c"].get("c_crossattn")
                if (args.get("cond_or_uncond", [0]) != [0] or context is None
                        or context.shape[:2] != (1, text_len) or patch_grid(args["input"], core.patch) != grid):
                    raise ValueError("Collection requires matching positive batch1 conditioning and patch grid")
                step = report["phase1_nfe"] + 1
                if step > 2:
                    raise RuntimeError("Experimental collection attempted more than two model evaluations")
                collector.begin_forward(args["timestep"], step)
                before = tensor_hash(args["input"]) if audit_tensors else None
                report["phase1_nfe"] += 1
                output = apply_model(args["input"], args["timestep"], **args["c"])
                if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
                    raise ValueError("Experimental model returned a nonfinite or unsupported prediction")
                output_before = tensor_hash(output) if audit_tensors else None
                after = tensor_hash(args["input"]) if audit_tensors else None
                if before != after:
                    raise RuntimeError("Collection model evaluation mutated its input")
                row = {"step": step, "tensor_audit_enabled": audit_tensors,
                    "scope": "observer returns the original model output; independent native no-hook parity remains unverified"}
                if audit_tensors:
                    row.update({"input_before_sha256": before, "input_after_sha256": after,
                        "output_before_sha256": output_before, "output_after_sha256": tensor_hash(output)})
                report["forward_integrity"].append(row)
                return output
            probe.set_injections(COLLECTION_KEY, [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)])
            probe.model_options["model_function_wrapper"] = forward
            collection_started = time.perf_counter()
            with torch.inference_mode():
                comfy.sample.sample(probe, noise.clone(), steps, 1., "euler", "simple", positive, negative,
                                    image.clone(), sigmas=used_sigmas.clone(), seed=seed)
            report["collection_seconds"] = time.perf_counter() - collection_started
            expected = {(step, block) for step in config["selected_steps"] for block in config["selected_blocks"]}
            if report["phase1_nfe"] != 2 or collector.observed != expected:
                raise RuntimeError("Incomplete experimental collection; sigma/token/block alignment did not match every tap")
            if tensor_hash(noise) != report["initial_noise_sha256"] or tensor_hash(image) != report["initial_latent_sha256"]:
                raise RuntimeError("Experimental collection modified its initial noise or latent")
            algorithm_started = time.perf_counter()
            with torch.inference_mode():
                banks, algorithm_report = accumulator.finalize()
            report["algorithm_seconds"] = collector.algorithm_seconds + time.perf_counter() - algorithm_started
            report["tap_manifest"] = collector.manifest
            report["algorithm"] = algorithm_report
            _memory_finish(report["memory"])
            report["elapsed_seconds"] = time.perf_counter() - started
            suite = {"experiment_schema_version": EXPERIMENT_SCHEMA_VERSION, "banks": banks, "report": report,
                     "evidence_maps": getattr(accumulator, "evidence_maps", {})}
        finally:
            _cleanup(lambda: comfy.model_management.unload_model_and_clones(probe, unload_additional_models=False),
                     lambda: collector.remove() if collector is not None else None)
    report["owned_hooks_removed"] = True
    report["elapsed_seconds"] = time.perf_counter() - started
    return suite


def select_experimental_mask(suite, variant):
    if not isinstance(suite, dict) or suite.get("experiment_schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError("Unsupported experimental mask suite")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown experimental mask variant: {variant}")
    status = suite.get("report", {}).get("algorithm", {}).get("variants", {}).get(variant, {})
    if status.get("status") != "ok" or variant not in suite.get("banks", {}):
        raise ValueError(f"Variant {variant} failed: {status.get('reason', status.get('error', 'candidate unavailable'))}")
    _validate_partition(suite["banks"][variant])
    bank = copy.deepcopy(suite["banks"][variant])
    return bank, {"variant": variant, "candidate": copy.deepcopy(status), "experiment": copy.deepcopy(suite["report"])}
