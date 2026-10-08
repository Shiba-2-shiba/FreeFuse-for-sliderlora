"""Measure standard ComfyUI LoRA vs full-sequence hook on the actual model."""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
from pathlib import Path
import sys

from comfy_environment import bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--style-lora")
    parser.add_argument("--style-strength", type=float, default=.8)
    parser.add_argument("--strength", type=float, default=4.)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", help="Write JSON measurements")
    args = parser.parse_args()
    if args.repeats < 2: parser.error("--repeats must be at least 2")
    report = {"status": "failed", "probes": []}
    model = None
    native = None
    try:
        revision = bootstrap(args.comfy_root, cpu=args.cpu)
        import math
        import torch
        import comfy.sd
        import comfy.model_management
        from comfy.ldm.krea2.model import SingleStreamDiT
        from safetensors.torch import load_file
        from slider_fuse.diagnostics import file_hash, environment_info, tensor_metrics
        from slider_fuse.lora import load_adapters, RoutingState, SliderHook, core_guard, compare_probe_outputs
        from slider_fuse.sampling import apply_native_slider, tensor_hash

        if not math.isfinite(args.strength) or args.strength == 0 or not math.isfinite(args.style_strength):
            raise ValueError("Use finite, nonzero Slider strength and finite style strength")
        model = comfy.sd.load_diffusion_model(args.model)
        if model is None or not isinstance(model.model.diffusion_model, SingleStreamDiT):
            raise ValueError("Expected native Krea2 model")
        if args.style_lora and args.style_strength:
            model, _ = comfy.sd.load_lora_for_models(model, None, load_file(args.style_lora, device="cpu"), args.style_strength, 0.)
        core = model.model.diffusion_model
        source = load_file(args.lora, device="cpu")
        adapters = load_adapters(source, core)
        report.update(environment=environment_info(), comfy_revision=revision,
            asset_sha256={"model": file_hash(args.model), "lora": file_hash(args.lora),
                          "style": file_hash(args.style_lora) if args.style_lora else None},
            strength=args.strength, style_strength=args.style_strength if args.style_lora else 0.,
            matched_modules=len(adapters), lora_keys={name: {"rank": a.down.shape[0], "alpha": a.alpha,
                "alpha_over_rank": a.scale} for name, a in adapters.items()})

        def fingerprint(module):
            weight = module.weight
            result = {"data": tensor_hash(getattr(weight, "_qdata", weight))}
            params = getattr(weight, "_params", None)
            if params is not None:
                keys = [f.name for f in fields(params)] if is_dataclass(params) else vars(params)
                for key in keys:
                    value = getattr(params, key)
                    result[key] = tensor_hash(value) if isinstance(value, torch.Tensor) else str(value)
            return result

        selected = {}
        for name in sorted(adapters, key=lambda n: (int(n.split(".")[1]), n)):
            selected.setdefault(name.rsplit(".", 1)[-1], name)
        with core_guard(core), torch.inference_mode():
            for name in selected.values():
                comfy.model_management.load_models_gpu([model])
                module = core.get_submodule(name); adapter = adapters[name]
                if getattr(module, "quant_format", None) not in (None, "int8_tensorwise"):
                    raise ValueError("Unsupported native quantization")
                x = torch.randn(1, 6, adapter.down.shape[1], generator=torch.Generator().manual_seed(42))
                x = x.to(device=model.load_device, dtype=model.model.get_dtype())
                before = fingerprint(module)
                base = [module(x).detach().cpu() for _ in range(args.repeats)]
                state = RoutingState(phase="route", cap_len=2, mask=torch.tensor([[[1., 0.], [0., 1.]]]))
                hook = SliderHook(module, adapter, args.strength, state, name)
                try:
                    hook.inject(); routed = module(x).detach().cpu()
                finally: hook.eject()
                formula = compare_probe_outputs(base[0], base[1], routed,
                    (adapter.delta(x[:, 2:]) * args.strength).detach().cpu(), state.mask.reshape(1, 4, 1), 2)
                if not formula["passed"]: raise RuntimeError(f"Independent routed formula probe failed: {name}")
                state = RoutingState(phase="route", cap_len=2, mask=torch.ones(1, 2, 2), image_scope="all", text_scope="all")
                state.configure_target_text(1., (0,), (1,), 2)
                hook = SliderHook(module, adapter, args.strength, state, name)
                try:
                    hook.inject(); hooked = [module(x).detach().cpu() for _ in range(args.repeats)]
                finally: hook.eject()
                comfy.model_management.unload_model_and_clones(model, unload_additional_models=False)
                native = apply_native_slider(model, source, args.strength, adapters)
                try:
                    comfy.model_management.load_models_gpu([native])
                    standard = [module(x).detach().cpu() for _ in range(args.repeats)]
                finally:
                    comfy.model_management.unload_model_and_clones(native, unload_additional_models=False)
                comfy.model_management.load_models_gpu([model])
                restored = module(x).detach().cpu()
                after = fingerprint(module)
                # Reverse order exercises native -> hook -> native on the same input.
                hook = SliderHook(module, adapter, args.strength, state, name)
                try:
                    hook.inject(); reversed_hook = module(x).detach().cpu()
                finally: hook.eject()
                comfy.model_management.unload_model_and_clones(model, unload_additional_models=False)
                try:
                    comfy.model_management.load_models_gpu([native])
                    reversed_native = module(x).detach().cpu()
                finally:
                    comfy.model_management.unload_model_and_clones(native, unload_additional_models=False)
                entry = {"module": name, "quant_format": getattr(module, "quant_format", None),
                    "input_sha256": tensor_hash(x), "dtype": str(x.dtype), "formula_probe": formula,
                    "native_vs_hook": tensor_metrics(standard[0], hooked[0]),
                    "base_repeat": tensor_metrics(base[0], base[1]),
                    "native_repeat": tensor_metrics(standard[0], standard[1]),
                    "hook_repeat": tensor_metrics(hooked[0], hooked[1]),
                    "restored_base": tensor_metrics(base[0], restored), "packed_weight_unchanged": before == after,
                    "reverse_hook": tensor_metrics(hooked[0], reversed_hook),
                    "reverse_native": tensor_metrics(standard[0], reversed_native)}
                report["probes"].append(entry)
                if before != after: raise RuntimeError(f"Packed base weights were not restored: {name}")
                adapter.clear()
        report["status"] = "measured_only"
        report["note"] = "Independent formula probe passed; standard/hook measurements are not a parity or image-quality pass. Use diagnostic Sampler first predictions as well."
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        if model is not None:
            import comfy.model_management
            comfy.model_management.unload_model_and_clones(model, unload_additional_models=False)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output: Path(args.output).write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0 if report["status"] == "measured_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
