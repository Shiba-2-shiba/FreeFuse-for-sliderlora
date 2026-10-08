"""Opt-in measurement and portable diagnostic artifacts; no ComfyUI import."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import uuid

import torch


@dataclass
class DiagnosticPayload:
    report: dict
    first_prediction: torch.Tensor
    effective_image_mask: torch.Tensor


def file_hash(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rms(value):
    if value is None or value.numel() == 0:
        return None
    result = float(value.detach().float().square().mean().sqrt())
    if not math.isfinite(result):
        raise ValueError("Non-finite diagnostic RMS")
    return result


def environment_info():
    root = None
    comfy = sys.modules.get("comfy")
    if getattr(comfy, "__file__", None):
        root = Path(comfy.__file__).resolve().parent.parent
    elif comfy is not None:
        # comfy is a namespace package on some installations.
        paths = list(getattr(comfy, "__path__", ()))
        if len(paths) == 1:
            root = Path(paths[0]).resolve().parent
    revision = None
    if root is not None:
        try:
            revision = subprocess.check_output(["git", "-c", f"safe.directory={root.as_posix()}",
                "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return {"python": sys.version.split()[0], "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "comfy_revision": revision,
            "comfy_revision_unavailable_reason": "ComfyUI root or Git revision unavailable" if revision is None else None}


class DiagnosticRecorder:
    def __init__(self, selected_modules):
        self.selected_modules = set(selected_modules)
        self.adapter_stats = []
        self._seen = set()
        self.first_prediction = None
        self.first_inputs = {}

    def record_inputs(self, image, context, timestep):
        if self.first_inputs:
            return
        from .sampling import tensor_hash
        self.first_inputs = {"first_input_sha256": tensor_hash(image),
            "conditioning_sha256": tensor_hash(context), "first_timestep_sha256": tensor_hash(timestep),
            "first_input_shape": list(image.shape), "first_input_dtype": str(image.dtype)}

    def record_prediction(self, prediction):
        if self.first_prediction is None:
            if not isinstance(prediction, torch.Tensor) or not torch.isfinite(prediction).all():
                raise ValueError("Non-finite or unsupported first model prediction")
            self.first_prediction = prediction.detach().cpu().clone()

    def record_linear(self, name, x, base, result, image_delta, text_delta, state, adapter, strength):
        if name not in self.selected_modules or name in self._seen:
            return
        self._seen.add(name)
        cap = state.cap_len
        image_selection = state.effective_image_mask(base).squeeze(-1).bool()
        positions = (state.target_text_indices(base) if state.text_scope != "none" and state.target_text_scale
                     else torch.empty(0, device=base.device, dtype=torch.long))

        def stats(reference, output, delta, selection):
            reference = reference[selection]
            actual = output[selection] - reference
            computed = torch.zeros_like(reference) if delta is None else delta[selection]
            base_rms, delta_rms = rms(reference), rms(computed)
            return {"selected_row_count": int(selection.sum()),
                "unselected_row_count": int(selection.numel() - selection.sum()),
                "base_rms": base_rms, "computed_delta_rms": delta_rms,
                "actual_delta_rms": rms(actual),
                "delta_over_base_rms": delta_rms / base_rms if base_rms else None,
                "ratio_unavailable_reason": "no selected rows or zero base RMS" if not base_rms else None,
                "actual_delta_max_abs": float(actual.detach().float().abs().max()) if actual.numel() else None,
                "finite": bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())}

        text_selection = torch.zeros(base.shape[:2], dtype=torch.bool, device=base.device)[:, :cap]
        text_selection[:, positions] = True
        full_text_delta = torch.zeros_like(base[:, :cap])
        if text_delta is not None:
            full_text_delta.index_copy_(1, positions, text_delta)
        self.adapter_stats.append({"module": name, "rank": adapter.down.shape[0],
            "alpha": adapter.alpha, "alpha_over_rank": adapter.scale, "strength": strength,
            "input_dtype": str(x.dtype), "adapter_storage_dtype": str(adapter.down.dtype),
            "adapter_compute_dtype": str(x.dtype), "base_dtype": str(base.dtype), "output_dtype": str(result.dtype),
            "image": stats(base[:, cap:], result[:, cap:], image_delta, image_selection),
            "text": stats(base[:, :cap], result[:, :cap], full_text_delta, text_selection)})


def tensor_metrics(reference, actual, *, atol=None, rtol=None):
    if (atol is None) != (rtol is None):
        raise ValueError("Specify both atol and rtol")
    if atol is not None and (not math.isfinite(atol) or not math.isfinite(rtol) or min(atol, rtol) < 0):
        raise ValueError("Tolerances must be finite and nonnegative")
    if reference.shape != actual.shape or reference.numel() == 0:
        raise ValueError("Tensor shape mismatch or empty tensor")
    a, b = reference.detach().cpu().double(), actual.detach().cpu().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Non-finite comparison tensor")
    diff = b - a
    norm = float(a.norm())
    return {"mae": float(diff.abs().mean()), "rmse": float(diff.square().mean().sqrt()),
        "max_abs": float(diff.abs().max()), "relative_l2": float(diff.norm()) / norm if norm else None,
        "relative_l2_unavailable_reason": "zero reference norm" if not norm else None,
        "exact_equal": bool(torch.equal(reference, actual)),
        "within_declared_tolerance": bool(torch.allclose(b, a, atol=atol, rtol=rtol)) if atol is not None else None,
        "status": "measured_only" if atol is None else "within_tolerance" if torch.allclose(b, a, atol=atol, rtol=rtol) else "mismatch",
        "atol": atol, "rtol": rtol}


def linked_node(prompt, node_id, field, expected_type=None, slot=0):
    try:
        link = prompt[str(node_id)]["inputs"][field]
        if not isinstance(link, list) or len(link) != 2 or link[1] != slot:
            raise ValueError(f"Unsupported link {node_id}.{field}")
        ident = str(link[0]); node = prompt[ident]
        if expected_type is not None and node["class_type"] != expected_type:
            raise ValueError(f"Expected {expected_type} at {node_id}.{field}")
        return ident, node["inputs"]
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"Missing provenance link {node_id}.{field}") from error


def generation_config(prompt, sampler_id, vae_id, *, standard_case=None):
    """Trace only the documented evaluation graphs, never search for a loader."""
    try:
        sampler = prompt[str(sampler_id)]
        expected = "KSampler" if standard_case else "Krea2SliderFuseDiagnosticSampler"
        if sampler["class_type"] != expected:
            raise ValueError(f"Expected {expected}")
        s = sampler["inputs"]
        encode_id, encode = linked_node(prompt, sampler_id, "positive", "Krea2SliderFuseEncode")
        _, clip = linked_node(prompt, encode_id, "clip", "CLIPLoader")
        _, negative = linked_node(prompt, sampler_id, "negative", "ConditioningZeroOut")
        if negative["conditioning"] != [encode_id, 0]:
            raise ValueError("Negative must zero the matching positive")
        _, latent = linked_node(prompt, sampler_id, "latent_image" if standard_case else "latent", "EmptySD3LatentImage")
        vae_node = prompt[str(vae_id)]
        if vae_node["class_type"] != "VAELoader": raise ValueError("Expected VAELoader")
        loaders = []
        current, _ = linked_node(prompt, sampler_id, "model")
        seen = set()
        while prompt[current]["class_type"] == "LoraLoaderModelOnly":
            if current in seen: raise ValueError("Cyclic model provenance")
            seen.add(current)
            loader = prompt[current]["inputs"]
            loaders.append({"lora_name": loader["lora_name"], "strength": loader["strength_model"]})
            current, _ = linked_node(prompt, current, "model")
        unet = prompt[current]
        if unet["class_type"] != "UNETLoader": raise ValueError("Unsupported model provenance")
        slider_name = s.get("lora_name")
        slider_strength = s.get("strength")
        if standard_case:
            if standard_case not in ("standard_zero", "standard_global"):
                raise ValueError("Unknown standard case")
            if s["sampler_name"] != "euler" or s["scheduler"] != "simple" or s["denoise"] != 1.:
                raise ValueError("Unsupported standard sampler settings")
            if standard_case == "standard_global":
                if len(loaders) != 2: raise ValueError("Expected one style and one global Slider")
                slider = loaders.pop(0); slider_name = slider["lora_name"]; slider_strength = slider["strength"]
            else:
                slider_name = None; slider_strength = 0.
        else:
            subjects_id, subject = linked_node(prompt, sampler_id, "subjects", "Krea2SliderFuseSubjects")
            if subject["prompt_info"] != [encode_id, 1] or s["prompt_info"] != [encode_id, 1]:
                raise ValueError("Mismatched prompt_info provenance")
        if len(loaders) != 1: raise ValueError("Expected exactly one global style Loader")
        if slider_name is not None and any(l["lora_name"] == slider_name for l in loaders):
            raise ValueError("Duplicate Slider in upstream MODEL")
        result = {"checkpoint": unet["inputs"]["unet_name"], "weight_dtype": unet["inputs"]["weight_dtype"],
            "clip_name": clip["clip_name"], "clip_type": clip["type"], "clip_device": clip["device"],
            "vae_name": vae_node["inputs"]["vae_name"], "style_loras": list(reversed(loaders)),
            "prompt": encode["prompt"], "width": latent["width"], "height": latent["height"],
            "batch_size": latent["batch_size"], "seed": s["seed"], "steps": s["steps"], "cfg": s["cfg"],
            "sampler": "euler", "scheduler": "simple", "denoise": 1.,
            "slider_lora": slider_name, "strength": slider_strength}
        if result["clip_type"] != "krea2" or result["batch_size"] != 1:
            raise ValueError("Unsupported evaluation conditioning or batch")
        return result
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Incomplete evaluation provenance") from error


def diagnostic_provenance(prompt, save_node_id, report):
    if not isinstance(prompt, dict) or save_node_id is None:
        raise ValueError("Hidden prompt and unique_id are required for diagnostic saving")
    try:
        if prompt[str(save_node_id)]["class_type"] != "Krea2SliderFuseDiagnosticSave":
            raise ValueError("Expected diagnostic Save node")
        sampler_id, settings = linked_node(prompt, save_node_id, "latent", "Krea2SliderFuseDiagnosticSampler")
        diagnostic_id, _ = linked_node(prompt, save_node_id, "diagnostics", "Krea2SliderFuseDiagnosticSampler", slot=2)
        decode_id, _ = linked_node(prompt, save_node_id, "images", "VAEDecode")
        decoded_sampler_id, _ = linked_node(prompt, decode_id, "samples", "Krea2SliderFuseDiagnosticSampler")
        if diagnostic_id != sampler_id or decoded_sampler_id != sampler_id:
            raise ValueError("Images, latent and diagnostics must use the same Sampler")
        vae_id, _ = linked_node(prompt, decode_id, "vae", "VAELoader")
        generation = generation_config(prompt, sampler_id, vae_id)
        for key in ("backend", "image_scope", "text_scope", "strength", "seed", "steps", "cfg", "trial_id"):
            if settings[key] != report[key]: raise ValueError(f"Diagnostic provenance mismatch: {key}")
        if generation["prompt"] != report["prompt"]:
            raise ValueError("Diagnostic prompt mismatch")
        return generation
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Incomplete diagnostic provenance") from error


def validate_prefix(prefix, *, flat=False):
    windows = PureWindowsPath(prefix)
    normalized = prefix.replace("\\", "/")
    if (not prefix or windows.drive or windows.is_absolute() or normalized.startswith("/")
            or any(part in ("..", ".", "") for part in normalized.split("/"))
            or (flat and "/" in normalized)):
        raise ValueError("Unsafe diagnostic filename prefix")


def save_diagnostic_artifacts(directory, basename, latent, images, payload, prompt, extra_pnginfo, save_node_id):
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from safetensors.torch import save_file
    from .sampling import tensor_hash

    validate_prefix(basename, flat=True)
    generation = diagnostic_provenance(prompt, save_node_id, payload.report)
    for key, value in (("final_latent", latent), ("first_prediction", payload.first_prediction),
                       ("effective_image_mask", payload.effective_image_mask)):
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Non-finite or invalid {key}")
        if tensor_hash(value) != payload.report[key + "_sha256"]:
            raise ValueError(f"Mismatched {key} artifact")
    if images.ndim != 4 or images.shape[0] != 1 or images.shape[-1] != 3 or not torch.isfinite(images).all():
        raise ValueError("Expected finite batch1 RGB image")
    mask = payload.effective_image_mask
    if mask.ndim != 3 or mask.shape[0] != 1 or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Expected a binary effective image mask")
    # Validate JSON before creating files. Tensors stay in the dedicated payload.
    json.dumps(payload.report, allow_nan=False)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    counter = 0
    while True:
        stem = basename if counter == 0 else f"{basename}{counter:02}_"
        names = {"image": stem + ".png", "effective_mask": stem + "_effective_mask.png",
                 "tensors": stem + ".safetensors", "report": stem + ".json"}
        lock = directory / (stem + ".lock")
        try:
            with lock.open("x"): pass
        except FileExistsError:
            counter += 1; continue
        if any((directory / name).exists() for name in names.values()):
            lock.unlink(); counter += 1; continue
        break
    artifact_id = uuid.uuid4().hex
    report = dict(payload.report, artifact_id=artifact_id, artifacts=names, generation=generation,
                  complete=True, provenance_level="diagnostic_run")
    temporary = []
    try:
        metadata = PngInfo()
        metadata.add_text("artifact_id", artifact_id)
        metadata.add_text("run_id", report["run_id"])
        metadata.add_text("prompt", json.dumps(prompt, ensure_ascii=False))
        for key, value in (extra_pnginfo or {}).items():
            if key not in ("prompt", "artifact_id", "run_id"):
                metadata.add_text(key, json.dumps(value, ensure_ascii=False))
        png = (images[0].detach().cpu().clamp(0, 1) * 255).to(torch.uint8).numpy()
        for kind, image in (("image", Image.fromarray(png)),
                            ("effective_mask", Image.fromarray((mask[0].cpu() * 255).to(torch.uint8).numpy()))):
            temp = directory / (names[kind] + ".tmp")
            temporary.append(temp)
            image.save(temp, format="PNG", pnginfo=metadata)
            os.replace(temp, directory / names[kind])
        temp = directory / (names["tensors"] + ".tmp")
        temporary.append(temp)
        save_file({"final_latent": latent.detach().cpu().contiguous(),
                   "first_prediction": payload.first_prediction.detach().cpu().contiguous()}, str(temp),
                  metadata={"artifact_id": artifact_id, "run_id": report["run_id"]})
        os.replace(temp, directory / names["tensors"])
        report["artifact_sha256"] = {key: file_hash(directory / name) for key, name in names.items() if key != "report"}
        temp = directory / (names["report"] + ".tmp")
        temporary.append(temp)
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temp, directory / names["report"])
        return report
    finally:
        for temp in temporary:
            temp.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
