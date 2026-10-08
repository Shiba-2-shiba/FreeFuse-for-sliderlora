"""Opt-in measurement and portable diagnostic artifacts; no ComfyUI import."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import uuid

import torch

DIAGNOSTIC_SAMPLERS = ("Krea2SliderFuseDiagnosticSampler", "Krea2SliderFusePredictionMixSampler")


@dataclass
class DiagnosticPayload:
    report: dict
    first_prediction: torch.Tensor
    effective_image_mask: torch.Tensor
    extra_tensors: dict[str, torch.Tensor] = field(default_factory=dict)


def implementation_info():
    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"],
                                             text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    # The source digest also distinguishes local edits sharing the same HEAD.
    digest = hashlib.sha256()
    for path in sorted((root / "slider_fuse").glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return {"implementation_revision": revision, "implementation_dirty": dirty,
            "implementation_source_sha256": digest.hexdigest()}


def region_measurement(reference, output, selection, computed=None):
    """GPU scalar reductions; serialization is deferred to finalize()."""
    a = reference.detach().float()[selection]
    b = output.detach().float()[selection]
    delta = b - a
    c = torch.zeros_like(a) if computed is None else computed.detach().float()[selection]
    finite = torch.isfinite(a) & torch.isfinite(b) & torch.isfinite(c)
    clean = torch.where(finite, delta, 0.)
    clean_c = torch.where(finite, c, 0.)
    return torch.stack([delta.new_tensor(a.numel()), (~finite).sum(), clean.square().sum(),
                        clean.abs().max() if clean.numel() else delta.new_zeros(()),
                        (delta != 0).sum(), clean_c.square().sum(),
                        ((c != 0) & (delta == 0)).sum()])


def serialize_measurement(values):
    count, nonfinite, squared, maximum, changed, computed_squared, dropped = values
    return {"element_count": int(count), "nonfinite_count": int(nonfinite),
            "rms": math.sqrt(squared / count) if count else None,
            "max_abs": maximum if count else None, "changed_elements": int(changed),
            "computed_delta_rms": math.sqrt(computed_squared / count) if count else None,
            "rounded_away_elements": int(dropped),
            "unavailable_reason": None if count else "empty region"}


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
    def __init__(self, selected_modules, *, level="summary"):
        if level not in ("summary", "audit"):
            raise ValueError("Diagnostic level must be summary or audit")
        self.level = level
        self.selected_modules = set(selected_modules)
        self.adapter_stats = []
        self._seen = set()
        self.first_prediction = None
        self.first_inputs = {}
        self.linear_audit = []
        self.step_trace = []
        self._trace_inputs = []; self._trace_predictions = []; self._trace_sigmas = []
        self._eval_index = None; self._sigma = None

    def begin_step(self, eval_index, sigma):
        self._eval_index = eval_index
        self._sigma = float(sigma.detach().cpu().reshape(-1)[0]) if self.level == "audit" else None

    def record_step(self, prediction, latent_input, sigma):
        if self.level != "audit":
            return
        from .sampling import tensor_hash
        if any(not torch.isfinite(v).all() for v in (prediction, latent_input, sigma)):
            raise ValueError("Non-finite diagnostic step")
        self.step_trace.append({"eval_index": self._eval_index, "sigma": self._sigma,
            "input_sha256": tensor_hash(latent_input), "prediction_sha256": tensor_hash(prediction)})
        self._trace_inputs.append(latent_input.detach().cpu().clone())
        self._trace_predictions.append(prediction.detach().cpu().clone())
        self._trace_sigmas.append(sigma.detach().cpu().reshape(-1)[0].clone())

    def finalize(self):
        rows = []
        if self.linear_audit:
            measures = [v for row in self.linear_audit for section in ("image", "text")
                        for v in row[section].values()]
            values = torch.stack(measures).detach().cpu().double().tolist()
            iterator = iter(values)
            for row in self.linear_audit:
                rows.append(dict(row, **{section: {name: serialize_measurement(next(iterator))
                    for name in row[section]} for section in ("image", "text")}))
        tensors = {}
        if self._trace_inputs:
            tensors = {"trace_inputs": torch.stack(self._trace_inputs),
                       "trace_predictions": torch.stack(self._trace_predictions),
                       "trace_sigmas": torch.stack(self._trace_sigmas)}
        return {"diagnostic_level": self.level, "linear_audit": rows if self.level == "audit" else None,
                "step_trace": self.step_trace if self.level == "audit" else None}, tensors

    def _audit_linear(self, name, base, result, image_delta, text_delta, state):
        cap = state.cap_len
        selection = state.effective_image_mask(base).squeeze(-1).bool()
        if state.audit_partition is None:
            raise ValueError("Linear audit requires a reference partition")
        if base.device not in state._audit_partition_cache:
            state._audit_partition_cache[base.device] = {
                k: v.reshape(1, -1).to(device=base.device, dtype=torch.bool)
                for k, v in state.audit_partition.items()}
        image_regions = dict(state._audit_partition_cache[base.device], unselected=~selection,
                             selected=selection)
        target = torch.zeros(base.shape[:2], device=base.device, dtype=torch.bool)[:, :cap]
        protected = torch.zeros_like(target)
        target[:, list(state.target_text_positions)] = True
        protected[:, list(state.protected_text_positions)] = True
        selected = torch.zeros_like(target)
        if state.text_scope != "none" and state.target_text_scale:
            selected[:, state.target_text_indices(base)] = True
        full_delta = torch.zeros_like(base[:, :cap])
        if text_delta is not None:
            full_delta.index_copy_(1, state.target_text_indices(base), text_delta)
        self.linear_audit.append({"eval_index": self._eval_index, "sigma": self._sigma,
            "module": name, "call_index": len(self.linear_audit),
            "image": {k: region_measurement(base[:, cap:], result[:, cap:], v, image_delta)
                      for k, v in image_regions.items()},
            "text": {k: region_measurement(base[:, :cap], result[:, :cap], v, full_delta)
                     for k, v in {"target_phrase": target, "protected_phrase": protected,
                                  "other": ~(target | protected), "selected": selected,
                                  "unselected": ~selected}.items()}})

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
        if self.level == "audit":
            self._audit_linear(name, base, result, image_delta, text_delta, state)
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
        allowed = (expected_type,) if isinstance(expected_type, str) else expected_type
        if allowed is not None and node["class_type"] not in allowed:
            raise ValueError(f"Expected {expected_type} at {node_id}.{field}")
        return ident, node["inputs"]
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"Missing provenance link {node_id}.{field}") from error


def generation_config(prompt, sampler_id, vae_id, *, standard_case=None):
    """Trace only the documented evaluation graphs, never search for a loader."""
    try:
        sampler = prompt[str(sampler_id)]
        expected = ("KSampler",) if standard_case else DIAGNOSTIC_SAMPLERS
        if sampler["class_type"] not in expected:
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
        sampler_id, settings = linked_node(prompt, save_node_id, "latent", DIAGNOSTIC_SAMPLERS)
        diagnostic_id, _ = linked_node(prompt, save_node_id, "diagnostics", DIAGNOSTIC_SAMPLERS, slot=2)
        decode_id, _ = linked_node(prompt, save_node_id, "images", "VAEDecode")
        decoded_sampler_id, _ = linked_node(prompt, decode_id, "samples", DIAGNOSTIC_SAMPLERS)
        if diagnostic_id != sampler_id or decoded_sampler_id != sampler_id:
            raise ValueError("Images, latent and diagnostics must use the same Sampler")
        vae_id, _ = linked_node(prompt, decode_id, "vae", "VAELoader")
        generation = generation_config(prompt, sampler_id, vae_id)
        mix = prompt[sampler_id]["class_type"] == "Krea2SliderFusePredictionMixSampler"
        if mix and report["backend"] != "prediction_mix":
            raise ValueError("Prediction mix backend mismatch")
        keys = (("mix_scope",) if mix else ("backend", "image_scope", "text_scope"))
        for key in keys + ("strength", "seed", "steps", "cfg", "trial_id"):
            if settings[key] != report[key]: raise ValueError(f"Diagnostic provenance mismatch: {key}")
        default_level = "audit" if mix else "summary"
        if settings.get("diagnostic_level", default_level) != report.get("diagnostic_level", default_level):
            raise ValueError("Diagnostic level provenance mismatch")
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


def validate_audit_coverage(report):
    """A complete audit must cover every declared projection and sampler step."""
    if report.get("diagnostic_schema_version", 1) != 2 or report.get("diagnostic_level") != "audit":
        return
    steps = report.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("Invalid audit step count")
    backend = report.get("backend")
    if backend not in ("hook", "native", "prediction_mix"):
        raise ValueError("Unknown audit backend")
    if backend == "native" and (report.get("linear_audit") or report.get("prediction_mix_audit")):
        raise ValueError("Native backend has no direct audit instrumentation")
    if backend == "hook":
        matched = report.get("matched_modules")
        rows = report.get("linear_audit")
        if (isinstance(matched, bool) or not isinstance(matched, int) or matched < 1
                or not isinstance(rows, list) or len(rows) != steps * matched
                or report.get("lora_linear_calls") != steps * matched
                or report.get("reached_modules") != matched):
            raise ValueError("Incomplete all-layer linear audit coverage")
        per_step = [set() for _ in range(steps)]
        for i, row in enumerate(rows):
            step = row.get("eval_index")
            module = row.get("module")
            if (isinstance(step, bool) or not isinstance(step, int) or not 0 <= step < steps
                    or not isinstance(module, str) or row.get("call_index") != i
                    or module in per_step[step]):
                raise ValueError("Duplicate/invalid linear audit step or module")
            per_step[step].add(module)
        expected = set(report.get("matched_module_names", per_step[0]))
        if len(expected) != matched or any(names != expected for names in per_step):
            raise ValueError("Incomplete per-step linear audit module coverage")
    elif backend == "prediction_mix":
        scope = report.get("mix_scope")
        if scope not in ("none", "target_mask", "all"):
            raise ValueError("Invalid prediction-mix audit scope")
        partial = scope == "target_mask" and report.get("strength") != 0
        slider_only = scope == "all" and report.get("strength") != 0
        expected_calls = ({"base": steps, "slider": steps} if partial else
                          {"base": 0, "slider": steps} if slider_only else {"base": steps, "slider": 0})
        if report.get("sampler_nfe") != steps or report.get("branch_nfe") != expected_calls:
            raise ValueError("Invalid prediction-mix audit branch coverage")
        rows = report.get("prediction_mix_audit")
        if not partial:
            if rows is not None and (not isinstance(rows, list) or rows):
                raise ValueError("Endpoint audit cannot measure an unexecuted branch")
            return
        if (not isinstance(rows, list) or len(rows) != steps
                or any(row.get("eval_index") != i for i, row in enumerate(rows))):
            raise ValueError("Incomplete prediction-mix audit step coverage")
        for row in rows:
            if not {"base_excluded", "slider_selected"} <= set(row):
                raise ValueError("Incomplete prediction-mix audit regions")


def save_diagnostic_artifacts(directory, basename, latent, images, payload, prompt, extra_pnginfo, save_node_id):
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from safetensors.torch import save_file
    from .sampling import tensor_hash

    validate_prefix(basename, flat=True)
    generation = diagnostic_provenance(prompt, save_node_id, payload.report)
    validate_audit_coverage(payload.report)
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
    stored_tensors = {"final_latent": latent.detach().cpu().contiguous(),
                      "first_prediction": payload.first_prediction.detach().cpu().contiguous()}
    reference_names = {"reference_" + k + "_mask" for k in ("target", "protected", "background")}
    allowed = {"trace_inputs", "trace_predictions", "trace_sigmas",
               "trace_base_predictions", "trace_slider_predictions"} | reference_names
    for key, value in payload.extra_tensors.items():
        if key not in allowed or not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid diagnostic trace tensor: {key}")
        expected = (tuple(payload.effective_image_mask.shape) if key in reference_names else
                    (payload.report["steps"],) if key == "trace_sigmas" else
                    (payload.report["steps"],) + tuple(payload.first_prediction.shape))
        if tuple(value.shape) != expected:
            raise ValueError(f"Diagnostic trace dimensions mismatch: {key}")
        stored_tensors[key] = value.detach().cpu().contiguous()
    references = set(payload.extra_tensors) & reference_names
    if references:
        if references != reference_names:
            raise ValueError("Incomplete reference mask partition")
        for name in ("target", "protected", "background"):
            value = stored_tensors["reference_" + name + "_mask"]
            if not ((value == 0) | (value == 1)).all() or tensor_hash(value) != payload.report["reference_partition_sha256"][name]:
                raise ValueError("Reference mask hash/binary mismatch")
        if not torch.equal(sum(stored_tensors[k] for k in reference_names), torch.ones_like(payload.effective_image_mask)):
            raise ValueError("Reference masks are not an exclusive partition")
    if any(k.startswith("trace_") for k in payload.extra_tensors):
        if not {"trace_inputs", "trace_predictions", "trace_sigmas"} <= set(payload.extra_tensors):
            raise ValueError("Incomplete diagnostic step trace")
        rows = payload.report.get("step_trace")
        if not isinstance(rows, list) or len(rows) != payload.report["steps"]:
            raise ValueError("Incomplete diagnostic step manifest")
        for i, row in enumerate(rows):
            if (row["eval_index"] != i or row["input_sha256"] != tensor_hash(stored_tensors["trace_inputs"][i])
                    or row["prediction_sha256"] != tensor_hash(stored_tensors["trace_predictions"][i])
                    or row["sigma"] != float(stored_tensors["trace_sigmas"][i])):
                raise ValueError("Diagnostic trace hash or sigma mismatch")
        if not torch.equal(stored_tensors["trace_predictions"][0], payload.first_prediction):
            raise ValueError("First prediction does not match trace")
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
    if report.get("diagnostic_schema_version") == 2:
        report["tensor_manifest"] = {k: {"sha256": tensor_hash(v), "shape": list(v.shape), "dtype": str(v.dtype)}
                                     for k, v in stored_tensors.items()}
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
        save_file(stored_tensors, str(temp),
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
