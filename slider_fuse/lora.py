"""Plain additive LoRA, evaluated outside quantized base weight storage."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
import re
import threading
import weakref

import torch
from torch.nn import functional as F

_LAYER = re.compile(r"^blocks\.\d+\.attn\.(?:wq|wk|wv|gate|wo)$")
_LOCK = threading.Lock()
_ACTIVE = weakref.WeakSet()


@contextmanager
def core_guard(core):
    with _LOCK:
        if core in _ACTIVE:
            raise RuntimeError("This Krea2 model core is already used by Slider FreeFuse; parallel shared-core sampling is unsupported")
        _ACTIVE.add(core)
    try:
        yield
    finally:
        with _LOCK:
            _ACTIVE.discard(core)


@dataclass
class Adapter:
    down: torch.Tensor
    up: torch.Tensor
    alpha: float
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def scale(self) -> float:
        return self.alpha / self.down.shape[0]

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        key = (x.device, x.dtype)
        if key not in self._cache:
            self._cache[key] = (self.down.to(device=x.device, dtype=x.dtype), self.up.to(device=x.device, dtype=x.dtype))
        down, up = self._cache[key]
        return F.linear(F.linear(x, down), up) * self.scale

    def clear(self):
        self._cache.clear()


def load_adapters(source: dict[str, torch.Tensor], core) -> dict[str, Adapter]:
    """Consume every safetensors key; accept the training node's standard format."""
    if not source:
        raise ValueError("Slider LoRA has no tensors")
    known = {}
    for name, module in core.named_modules():
        if _LAYER.fullmatch(name):
            known["lora_unet_" + name.replace(".", "_")] = (name, module)
    grouped = {}
    for key, tensor in source.items():
        match = re.fullmatch(r"(.+)\.(lora_down\.weight|lora_up\.weight|alpha)", key)
        if not match or match[1] not in known:
            raise ValueError(f"Unsupported/unmapped Slider key: {key}. Initial support: blocks.N.attn wq/wk/wv/gate/wo, plain lora_unet format")
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite or invalid Slider tensor: {key}")
        grouped.setdefault(match[1], {})[match[2]] = tensor.detach()
    result = {}
    for prefix, values in grouped.items():
        if not {"lora_down.weight", "lora_up.weight"} <= set(values):
            raise ValueError(f"Incomplete down/up pair: {prefix}")
        down, up = values["lora_down.weight"], values["lora_up.weight"]
        name, module = known[prefix]
        shape = getattr(getattr(module, "weight", None), "shape", None)
        if down.ndim != 2 or up.ndim != 2 or down.shape[0] < 1 or up.shape[1] != down.shape[0] or shape != (up.shape[0], down.shape[1]):
            raise ValueError(f"Slider shape/rank does not match {name}")
        if not down.is_floating_point() or not up.is_floating_point():
            raise ValueError("Slider matrices must be floating point, even with an INT8 base")
        alpha_tensor = values.get("alpha", torch.tensor(float(down.shape[0])))
        if alpha_tensor.numel() != 1:
            raise ValueError(f"alpha must be scalar: {prefix}")
        alpha = float(alpha_tensor)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"alpha must be positive: {prefix}")
        result[name] = Adapter(down.cpu(), up.cpu(), alpha)
    return result


def compare_probe_outputs(reference, repeat, actual, expected_delta, image_mask, cap_len) -> dict:
    """A no-op or wrongly scaled native route must not pass the real-machine gate."""
    if (reference.shape != repeat.shape or actual.shape != reference.shape or reference.ndim != 3
            or expected_delta.shape != reference[:, cap_len:].shape
            or image_mask.shape != (1, reference.shape[1] - cap_len, 1)):
        raise ValueError("Probe output/mask dimensions do not match [text, image]")
    if any(not torch.isfinite(value).all() for value in (reference, repeat, actual, expected_delta)):
        return {"passed": False, "error": "Non-finite native probe output"}
    expected = reference.clone()
    expected[:, cap_len:] += expected_delta.to(reference) * image_mask.to(reference)
    selected = torch.zeros_like(reference, dtype=torch.bool)
    selected[:, cap_len:] = (image_mask.to(reference.device) > 0).expand_as(expected_delta)
    repeat_error = float((repeat - reference).abs().max())
    tolerance = max(1e-6, repeat_error * 5)
    outside = float((actual - reference)[~selected].abs().max()) if (~selected).any() else 0.
    target = float((actual - reference)[selected].abs().max()) if selected.any() else 0.
    expected_target = float((expected - reference)[selected].abs().max()) if selected.any() else 0.
    expected_error = float((actual - expected).abs().max())
    return {"passed": outside <= tolerance and expected_error <= tolerance and target > tolerance and expected_target > tolerance,
            "repeat_error": repeat_error, "tolerance": tolerance, "outside_max_error": outside,
            "target_max_change": target, "expected_target_max_change": expected_target, "expected_max_error": expected_error}


@dataclass
class RoutingState:
    phase: str = "off"
    cap_len: int | None = None
    mask: torch.Tensor | None = None
    reached: set[str] = field(default_factory=set)
    calls: int = 0
    _mask_cache: dict = field(default_factory=dict, repr=False)

    def image_mask(self, output: torch.Tensor) -> torch.Tensor:
        if self.mask is None or self.cap_len is None or self.cap_len < 1:
            raise ValueError("Runtime text length and target mask have not been captured")
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != self.cap_len + self.mask.numel():
            raise ValueError("LoRA output sequence does not match [text, image] mask layout")
        key = (output.device, output.dtype)
        if key not in self._mask_cache:
            self._mask_cache[key] = self.mask.reshape(1, -1, 1).to(output)
        return self._mask_cache[key]

    def clear(self):
        self.phase = "off"; self.cap_len = None; self.mask = None
        self._mask_cache.clear()


class SliderHook:
    def __init__(self, module, adapter: Adapter, strength: float, state: RoutingState, name: str):
        if not math.isfinite(strength):
            raise ValueError("Slider strength must be finite")
        self.module, self.adapter, self.strength, self.state, self.name = module, adapter, strength, state, name
        self.original_forward = None
        self._original_attribute = None
        self._had_attribute = False

    def inject(self):
        if self.original_forward is None:
            self._had_attribute = "forward" in self.module.__dict__
            self._original_attribute = self.module.__dict__.get("forward")
            self.original_forward = self.module.forward
            self.module.forward = self.forward

    def eject(self):
        if self.original_forward is not None:
            if self._had_attribute:
                self.module.forward = self._original_attribute
            else:
                del self.module.forward
            self.original_forward = None
            self._original_attribute = None
        self.adapter.clear()

    def forward(self, x, *args, **kwargs):
        base = self.original_forward(x, *args, **kwargs)
        if self.state.phase != "route":
            return base
        self.state.reached.add(self.name); self.state.calls += 1
        mask = self.state.image_mask(base)
        if self.strength == 0 or not bool(self.state.mask.any()):
            return base
        cap = self.state.cap_len
        delta = self.adapter.delta(x[:, cap:]).to(base.dtype) * self.strength * mask
        result = base.clone()
        result[:, cap:] = base[:, cap:] + delta
        return result
