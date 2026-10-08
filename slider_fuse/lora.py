"""Plain additive LoRA, evaluated outside quantized base weight storage.

SPDX-License-Identifier: Apache-2.0
Reversible injection/routing is adapted from FreeFuse-for-anima
anima_freefuse/lora.py at 1b924b5dd1f7266fa6e5869331e67a2033e7ec2f.
Modified for FreeFuse-for-sliderlora: native Krea2 key validation, additive
down/up evaluation, image/optional target-text routing and exact forward restore.
See LICENSE, NOTICE and THIRD_PARTY_NOTICES.md for licensing and attribution.
"""
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


def validate_target_text_scale(scale):
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("target_text_scale must be a finite number in [0,1]")


def _validate_text_positions(target, protected, cap_len):
    if isinstance(cap_len, bool) or not isinstance(cap_len, int) or cap_len < 1:
        raise ValueError("Text routing requires a positive runtime text length")
    for name, positions in (("target", target), ("protected", protected)):
        if (not positions or any(isinstance(p, bool) or not isinstance(p, int) or not 0 <= p < cap_len for p in positions)
                or len(set(positions)) != len(positions)):
            raise ValueError(f"Invalid {name} text positions; use unique indices inside the text prefix")
    if set(target) & set(protected):
        raise ValueError("Target/protected text positions overlap")


@dataclass
class RoutingState:
    phase: str = "off"
    cap_len: int | None = None
    mask: torch.Tensor | None = None
    reached: set[str] = field(default_factory=set)
    calls: int = 0
    _mask_cache: dict = field(default_factory=dict, repr=False)
    target_text_scale: float = 0.
    target_text_positions: tuple[int, ...] = ()
    protected_text_positions: tuple[int, ...] = ()
    expected_text_len: int | None = None
    target_text_calls: int = 0
    _text_index_cache: dict = field(default_factory=dict, repr=False)
    image_scope: str = "target_mask"
    text_scope: str = "target_phrase"
    text_linear_calls: int = 0
    recorder: object = field(default=None, repr=False)

    def __post_init__(self):
        self.validate_scopes()

    def validate_scopes(self):
        if self.image_scope not in ("target_mask", "all") or self.text_scope not in ("none", "target_phrase", "all"):
            raise ValueError("Invalid diagnostic image/text scope")

    def configure_target_text(self, scale, target_positions, protected_positions, text_len):
        validate_target_text_scale(scale)
        _validate_text_positions(target_positions, protected_positions, text_len)
        self.target_text_scale = float(scale)
        self.target_text_positions = tuple(target_positions)
        self.protected_text_positions = tuple(protected_positions)
        self.expected_text_len = text_len
        self._text_index_cache.clear()

    def target_text_indices(self, output):
        self.validate_scopes()
        if self.cap_len != self.expected_text_len:
            raise ValueError("Runtime text boundary differs from target phrase positions")
        validate_target_text_scale(self.target_text_scale)
        _validate_text_positions(self.target_text_positions, self.protected_text_positions, self.cap_len)
        key = (output.device, self.text_scope)
        if key not in self._text_index_cache:
            positions = range(self.cap_len) if self.text_scope == "all" else self.target_text_positions
            self._text_index_cache[key] = torch.tensor(list(positions), device=output.device, dtype=torch.long)
        return self._text_index_cache[key]

    def image_mask(self, output: torch.Tensor) -> torch.Tensor:
        if self.mask is None or self.cap_len is None or self.cap_len < 1:
            raise ValueError("Runtime text length and target mask have not been captured")
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != self.cap_len + self.mask.numel():
            raise ValueError("LoRA output sequence does not match [text, image] mask layout")
        key = (output.device, output.dtype)
        if key not in self._mask_cache:
            self._mask_cache[key] = self.mask.reshape(1, -1, 1).to(output)
        return self._mask_cache[key]

    def effective_image_mask(self, output: torch.Tensor) -> torch.Tensor:
        self.validate_scopes()
        mask = self.image_mask(output)
        return torch.ones_like(mask) if self.image_scope == "all" else mask

    def clear(self):
        self.phase = "off"; self.cap_len = None; self.mask = None
        self._mask_cache.clear()
        self.target_text_scale = 0.
        self.target_text_positions = (); self.protected_text_positions = (); self.expected_text_len = None
        self._text_index_cache.clear()
        self.image_scope = "target_mask"; self.text_scope = "target_phrase"
        self.recorder = None


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
        mask = self.state.effective_image_mask(base)
        if self.strength == 0 or not bool(self.state.mask.any()):
            if self.state.recorder is not None:
                self.state.recorder.record_linear(self.name, x, base, base, None, None,
                                                 self.state, self.adapter, self.strength)
            return base
        cap = self.state.cap_len
        delta = self.adapter.delta(x[:, cap:]).to(base.dtype) * self.strength * mask
        result = base.clone()
        result[:, cap:] = base[:, cap:] + delta
        text_delta = None
        if self.state.target_text_scale and self.state.text_scope != "none":
            positions = self.state.target_text_indices(base)
            text_delta = self.adapter.delta(x.index_select(1, positions)).to(base.dtype) * self.strength * self.state.target_text_scale
            result.index_copy_(1, positions, base.index_select(1, positions) + text_delta)
            self.state.text_linear_calls += 1
            if self.state.text_scope == "target_phrase":
                self.state.target_text_calls += 1
        if self.state.recorder is not None:
            self.state.recorder.record_linear(self.name, x, base, result, delta, text_delta,
                                             self.state, self.adapter, self.strength)
        return result
