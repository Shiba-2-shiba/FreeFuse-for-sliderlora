"""Krea2 observation and the FreeFuse two-stage concept similarity calculation.

SPDX-License-Identifier: Apache-2.0
Adapted from yaoliliu/FreeFuse at f5570195e84d3bc8e7f6702fcb7b012b89372b8e:
freefuse_comfyui/freefuse_core/{krea2_support.py,attention_replace.py}.
Modified for FreeFuse-for-sliderlora: two explicit subject roles, strict
token/grid/sigma validation, chunked similarity reduction and no attention bias.
See LICENSE, NOTICE and THIRD_PARTY_NOTICES.md for licensing and attribution.
Observation never replaces the model's attention result or permission matrix.
"""
from __future__ import annotations

import math

import torch


def similarity_maps(img_q, txt_k, hidden, positions, top_k_ratio=.3, temperature=4000.) -> dict[str, torch.Tensor]:
    if not math.isfinite(top_k_ratio) or not 0 < top_k_ratio <= 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("top_k_ratio must be in (0,1] and temperature must be positive")
    if (img_q.ndim != 4 or txt_k.ndim != 4 or hidden.ndim != 3 or img_q.shape[0] != 1
            or txt_k.shape[0] != 1 or hidden.shape[:2] != img_q.shape[:2] or img_q.shape[2:] != txt_k.shape[2:]):
        raise ValueError("Unexpected single-image Q/K/hidden layout")
    if set(positions) != {"target", "protected"}:
        raise ValueError("Collect target AND protected, regardless of adapter presence")
    scores = {}
    for name, span in positions.items():
        if not span or any(isinstance(p, bool) or not isinstance(p, int) or not 0 <= p < txt_k.shape[1] for p in span):
            raise ValueError(f"Invalid {name} text position; refusing to clamp token indices")
        keys = txt_k[:, list(span)].float()
        weights = torch.einsum("bihd,bjhd->bhij", img_q.float(), keys) / 1000.
        scores[name] = weights.softmax(dim=2).mean(dim=1).mean(dim=-1)
    features = hidden.float()
    result = {}
    count = max(1, int(features.shape[1] * top_k_ratio))
    for name, score in scores.items():
        other = "protected" if name == "target" else "target"
        top = torch.topk(score * 2 - scores[other], count, dim=-1).indices
        core = torch.gather(features, 1, top.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
        # Bound the temporary core-token x all-image-token matrix.
        total = features.new_zeros((1, features.shape[1]))
        for chunk in core.split(128, dim=1):
            total += torch.bmm(chunk, features.transpose(1, 2)).sum(dim=1)
        result[name] = (total / count / temperature).softmax(dim=-1).detach()
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Attention observation produced non-finite maps")
    return result


class AttentionCollector:
    def __init__(self, positions, grid, collect_step, collect_block, top_k_ratio, temperature, sigmas):
        if not 1 <= collect_step < len(sigmas):
            raise ValueError("collect_step must be in 1..total steps")
        self.positions, self.grid = positions, grid
        self.collect_step, self.collect_block = collect_step, collect_block
        self.top_k_ratio, self.temperature = top_k_ratio, temperature
        self.sigmas = sigmas.detach().float().cpu()
        self.active = False
        self.collect_now = False
        self.cap_len = None
        self.maps = {}
        self.observation = {}
        self._handles = []

    def begin_forward(self, sigma: torch.Tensor, text_len: int):
        if self.cap_len is not None and self.cap_len != text_len:
            raise ValueError("Runtime text token count changed during observation")
        self.cap_len = text_len
        values = sigma.detach().float().cpu().reshape(-1)
        if values.numel() != 1:
            raise ValueError("Observation supports a single positive image batch")
        self.collect_now = bool(torch.isclose(values[0], self.sigmas[self.collect_step - 1], rtol=1e-5, atol=1e-6))
        if self.collect_now:
            self.observation = {"sigma": float(values[0]), "step": self.collect_step,
                                "block": self.collect_block, "cap_len": text_len,
                                "grid": list(self.grid), "positions": self.positions}

    def install(self, core, state):
        if self._handles:
            raise RuntimeError("Collector is already installed")
        def text_hook(_module, _args, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise ValueError("Unexpected Krea2 txtfusion output")
            cap = output.shape[1]
            if self.cap_len is not None and cap != self.cap_len:
                raise ValueError("txtfusion text length differs from encoded token count")
            state.cap_len = cap
        self._handles.append(core.txtfusion.register_forward_hook(text_hook))
        self._handles.append(core.blocks[self.collect_block].register_forward_pre_hook(self._observe, with_kwargs=True))

    def _observe(self, block, args, kwargs):
        if not self.active or not self.collect_now or self.maps:
            return None
        if len(args) < 3:
            raise ValueError("Unsupported native Krea2 block signature")
        from comfy.ldm.flux.math import apply_rope
        from comfy.ldm.modules.attention import optimized_attention_masked

        x, vec, freqs = args[:3]
        mask = kwargs.get("mask", args[3] if len(args) > 3 else None)
        options = kwargs.get("transformer_options", {})
        if self.cap_len is None or x.shape[1] - self.cap_len != self.grid[0] * self.grid[1]:
            raise ValueError("Observed image tokens differ from the Krea2 patch grid")
        scale, shift, _gate, *_rest = block.mod(vec)
        value = (1 + scale) * block.prenorm(x) + shift
        attn = block.attn
        q = attn.wq(value).reshape(1, value.shape[1], attn.heads, -1).transpose(1, 2)
        k = attn.wk(value).reshape(1, value.shape[1], attn.kvheads, -1).transpose(1, 2)
        v = attn.wv(value).reshape(1, value.shape[1], attn.kvheads, -1).transpose(1, 2)
        q, k = attn.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)
        if attn.heads % attn.kvheads:
            raise ValueError("Unsupported Krea2 GQA head ratio")
        ratio = attn.heads // attn.kvheads
        if ratio != 1:
            k = k.repeat_interleave(ratio, dim=1); v = v.repeat_interleave(ratio, dim=1)
        output = optimized_attention_masked(q, k, v, attn.heads, mask=mask, skip_reshape=True,
                                            preferred_attention=getattr(attn, "comfy_attention", None), transformer_options=options)
        self.maps = similarity_maps(q.transpose(1, 2)[:, self.cap_len:], k.transpose(1, 2)[:, :self.cap_len],
                                    output[:, self.cap_len:], self.positions, self.top_k_ratio, self.temperature)
        self.observation["image_len"] = self.grid[0] * self.grid[1]
        self.observation["q_heads"] = attn.heads
        self.observation["kv_heads"] = attn.kvheads
        return None

    def remove(self):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def reset(self):
        self.active = False; self.collect_now = False
        self.maps.clear()
