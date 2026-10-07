"""Exclusive target/protected/background masks, without equal-area quotas."""
from __future__ import annotations

import torch
from torch.nn import functional as F

GROUPS = ("target", "protected", "background")


def patch_grid(latent: torch.Tensor, patch: int) -> tuple[int, int]:
    if (not isinstance(latent, torch.Tensor) or latent.ndim not in (4, 5)
            or latent.shape[0] != 1 or (latent.ndim == 5 and latent.shape[2] != 1)):
        raise ValueError("Only a single still image, batch size 1, is supported")
    if not isinstance(patch, int) or isinstance(patch, bool) or patch < 1:
        raise ValueError("Krea2 patch size must be a positive integer")
    h, w = latent.shape[-2:]
    if min(h, w) < 1:
        raise ValueError("Latent canvas is empty")
    return ((h + patch - 1) // patch, (w + patch - 1) // patch)


def _bank(masks: dict[str, torch.Tensor], grid: tuple[int, int], **extra) -> dict:
    diagnostics = {}
    for name, mask in masks.items():
        indices = torch.nonzero(mask[0] > 0, as_tuple=False)
        bounds = None
        if indices.numel():
            low = indices.amin(0).tolist(); high = (indices.amax(0) + 1).tolist()
            bounds = [low[1], low[0], high[1], high[0]]
        diagnostics[name] = {"coverage": float(mask.float().mean()), "bbox_xyxy": bounds}
    return {"masks": masks, "grid": grid, "diagnostics": diagnostics, **extra}


def generate_masks(maps: dict[str, torch.Tensor], grid: tuple[int, int]) -> dict:
    if set(maps) != {"target", "protected"}:
        raise ValueError("Both target and protected concept maps are required")
    h, w = grid
    if min(h, w) < 1:
        raise ValueError("Mask grid is empty")
    scores = []
    for name in ("target", "protected"):
        values = maps[name].detach().float()
        if values.ndim == 3 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        if values.shape != (1, h * w):
            raise ValueError(f"{name} map must match the observed single-image patch grid")
        if not torch.isfinite(values).all():
            raise ValueError(f"{name} map has NaN/Inf")
        low, high = values.amin(), values.amax()
        if high - low <= torch.maximum(torch.maximum(low.abs(), high.abs()), low.new_tensor(1.)) * 1e-6:
            raise ValueError(f"{name} map is spatially constant/diffuse; use manual masks or another collection setting")
        scores.append((values - low) / (high - low))
    target, protected = scores
    maximum = torch.maximum(target, protected)
    foreground = maximum > (1 - maximum)
    masks = {"target": (foreground & (target - protected > 1e-6)).float().reshape(1, h, w),
             "protected": (foreground & (protected - target > 1e-6)).float().reshape(1, h, w)}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"Automatic {name} mask is empty; use manual masks or another collection setting")
    masks["background"] = 1 - masks["target"] - masks["protected"]
    return _bank(masks, grid, mode="auto", raw_maps={k: v.detach().cpu() for k, v in maps.items()})


def manual_masks(target: torch.Tensor, protected: torch.Tensor, grid: tuple[int, int]) -> dict:
    if min(grid) < 1:
        raise ValueError("Mask grid is empty")
    converted = {}
    shape = None
    for name, source in (("target", target), ("protected", protected)):
        if not isinstance(source, torch.Tensor) or source.ndim not in (2, 3):
            raise ValueError("Each manual MASK must have shape H,W or 1,H,W")
        if source.ndim == 3 and source.shape[0] != 1:
            raise ValueError("Manual mask batch must be 1")
        if min(source.shape[-2:]) < 1 or not torch.isfinite(source).all() or source.min() < 0 or source.max() > 1:
            raise ValueError("Manual mask must be finite and in [0,1]")
        if shape is not None and source.shape[-2:] != shape:
            raise ValueError("Both manual masks must use the same canvas")
        shape = source.shape[-2:]
        converted[name] = (F.interpolate(source.detach().cpu().float().reshape(1, 1, *shape), size=grid,
                                         mode="area")[0] >= .5).float()
    if (converted["target"] * converted["protected"]).any():
        raise ValueError("Manual masks overlap on the Krea2 patch grid")
    if any(not value.any() for value in converted.values()):
        raise ValueError("A manual mask is empty on the Krea2 patch grid")
    converted["background"] = 1 - converted["target"] - converted["protected"]
    return _bank(converted, grid, mode="manual")
