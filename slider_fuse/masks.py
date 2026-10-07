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
        sizes = _component_sizes(mask[0])
        count = sum(sizes)
        diagnostics[name] = {"coverage": float(mask.float().mean()), "bbox_xyxy": bounds,
                             "component_count_8": len(sizes), "foreground_tokens": count,
                             "largest_component_tokens": max(sizes, default=0),
                             "largest_component_fraction": max(sizes, default=0) / count if count else 0.}
    return {"masks": masks, "grid": grid, "diagnostics": diagnostics, **extra}


def _component_sizes(mask: torch.Tensor) -> list[int]:
    """Diagnostic only: 8-connected components do not alter routing masks."""
    rows = mask.detach().cpu().bool().tolist()
    h, w = mask.shape
    seen = set(); sizes = []
    for y in range(h):
        for x in range(w):
            if not rows[y][x] or (y, x) in seen:
                continue
            stack = [(y, x)]; seen.add((y, x)); size = 0
            while stack:
                v, u = stack.pop(); size += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = v + dy, u + dx
                        if (0 <= ny < h and 0 <= nx < w and rows[ny][nx] and (ny, nx) not in seen):
                            seen.add((ny, nx)); stack.append((ny, nx))
            sizes.append(size)
    return sizes


def _map_diagnostics(maps: dict[str, torch.Tensor]) -> dict:
    values = {name: value.detach().cpu().double().reshape(-1) for name, value in maps.items()}
    result = {}
    for name, vector in values.items():
        low, high, mean = float(vector.min()), float(vector.max()), float(vector.mean())
        probabilities = vector.clamp_min(0)
        mass = probabilities.sum()
        entropy = None
        if mass > 0 and vector.numel() > 1:
            probabilities = probabilities / mass
            entropy = float(-(probabilities * probabilities.clamp_min(1e-300).log()).sum() / torch.log(vector.new_tensor(vector.numel())))
        result[name] = {"min": low, "max": high, "mean": mean, "range": high - low,
                        "range_over_mean": (high - low) / max(abs(mean), 1e-300), "normalized_entropy": entropy}
    target = values["target"] - values["target"].mean()
    protected = values["protected"] - values["protected"].mean()
    denominator = target.norm() * protected.norm()
    result["target_protected_correlation"] = float((target * protected).sum() / denominator) if denominator > 0 else None
    return result


def similarity_preview(bank: dict, name: str) -> torch.Tensor:
    """Independent min/max display, not a segmentation or calibrated confidence."""
    mask = bank["masks"][name]
    raw = bank.get("raw_maps", {}).get(name)
    if raw is None:
        return torch.zeros_like(mask)
    values = raw.detach().cpu().float()
    if values.numel() != mask.numel() or not torch.isfinite(values).all():
        raise ValueError("Raw similarity preview does not match the mask grid")
    low, high = values.min(), values.max()
    normalized = (values - low) / (high - low) if high > low else torch.zeros_like(values)
    return normalized.reshape_as(mask)


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
    return _bank(masks, grid, mode="auto", raw_maps={k: v.detach().cpu() for k, v in maps.items()},
                 map_diagnostics=_map_diagnostics(maps))


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
