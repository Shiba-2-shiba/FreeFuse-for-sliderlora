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


def _components(mask: torch.Tensor, connectivity: int = 8) -> list[list[int]]:
    """Row-major ordered components on a CPU binary grid."""
    rows = mask.detach().cpu().bool().tolist()
    h, w = mask.shape
    neighbors = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                 if (dy or dx) and (connectivity == 8 or abs(dy) + abs(dx) == 1)]
    seen = set(); components = []
    for y in range(h):
        for x in range(w):
            if not rows[y][x] or (y, x) in seen:
                continue
            stack = [(y, x)]; seen.add((y, x)); component = []
            while stack:
                v, u = stack.pop(); component.append(v * w + u)
                for dy, dx in neighbors:
                    ny, nx = v + dy, u + dx
                    if (0 <= ny < h and 0 <= nx < w and rows[ny][nx] and (ny, nx) not in seen):
                        seen.add((ny, nx)); stack.append((ny, nx))
            components.append(component)
    return components


def _component_sizes(mask: torch.Tensor) -> list[int]:
    return [len(component) for component in _components(mask)]


def validate_postprocess_settings(max_hole_area=0, dilate_radius=0):
    for name, value, maximum in (("fill_holes_max_area", max_hole_area, 64), ("mask_dilate_radius", dilate_radius, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in 0..{maximum}")


def _validate_partition(bank):
    masks = bank.get("masks", {})
    if set(masks) != set(GROUPS):
        raise ValueError("Postprocessing requires target/protected/background masks")
    target = masks["target"]
    if not isinstance(target, torch.Tensor) or target.ndim != 3 or target.shape[0] != 1 or min(target.shape[-2:]) < 1:
        raise ValueError("Postprocessing masks must have shape [1,H,W]")
    if tuple(bank.get("grid", ())) != tuple(target.shape[-2:]):
        raise ValueError("Postprocessing grid differs from mask dimensions")
    for name, mask in masks.items():
        if (not isinstance(mask, torch.Tensor) or mask.shape != target.shape or mask.device != target.device
                or mask.dtype != target.dtype or not mask.is_floating_point()):
            raise ValueError("Postprocessing masks must share shape, floating dtype and device")
        if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError(f"Postprocessing {name} mask must be finite and binary")
    if not torch.equal(sum(masks.values()), torch.ones_like(target)):
        raise ValueError("Postprocessing masks must form an exclusive, complete partition")
    if not target.any() or not masks["protected"].any():
        raise ValueError("Postprocessing cannot repair an empty target/protected mask")
    return masks


def validate_selection_radius(radius=0):
    if isinstance(radius, bool) or not isinstance(radius, int) or not 0 <= radius <= 16:
        raise ValueError("selection_dilate_radius must be an integer in 0..16 token cells")


def prediction_selection_mask(bank, radius=0):
    """Expand only the largest target component through background; keep the partition."""
    validate_selection_radius(radius)
    source = _validate_partition(bank)
    target = source["target"].detach().clone()
    if radius == 0:
        return target
    components = _components(target[0].cpu().bool())
    main = max(components, key=lambda c: (len(c), -min(c)))
    grown = torch.zeros_like(target[0], device="cpu", dtype=torch.bool)
    grown.flatten()[main] = True
    background = source["background"][0].detach().cpu().bool()
    # One-cell iterations cannot jump across a protected barrier.
    for _ in range(radius):
        neighbors = F.max_pool2d(grown.float()[None, None], 3, stride=1, padding=1)[0, 0].bool()
        grown |= neighbors & background
    return torch.maximum(target, grown[None].to(target))


def postprocess_masks(bank, *, max_hole_area=0, dilate_radius=0):
    """Fill background-only holes, then optionally dilate the largest target component.

    Protection is immutable. Both zero settings preserve mask values exactly.
    Input banks and observed similarity maps are never modified.
    """
    validate_postprocess_settings(max_hole_area, dilate_radius)
    source = _validate_partition(bank)
    original = {name: mask.detach().clone() for name, mask in source.items()}
    target = original["target"][0].cpu().bool()
    protected = original["protected"][0].cpu().bool()
    background = original["background"][0].cpu().bool()
    updated = target.clone()
    filled = torch.zeros_like(target)
    dilated = torch.zeros_like(target)
    h, w = target.shape
    report = {"enabled": bool(max_hole_area or dilate_radius), "fill_holes_max_area": max_hole_area,
              "mask_dilate_radius": dilate_radius, "external_components_skipped": None,
              "enclosed_holes_total": None, "oversize_holes_skipped": None,
              "protected_holes_blocked": None, "filled_hole_count": 0,
              "dilation_candidate_count": None, "dilation_protected_blocked_count": None}
    if max_hole_area:
        report.update(external_components_skipped=0, enclosed_holes_total=0,
                      oversize_holes_skipped=0, protected_holes_blocked=0)
        for component in _components(~target, connectivity=4):
            if any(index // w in (0, h - 1) or index % w in (0, w - 1) for index in component):
                report["external_components_skipped"] += 1
                continue
            report["enclosed_holes_total"] += 1
            if len(component) > max_hole_area:
                report["oversize_holes_skipped"] += 1
                continue
            if protected.flatten()[component].any():
                report["protected_holes_blocked"] += 1
                continue
            if not background.flatten()[component].all():
                raise ValueError("Unclassified pixels in an enclosed target hole")
            filled.flatten()[component] = True
            report["filled_hole_count"] += 1
        updated |= filled
    if dilate_radius:
        components = _components(updated)
        main = max(components, key=lambda component: (len(component), -min(component)))
        main_mask = torch.zeros_like(updated)
        main_mask.flatten()[main] = True
        candidate = F.max_pool2d(main_mask.float()[None, None], 3, stride=1, padding=1)[0, 0].bool() & ~updated
        report["dilation_candidate_count"] = int(candidate.sum())
        report["dilation_protected_blocked_count"] = int((candidate & protected).sum())
        dilated = candidate & background
        updated |= dilated
    new_target = updated[None].to(original["target"])
    processed = {"target": new_target, "protected": original["protected"].clone(),
                 "background": 1 - new_target - original["protected"]}
    _validate_partition({"masks": processed, "grid": bank["grid"]})
    added_cells = filled | dilated
    added = added_cells[None].to(original["target"])
    if (not torch.equal(processed["protected"], source["protected"])
            or not torch.equal(processed["target"] - source["target"], added)
            or bool((added * source["protected"]).any())):
        raise RuntimeError("Postprocessing violated target/protection ownership")
    result = dict(bank)
    result.update(_bank(processed, tuple(bank["grid"])))
    report.update(filled_token_count=int(filled.sum()), dilated_token_count=int(dilated.sum()),
                  added_token_count=int(added_cells.sum()), protected_changed_token_count=0, partition_valid=True,
                  before=_bank(original, tuple(bank["grid"]))["diagnostics"], after=result["diagnostics"])
    result.update(original_masks=original, added_target_mask=added, postprocess_diagnostics=report)
    return result


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
