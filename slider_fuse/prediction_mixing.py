"""Exact binary selection of compatible sampler predictions."""
from __future__ import annotations

import torch


def _validate_prediction(value):
    if (not isinstance(value, torch.Tensor) or value.ndim not in (4, 5)
            or value.shape[0] != 1 or (value.ndim == 5 and value.shape[2] != 1)
            or min(value.shape) < 1 or not value.is_floating_point()
            or not torch.isfinite(value).all()):
        raise ValueError("Prediction must be a finite batch1 still-image floating tensor")


def prediction_mask(token_mask: torch.Tensor, prediction: torch.Tensor, *, patch: int) -> torch.Tensor:
    _validate_prediction(prediction)
    if isinstance(patch, bool) or not isinstance(patch, int) or patch < 1:
        raise ValueError("Patch must be a positive integer")
    h, w = prediction.shape[-2:]
    if (not isinstance(token_mask, torch.Tensor)
            or token_mask.shape != (1, (h + patch - 1) // patch, (w + patch - 1) // patch)
            or not torch.isfinite(token_mask).all()
            or not ((token_mask == 0) | (token_mask == 1)).all()):
        raise ValueError("Token mask must be binary and match the prediction patch grid")
    mask = token_mask.to(device=prediction.device, dtype=torch.bool)
    mask = mask.repeat_interleave(patch, -2).repeat_interleave(patch, -1)[..., :h, :w]
    return mask.reshape((1,) * (prediction.ndim - 2) + (h, w))


def mix_predictions(base: torch.Tensor, slider: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    _validate_prediction(base); _validate_prediction(slider)
    if base.shape != slider.shape or base.dtype != slider.dtype or base.device != slider.device:
        raise ValueError("Branch prediction shape, dtype and device must match")
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.device != base.device
            or mask.shape != (1,) * (base.ndim - 2) + base.shape[-2:]):
        raise ValueError("Prediction selector must be a matching binary spatial mask")
    return torch.where(mask, slider, base)
