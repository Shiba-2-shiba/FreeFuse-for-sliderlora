"""Experimental reclassification with independently supplied semantic anchors."""
import torch
from torch.nn import functional as F

from .experimental_masks import _seed_pool, _prototypes, _CandidateFailure


def anchored_candidate(features, late_logits, anchor_context, config):
    if anchor_context.shape != late_logits.shape or anchor_context.shape[1] != 3:
        raise ValueError("Anchors and late context must share the three-role spatial grid")
    if not torch.isfinite(anchor_context).all() or (anchor_context<0).any():
        raise ValueError("Anchors must be finite nonnegative role probabilities")
    if not torch.allclose(anchor_context.sum(-1),torch.ones(len(anchor_context),device=anchor_context.device),atol=1e-5):
        raise ValueError("Anchor role probabilities must sum to one")
    normalized=F.normalize(features.float()[0],dim=-1)
    late=(late_logits/config["context_temperature"]).softmax(-1)
    roles=("target","protected","background");seeds={};indices={};counts={};similarities=[]
    for role,name in enumerate(roles):
        pool=_seed_pool(anchor_context,role,config)
        centers,chosen=_prototypes(normalized,pool,config["prototypes"])
        similarities.append((normalized@centers.T).amax(-1))
        seed=torch.zeros(len(late),dtype=torch.bool,device=features.device);seed[pool]=True
        seeds[name]=seed;indices[name]=chosen;counts[name]=len(pool)
    feature_scores=torch.stack(similarities,-1)
    scores=(feature_scores/config["feature_temperature"]+late.clamp_min(1e-12).log()).softmax(-1)
    for role,name in enumerate(roles):scores[seeds[name]]=F.one_hot(torch.tensor(role,device=scores.device),3).to(scores)
    if not torch.isfinite(scores).all():raise _CandidateFailure("nonfinite_scores","Anchor scoring produced nonfinite values")
    return {"scores":scores,"seeds":seeds,"prototype_indices":indices,"seed_counts":counts,"roles":roles,
            "growth_allowed":(anchor_context>1/3)&(feature_scores>.25)}


_ROLES = ("target", "protected", "background")


def _validate_anchor(anchor):
    if (not isinstance(anchor, torch.Tensor) or anchor.ndim != 2 or anchor.shape[1] != 3
            or not len(anchor) or not torch.isfinite(anchor).all() or (anchor < 0).any()
            or not torch.allclose(anchor.sum(-1), torch.ones(len(anchor), device=anchor.device), atol=1e-5)):
        raise ValueError("Anchors must be finite three-role probabilities on a nonempty grid")


def build_seed_pools(anchor_context, bootstrap_masks, config):
    """Keep prototype support and hard locks separate, with ordered intersections."""
    _validate_anchor(anchor_context)
    target, protected = (bootstrap_masks[r] for r in _ROLES[:2])
    _binary_masks(target, protected)
    if target.numel() != len(anchor_context) or target.device != anchor_context.device:
        raise ValueError("Bootstrap and anchor grids/devices must match")
    if (target.bool() & protected.bool()).any():
        raise ValueError("Bootstrap target and protected roles overlap")
    full = {name: _seed_pool(anchor_context, role, config) for role, name in enumerate(_ROLES)}
    supported, locks, failures = {}, {}, {}
    for name in _ROLES:
        pool = full[name]
        selected = pool if name == 'background' else pool[bootstrap_masks[name].flatten().bool()[pool]]
        supported[name] = selected.clone()
        if not len(selected):
            failures[name] = 'empty_supported_pool'
        lock = torch.zeros(len(anchor_context), dtype=torch.bool, device=anchor_context.device)
        lock[selected[:3]] = True
        locks[name] = lock
    return {'full': full, 'supported': supported, 'locks': locks, 'supported_failures': failures}


def _normalized_features(features):
    if (not isinstance(features, torch.Tensor) or features.ndim != 3 or features.shape[0] != 1
            or not features.shape[1] or not features.shape[2] or not torch.isfinite(features).all()):
        raise ValueError("Features must be finite [1,N,C] tensors")
    return F.normalize(features.float()[0], dim=-1)


def fit_role_prototypes(features, pools, config):
    normalized = _normalized_features(features)
    if set(pools) != set(_ROLES):
        raise ValueError("Prototype pools must supply all three roles")
    centers, indices, counts = {}, {}, {}
    occupied = torch.zeros(len(normalized), dtype=torch.bool, device=features.device)
    for name in _ROLES:
        pool = pools[name]
        if (not isinstance(pool, torch.Tensor) or pool.ndim != 1 or pool.dtype != torch.long
                or pool.device != features.device or not len(pool)):
            raise ValueError("Invalid or empty prototype pool for " + name)
        if ((pool < 0).any() or (pool >= len(normalized)).any() or len(pool.unique()) != len(pool)
                or occupied[pool].any()):
            raise ValueError("Prototype pool indices are duplicate, overlapping or outside the grid")
        occupied[pool] = True
        centers[name], indices[name] = _prototypes(normalized, pool, config['prototypes'])
        counts[name] = len(pool)
    return {'centers': centers, 'prototype_indices': indices, 'prototype_pool_counts': counts}


def score_with_seed_controls(features, late_logits, anchor_context, prototype_bank, lock_masks, config):
    """Score with supplied prototypes, clamping only explicitly supplied locks."""
    _validate_anchor(anchor_context)
    normalized = _normalized_features(features)
    if (late_logits.shape != anchor_context.shape or len(normalized) != len(anchor_context)
            or late_logits.device != features.device or anchor_context.device != features.device
            or not torch.isfinite(late_logits).all()):
        raise ValueError("Late/anchor/features must share a finite spatial grid and device")
    if set(lock_masks) != set(_ROLES) or set(prototype_bank['centers']) != set(_ROLES):
        raise ValueError("All three roles must have centers and lock masks")
    occupied = torch.zeros(len(normalized), dtype=torch.bool, device=features.device)
    similarities, locks = [], {}
    for name in _ROLES:
        lock, centers = lock_masks[name], prototype_bank['centers'][name]
        if (not isinstance(lock, torch.Tensor) or lock.dtype != torch.bool
                or lock.shape != occupied.shape or lock.device != features.device):
            raise ValueError("Lock masks must be boolean vectors on the feature grid")
        if (occupied & lock).any():
            raise ValueError("Role lock masks overlap")
        occupied |= lock
        if (centers.ndim != 2 or centers.shape[1] != normalized.shape[1] or not len(centers)
                or centers.device != features.device or not torch.isfinite(centers).all()):
            raise ValueError("Prototype centers must be finite and match feature dimensions/device")
        similarities.append((normalized @ centers.T).amax(-1))
        locks[name] = lock.clone()
    feature_scores = torch.stack(similarities, -1)
    late = (late_logits / config['context_temperature']).softmax(-1)
    prelock = (feature_scores / config['feature_temperature'] + late.clamp_min(1e-12).log()).softmax(-1)
    scores = prelock.clone()
    for role, name in enumerate(_ROLES):
        scores[locks[name]] = F.one_hot(torch.tensor(role, device=scores.device), 3).to(scores)
    if not torch.isfinite(scores).all():
        raise _CandidateFailure('nonfinite_scores', 'Seed-controlled scoring produced nonfinite values')
    counts = {name: int(mask.sum()) for name, mask in locks.items()}
    return {'scores': scores, 'prelock_scores': prelock, 'seeds': locks,
            'lock_masks': locks, 'seed_counts': counts, 'lock_counts': dict(counts),
            'prototype_indices': {r: list(prototype_bank['prototype_indices'][r]) for r in _ROLES},
            'prototype_pool_counts': dict(prototype_bank['prototype_pool_counts']), 'roles': _ROLES,
            'growth_allowed': (anchor_context > 1/3) & (feature_scores > .25)}


def _binary_masks(*values):
    first = values[0]
    if not isinstance(first, torch.Tensor) or first.ndim != 3 or first.shape[0] != 1:
        raise ValueError("Masks must have shape [1,H,W]")
    for value in values:
        if (not isinstance(value, torch.Tensor) or value.shape != first.shape
                or value.device != first.device or not torch.isfinite(value).all()
                or not ((value == 0) | (value == 1)).all()):
            raise ValueError("Masks must be finite binary tensors on the same grid/device")


def compose_paired_masks(base_target, slider_target, pilot_target, protected, *, include_pilot):
    """Separate the paired union from the optional historical pilot support."""
    _binary_masks(base_target, slider_target, pilot_target, protected)
    if type(include_pilot) is not bool:
        raise ValueError("include_pilot must be boolean")
    raw = base_target.bool() | slider_target.bool()
    if include_pilot:
        raw = raw | pilot_target.bool()
    target = raw & ~protected.bool()
    if not target.any() or not protected.any():
        raise ValueError("An empty target/protected mask is not a valid candidate")
    return {"target": target.float(), "protected": protected.float().clone(),
            "background": (~(target | protected.bool())).float(),
            "raw_target": raw.float(), "collision": (raw & protected.bool()).float(),
            "added_to_pilot": (target & ~pilot_target.bool()).float(),
            "removed_from_pilot": (~target & pilot_target.bool()).float()}


def apply_diagnostic_release(raw_target, protected, release_mask):
    """Release only explicitly supplied collision cells; never infer anatomy."""
    _binary_masks(raw_target, protected, release_mask)
    collision = raw_target.bool() & protected.bool()
    if (release_mask.bool() & ~collision).any():
        raise ValueError("Release mask must be a subset of the raw/protected collision")
    new_protected = protected.bool() & ~release_mask.bool()
    target = raw_target.bool() & ~new_protected
    if not target.any() or not new_protected.any():
        raise ValueError("Diagnostic release leaves an empty partition role")
    return {"target": target.float(), "protected": new_protected.float(),
            "background": (~(target | new_protected)).float()}
