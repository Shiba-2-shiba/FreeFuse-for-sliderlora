"""Bounded, observation-only mask hypotheses; never used by the production sampler.

The baseline is the unchanged FreeFuse two-stage calculation. All other scores
are *heuristics*, not calibrated probabilities or research-validated defaults.
Multi-prototype construction is an engineering hypothesis. Graph refinement is
inspired by suppressed-affinity/reseeding ideas (iSeg), not an implementation or
validation of that paper. The graph is conditional image-to-image Q/K attention,
not the model's joint text/image attention probabilities.

Defaults deliberately reject weak evidence: context seed confidence .65 and
margin .15; output confidence .55 and margin .10. top_k_ratio caps seed pools;
it does not prescribe object area. Unknown cells route to the base/background
mask but are reported separately from real prompt-background evidence.
"""
from __future__ import annotations

import copy
import math

import torch
from torch.nn import functional as F

from .attention import similarity_maps
from .masks import _bank, generate_masks

VARIANTS = ("baseline", "centroid_control", "multi_proto", "multi_proto_bg", "adaln_bg", "ensemble_bg", "propagated_bg")
_ROLES = ("target", "protected", "background")
_DEFAULTS = dict(collect_step=2, collect_block=18, selected_steps=[1, 2], selected_blocks=[16, 18],
                 top_k_ratio=.2, temperature=10000., prototypes=3, seed_confidence=.65, seed_margin=.15,
                 mask_confidence=.55, mask_margin=.10, context_temperature=1., feature_temperature=.15,
                 propagation_iterations=1, propagation_heads=[0], affinity_chunk_size=128,
                 affinity_threshold=.05, propagation_strength=.5)


def _integer(name, value, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")


def validate_experimental_config(config):
    """Validate a small JSON config; return an independent, fully defaulted dict."""
    if not isinstance(config, dict) or set(config) - set(_DEFAULTS):
        raise ValueError("experimental config must be a dict with supported keys only")
    result = copy.deepcopy(_DEFAULTS)
    result.update(copy.deepcopy(config))
    for name, low, high in (("collect_step", 1, 1000), ("collect_block", 0, 255), ("prototypes", 1, 8),
                           ("propagation_iterations", 0, 3), ("affinity_chunk_size", 1, 512)):
        _integer(name, result[name], low, high)
    for name, low, high, strict_low in (("top_k_ratio", 0, 1, True), ("temperature", 1e-6, 1e9, False),
                                      ("seed_confidence", .5, 1, False), ("seed_margin", 0, 1, False),
                                      ("mask_confidence", .5, 1, False), ("mask_margin", 0, 1, False),
                                      ("context_temperature", 1e-6, 1e9, False), ("feature_temperature", 1e-6, 1e9, False),
                                      ("affinity_threshold", 0, 1, False), ("propagation_strength", 0, 1, False)):
        value = result[name]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or not low <= value <= high or (strict_low and value == low)):
            raise ValueError(f"{name} is outside its finite numeric domain")
        result[name] = float(value)
    for name, low, high in (("selected_steps", 1, result["collect_step"]), ("selected_blocks", 0, 255), ("propagation_heads", 0, 255)):
        values = result[name]
        if not isinstance(values, list) or not values or len(values) > 32:
            raise ValueError(f"{name} must be a nonempty bounded JSON list")
        for value in values:
            _integer(name, value, low, high)
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must contain unique indices")
        result[name] = sorted(values)
    if result["collect_step"] not in result["selected_steps"] or result["collect_block"] not in result["selected_blocks"]:
        raise ValueError("selected steps/blocks must include the primary collection tap")
    if len(result["selected_steps"]) * len(result["selected_blocks"]) > 6:
        raise ValueError("At most six observation taps are supported")
    return result


class _CandidateFailure(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def _failure(error):
    return {"status": "failed", "error_code": getattr(error, "code", "invalid_candidate"), "error": str(error)}


def _cpu(value):
    return value.detach().float().cpu().clone()


def _permission_mask(mask, n, t, heads, device):
    if mask is None:
        return None, None
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool and not mask.is_floating_point():
        raise ValueError("attention mask must be boolean permissions or additive floating biases")
    if mask.ndim == 2:
        mask = mask[None]
    elif mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 3 or mask.shape[0] not in (1, heads) or mask.shape[-2:] not in ((n, n), (n + t, n + t)):
        raise ValueError("Unsupported image-image or joint attention mask shape")
    if mask.is_floating_point() and (torch.isnan(mask).any() or torch.isposinf(mask).any()):
        raise ValueError("attention mask has NaN or positive infinity")
    text_permissions = None
    if mask.shape[-1] == n + t:
        text_permissions = mask[:, t:, :t].detach().to(device)
        mask = mask[:, t:, t:]
    return mask.detach().to(device), text_permissions


def _seed_pool(context, role, config):
    others = torch.cat((context[:, :role], context[:, role + 1:]), dim=1).amax(dim=1)
    margin = context[:, role] - others
    eligible = (context[:, role] >= config["seed_confidence"]) & (margin >= config["seed_margin"]) & (margin > 0)
    indices = torch.where(eligible)[0]
    if not len(indices):
        raise _CandidateFailure("empty_seed_pool", f"No high-confidence context seeds for {_ROLES[role]}")
    cap = max(1, int(len(context) * config["top_k_ratio"]))
    # Stable row-major tie breaks are explicit; no RNG or equal-area allocation.
    rank = torch.argsort(margin[indices], descending=True, stable=True)
    return indices[rank[:cap]]


def _prototypes(features, pool, count):
    points = features[pool]
    count = min(count, len(pool))
    if count == 1:
        return F.normalize(points.mean(dim=0, keepdim=True), dim=-1), [int(pool[0])]
    selected = [0]
    distance = 1 - points @ points[0]
    for _ in range(1, count):
        distance[selected] = -1
        choice = int(torch.argmax(distance))
        if float(distance[choice]) <= 1e-6:
            break
        selected.append(choice)
        distance = torch.minimum(distance, 1 - points @ points[choice])
    centers = points[selected]
    assignment = (points @ centers.T).argmax(dim=1)
    prototypes = []
    for index in range(len(selected)):
        members = points[assignment == index]
        if not len(members):
            members = centers[index:index + 1]
        prototypes.append(F.normalize(members.mean(dim=0), dim=-1))
    return torch.stack(prototypes), [int(pool[index]) for index in selected]


def _candidate(features, context_logits, roles, config, prototype_count):
    if len(roles) == 3 and context_logits.shape[1] != 3:
        raise _CandidateFailure("missing_background_tokens", "Background variants require exact real prompt-token positions")
    logits = context_logits[:, :len(roles)]
    context = (logits / config["context_temperature"]).softmax(dim=-1)
    normalized = F.normalize(features.float()[0], dim=-1)
    similarities, seeds, prototype_indices, counts = [], {}, {}, {}
    for role, name in enumerate(roles):
        pool = _seed_pool(context, role, config)
        centers, indices = _prototypes(normalized, pool, prototype_count)
        similarities.append((normalized @ centers.T).amax(dim=-1))
        seed = torch.zeros(len(context), dtype=torch.bool, device=features.device)
        seed[pool] = True
        seeds[name] = seed
        prototype_indices[name] = indices
        counts[name] = int(len(pool))
    feature_scores = torch.stack(similarities, dim=-1)
    # Matching calibration for every new variant, including the centroid control.
    scores = (feature_scores / config["feature_temperature"] + context.clamp_min(1e-12).log()).softmax(dim=-1)
    for role, name in enumerate(roles):
        scores[seeds[name]] = F.one_hot(torch.tensor(role, device=scores.device), len(roles)).to(scores)
    if not torch.isfinite(scores).all():
        raise _CandidateFailure("nonfinite_scores", "Prototype scoring produced nonfinite values")
    return {"scores": scores, "seeds": seeds, "prototype_indices": prototype_indices,
            "seed_counts": counts, "roles": roles,
            "growth_allowed": (context > 1 / len(roles)) & (feature_scores > .25)}


def _labels(scores, config):
    best, labels = scores.max(dim=-1)
    runner_up = scores.topk(2, dim=-1).values[:, 1]
    confident = (best >= config["mask_confidence"]) & (best - runner_up >= config["mask_margin"]) & (best > runner_up)
    return torch.where(confident, labels, -1)


def _candidate_bank(candidate, grid, config):
    scores, roles = candidate["scores"], candidate["roles"]
    labels = _labels(scores, config)
    masks = {name: _cpu((labels == role).reshape(1, *grid)) for role, name in enumerate(roles)}
    for name in ("target", "protected"):
        if not masks[name].any():
            raise _CandidateFailure("empty_mask", f"Automatic {name} candidate is empty; no fallback mask was substituted")
    real_background = masks.get("background", torch.zeros((1, *grid)))
    masks["background"] = 1 - masks["target"] - masks["protected"]
    uncertain = _cpu((labels == -1).reshape(1, *grid))
    return _bank(masks, grid, mode="experimental", raw_maps={name: _cpu(scores[:, role][None]) for role, name in enumerate(roles)},
                 uncertain_mask=uncertain, real_background_mask=real_background,
                 seed_masks={name: _cpu(value.reshape(1, *grid)) for name, value in candidate["seeds"].items()})


def _summary(candidate, bank):
    return {"status": "ok", "confidence_semantics": "heuristic_not_calibrated", "seed_counts": candidate["seed_counts"],
            "prototype_indices": candidate["prototype_indices"], "uncertain_tokens": int(bank["uncertain_mask"].sum()),
            "real_background_tokens": int(bank["real_background_mask"].sum()), "diagnostics": bank["diagnostics"]}


def _propagate(candidate, q, k, permissions, config):
    scores = candidate["scores"].clone()
    initial = scores.clone()
    labels = _labels(initial, config)
    barrier = labels >= 0
    n, heads, width = q.shape[1:]
    selected = config["propagation_heads"]
    chunk = min(config["affinity_chunk_size"], n)
    weak_removed = zero_rows = edges = 0
    for _iteration in range(config["propagation_iterations"]):
        updated = torch.empty_like(scores)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            affinity = q.new_zeros((stop - start, n), dtype=torch.float32)
            for head in selected:
                logits = q[0, start:stop, head].float() @ k[0, :, head].float().T / math.sqrt(width)
                if not torch.isfinite(logits).all():
                    raise _CandidateFailure("nonfinite_affinity", "Image Q/K affinity overflowed; propagation was not substituted")
                if permissions is not None:
                    permitted = permissions[0 if permissions.shape[0] == 1 else head, start:stop]
                    logits = logits.masked_fill(~permitted, -torch.inf) if permitted.dtype == torch.bool else logits + permitted.float()
                if torch.isnan(logits).any() or torch.isposinf(logits).any():
                    raise _CandidateFailure("nonfinite_affinity", "Masked image Q/K affinity has nonfinite positive values")
                affinity += logits.softmax(dim=-1).nan_to_num(0.) / len(selected)
            # ReLU subtraction relative to each row maximum; scale-free at large N.
            threshold = affinity.amax(dim=-1, keepdim=True) * config["affinity_threshold"]
            suppressed = (affinity - threshold).clamp_min(0)
            weak_removed += int(((affinity > 0) & (suppressed == 0)).sum())
            # Confident opposing role evidence cannot exchange labels across an edge.
            conflict = ((labels[start:stop, None] >= 0) & (labels[None] >= 0)
                        & (labels[start:stop, None] != labels[None]))
            suppressed.masked_fill_(conflict, 0)
            mass = suppressed.sum(dim=-1, keepdim=True)
            empty = mass[:, 0] == 0
            zero_rows += int(empty.sum())
            edges += int((suppressed > 0).sum())
            walk = (suppressed / mass.clamp_min(1e-30)) @ scores
            walk[empty] = initial[start:stop][empty]  # restart, never invent a prohibited self-edge
            mixed = (1 - config["propagation_strength"]) * initial[start:stop] + config["propagation_strength"] * walk
            # No growth into a role unsupported by both context and feature evidence.
            mixed = torch.where(candidate["growth_allowed"][start:stop], mixed, torch.minimum(mixed, initial[start:stop]))
            total = mixed.sum(dim=-1, keepdim=True)
            mixed = torch.where(total > 0, mixed / total.clamp_min(1e-30), initial[start:stop])
            unsupported_growth = ((mixed > initial[start:stop] + 1e-7) & ~candidate["growth_allowed"][start:stop]).any(dim=-1)
            mixed[unsupported_growth] = initial[start:stop][unsupported_growth]
            mixed[barrier[start:stop]] = initial[start:stop][barrier[start:stop]]
            updated[start:stop] = mixed
        scores = updated
        for role, name in enumerate(candidate["roles"]):
            scores[candidate["seeds"][name]] = initial[candidate["seeds"][name]]
    result = dict(candidate, scores=scores)
    final_labels = _labels(scores, config)
    info = {"kind": "conditional_image_image_qk", "heads": selected, "head_selection": "provisional_not_validated",
            "iterations": config["propagation_iterations"], "affinity_threshold_semantics": "relu(weight - threshold * row_max), then row_normalize",
            "empty_row_policy": "restart_initial_scores", "mask_semantics": "boolean_true_allowed_or_additive_bias",
            "largest_affinity_chunk_shape": [chunk, n], "weak_edges_removed": weak_removed, "retained_edges_across_iterations": edges,
            "zero_rows_across_iterations": zero_rows, "seed_changes": 0,
            "barrier_violations": int((final_labels[barrier] != labels[barrier]).sum()),
            "growth_support": "context_above_uniform_and_cosine_above_0.25; unsupported_postnormalization_growth_restarts_row"}
    return result, info


class ExperimentalMaskAccumulator:
    """Compute on-device at observation time; retain only detached CPU score maps."""

    def __init__(self, config, grid, positions):
        self.config = validate_experimental_config(config)
        if not isinstance(grid, (tuple, list)) or len(grid) != 2:
            raise ValueError("grid must contain two positive integers")
        for value in grid:
            _integer("grid", value, 1, 65536)
        self.grid = tuple(grid)
        if not isinstance(positions, dict) or set(positions) != set(_ROLES):
            raise ValueError("positions require target, protected, and explicit background entries")
        seen = set()
        self.positions = {}
        for role in _ROLES:
            indices = positions[role]
            if not isinstance(indices, (tuple, list)) or (role != "background" and not indices):
                raise ValueError(f"Invalid {role} positions")
            for index in indices:
                _integer("text position", index, 0, 1000000)
            if len(set(indices)) != len(indices) or set(indices) & seen:
                raise ValueError("Role positions must be unique and disjoint")
            self.positions[role] = tuple(indices)
            seen.update(indices)
        self.evidence_maps = {}
        self._observations = {}
        self._banks = {}
        self._reports = {}
        self._ensemble = {}
        self._text_length = None

    def _validate_inputs(self, step, block, img_q, txt_k, img_k, attention_output, attention_input, attention_mask):
        _integer("step", step, 1, 1000)
        _integer("block", block, 0, 255)
        if step not in self.config["selected_steps"] or block not in self.config["selected_blocks"]:
            raise ValueError("Observation tap is not selected")
        if (step, block) in self._observations:
            raise ValueError("Observation tap was already collected")
        values = (img_q, txt_k, img_k, attention_output, attention_input)
        for value in values:
            if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("Observation tensors must be finite floating tensors")
        if (img_q.ndim != 4 or img_q.shape[0] != 1 or min(img_q.shape[1:]) < 1
                or img_q.shape[1] != math.prod(self.grid) or img_k.shape != img_q.shape
                or txt_k.ndim != 4 or txt_k.shape[0] != 1 or txt_k.shape[1] < 1 or txt_k.shape[2:] != img_q.shape[2:]
                or attention_output.ndim != 3 or attention_output.shape[:2] != img_q.shape[:2] or attention_output.shape[-1] < 1
                or attention_input.shape != attention_output.shape or any(value.device != img_q.device for value in values)):
            raise ValueError("Unexpected single-image Q/K/feature layout, device or grid")
        if any(index >= txt_k.shape[1] for indices in self.positions.values() for index in indices):
            raise ValueError("Text position exceeds exact runtime token count")
        if self._text_length is not None and self._text_length != txt_k.shape[1]:
            raise ValueError("Runtime text token count changed between taps")
        if max(self.config["propagation_heads"]) >= img_q.shape[2]:
            raise ValueError("propagation_heads exceeds expanded Q/K head count")
        permissions = _permission_mask(attention_mask, img_q.shape[1], txt_k.shape[1], img_q.shape[2], img_q.device)
        self._text_length = txt_k.shape[1]
        return permissions

    def _context(self, q, keys, step, block, permissions):
        logits = []
        summary = {"head_count": q.shape[2], "semantic_head_selection": "all_allowed_heads_and_tokens_mean",
                   "text_permission_source": "joint_mask" if permissions is not None else "unprovided_assume_all_text_links",
                   "logit_scale": "q_dot_k_div_sqrt_head_dim", "context_temperature_semantics": "provisional_heuristic_scale",
                   "role_tokens": {}}
        for role in _ROLES:
            positions = self.positions[role]
            if not positions:
                summary["role_tokens"][role] = []
                continue
            values = torch.einsum("nhd,thd->hnt", q[0].float(), keys[0, list(positions)].float()) / math.sqrt(q.shape[-1])
            if not torch.isfinite(values).all():
                raise ValueError("Context Q/K products are nonfinite")
            allowed = torch.ones_like(values, dtype=torch.bool)
            if permissions is not None:
                permitted = permissions[:, :, list(positions)]
                if permitted.dtype == torch.bool:
                    allowed &= permitted
                else:
                    allowed &= torch.isfinite(permitted)
                    values = values + permitted.masked_fill(~torch.isfinite(permitted), 0).float()
            values = values.masked_fill(~allowed, 0)
            counts = allowed.sum(dim=(0, 2))
            role_logits = values.sum(dim=(0, 2)) / counts.clamp_min(1)
            logits.append(role_logits.masked_fill(counts == 0, -1e9))
            prefix = f"step_{step}.block_{block}.{role}"
            self.evidence_maps[f"{prefix}.head_scores"] = _cpu(values.sum(dim=-1) / allowed.sum(dim=-1).clamp_min(1))
            self.evidence_maps[f"{prefix}.token_scores"] = _cpu((values.sum(dim=0) / allowed.sum(dim=0).clamp_min(1)).T)
            self.evidence_maps[f"{prefix}.head_allowed_fraction"] = _cpu(allowed.float().mean(dim=-1))
            self.evidence_maps[f"{prefix}.token_allowed_fraction"] = _cpu(allowed.float().mean(dim=0).T)
            token_summary = []
            for offset, token in enumerate(positions):
                per_head = values[:, :, offset]
                token_summary.append({"token_index": token, "head_spatial_mean": per_head.mean(dim=-1).cpu().tolist(),
                                      "head_spatial_std": per_head.std(dim=-1, unbiased=False).cpu().tolist(),
                                      "mean_head_disagreement": float(per_head.std(dim=0, unbiased=False).mean())})
            summary["role_tokens"][role] = token_summary
        stacked = torch.stack(logits, dim=-1)
        context = (stacked / self.config["context_temperature"]).softmax(dim=-1)
        summary.update(max_absolute_allowed_logit=float(stacked[stacked > -1e9].abs().max()) if (stacked > -1e9).any() else 0.,
                       context_saturation_fraction=float((context.amax(dim=-1) >= .99).float().mean()),
                       feature_logit_bound=1 / self.config["feature_temperature"])
        return stacked, summary

    def _save_candidate(self, name, candidate, extra=None):
        try:
            bank = _candidate_bank(candidate, self.grid, self.config)
            self._banks[name] = bank
            self._reports[name] = _summary(candidate, bank)
            if extra:
                self._reports[name].update(extra)
        except _CandidateFailure as error:
            self._reports[name] = _failure(error)

    @torch.no_grad()
    def observe(self, *, step, block, img_q, txt_k, img_k, attention_output, attention_input, attention_mask=None):
        permissions, text_permissions = self._validate_inputs(step, block, img_q, txt_k, img_k, attention_output, attention_input, attention_mask)
        context, context_summary = self._context(img_q, txt_k, step, block, text_permissions)
        primary = step == self.config["collect_step"] and block == self.config["collect_block"]
        self._observations[step, block] = {"step": step, "block": block, "image_tokens": img_q.shape[1],
                                          "text_tokens": txt_k.shape[1], "context_summary": context_summary}
        if primary:
            try:
                raw = similarity_maps(img_q, txt_k, attention_output, {role: self.positions[role] for role in _ROLES[:2]},
                                      self.config["top_k_ratio"], self.config["temperature"])
                baseline = generate_masks(raw, self.grid)
                baseline["masks"] = {name: _cpu(value) for name, value in baseline["masks"].items()}
                baseline.update(uncertain_mask=baseline["masks"]["background"].clone(),
                                real_background_mask=torch.zeros((1, *self.grid)), seed_masks={})
                self._banks["baseline"] = baseline
                self._reports["baseline"] = {"status": "ok", "algorithm": "unchanged_similarity_maps_plus_generate_masks",
                                              "background_semantics": "legacy_residual_not_real_background", "diagnostics": baseline["diagnostics"]}
            except ValueError as error:
                self._reports["baseline"] = _failure(error)
            for name, features, roles, count in (("centroid_control", attention_output, _ROLES[:2], 1),
                                                 ("multi_proto", attention_output, _ROLES[:2], self.config["prototypes"]),
                                                 ("adaln_bg", attention_input, _ROLES, self.config["prototypes"])):
                try:
                    candidate = _candidate(features, context, roles, self.config, count)
                    self._save_candidate(name, candidate)
                except _CandidateFailure as error:
                    self._reports[name] = _failure(error)
        try:
            candidate = _candidate(attention_output, context, _ROLES, self.config, self.config["prototypes"])
            # Only detached scores/seeds remain after this observation returns.
            self._ensemble[step, block] = {"scores": _cpu(candidate["scores"]), "roles": _ROLES,
                                           "seeds": {name: _cpu(value).bool() for name, value in candidate["seeds"].items()},
                                           "seed_counts": candidate["seed_counts"], "prototype_indices": candidate["prototype_indices"]}
            if primary:
                self._save_candidate("multi_proto_bg", candidate)
                try:
                    propagated, info = _propagate(candidate, img_q, img_k, permissions, self.config)
                    self._save_candidate("propagated_bg", propagated, {"propagation": info})
                except _CandidateFailure as error:
                    self._reports["propagated_bg"] = _failure(error)
        except _CandidateFailure as error:
            self._ensemble[step, block] = _failure(error)
            if primary:
                self._reports["multi_proto_bg"] = _failure(error)
                self._reports["propagated_bg"] = _failure(error)

    def finalize(self):
        expected = [(step, block) for step in self.config["selected_steps"] for block in self.config["selected_blocks"]]
        missing = [list(tap) for tap in expected if tap not in self._observations]
        if missing:
            self._reports["ensemble_bg"] = _failure(_CandidateFailure("missing_taps", f"Missing selected observation taps: {missing}"))
        else:
            failed = next((value for key in expected if (value := self._ensemble[key]).get("status") == "failed"), None)
            if failed:
                self._reports["ensemble_bg"] = dict(failed)
            else:
                candidates = [self._ensemble[key] for key in expected]
                seeds = {role: torch.stack([value["seeds"][role] for value in candidates]).all(dim=0) for role in _ROLES}
                ensemble = {"scores": torch.stack([value["scores"] for value in candidates]).mean(dim=0), "roles": _ROLES,
                            "seeds": seeds, "seed_counts": {role: int(mask.sum()) for role, mask in seeds.items()},
                            "prototype_indices": {role: [] for role in _ROLES}}
                self._save_candidate("ensemble_bg", ensemble, {"aggregation": "arithmetic_mean_of_same_output_feature_multi_proto_bg_scores",
                                                               "tap_count": len(expected), "seeds": "intersection_across_taps"})
        for name in VARIANTS:
            if name not in self._reports:
                self._reports[name] = _failure(_CandidateFailure("missing_primary_tap", "Primary observation tap was not collected"))
        report = {"config": copy.deepcopy(self.config), "positions": {role: list(value) for role, value in self.positions.items()},
                  "grid": list(self.grid), "variants": copy.deepcopy(self._reports), "missing_taps": missing,
                  "observations": [self._observations[key] for key in sorted(self._observations)],
                  "limitations": "heuristic scores and provisional thresholds; no calibrated confidence, equal-area quotas, forced body completion, or validation claims",
                  "provenance": {"multi_prototype": "engineering_hypothesis", "propagation": "iSeg_inspired_suppressed_affinity_and_reseeding_not_paper_reproduction"}}
        return dict(self._banks), report
