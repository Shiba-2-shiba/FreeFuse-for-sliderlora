"""Observation-only step/block sweep, reusing the exact existing k3 computation."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import uuid

import torch

from .experimental_artifacts import save_experimental_artifacts, validate_experimental_prefix
from .experimental_collection import _collect_mask_observations, select_experimental_mask
from .experimental_masks import (ExperimentalMaskAccumulator, VARIANTS, _candidate,
                                 _CandidateFailure, _failure, validate_experimental_config)


def validate_sweep_config(value):
    if not isinstance(value, dict):
        raise ValueError("Sweep configuration must be an object")
    selected = value.get("selected_steps", [1, 2, 3])
    blocks = value.get("selected_blocks", list(range(14, 28)))
    if selected != [1, 2, 3] or any(isinstance(step, bool) for step in selected):
        raise ValueError("Sweep observes exactly steps [1,2,3]")
    if (not isinstance(blocks, list) or not blocks or len(blocks) > 28
            or any(isinstance(b, bool) or not isinstance(b, int) or not 0 <= b < 28 for b in blocks)
            or len(set(blocks)) != len(blocks)):
        raise ValueError("Sweep blocks must be unique integer indices in 0..27")
    primary = value.get("collect_block", 18 if 18 in blocks else blocks[0])
    if primary not in blocks or value.get("collect_step", 3) != 3:
        raise ValueError("Sweep primary block must be selected and prefix must be three steps")
    normalized = validate_experimental_config({**value, "collect_step": 3, "collect_block": primary,
                                               "selected_steps": [3], "selected_blocks": [primary]})
    normalized.update(selected_steps=[1, 2, 3], selected_blocks=sorted(blocks))
    return normalized


def _parse_config(text):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate sweep configuration key: {key}")
            result[key] = value
        return result
    if not isinstance(text, str):
        raise ValueError("config_json must be a JSON object string")
    return validate_sweep_config(json.loads(text, object_pairs_hook=unique_pairs))


class SweepAccumulator:
    """Compute each primary k3 mask independently; retain detached spatial evidence."""
    def __init__(self, config, grid, positions):
        self.config = validate_sweep_config(config)
        self.grid, self.positions = tuple(grid), positions
        self.taps = {}
        self.evidence_maps = {}

    @torch.no_grad()
    def observe(self, *, step, block, **data):
        key = f"step_{step}.block_{block}"
        if step not in self.config["selected_steps"] or block not in self.config["selected_blocks"] or key in self.taps:
            raise ValueError("Unexpected or repeated sweep tap")
        config = {**self.config, "collect_step": step, "collect_block": block,
                  "selected_steps": [step], "selected_blocks": [block]}
        inner = ExperimentalMaskAccumulator(config, self.grid, self.positions)
        _, permissions = inner._validate_inputs(step, block, **data)
        context, context_summary = inner._context(data["img_q"], data["txt_k"], step, block, permissions)
        try:
            candidate = _candidate(data["attention_output"], context,
                                   ("target", "protected", "background"), config, config["prototypes"])
            inner._save_candidate("multi_proto_bg", candidate)
        except _CandidateFailure as error:
            inner._reports["multi_proto_bg"] = _failure(error)
        variants = {name: _failure(_CandidateFailure("not_requested", "Sweep computes only multi_proto_bg")) for name in VARIANTS}
        variants.update(inner._reports)
        algorithm = {"config": copy.deepcopy(config), "positions": copy.deepcopy(self.positions),
                     "grid": list(self.grid), "variants": variants, "missing_taps": [],
                     "observations": [{"step": step, "block": block, "context_summary": context_summary}],
                     "limitations": "Same provisional k3 scores/thresholds as existing masks; no calibrated confidence or segmentation labels"}
        self.taps[key] = {"banks": inner._banks, "algorithm": algorithm, "evidence_maps": inner.evidence_maps}

    def finalize(self):
        expected = {f"step_{s}.block_{b}" for s in self.config["selected_steps"] for b in self.config["selected_blocks"]}
        missing = sorted(expected - set(self.taps))
        if missing:
            raise RuntimeError(f"Missing sweep taps: {missing}")
        return self.taps, {"config": copy.deepcopy(self.config), "missing_taps": missing,
                           "tap_status": {key: tap["algorithm"]["variants"]["multi_proto_bg"] for key, tap in self.taps.items()}}


def collect_attention_sweep(model, positive, negative, prompt_info, subjects, latent, *, seed=42, steps=8,
                            trial_id=0, config_json="{}", background_phrase="", background_occurrence=0,
                            audit_tensors=False):
    config = _parse_config(config_json)
    shared = _collect_mask_observations(model, positive, negative, prompt_info, subjects, latent,
        seed=seed, steps=steps, trial_id=trial_id, config=config, collection_steps=3,
        accumulator_factory=SweepAccumulator, background_phrase=background_phrase,
        background_occurrence=background_occurrence, audit_tensors=audit_tensors)
    report = shared["report"]
    report.update(sweep_schema_version=1, collection_steps=3,
                  timing_scope="One shared three-forward sweep including synchronous per-tap k3/observation; algorithm time is an overlapping subset",
                  nfe_scope="Three model calls for the entire sweep, never sum this across tap suites")
    taps = {}
    for key, tap in shared["banks"].items():
        step, block = map(int, (key.split(".")[0].split("_")[1], key.split(".")[1].split("_")[1]))
        per_tap = copy.deepcopy(report)
        per_tap.update(primary_tap={"step": step, "block": block}, shared_collection_id=report["run_id"],
                       normalized_config=tap["algorithm"]["config"], algorithm=tap["algorithm"],
                       tap_manifest=[row for row in report["tap_manifest"] if (row["step"], row["block"]) == (step, block)])
        taps[key] = {"experiment_schema_version": 1, "banks": tap["banks"], "report": per_tap,
                     "evidence_maps": tap["evidence_maps"]}
    return {"sweep_schema_version": 1, "report": report, "taps": taps}


def select_attention_sweep(sweep, step, block):
    if not isinstance(sweep, dict) or sweep.get("sweep_schema_version") != 1:
        raise ValueError("Unsupported attention sweep")
    if isinstance(step, bool) or not isinstance(step, int) or isinstance(block, bool) or not isinstance(block, int):
        raise ValueError("Sweep tap indices must be integers")
    key = f"step_{step}.block_{block}"
    if key not in sweep["taps"]:
        raise ValueError(f"Sweep tap was not collected: {key}")
    return select_experimental_mask(sweep["taps"][key], "multi_proto_bg")


def save_attention_sweep(directory, prefix, sweep):
    validate_experimental_prefix(prefix)
    if sweep.get("sweep_schema_version") != 1:
        raise ValueError("Unsupported attention sweep")
    root = Path(directory).resolve()
    bundle = root / f"{prefix}_{uuid.uuid4().hex}"
    bundle.mkdir(mode=0o700, exist_ok=False)
    taps = {}
    for key, suite in sweep["taps"].items():
        safe_key = key.replace(".", "_")
        saved = save_experimental_artifacts(bundle, safe_key, suite)
        manifest = bundle / saved["manifest"]
        taps[key] = {"manifest": saved["manifest"], "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                     "status": suite["report"]["algorithm"]["variants"]["multi_proto_bg"]}
    manifest = {"sweep_schema_version": 1, "complete": True, "report": sweep["report"], "taps": taps}
    with (bundle / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    return {"bundle": bundle.name, "manifest": f"{bundle.name}/manifest.json", "tap_count": len(taps)}
