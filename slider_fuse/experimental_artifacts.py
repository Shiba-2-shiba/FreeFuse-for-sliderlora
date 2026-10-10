"""Save VAE-free experiment evidence in a separate versioned artifact schema."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import uuid

import torch

from .experimental_masks import VARIANTS
from .masks import _validate_partition
from .sampling import tensor_hash


def validate_experimental_prefix(prefix):
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", prefix):
        raise ValueError("Experiment filename_prefix must be a flat safe name containing only letters, digits, _ and -")
    return prefix


def _snapshot_tensors(suite):
    if not isinstance(suite, dict) or suite.get("experiment_schema_version") != 1:
        raise ValueError("Unsupported experimental mask suite")
    banks = suite.get("banks")
    if not isinstance(banks, dict) or set(banks) - set(VARIANTS):
        raise ValueError("Experimental suite has an invalid banks mapping or unknown variants")
    tensors = {}; png_keys = []
    statuses = suite.get("report", {}).get("algorithm", {}).get("variants", {})
    if set(statuses) != set(VARIANTS) or any(status.get("status") not in ("ok", "failed") for status in statuses.values()):
        raise ValueError("Experimental suite must record candidate statuses")
    if {name for name,status in statuses.items() if status.get("status") == "ok"} != set(banks):
        raise ValueError("Successful candidate set differs from saved banks")
    def add(key, value, *, shape=None, binary=False):
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.requires_grad:
            raise ValueError("Save only owned detached CPU map tensors")
        if shape is not None and tuple(value.shape) != tuple(shape):
            raise ValueError(f"Unexpected map shape for {key}")
        if value.ndim not in (2, 3) or not torch.isfinite(value).all():
            raise ValueError(f"Expected finite spatial map tensor for {key}")
        if binary and not ((value == 0) | (value == 1)).all():
            raise ValueError(f"Expected binary map tensor for {key}")
        tensors[key] = value.detach().float().contiguous().clone()
        if binary: png_keys.append(key)
    for variant, bank in banks.items():
        if statuses.get(variant, {}).get("status") != "ok":
            raise ValueError(f"Candidate {variant} has no successful algorithm status")
        masks = _validate_partition(bank)
        grid = tuple(bank["grid"]); n = grid[0] * grid[1]
        for role, mask in masks.items():
            add(f"{variant}.masks.{role}", mask, shape=(1, *grid), binary=True)
        for group in ("raw_maps", "seed_masks"):
            maps = bank.get(group)
            required = {"target", "protected"} if group == "raw_maps" else set()
            if group == "raw_maps" and variant.endswith("_bg"):
                required.add("background")
            if not isinstance(maps, dict) or set(maps) - {"target", "protected", "background"} or not required <= set(maps):
                raise ValueError(f"Missing or unexpected {group} roles")
            for role, value in maps.items():
                shape = (1, n) if group == "raw_maps" else (1, *grid)
                add(f"{variant}.{group}.{role}", value, shape=shape, binary=group == "seed_masks")
        for key in ("uncertain_mask", "real_background_mask"):
            if key not in bank:
                raise ValueError(f"Missing required {key}")
            add(f"{variant}.{key}", bank[key], shape=(1,*grid), binary=True)
        if not torch.equal(bank["uncertain_mask"] + bank["real_background_mask"], masks["background"]):
            raise ValueError("Uncertain and real background masks must partition the routing background")
    evidence = suite.get("evidence_maps", {})
    if not isinstance(evidence, dict):
        raise ValueError("Evidence maps must be a mapping")
    for key, value in evidence.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            raise ValueError("Unsafe evidence map name")
        add("evidence." + key, value)
    return tensors, png_keys


def save_experimental_artifacts(directory, filename_prefix, suite):
    """Exclusively create one new bundle and remove it on handled write failures."""
    from PIL import Image
    from safetensors.torch import save_file
    started = time.perf_counter()
    validate_experimental_prefix(filename_prefix)
    tensors, png_keys = _snapshot_tensors(suite)
    report = copy.deepcopy(suite["report"])
    # Validate JSON before creating a directory, and reject NaN/Inf metadata.
    json.dumps(report, allow_nan=False)
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError("Experiment output directory must already exist")
    artifact_id = uuid.uuid4().hex
    bundle_name = f"{filename_prefix}_{artifact_id}"
    bundle = root / bundle_name
    bundle.mkdir(mode=0o700, exist_ok=False)
    artifacts = {"manifest": f"{bundle_name}/manifest.json", "tensors": f"{bundle_name}/maps.safetensors", "mask_pngs": {}}
    try:
        for key in png_keys:
            name = key + ".png"
            value = tensors[key]
            with (bundle/name).open("xb") as output:
                Image.fromarray((value[0].numpy()*255).astype("uint8"), mode="L").save(output, format="PNG")
            artifacts["mask_pngs"][key] = f"{bundle_name}/{name}"
        tensor_path = bundle/"maps.safetensors"
        # A newly exclusively created bundle owns this as-yet nonexistent path.
        save_file(tensors, str(tensor_path), metadata={"experiment_schema_version":"1", "artifact_id":artifact_id})
        file_hashes = {}
        for path in bundle.iterdir():
            file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {"experiment_schema_version":1, "artifact_id":artifact_id, "complete":True,
            "report":report, "artifacts":artifacts, "file_sha256":file_hashes,
            "tensor_sha256":{key:tensor_hash(value) for key,value in tensors.items()},
            "tensor_shapes":{key:list(value.shape) for key,value in tensors.items()},
            "save_seconds":time.perf_counter()-started,
            "publication": "exclusive bundle with rollback on handled errors; not crash-atomic"}
        with (bundle/"manifest.json").open("x", encoding="utf-8") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    except BaseException:
        shutil.rmtree(bundle)
        raise
    return {**artifacts, "artifact_id":artifact_id, "save_seconds":manifest["save_seconds"]}
