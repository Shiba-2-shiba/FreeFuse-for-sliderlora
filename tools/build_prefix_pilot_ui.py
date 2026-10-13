"""Build conventional ComfyUI 0.4 workflows from the four paired-prefix API graphs.

This stdlib-only converter does not import ComfyUI, submit a graph, or run models.
It writes only the four named UI files. Each replacement is atomic; the complete
set is not a filesystem transaction. API inputs and seed controls are preserved.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CASES = ("park", "strong_overlap")
MODES = ("fast", "compare")
PILOT = "Krea2SliderFusePairedPrefixPilot"
SAVE = "Krea2SliderFusePrefixPilotSave"
SAMPLER = "Krea2SliderFusePredictionMixSampler"
COLLECT = "Krea2SliderFuseExperimentalMaskCollect"
SEED_NODES = {PILOT, SAMPLER, COLLECT}


def _templates():
    templates = {}
    for stem in ("krea2_female_slider_collection_comparison", "krea2_female_slider_attention_validation"):
        source = json.loads((ROOT / "workflows" / (stem + ".json")).read_text(encoding="utf-8"))
        for node in source["nodes"]:
            if node["type"] != "Note":
                templates.setdefault(node["type"], node)
    pilot = deepcopy(templates[SAMPLER])
    pilot["outputs"] = [
        {"name": "mask_bank", "type": "KREA2_SLIDER_FUSE_MASKS", "links": []},
        {"name": "probe", "type": "KREA2_SLIDER_FUSE_PAIRED_PROBE", "links": []},
    ]
    pilot["widgets_values_named"].update(background_phrase="", pilot_mode="prefix", source_identity_json="{}")
    pilot["size"] = [570, 840]
    templates[PILOT] = pilot
    templates[SAVE] = {
        "inputs": [{"name": "probe", "type": "KREA2_SLIDER_FUSE_PAIRED_PROBE", "link": None},
                   {"name": "mask_bank", "type": "KREA2_SLIDER_FUSE_MASKS", "link": None}],
        "outputs": [], "widgets_values_named": {"filename_prefix": ""}, "size": [420, 180],
    }
    templates["Krea2SliderFuseRetainedMask"] = {
        "inputs": [], "outputs": [{"name": name, "type": "MASK", "links": []} for name in ("target", "protected")],
        "widgets_values_named": {"report_path": ""}, "size": [420, 150],
    }
    return templates


def _is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int


def _validate_and_order(api, templates):
    for sid, entry in api.items():
        if not isinstance(sid, str) or not sid.isdecimal() or str(int(sid)) != sid:
            raise ValueError("UI node IDs must be canonical nonnegative integer strings")
        kind = entry["class_type"]
        if kind not in templates:
            raise ValueError("Unsupported prefix workflow node: " + kind)
        template = templates[kind]
        ports = {p["name"]: p for p in template["inputs"]}
        widgets = template["widgets_values_named"]
        for name, value in entry["inputs"].items():
            if name in ports:
                if not _is_link(value) or value[0] not in api:
                    raise ValueError(f"Invalid source for {sid}:{name}")
                source_kind = api[value[0]]["class_type"]
                if source_kind not in templates:
                    raise ValueError("Unsupported source node: " + source_kind)
                outputs = templates[source_kind]["outputs"]
                if not 0 <= value[1] < len(outputs) or outputs[value[1]]["type"] != ports[name]["type"]:
                    raise ValueError(f"Invalid output slot/type for {sid}:{name}")
            elif name not in widgets or _is_link(value):
                raise ValueError(f"Unrepresented API input {sid}:{name}")
    order, visited, active = [], set(), set()
    def visit(sid):
        if sid in active:
            raise ValueError("Cycle in prefix workflow")
        if sid in visited:
            return
        active.add(sid)
        for value in api[sid]["inputs"].values():
            if _is_link(value):
                visit(value[0])
        active.remove(sid)
        visited.add(sid)
        order.append(sid)
    for sid in api:
        visit(sid)
    return order


def _position(sid, mode):
    setup = {"1": [40, 60], "2": [40, 300], "3": [530, 60], "4": [530, 320],
        "5": [1040, 60], "6": [1040, 260], "7": [40, 650], "9": [530, 760],
        "8": [1540, 60], "18": [2220, 60], "60": [2220, 380],
        "61": [2700, 380], "62": [3180, 380]}
    if sid in setup:
        return setup[sid]
    row, offset = divmod(int(sid), 100)
    base_y = 1320 if mode == "fast" or row == 1 else 2380
    positions = {0: [40, 0], 5: [680, 0], 10: [680, 300], 11: [1160, 300],
                 20: [1650, 0], 21: [2200, 0], 22: [2640, 0]}
    if row not in (1, 2) or offset not in positions:
        raise ValueError("Unsupported prefix workflow layout node: " + sid)
    x, y = positions[offset]
    return [x, base_y + y]


def _title(sid, entry):
    kind, inputs = entry["class_type"], entry["inputs"]
    if kind == PILOT:
        return ("NON-FINAL mask pilot: first 3 of 8 steps / 6 NFE" if inputs["pilot_mode"] == "prefix"
                else "Full-reference mask pilot: 8 steps / 16 NFE")
    if kind == SAMPLER:
        return "Final replay: original noise / 8 steps / Slider +4"
    if kind == COLLECT:
        return "ONE shared cold bootstrap: first 2 of 8 steps"
    if kind == SAVE:
        return "Save masks + exact pilot evidence (no VAE)"
    if kind == "VAEDecode":
        return "Decode final 8-step replay only"
    if kind == "Krea2SliderFuseDiagnosticSave":
        return "Save final replay image + latent + diagnostics"
    return entry.get("_meta", {}).get("title", kind.removeprefix("Krea2SliderFuse"))


def make_ui(api, *, case_id, mode):
    """Convert a bare/enveloped graph; fail rather than lose literals or links."""
    if mode not in MODES or case_id not in CASES:
        raise ValueError("Unknown prefix workflow case or mode")
    api = api.get("prompt", api)
    templates = _templates()
    order = _validate_and_order(api, templates)
    nodes = {}
    for index, sid in enumerate(order):
        entry = api[sid]
        kind = entry["class_type"]
        node = deepcopy(templates[kind])
        node.update(id=int(sid), type=kind, flags={}, order=index, mode=0,
                    pos=_position(sid, mode), title=_title(sid, entry), properties={"Node name for S&R": kind})
        for port in node["inputs"]:
            port["link"] = None
        for output in node["outputs"]:
            output["links"] = []
        named = {name: deepcopy(entry["inputs"][name]) for name in node["widgets_values_named"] if name in entry["inputs"]}
        values = list(named.values())
        if kind in SEED_NODES:
            values.insert(list(named).index("seed") + 1, "fixed")
        node.update(widgets_values=values, widgets_values_named=named)
        nodes[sid] = node
    links = []
    for sid in order:
        for slot, port in enumerate(nodes[sid]["inputs"]):
            source = api[sid]["inputs"].get(port["name"])
            if source is None:
                continue
            origin, output = source
            lid = len(links) + 1
            port["link"] = lid
            nodes[origin]["outputs"][output]["links"].append(lid)
            links.append([lid, int(origin), output, int(sid), slot, port["type"]])
    cost = ("Expected cold cost: bootstrap 2 + prefix pilot 6 + final replay 16 = 24 model NFE. "
            "One paired prefix and one full replay are enabled." if mode == "fast" else
            "Expected cold cost: bootstrap 2 + full-reference pilot 16 + prefix pilot 6 + two final replays 32 = 56 model NFE. "
            "Both pilot paths share ONE bootstrap; compare the saved exact evidence offline.")
    # Keep the arithmetic directly readable even with long explanatory labels.
    arithmetic = "2 + 6 + 16 = 24" if mode == "fast" else "2 + 16 + 6 + 32 = 56"
    notes = [
        "EXPERIMENTAL PAIRED PREFIX | " + case_id + " | " + mode.upper() + "\n\n"
        "The prefix is NON-FINAL: only masks and validation evidence leave the pilot. It executes the first 3 updates "
        "of the same full 8-step Euler/simple schedule, with base + Slider branches (6 model NFE). "
        "Never substitute a separately generated 3-step schedule. The cold bootstrap uses the first 2 forwards "
        "of that full schedule and selects multi_proto_bg. No inherited OFF image or partial pilot latent is decoded.",
        cost + "\n\n" + arithmetic + " (expected, not measured). NFE is not wall time or memory. "
        "Taps, feature transfer, mask construction, audit and saves have overhead. Cache reuse changes new work. "
        "Keep the same seed, model, prompt, original noise and original empty latent for every replay: 8 steps, "
        "one local Slider +4, radius 0 and fill holes 0. Seed control_after_generate is fixed; change trial_id to remeasure.",
        "IMPORT CHECKLIST\n\nSelect the available model, text encoder, VAE, style LoRA and Slider in the visible loaders/widgets. "
        "Any changes create a different experiment: retain the exact queued API graph and source identity. "
        "These files are not GPU-validated and do not establish image quality, faster wall time, or pose. "
        "Human inspection of final replay images is required for pose, protected identity, target change, contact anatomy "
        "and background leakage. " + ("Prior strong-overlap pose attempts failed 6/6; a new pose success is not established."
        if case_id == "strong_overlap" else "A prompt or mask overlap cannot establish a successful pose."),
    ]
    note_start = max(9000, max(map(int, nodes), default=0) + 1)
    note_nodes = [{"id": note_start+i, "type": "Note", "pos": [40+i*1200, -610], "size": [1120, 480],
        "flags": {}, "order": len(nodes)+i, "mode": 0, "inputs": [], "outputs": [], "properties": {},
        "widgets_values": [text], "title": "Paired-prefix pilot instructions"} for i, text in enumerate(notes)]
    groups = [{"title": "Shared setup + ONE cold bootstrap (2 NFE)", "bounding": [10, 10, 3600, 1130],
        "color": "#526b87", "font_size": 24, "flags": {}}]
    branches = [("NON-FINAL prefix pilot + original-noise final replay", 1260)] if mode == "fast" else [
        ("Full-reference pilot + original-noise final replay", 1260),
        ("NON-FINAL prefix pilot + original-noise final replay", 2320)]
    groups.extend({"title": title, "bounding": [10, y, 3150, 980], "color": "#526b87", "font_size": 24, "flags": {}}
                  for title, y in branches)
    return {"last_node_id": note_start+len(note_nodes)-1, "last_link_id": len(links),
        "nodes": list(nodes.values())+note_nodes, "links": links, "groups": groups, "config": {},
        "extra": {"ds": {"scale": .35, "offset": [100, 660]}}, "version": .4}


def write_workflows(output_dir=None, api_dir=None):
    source = Path(api_dir) if api_dir else ROOT / "workflows"
    destination = Path(output_dir) if output_dir else ROOT / "workflows"
    prepared = []
    for case_id in CASES:
        for mode in MODES:
            stem = f"krea2_prefix_pilot_{case_id}_{mode}"
            api = json.loads((source / (stem+"_api.json")).read_text(encoding="utf-8"))
            ui = make_ui(api, case_id=case_id, mode=mode)
            prepared.append((destination / (stem+".json"), json.dumps(ui, indent=2, ensure_ascii=False, allow_nan=False)+"\n"))
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for path, text in prepared:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=destination,
                    prefix="."+path.name+".", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(text)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        written.append(str(path))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-dir", type=Path, help="Directory containing the four frozen API graph files")
    parser.add_argument("--output-dir", type=Path, help="UI output directory; defaults to workflows/")
    args = parser.parse_args(argv)
    paths = write_workflows(output_dir=args.output_dir, api_dir=args.api_dir)
    print(json.dumps({"status": "generated_not_gpu_validated", "files": paths}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
