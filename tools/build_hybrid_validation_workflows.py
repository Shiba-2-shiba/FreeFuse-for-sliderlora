"""Generate bounded A/B/C and manual-D ComfyUI evaluation graphs.

No model inference, runtime edits, mask synthesis, or resizing is performed.
Only the five named generated JSON files are replaced in --output-dir. Each
replacement is atomic, but the five-file set is not a filesystem transaction.
Default generation uses only the standard library. Optional --input-dir mask
preflight additionally needs Pillow and reads (never modifies) the two PNGs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "krea2_female_slider_collection_comparison"
STEM = "krea2_female_slider_hybrid_validation"
MANUAL = "krea2_female_slider_hybrid_manual_d"
SAMPLER = "Krea2SliderFusePredictionMixSampler"
DEFAULT_CASE = "woman_front_strong_overlap"
HISTORICAL = {
    "original": "test-results/u456-20261009/u5_original_seed42_target_mask_strength4/workflow.json",
    "man_front": "test-results/overlap-20261009/man_front_r4/workflow.json",
    "park": "test-results/u456-20261009/u5_park_seed444444_target_mask_strength4/workflow.json",
}
HISTORICAL_PROMPTS = {
    'man_front': (
        'Photorealistic full-body studio group photograph of exactly two fully clothed adults posing '
        'closely together, with clearly overlapping silhouettes. An adult woman in a sage-green top and '
        'black trousers stands slightly behind and to the left of an adult man in a blue T-shirt and '
        'black trousers. The man is in the foreground, nearer the camera, and partially occludes the '
        "woman's right shoulder, right upper arm and the right side of her torso. The woman's left hand "
        "rests lightly on the man's left upper arm. Their shoulders, arms and torsos visibly overlap at "
        'the center, with no gap between their silhouettes. Both faces remain clearly visible and look at '
        'the camera. Exactly two separate heads, anatomically correct hands and limbs, both entire '
        'figures visible from head to shoes. One continuous pale gray concrete wall and floor, soft even '
        'daylight, consistent exposure and natural shadows.'
    ),
    'park': (
        'Photorealistic full-body photograph of exactly two fully clothed adults standing side by side on '
        'one continuous park path. On the left is an adult woman in a sage-green top and black trousers. '
        'On the right is an adult man in a blue T-shirt and black trousers. Both face the camera with '
        'arms relaxed at their sides and a small gap between them. Both entire figures are visible from '
        'head to shoes, at the same camera distance and consistent scale. Mature leafy green trees, a '
        'park bench in the distance, continuous stone path and grass, soft daylight, coherent '
        'perspective, natural ground shadows and anatomically correct hands and limbs.'
    ),
}
NEW_PROMPT = (
    "Photorealistic full-body studio group photograph of exactly two fully clothed adults "
    "standing closely together with strongly overlapping silhouettes. An adult woman in a "
    "sage-green top and black trousers stands in the foreground, nearer the camera. An adult "
    "man in a blue T-shirt and black trousers stands directly behind her, staggered slightly "
    "to her right. The woman's body substantially occludes the front of the man's torso, "
    "while his face remains fully visible beside her head. Their shoulders, arms and torsos "
    "visibly overlap at the center, with no gap between their silhouettes. The man's hand "
    "rests lightly on the woman's upper arm in modest, natural contact; the other hands are "
    "relaxed and clearly separated. Both faces look at the camera. Exactly two separate "
    "heads, anatomically correct hands and limbs, both standing figures framed from head "
    "to shoes with both pairs of feet visible and space below the shoes. One continuous "
    "pale gray concrete wall and floor, soft even daylight, consistent exposure and natural shadows."
)
POSE_CRITERIA = [
    "Exactly two fully clothed adults, both standing, with full-body framing and both pairs of feet visible.",
    "Woman in the foreground; man behind her with only a slight horizontal offset.",
    "The woman's body substantially occludes the man's torso; the man's face remains clearly visible.",
    "Hands make modest natural contact; no extra heads or fused/extra limbs.",
    "A human must inspect generated images: prompt wording and token-mask intersection do not establish this pose.",
]
REGROW_WARNING = (
    "Initial subtraction is not a persistent exclusion zone: radius>0 may regrow into an "
    "initially excluded area when P_off misses it. Compare the before-dilation, effective "
    "and selection-added masks; protected/background pixels can still change indirectly."
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_cases():
    cases = []
    for case_id, evidence_name in HISTORICAL.items():
        if case_id == "original":
            prompt = read_json(ROOT / "workflows" / (BASE + "_api.json"))["4"]["inputs"]["prompt"]
            source = {"path": "workflows/" + BASE + "_api.json", "node_id": "4"}
        else:
            prompt = HISTORICAL_PROMPTS[case_id]
            source = {"path": "tools/build_hybrid_validation_workflows.py", "constant": "HISTORICAL_PROMPTS[" + case_id + "]"}
        source.update(status="historical_exact", provenance="Exact prompt transcribed from the provided historical workflow: " + evidence_name + "; the evidence archive is not a runtime dependency or included deliverable.")
        cases.append({
            "case_id": case_id, "prompt": prompt, "prompt_source": source,
            "seeds": [444444 if case_id == "park" else 42], "optional_seeds": [],
            "note": "Historical prompt and recommended seed only; this evaluation reruns matched controls and does not reproduce every historical run setting.",
        })
    cases.append({
        "case_id": DEFAULT_CASE, "prompt": NEW_PROMPT,
        "prompt_source": {"path": "tools/build_hybrid_validation_workflows.py", "constant": "NEW_PROMPT", "status": "new_unexecuted"},
        "seeds": [42, 444444], "optional_seeds": [444444],
        "human_pose_gate": {"required": True, "status": "not_evaluated", "criteria": POSE_CRITERIA},
    })
    return cases


def input_filename(value):
    if not isinstance(value, str) or not value or value != value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("Mask filename must be a nonempty ComfyUI input-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("..", ".") for part in value.split("/")) or any(c in value for c in ("\\", ":", "[", "]", "\n", "\r")):
        raise ValueError("Mask filename must stay inside ComfyUI input, using forward-slash relative paths")
    if path.suffix.lower() != ".png":
        raise ValueError("Manual masks must be PNG files")
    return value


def preflight_masks(input_dir, requirements):
    """Check binary grayscale and equal canvas sizes, not semantic alignment."""
    root = Path(input_dir).resolve()
    paths = []
    for requirement in requirements:
        path = (root / requirement["filename"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Mask path resolves outside ComfyUI input: " + requirement["filename"])
        if not path.is_file():
            raise FileNotFoundError("Required manual mask is missing: " + requirement["filename"])
        paths.append(path)
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required for --input-dir mask validation; no workflow files were written") from error
    sizes = []
    for path in paths:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG" or image.mode not in ("1", "L", "RGB") or "transparency" in image.info or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Mask must be a static binary 1/L/RGB PNG without alpha, transparency or palette: " + path.name)
            if image.mode == "RGB":
                red, green, blue = (channel.tobytes() for channel in image.split())
                valid = red == green == blue and set(red) <= {0, 255}
            elif image.mode == "L":
                valid = set(image.tobytes()) <= {0, 255}
            else:
                valid = True  # PNG mode 1 is binary by construction.
            if not valid:
                raise ValueError("Mask must have binary black/white equal RGB channels: " + path.name)
            sizes.append(image.size)
    if sizes[0] != sizes[1]:
        raise ValueError("Manual support and saved P_off must have identical canvas dimensions; no automatic resizing")
    for requirement, size in zip(requirements, sizes):
        requirement.update(provided=True, dimensions=list(size))


def add_node(api, sid, class_type, inputs, title):
    api[str(sid)] = {"class_type": class_type, "inputs": inputs, "_meta": {"title": title}}


def add_mask_export(api, first_id, source, prefix, name):
    add_node(api, first_id, "MaskToImage", {"mask": source}, name + ": mask to image")
    add_node(api, first_id + 1, "SaveImage", {"images": [str(first_id), 0], "filename_prefix": prefix + name}, name + ": ordinary PNG")


def make_ui(api, base_ui, manual, radius):
    """Rebuild all links/widgets from API values using the existing graph schema."""
    originals = {str(n["id"]): n for n in base_ui["nodes"] if n["type"] != "Note"}
    templates = {}
    for node in originals.values():
        templates.setdefault(node["type"], node)
    templates["LoadImageMask"] = {
        "type": "LoadImageMask", "size": [390, 330], "inputs": [],
        "outputs": [{"name": "MASK", "type": "MASK", "links": []}],
        "widgets_values_named": {"image": "", "channel": "red"},
    }
    order, visited, active = [], set(), set()
    def visit(sid):
        if sid in active:
            raise ValueError("Cycle in generated API graph")
        if sid in visited:
            return
        active.add(sid)
        for value in api[sid]["inputs"].values():
            if isinstance(value, list):
                visit(value[0])
        active.remove(sid)
        visited.add(sid)
        order.append(sid)
    for sid in api:
        visit(sid)
    nodes = {}
    manual_positions = {
        "1": [40, 100], "2": [40, 290], "3": [500, 100], "4": [500, 320],
        "5": [970, 100], "7": [970, 280], "9": [970, 470],
        "200": [40, 930], "201": [500, 930], "202": [970, 960],
        "20": [1430, 880], "21": [1880, 770], "22": [2360, 770], "23": [2820, 770],
        "100": [2350, 1270], "101": [2820, 1270], "102": [3270, 1250],
        "103": [2820, 1650], "104": [3270, 1630],
        "105": [2820, 2030], "106": [3270, 2010],
        "107": [2820, 2410], "108": [3270, 2390],
        "203": [40, 1460], "204": [500, 1440], "205": [40, 1870], "206": [500, 1850],
        "207": [970, 1460], "208": [1430, 1440],
    }
    for index, sid in enumerate(order):
        entry = api[sid]
        template = originals.get(sid)
        if not template or template["type"] != entry["class_type"]:
            template = templates[entry["class_type"]]
        node = deepcopy(template)
        node.update(id=int(sid), type=entry["class_type"], flags={}, order=index, mode=0,
                    title=entry.get("_meta", {}).get("title", entry["class_type"]),
                    properties={"Node name for S&R": entry["class_type"]})
        if manual:
            node["pos"] = manual_positions[sid]
        elif int(sid) >= 100:
            block, offset = divmod(int(sid) - 100, 10)
            column = 0 if offset == 0 else (460 if offset % 2 else 920) + (940 if offset >= 5 else 0)
            node["pos"] = [4400 + column, 100 + block * 820 + (390 if offset in (3, 4, 7, 8) else 0)]
        for port in node["inputs"]:
            port["link"] = None
        for port in node["outputs"]:
            port["links"] = []
        widget_names = list(template["widgets_values_named"])
        named = {name: entry["inputs"][name] for name in widget_names}
        values = list(named.values())
        if entry["class_type"] == SAMPLER:
            values.insert(widget_names.index("seed") + 1, "fixed")
        elif entry["class_type"] == "LoadImageMask":
            values.append("image")  # Frontend upload button, not an API input.
        node.update(widgets_values=values, widgets_values_named=named)
        nodes[sid] = node
    links = []
    for sid in order:
        for slot, port in enumerate(nodes[sid]["inputs"]):
            source = api[sid]["inputs"].get(port["name"])
            if source is None:
                continue
            origin, out_slot = source
            lid = len(links) + 1
            port["link"] = lid
            nodes[origin]["outputs"][out_slot]["links"].append(lid)
            links.append([lid, int(origin), out_slot, int(sid), slot, port["type"]])
    if manual:
        notes = [
            "D: manual support control. Human full-body support is not provided. REQUIRED_ filenames are intentional placeholders: missing files must fail LoadImageMask validation. No guessed support mask is generated.",
            f"Both LoadImageMask nodes use red: white=1, black=0. Use binary grayscale PNGs without alpha/palette and identical canvas dimensions. Copy the saved P_off from the same case / seed / model / settings into ComfyUI input. Token-grid P_off and 1024px support cannot be combined without an explicit aligned preparation step. Run --input-dir preflight. D uses original style MODEL + empty latent, one local Slider +4, radius={radius}; A/C controls use the same settings.",
            "D = clamp(human_support - P_off, 0, 1), with P_off also connected as protected_mask. " + REGROW_WARNING,
        ]
    else:
        notes = [
            "A: OFF target/OFF protected. B: ON target/ON protected. C: clamp(ON target - OFF protected), OFF protected. Both collector Samplers have own strength=0; only ON's upstream MODEL contains global Slider +4. All FINAL nodes use style-only MODEL, empty latent, and local Slider +4.",
            f"Fixed matched seed, steps=8, CFG=1, radius={radius}. Save effective selection from final MaskPreview slot 7 and added margin from slot 8. Raw collector target is not the final effective selection. All saves are active: cold default A/B/C costs 68 model evaluations. DiagnosticSave retains correlated image/latent/report artifacts. Change trial_id on all samplers for uncached remeasurement.",
            "New strong-overlap case has a HUMAN pose gate: woman foreground, man behind, substantial man's torso occlusion, man's face visible, both full-body/feet visible. Prompt and token intersection do not establish generated pose. GPU/image/pose validation not run. " + REGROW_WARNING,
        ]
    note_nodes = []
    for index, text in enumerate(notes):
        note_nodes.append({"id": 9000 + index, "type": "Note", "pos": [40 + index * 1350, -510], "size": [1250, 420], "flags": {}, "order": len(nodes) + index, "mode": 0, "inputs": [], "outputs": [], "properties": {}, "widgets_values": [text], "title": "Validation instructions"})
    groups = [] if manual else deepcopy(base_ui.get("groups", []))
    if not manual:
        groups.append({"title": "FINAL effective selection + added margin (A/B/C)", "bounding": [4360, 35, 2390, 2400], "color": "#725194", "font_size": 24, "flags": {}})
    return {"last_node_id": 9002, "last_link_id": len(links), "nodes": list(nodes.values()) + note_nodes, "links": links, "groups": groups, "config": {}, "extra": {"ds": {"scale": 0.28, "offset": [160, 640]}}, "version": 0.4}


def write_workflows(output_dir=None, case_id=DEFAULT_CASE, seed=None, radius=4,
                    support_mask=None, protected_mask=None, input_dir=None):
    if type(radius) is not int or radius not in (0, 2, 4):
        raise ValueError("radius must be one of 0, 2, or 4")
    cases = build_cases()
    by_case = {case["case_id"]: case for case in cases}
    if case_id not in by_case:
        raise ValueError("Unknown evaluation case: " + str(case_id))
    selected = by_case[case_id]
    if seed is None:
        seed = selected["seeds"][0]
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed must be an integer from 0 through 2**64-1")
    support_mask = input_filename(f"REQUIRED_HUMAN_FULLBODY_SUPPORT_{case_id}_seed{seed}.png" if support_mask is None else support_mask)
    protected_mask = input_filename(f"REQUIRED_SAVED_P_OFF_{case_id}_seed{seed}.png" if protected_mask is None else protected_mask)
    requirements = [
        {"node_id": "200", "filename": support_mask, "required": True, "provided": False, "role": "human_authored_fullbody_support", "polarity": "white=1 intended slider support; black=0 excluded", "source": "User draws after inspecting the same-case OFF reference; no mask is supplied by this package."},
        {"node_id": "201", "filename": protected_mask, "required": True, "provided": False, "role": "saved_off_protected_mask", "polarity": "white=1 protected man; black=0 unprotected", "source": "Copy raw_P_off from the matching A/B/C collector run, same case/seed/model/settings, into ComfyUI input."},
    ]
    if input_dir is not None:
        preflight_masks(input_dir, requirements)
    api = read_json(ROOT / "workflows" / (BASE + "_api.json"))
    base_ui = read_json(ROOT / "workflows" / (BASE + ".json"))
    prefix = f"slider_hybrid_validation/{case_id}/seed{seed}/r{radius}/"
    api["4"]["inputs"]["prompt"] = selected["prompt"]
    for sid, node in api.items():
        if node["class_type"] == SAMPLER:
            node["inputs"]["seed"] = seed
            if sid in ("21", "31", "42"):
                node["inputs"]["selection_dilate_radius"] = radius
        if "filename_prefix" in node["inputs"]:
            node["inputs"]["filename_prefix"] = prefix + node["inputs"]["filename_prefix"].split("/")[-1]
    for branch, sampler, preview in (("A", "21", 100), ("B", "31", 110), ("C", "42", 120)):
        add_node(api, preview, "Krea2SliderFuseMaskPreview", {"mask_bank": [sampler, 1]}, branch + ": FINAL bank, not collector bank")
        add_mask_export(api, preview + 1, [str(preview), 7], prefix, branch + "_effective_prediction_mask")
        add_mask_export(api, preview + 3, [str(preview), 8], prefix, branch + "_selection_added_mask")
        add_mask_export(api, preview + 5, [str(preview), 0], prefix, branch + "_reference_target_before_dilation")
        add_mask_export(api, preview + 7, [str(preview), 1], prefix, branch + "_reference_protected")
    manual = {sid: deepcopy(api[sid]) for sid in ("1", "2", "3", "4", "5", "7", "9", "20", "21", "22", "23", "100")}
    manual["20"]["inputs"].update(target_mask=["202", 0], protected_mask=["201", 0])
    manual["20"]["_meta"]["title"] = "D: human support minus saved OFF protection"
    manual["21"]["_meta"]["title"] = "D: FINAL style baseline + local Slider"
    manual["23"]["inputs"]["filename_prefix"] = prefix + "D_manual_support"
    manual["23"]["_meta"]["title"] = "D: correlated final diagnostics"
    manual["100"]["_meta"]["title"] = "D: FINAL mask bank"
    add_node(manual, 200, "LoadImageMask", {"image": support_mask, "channel": "red"}, "REQUIRED human full-body support (not provided)")
    add_node(manual, 201, "LoadImageMask", {"image": protected_mask, "channel": "red"}, "REQUIRED saved P_off from matching case/seed")
    add_node(manual, 202, "MaskComposite", {"destination": ["200", 0], "source": ["201", 0], "x": 0, "y": 0, "operation": "subtract"}, "D target = clamp(human support - P_off)")
    for first_id, source, name in ((101, ["100", 7], "D_effective_prediction_mask"), (103, ["100", 8], "D_selection_added_mask"), (105, ["100", 0], "D_reference_target_before_dilation"), (107, ["100", 1], "D_reference_protected"), (203, ["200", 0], "D_human_support_input"), (205, ["201", 0], "D_saved_P_off_input"), (207, ["202", 0], "D_support_minus_P_off")):
        add_mask_export(manual, first_id, source, prefix, name)
    output_prefixes = {label: {sid: n["inputs"]["filename_prefix"] for sid, n in graph.items() if "filename_prefix" in n["inputs"]} for label, graph in (("abc", api), ("manual_d", manual))}
    base_index = read_json(ROOT / "workflows" / (BASE + ".index.json"))
    coordinated = deepcopy(base_index["coordinated_settings"])
    for group in coordinated.values():
        sid, name = group["api_inputs"][0]
        group["default"] = api[sid]["inputs"][name]
    manifest = {
        "name": "Bounded hybrid + strong-overlap evaluation; no runtime changes",
        "generator": "tools/build_hybrid_validation_workflows.py",
        "base_workflow": BASE + ".json", "source_revision": base_index["source_revision"],
        "workflows": {"abc": {"ui": STEM + ".json", "api": STEM + "_api.json"}, "manual_d": {"ui": MANUAL + ".json", "api": MANUAL + "_api.json"}},
        "validation": {"static_graph_and_schema": "tests/test_hybrid_validation_workflows.py", "gpu_execution": "not_run", "image_quality": "not_evaluated", "pose_gate": "not_evaluated"},
        "selected_case": {"case_id": case_id, "seed": seed, "radius": radius}, "cases": cases,
        "coordinated_settings": coordinated,
        "manual_d": {
            "required_files": requirements, "missing_input_behavior": "fail_validation_no_fallback",
            "input_validation": "binary_grayscale_equal_dimensions_human_alignment_not_verified" if input_dir is not None else "not_run_files_not_provided",
            "canvas_requirement": "Both PNGs must have identical width/height and represent the same aligned canvas. Saved P_off is token-grid sized; do not mix it with a 1024px support without explicit preparation. No automatic resizing.",
            "input_format": "PNG mode 1/L or equal-channel RGB; binary black=0/white=255; no alpha/palette. LoadImageMask channel=red means white=1 without alpha inversion.",
            "initial_target": "clamp(human_support - saved_P_off, 0, 1)",
            "protected_mask": "same saved_P_off, connected to Subjects.protected_mask",
            "final_radius": radius, "regrowth_warning": REGROW_WARNING,
            "comparison_controls": "A and C from the same case/seed/radius, shared style model/empty latent/settings; D adds only human-authored support.",
            "settings": deepcopy(manual["21"]["inputs"]),
            "files_are_not_semantically_verified": True,
        },
        "output_prefixes": output_prefixes,
        "mask_slots": {"effective_prediction_mask": 7, "selection_added_mask": 8, "reference_target_before_dilation": 0, "reference_protected": 1, "indexing": "zero_based"},
        "nfe_estimate": {"collector_each": 10, "final_each_partial_mask": 16, "abc_all_active_outputs": 68, "manual_d_all_active_outputs": 16, "abc_plus_manual_d": 84,
                         "assumptions": "Cold uncached execution; all 24 A/B/C terminal saves and all 8 D saves are active. Two collectors each do collect_step=2 plus steps=8 (10 each); A/B/C/D each use nonzero strength and a nonempty partial target at 2*8=16. Extra mask saves share sampler outputs and add 0 model NFE. Empty/full selection endpoints may skip a branch. Counts exclude VAE, mask operations and audit overhead. Muting only final image saves does not isolate a branch while other terminal mask/reference saves remain active."},
        "rerun_guidance": base_index["rerun_guidance"] + " Apply matching settings to D, and preserve the source A/B/C prefix plus the exact mask files with D artifacts. Generator overwrites only its five named JSON outputs; choose a separate --output-dir per case/seed/radius to retain variants.",
        "limitations": base_index["artifact_limits"] + [REGROW_WARNING, "No human support mask is bundled. The previous manual-half comparison is historical context, not a full-body manual control.", "A zero token-mask intersection does not prove protected appearance is unchanged or the generated pose meets the human gate."],
        "load_image_mask_sources": ["https://github.com/Comfy-Org/ComfyUI/blob/master/nodes.py", "https://github.com/Comfy-Org/ComfyUI_frontend/blob/main/apps/website/public/workflow-graphs/edit-selected-region.json"],
    }
    payloads = {STEM + "_api.json": api, STEM + ".json": make_ui(api, base_ui, False, radius), MANUAL + "_api.json": manual, MANUAL + ".json": make_ui(manual, base_ui, True, radius), STEM + ".index.json": manifest}
    rendered = {name: json.dumps(value, ensure_ascii=False, indent=2) + "\n" for name, value in payloads.items()}
    destination = Path(output_dir) if output_dir is not None else ROOT / "workflows"
    destination.mkdir(parents=True, exist_ok=True)
    pending = []
    try:
        for name, content in rendered.items():
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination, prefix=".hybrid-", delete=False) as stream:
                pending.append((Path(stream.name), destination / name))
                stream.write(content)
        for temporary, target in pending:
            os.replace(temporary, target)
    finally:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)
    return [str(destination / name) for name in payloads]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", dest="case_id", choices=[*HISTORICAL, DEFAULT_CASE], default=DEFAULT_CASE)
    parser.add_argument("--seed", type=int, default=None, help="Default: 444444 for park; 42 for all other cases")
    parser.add_argument("--radius", type=int, choices=(0, 2, 4), default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workflows")
    parser.add_argument("--support-mask", help="Human-authored binary PNG, relative to ComfyUI input")
    parser.add_argument("--protected-mask", help="Saved matching OFF P_off PNG, relative to ComfyUI input")
    parser.add_argument("--input-dir", type=Path, help="Preflight both PNGs: required files, binary grayscale, equal dimensions; no resizing")
    args = parser.parse_args()
    try:
        files = write_workflows(**vars(args))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, "Hybrid workflow generation failed: " + str(error) + "\n")
    print(json.dumps({"status": "generated_not_gpu_validated", "files": files}, indent=2))


if __name__ == "__main__":
    main()
