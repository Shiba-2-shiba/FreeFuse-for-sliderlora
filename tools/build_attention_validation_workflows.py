"""Build experimental shared-collection attention-mask comparison workflows.

The default graph stops after the two-forward prefix of the full eight-step
schedule. The optional edited graph adds seven existing manual PredictionMix
branches. No inference, dependency installation, completed OFF image, external
image, or SAM model is required by this generator. It imports only stdlib code.
Only the five named JSON artifacts are replaced; each replacement is atomic,
but the set is not a filesystem transaction. Use distinct output directories
for variants you want to retain.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASE = "krea2_female_slider_collection_comparison"
STEM = "krea2_female_slider_attention_validation"
EDITED = STEM + "_edited"
DEFAULT_CASE = "woman_front_strong_overlap"
COLLECT = "Krea2SliderFuseExperimentalMaskCollect"
SELECT = "Krea2SliderFuseExperimentalMaskSelect"
SAVE = "Krea2SliderFuseExperimentalMaskSave"
SAMPLER = "Krea2SliderFusePredictionMixSampler"
SUITE = "KREA2_SLIDER_FUSE_MASK_EXPERIMENT"
VARIANTS = ("baseline", "centroid_control", "multi_proto", "multi_proto_bg", "adaln_bg", "ensemble_bg", "propagated_bg")
CONFIG = dict(collect_step=2, collect_block=18, selected_steps=[1, 2], selected_blocks=[16, 18],
    top_k_ratio=.2, temperature=10000., prototypes=3, seed_confidence=.65, seed_margin=.15,
    mask_confidence=.55, mask_margin=.10, context_temperature=1., feature_temperature=.15,
    propagation_iterations=1, propagation_heads=[0], affinity_chunk_size=128,
    affinity_threshold=.05, propagation_strength=.5)
VARIANT_PURPOSES = {
    "baseline": "Existing single-prototype attention baseline, collected from the shared prefix.",
    "centroid_control": "One-prototype control using experimental calibration/seed rules; comparison with multi_proto isolates prototype count.",
    "multi_proto": "Multiple target/protected prototypes without a background competitor.",
    "multi_proto_bg": "Multiple prototypes plus an explicit background competitor.",
    "adaln_bg": "Feature-tap candidate with background competition.",
    "ensemble_bg": "Adds fixed step/block views to multi_proto_bg, with background competition.",
    "propagated_bg": "Adds one primary-tap Q/K propagation to multi_proto_bg, not to ensemble_bg; head 0 is provisional.",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_cases():
    """Reuse the historical prompt source verbatim; do not improve prompts here."""
    path = ROOT / "tools/build_hybrid_validation_workflows.py"
    spec = importlib.util.spec_from_file_location("attention_historical_prompts", path)
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    prior = {case["case_id"]: case for case in historical.build_cases()}
    cases = []
    for case_id in ("park", "man_front", DEFAULT_CASE):
        source = prior[case_id]
        case = {
            "case_id": case_id, "prompt": source["prompt"],
            "prompt_source": deepcopy(source["prompt_source"]),
            "primary_seeds": [444444] if case_id == "park" else ([42, 444444] if case_id == DEFAULT_CASE else [42]),
            "holdout_seeds": [123, 777],
            "holdout_policy": "Freeze config before inspecting holdouts; retain failures and every attempted seed. These are planned holdouts, not executed results.",
            "background_phrase": "park path" if case_id == "park" else "pale gray concrete wall",
            "background_occurrence": 0,
            "human_pose_gate": {"required": True, "status": "not_evaluated", "criteria": [
                "A human must inspect generated candidates or a separately retained matched image; mask-only output cannot establish pose or quality.",
                "Exactly two fully clothed adults; both full-body figures and both pairs of feet are visible.",
                "Check the intended scene and front/behind relation rather than accepting prompt wording as evidence.",
            ]},
        }
        if case_id == DEFAULT_CASE:
            case["human_pose_gate"]["criteria"] = deepcopy(historical.POSE_CRITERIA)
            case["prior_pose_evaluation"] = {
                "attempts": 6, "composition_not_met": 6,
                "source": "Prior user-reviewed strong-overlap evaluation supplied for this task; historical image archive is not bundled.",
                "interpretation": "No successful strong-pose reference is established. This is not a result for the new seven-method comparison.",
            }
        cases.append(case)
    return cases


def add_node(api, sid, node_type, inputs, title):
    api[str(sid)] = {"class_type": node_type, "inputs": inputs, "_meta": {"title": title}}


def make_ui(api, base_ui, edited):
    """Create conventional ComfyUI 0.4 nodes with complete API/widget parity."""
    templates = {}
    for node in base_ui["nodes"]:
        if node["type"] != "Note":
            templates.setdefault(node["type"], node)

    def template(node_type, input_ports, output_ports, widgets):
        templates[node_type] = {
            "type": node_type, "size": [420, 230],
            "inputs": [{"name": name, "type": kind, "link": None} for name, kind in input_ports],
            "outputs": [{"name": name, "type": kind, "links": []} for name, kind in output_ports],
            "widgets_values_named": {name: None for name in widgets},
        }
    template(COLLECT, [("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
        ("prompt_info", "KREA2_SLIDER_FUSE_PROMPT"), ("subjects", "KREA2_SLIDER_FUSE_SUBJECTS"), ("latent", "LATENT")],
        [("suite", SUITE)], ["seed", "steps", "trial_id", "config_json", "background_phrase", "background_occurrence", "audit_tensors"])
    templates[COLLECT]["size"] = [570, 760]
    template(SELECT, [("suite", SUITE)], [("mask_bank", "KREA2_SLIDER_FUSE_MASKS"), ("report", "STRING")], ["variant"])
    template(SAVE, [("suite", SUITE)], [], ["filename_prefix"])
    template("PreviewImage", [("images", "IMAGE")], [], [])
    templates["PreviewImage"]["size"] = [380, 360]

    order, visited, active = [], set(), set()
    def visit(sid):
        if sid in active:
            raise ValueError("Cycle in attention validation API graph")
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

    setup_positions = {"1": [40, 50], "2": [40, 260], "3": [500, 50], "4": [500, 280],
        "5": [1000, 50], "6": [1000, 240], "7": [40, 570], "8": [1500, 50], "9": [2200, 460], "10": [2150, 60]}
    candidate_positions = {0: [40, 0], 1: [480, 0], 2: [940, 0], 3: [1160, 0],
        4: [940, 380], 5: [1160, 380], 6: [940, 760], 7: [1160, 760],
        10: [1630, 0], 11: [2100, 0], 12: [2580, 0], 13: [3040, 0], 14: [2580, 350],
        15: [3040, 350], 16: [3300, 350], 17: [3040, 770], 18: [3300, 770]}
    nodes = {}
    for index, sid in enumerate(order):
        entry = api[sid]
        prototype = templates[entry["class_type"]]
        node = deepcopy(prototype)
        node.update(id=int(sid), type=entry["class_type"], flags={}, order=index, mode=0,
            title=entry["_meta"]["title"], properties={"Node name for S&R": entry["class_type"]})
        if sid in setup_positions:
            node["pos"] = setup_positions[sid]
        else:
            row, offset = divmod(int(sid) - 100, 30)
            x, y = candidate_positions[offset]
            node["pos"] = [x, 1200 + row * 1320 + y]
        for port in node["inputs"]:
            port["link"] = None
        for port in node["outputs"]:
            port["links"] = []
        named = {name: entry["inputs"][name] for name in prototype["widgets_values_named"]}
        values = list(named.values())
        if entry["class_type"] in (COLLECT, SAMPLER):
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
    notes = [
        "EXPERIMENTAL: one shared style-only collector runs exactly the first 2 forwards of the same 8-step Euler/simple schedule. Seven methods reuse these observations. No completed OFF image, external image, SAM, or VAE is needed for mask-only collection. Save suite once: its runtime manifest, raw maps and masks are the provenance record. Mask-only selectors/previews have no terminal consumers, so a failed candidate cannot prevent the suite report from being saved.",
        "This graph " + ("optionally generates all 7 edited candidates through the existing manual PredictionMix path. Cold partial-mask expectation: 2 + 7*16 = 114 model NFE. All edits restart from original empty latent with the same prompt/seed/model and one local Slider +4." if edited else "stops after masks and reports. Cold model-forward expectation: 2 NFE. A full edit is not generated. Inspect suite candidate statuses before using the separate edited graph; remove all terminal saves for any failed variant before running valid edits.") + " NFE is not time or memory: taps, prototypes, affinity, transfer, saving, and audit have real costs. Read actual runtime reports; cache reuse changes new work.",
        "Radius=0 and fill_holes=0 are fixed. Keep all attempted seeds; freeze settings before seeds 123/777 holdouts. Prior strong female-front pose failed 6/6 attempts. Prompt wording and mask overlap cannot verify pose. Human gate required; GPU execution and image quality are not validated by these graph files. Confidence scores are heuristic; uncertain cells routed to base are not established background.",
    ]
    note_nodes = [{"id": 9000 + i, "type": "Note", "pos": [40 + i * 1280, -530], "size": [1210, 460],
        "flags": {}, "order": len(nodes) + i, "mode": 0, "inputs": [], "outputs": [], "properties": {},
        "widgets_values": [text], "title": "Attention validation instructions"} for i, text in enumerate(notes)]
    groups = [{"title": variant + " | shared collection", "bounding": [10, 1140 + i * 1320, 3780 if edited else 1570, 1250],
        "color": "#526b87", "font_size": 24, "flags": {}} for i, variant in enumerate(VARIANTS)]
    return {"last_node_id": 9002, "last_link_id": len(links), "nodes": list(nodes.values()) + note_nodes,
        "links": links, "groups": groups, "config": {}, "extra": {"ds": {"scale": .35, "offset": [100, 630]}}, "version": .4}


def write_workflows(output_dir=None, case_id=DEFAULT_CASE, seed=None, trial_id=0):
    cases = build_cases()
    by_case = {case["case_id"]: case for case in cases}
    if case_id not in by_case:
        raise ValueError("Unknown evaluation case: " + str(case_id))
    selected = by_case[case_id]
    seed = selected["primary_seeds"][0] if seed is None else seed
    if type(seed) is not int or not 0 <= seed <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed must be an integer from 0 through 2**64-1")
    if type(trial_id) is not int or not 0 <= trial_id <= 0x7FFFFFFF:
        raise ValueError("trial_id must be an integer from 0 through 2**31-1")
    base = read_json(ROOT / "workflows" / (BASE + "_api.json"))
    base_ui = read_json(ROOT / "workflows" / (BASE + ".json"))
    prefix = f"slider_attention_validation/{case_id}/seed{seed}/trial{trial_id}/"
    suite_prefix = f"attn_validation_{case_id}_s{seed}_t{trial_id}_"
    api = {sid: deepcopy(base[sid]) for sid in ("1", "2", "3", "4", "5", "6", "7")}
    api["4"]["inputs"]["prompt"] = selected["prompt"]
    add_node(api, 8, COLLECT, dict(model=["2", 0], positive=["4", 0], negative=["5", 0],
        prompt_info=["4", 1], subjects=["6", 0], latent=["7", 0], seed=seed, steps=8, trial_id=trial_id,
        config_json=json.dumps(CONFIG, ensure_ascii=False, indent=2), background_phrase=selected["background_phrase"],
        background_occurrence=selected["background_occurrence"], audit_tensors=False), "ONE shared 2-forward prefix of 8-step schedule")
    add_node(api, 10, SAVE, dict(suite=["8", 0], filename_prefix=suite_prefix + "mask_only_suite"), "Save ALL seven candidates, raw maps and runtime manifest")
    records = []
    for index, variant in enumerate(VARIANTS):
        b = 100 + index * 30
        add_node(api, b, SELECT, dict(suite=["8", 0], variant=variant), variant + ": select same collection")
        add_node(api, b + 1, "Krea2SliderFuseMaskPreview", dict(mask_bank=[str(b), 0]), variant + ": candidate bank")
        records.append(dict(variant=variant, purpose=VARIANT_PURPOSES[variant], select_node=str(b),
            candidate_preview_node=str(b + 1), edited_sampler_node=str(b + 11), edited_save_node=str(b + 13),
            effective_prediction_mask_save_node=str(b + 16), selection_added_mask_save_node=str(b + 18)))

    edited = deepcopy(api)
    edited["10"]["inputs"]["filename_prefix"] = suite_prefix + "edited_suite"
    edited["9"] = deepcopy(base["9"])
    for index, record in enumerate(records):
        b, variant = 100 + index * 30, record["variant"]
        subjects = deepcopy(base["6"]["inputs"])
        subjects.update(target_mask=[str(b + 1), 0], protected_mask=[str(b + 1), 1])
        add_node(edited, b + 10, "Krea2SliderFuseSubjects", subjects, variant + ": manual candidate masks")
        sampling = deepcopy(base["21"]["inputs"])
        sampling.update(model=["2", 0], latent=["7", 0], subjects=[str(b + 10), 0], seed=seed, trial_id=trial_id,
            strength=4., steps=8, cfg=1., mask_mode="manual", mix_scope="target_mask", diagnostic_level="audit",
            collect_step=2, collect_block=18, top_k_ratio=.2, temperature=10000.,
            fill_holes_max_area=0, mask_dilate_radius=0, selection_dilate_radius=0)
        add_node(edited, b + 11, SAMPLER, sampling, variant + ": edited +4 / 8 steps / radius 0")
        add_node(edited, b + 12, "VAEDecode", dict(samples=[str(b + 11), 0], vae=["9", 0]), variant + ": edited image only")
        add_node(edited, b + 13, "Krea2SliderFuseDiagnosticSave", dict(images=[str(b + 12), 0],
            latent=[str(b + 11), 0], diagnostics=[str(b + 11), 2], filename_prefix=prefix + "edited/" + variant), variant + ": image, latent and audit")
        add_node(edited, b + 14, "Krea2SliderFuseMaskPreview", dict(mask_bank=[str(b + 11), 1]), variant + ": FINAL effective bank")
        for offset, slot, name in ((15, 7, "effective_prediction_mask"), (17, 8, "selection_added_mask")):
            add_node(edited, b + offset, "MaskToImage", dict(mask=[str(b + 14), slot]), variant + ": " + name)
            add_node(edited, b + offset + 1, "SaveImage", dict(images=[str(b + offset), 0],
                filename_prefix=prefix + "edited/" + variant + "_" + name), variant + ": save " + name)

    manifest = {
        "name": "Shared-prefix attention mask validation (experimental)", "generator": "tools/build_attention_validation_workflows.py",
        "workflows": {"mask_only": {"ui": STEM + ".json", "api": STEM + "_api.json"}, "edited": {"ui": EDITED + ".json", "api": EDITED + "_api.json"}},
        "selected_case": dict(case_id=case_id, seed=seed, trial_id=trial_id), "cases": cases,
        "variants": records, "collection_config": deepcopy(CONFIG),
        "ablation_pairs": [
            {"from": "baseline", "to": "centroid_control", "change": "Experimental calibration and seed rules; not a prototype-only comparison."},
            {"from": "centroid_control", "to": "multi_proto", "change": "Prototype count."},
            {"from": "multi_proto", "to": "multi_proto_bg", "change": "Explicit background competitor."},
            {"from": "multi_proto_bg", "to": "adaln_bg", "change": "Feature representation."},
            {"from": "multi_proto_bg", "to": "ensemble_bg", "change": "Additional step/block views."},
            {"from": "multi_proto_bg", "to": "propagated_bg", "change": "One primary-tap Q/K propagation; not ensemble plus propagation."},
        ],
        "validation": {"static_contracts": "tests/test_attention_validation_workflows.py", "gpu_execution": "not_run", "image_quality": "not_evaluated", "pose_gate": "not_evaluated"},
        "controls": {"model_node": "2", "initial_latent_node": "7", "prompt_node": "4", "collector_node": "8",
            "steps": 8, "collect_step": 2, "cfg": 1., "sampler": "euler", "scheduler": "simple", "slider_strength": 4.,
            "fill_holes_max_area": 0, "mask_dilate_radius": 0, "selection_dilate_radius": 0,
            "schedule_rule": "Use the prefix of the same full eight-step schedule; never substitute a separately generated two-step schedule.",
            "slider_rule": "Collector sees style-only model. Every optional edited branch applies one local Slider +4 to the original empty latent; collection latent is not reused."},
        "output_prefixes": {label: {sid: n["inputs"]["filename_prefix"] for sid, n in graph.items() if "filename_prefix" in n["inputs"]}
            for label, graph in (("mask_only", api), ("edited", edited))},
        "cost_accounting": {
            "expected_cold_mask_only_model_nfe": 2, "expected_partial_edit_model_nfe_each": 16,
            "expected_cold_all_edited_model_nfe": 114, "measurement_status": "not_run",
            "actual_model_nfe": None, "wall_seconds": None, "peak_memory_bytes": None, "suite_save_node": "10",
            "collection_report_fields": {"collection_id": "run_id", "model_nfe": ["phase1_nfe", "phase2_nfe"], "time": ["elapsed_seconds", "collection_seconds", "algorithm_seconds"], "memory": ["memory.peak_allocated_bytes", "memory.peak_reserved_bytes", "memory.scope", "memory.unavailable_reason"]},
            "edited_report_fields": {"model_nfe": "total_model_nfe", "link": "variant -> edited_save_node -> correlated diagnostic JSON; match queued graph, masks and collection_id"},
            "measurement_instructions": "Suite wall time and memory include all candidates and shared hooks, not isolated per-variant end-to-end costs. Per-variant algorithm timings omit shared work. Record the queue wall clock, exact submitted API, suite manifest, collection_id and all edited diagnostic paths. Change collector and every sampler trial_id together for cold cache-independent remeasurement. A cached suite report describes the earlier collection, not new work in this queue. Report cache status explicitly.",
            "memory_scope_warning": "CUDA peaks may be process-lifetime peaks rather than node-local incremental use. Preserve the runtime memory.scope and unavailable_reason; never add per-node peak memory values or present unavailable memory as zero.",
            "assumptions": "Cold uncached graph with every terminal output active; each of seven edits has nonzero strength and a nonempty partial mask, requiring 2*8 model evaluations. Endpoints or failures can change the count; actual runtime counts take precedence. NFE excludes VAE, observation hooks, feature transfers, affinity/prototype work, mask operations, audits and saving. Shared observations avoid duplicate model forwards but are not free in time or memory.",
        },
        "evaluation_record_template": {
            "case_id": case_id, "seed": seed, "trial_id": trial_id, "collection_id": None, "suite_manifest_path": None,
            "queued_api_path": None, "cache_status": "not_recorded", "actual_model_nfe": None, "wall_seconds": None,
            "peak_memory_bytes": None, "memory_scope": None, "mask_evaluation_json_path": None,
            "pose_gate": {"status": "not_evaluated", "passed": None, "human_reviewer": None, "evidence_image_paths": [], "notes": None},
            "variant_results": [dict(variant=variant, pose_gate=dict(status="not_evaluated", passed=None, human_reviewer=None, notes=None), edit_diagnostic_json_path=None, image_path=None,
                target_body_change=None, protected_identity=None, background_leakage=None, contact_anatomy=None,
                old_silhouette_residue=None, failure_reason=None) for variant in VARIANTS],
        },
        "evaluation_order": ["Run mask-only first. Inspect every candidate status in the saved suite; remove all image/mask save outputs for failed variants before optional edited execution.",
            "Validate runtime collection identity, actual counters, artifact hashes and cleanup evidence.",
            "Inspect target/protected masks; score against optional same-grid human annotations with tools/evaluate_attention_masks.py.",
            "If edited candidates are generated, apply the human pose gate first; composition_not_met stays in the report.",
            "Evaluate full-body change, protected appearance, background leakage, old-region residue and contact anatomy separately.",
            "Freeze configuration before holdouts; retain every attempted seed and all failures."],
        "limitations": ["This static manifest is a plan, not proof of queued execution; preserve runtime artifacts.",
            "No completed OFF reference image is generated or required; optional archived matched reference images may aid human assessment but their availability is not assumed.",
            "Different masks intentionally have different partitions; do not relax the existing diagnostic comparison CLI's same-partition requirements.",
            "Heuristic confidence is not calibrated probability. Base-routed uncertain cells are not established background.",
            "No forced equal area, no hard seed-area quota, no SAM/external-image dependency, no automatic pose/quality certification.",
            "Static protected masks are whole-person guesses, not guaranteed anatomical cores; selective small hand/arm adjustments are not guaranteed. Mask-only output cannot establish generated anatomy, occlusion or preservation of identity. Real GPU/INT8/UI-reload/quality validation remains separate."],
    }
    payloads = {STEM + "_api.json": api, STEM + ".json": make_ui(api, base_ui, False),
        EDITED + "_api.json": edited, EDITED + ".json": make_ui(edited, base_ui, True), STEM + ".index.json": manifest}
    rendered = {name: json.dumps(value, ensure_ascii=False, indent=2) + "\n" for name, value in payloads.items()}
    destination = Path(output_dir) if output_dir is not None else ROOT / "workflows"
    destination.mkdir(parents=True, exist_ok=True)
    pending = []
    try:
        for name, content in rendered.items():
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", dir=destination,
                prefix=".attention-", delete=False) as stream:
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
    parser.add_argument("--case", dest="case_id", choices=("park", "man_front", DEFAULT_CASE), default=DEFAULT_CASE)
    parser.add_argument("--seed", type=int, default=None, help="Default: 444444 for park; 42 otherwise")
    parser.add_argument("--trial-id", type=int, default=0, help="Change for uncached collection and edited sampling")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workflows")
    try:
        files = write_workflows(**vars(parser.parse_args()))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, "Attention workflow generation failed: " + str(error) + "\n")
    print(json.dumps({"status": "generated_not_gpu_validated", "files": files}, indent=2))


if __name__ == "__main__":
    main()
