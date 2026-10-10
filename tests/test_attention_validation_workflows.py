"""CPU/stdlib graph contracts; these do not claim GPU or image-quality success.

Tests catch: duplicated collection; a short replacement noise schedule; completed
OFF decoding; per-method sampling drift; disconnected report provenance;
postprocessing confounds; UI/API drift; and unearned pose/quality claims.
"""
import ast
import importlib.util
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
STEM = "krea2_female_slider_attention_validation"
EDITED = STEM + "_edited"
COLLECT = "Krea2SliderFuseExperimentalMaskCollect"
SELECT = "Krea2SliderFuseExperimentalMaskSelect"
SAVE = "Krea2SliderFuseExperimentalMaskSave"
SAMPLER = "Krea2SliderFusePredictionMixSampler"
VARIANTS = ("baseline", "centroid_control", "multi_proto", "multi_proto_bg", "adaln_bg", "ensemble_bg", "propagated_bg")
EXPECTED_CONFIG = dict(collect_step=2, collect_block=18, selected_steps=[1, 2], selected_blocks=[16, 18],
    top_k_ratio=.2, temperature=10000., prototypes=3, seed_confidence=.65, seed_margin=.15,
    mask_confidence=.55, mask_margin=.10, context_temperature=1., feature_temperature=.15,
    propagation_iterations=1, propagation_heads=[0], affinity_chunk_size=128,
    affinity_threshold=.05, propagation_strength=.5)
OUTPUTS = {
    "UNETLoader": ["MODEL"], "LoraLoaderModelOnly": ["MODEL"], "CLIPLoader": ["CLIP"],
    "Krea2SliderFuseEncode": ["CONDITIONING", "KREA2_SLIDER_FUSE_PROMPT"],
    "ConditioningZeroOut": ["CONDITIONING"], "Krea2SliderFuseSubjects": ["KREA2_SLIDER_FUSE_SUBJECTS"],
    "EmptySD3LatentImage": ["LATENT"], COLLECT: ["KREA2_SLIDER_FUSE_MASK_EXPERIMENT"],
    SELECT: ["KREA2_SLIDER_FUSE_MASKS", "STRING"], SAVE: [],
    "Krea2SliderFuseMaskPreview": ["MASK"] * 9, "MaskToImage": ["IMAGE"], "PreviewImage": [],
    "VAELoader": ["VAE"], "VAEDecode": ["IMAGE"], SAMPLER: ["LATENT", "KREA2_SLIDER_FUSE_MASKS", "KREA2_SLIDER_FUSE_DIAGNOSTICS"],
    "Krea2SliderFuseDiagnosticSave": [], "SaveImage": ["IMAGE"],
}


def is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int


class AttentionValidationWorkflowTests(unittest.TestCase):
    def read(self, stem=STEM, suffix="_api.json", folder=None):
        path = (Path(folder) if folder else ROOT / "workflows") / (stem + suffix)
        self.assertTrue(path.is_file(), "Missing attention validation deliverable: " + path.name)
        return json.loads(path.read_text(encoding="utf-8"))

    def generator(self):
        path = ROOT / "tools/build_attention_validation_workflows.py"
        self.assertTrue(path.is_file(), "Missing attention validation generator")
        spec = importlib.util.spec_from_file_location("attention_workflow_generator", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_mask_only_graph_has_no_completed_image_or_external_segmentation_dependency(self):
        api = self.read()
        classes = {n["class_type"] for n in api.values()}
        self.assertFalse(classes & {"VAELoader", "VAEDecode", SAMPLER, "KSampler", "LoadImage", "LoadImageMask"})
        self.assertTrue(classes <= set(OUTPUTS))
        saves = [n for n in api.values() if n["class_type"] == SAVE]
        self.assertEqual(len(saves), 1)
        self.assertEqual(saves[0]["inputs"]["suite"], ["8", 0])

    def test_mask_only_report_survives_failed_variant_by_avoiding_terminal_selectors(self):
        api = self.read()
        # Failed variants must remain reportable: only the suite saver is terminal.
        self.assertFalse({n["class_type"] for n in api.values()} & {"PreviewImage", "SaveImage", SAMPLER})
        ancestors = set()
        def collect(sid):
            if sid in ancestors:
                return
            ancestors.add(sid)
            for value in api[sid]["inputs"].values():
                if is_link(value):
                    collect(value[0])
        collect("10")
        self.assertEqual(ancestors, {str(i) for i in range(1, 9)} | {"10"})
        self.assertNotIn(SELECT, {api[sid]["class_type"] for sid in ancestors})

    def test_all_seven_methods_reuse_exactly_one_two_forward_eight_step_collection(self):
        for stem in (STEM, EDITED):
            api = self.read(stem)
            collectors = {sid: n for sid, n in api.items() if n["class_type"] == COLLECT}
            self.assertEqual(set(collectors), {"8"})
            inputs = collectors["8"]["inputs"]
            self.assertEqual(inputs["steps"], 8)
            self.assertIs(inputs.get("audit_tensors"), False)
            self.assertEqual(json.loads(inputs["config_json"]), EXPECTED_CONFIG)
            self.assertEqual(inputs["model"], ["2", 0])
            self.assertEqual(inputs["latent"], ["7", 0])
            self.assertEqual(inputs["positive"], ["4", 0])
            self.assertEqual(inputs["negative"], ["5", 0])
            self.assertEqual(inputs["prompt_info"], ["4", 1])
            self.assertEqual(inputs["subjects"], ["6", 0])
            selects = [n["inputs"] for n in api.values() if n["class_type"] == SELECT]
            self.assertCountEqual([n["variant"] for n in selects], VARIANTS)
            self.assertTrue(all(n["suite"] == ["8", 0] for n in selects))
            self.assertEqual(len([n for n in api.values() if n["class_type"] == SAVE]), 1)

    def test_edited_paths_have_single_local_slider_and_matched_zero_radius_controls(self):
        api = self.read(EDITED)
        baseline = self.read()
        for sid in ("1", "2", "3", "4", "5", "6", "7", "8"):
            self.assertEqual(api[sid], baseline[sid])
        self.assertEqual(len([n for n in api.values() if n["class_type"] == "LoraLoaderModelOnly"]), 1)
        samplers = {sid: n["inputs"] for sid, n in api.items() if n["class_type"] == SAMPLER}
        self.assertEqual(len(samplers), 7)
        reference = None
        for sid, inputs in samplers.items():
            for key, value in dict(model=["2", 0], latent=["7", 0], positive=["4", 0], negative=["5", 0],
                prompt_info=["4", 1], strength=4., steps=8, cfg=1., seed=42, trial_id=0,
                mask_mode="manual", mix_scope="target_mask", diagnostic_level="audit", collect_step=2,
                collect_block=18, top_k_ratio=.2, temperature=10000., fill_holes_max_area=0,
                mask_dilate_radius=0, selection_dilate_radius=0).items():
                self.assertEqual(inputs[key], value, sid + ":" + key)
            matched = {k: v for k, v in inputs.items() if k != "subjects"}
            if reference is None:
                reference = matched
            self.assertEqual(matched, reference)
            self.assertNotEqual(inputs["lora_name"], api["2"]["inputs"]["lora_name"])
            subjects = api[inputs["subjects"][0]]["inputs"]
            self.assertEqual(subjects["prompt_info"], ["4", 1])
            target, protected = subjects["target_mask"], subjects["protected_mask"]
            self.assertEqual(target, [protected[0], 0])
            self.assertEqual(protected[1], 1)
            preview = api[target[0]]
            self.assertEqual(preview["class_type"], "Krea2SliderFuseMaskPreview")
            selected = api[preview["inputs"]["mask_bank"][0]]
            self.assertEqual(selected["class_type"], SELECT)
        for node in api.values():
            if node["class_type"] == "VAEDecode":
                self.assertEqual(api[node["inputs"]["samples"][0]]["class_type"], SAMPLER)

    def test_edited_saves_correlate_image_latent_diagnostics_and_effective_mask(self):
        api, manifest = self.read(EDITED), self.read(suffix=".index.json")
        records = manifest["variants"]
        self.assertEqual([r["variant"] for r in records], list(VARIANTS))
        for record in records:
            sampler = record["edited_sampler_node"]
            save = api[record["edited_save_node"]]["inputs"]
            self.assertEqual(save["latent"], [sampler, 0])
            self.assertEqual(save["diagnostics"], [sampler, 2])
            self.assertEqual(api[save["images"][0]]["inputs"]["samples"], [sampler, 0])
            for name, slot in (("effective_prediction_mask", 7), ("selection_added_mask", 8)):
                image_save = api[record[name + "_save_node"]]["inputs"]
                convert = api[image_save["images"][0]]["inputs"]["mask"]
                self.assertEqual(convert[1], slot)
                self.assertEqual(api[convert[0]]["inputs"]["mask_bank"], [sampler, 1])

    def test_cases_preserve_historical_prompts_and_have_disjoint_holdout_seeds(self):
        manifest = self.read(suffix=".index.json")
        cases = {c["case_id"]: c for c in manifest["cases"]}
        self.assertEqual(set(cases), {"park", "man_front", "woman_front_strong_overlap"})
        spec = importlib.util.spec_from_file_location("historical", ROOT / "tools/build_hybrid_validation_workflows.py")
        historical = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(historical)
        prior = {c["case_id"]: c for c in historical.build_cases()}
        for cid, case in cases.items():
            self.assertEqual(case["prompt"], prior[cid]["prompt"])
            self.assertTrue(case["holdout_seeds"])
            self.assertFalse(set(case["primary_seeds"]) & set(case["holdout_seeds"]))
            self.assertEqual(len(case["holdout_seeds"]), len(set(case["holdout_seeds"])))
            self.assertIn(case["background_phrase"], case["prompt"])
            self.assertNotIn(case["background_phrase"], ("woman", "man"))
        self.assertEqual(cases["park"]["primary_seeds"], [444444])
        self.assertEqual(cases["man_front"]["primary_seeds"], [42])
        self.assertEqual(cases["woman_front_strong_overlap"]["primary_seeds"], [42, 444444])

    def test_human_pose_gate_retains_six_failed_attempts_without_claiming_new_success(self):
        manifest = self.read(suffix=".index.json")
        case = next(c for c in manifest["cases"] if c["case_id"] == "woman_front_strong_overlap")
        self.assertEqual(case["prior_pose_evaluation"]["attempts"], 6)
        self.assertEqual(case["prior_pose_evaluation"]["composition_not_met"], 6)
        gate = case["human_pose_gate"]
        self.assertTrue(gate["required"])
        self.assertEqual(gate["status"], "not_evaluated")
        criteria = " ".join(gate["criteria"]).lower()
        for term in ("woman", "foreground", "man", "behind", "torso", "face", "feet", "human"):
            self.assertIn(term, criteria)
        self.assertEqual(manifest["validation"]["gpu_execution"], "not_run")
        self.assertEqual(manifest["validation"]["image_quality"], "not_evaluated")
        self.assertFalse(manifest["evaluation_record_template"]["pose_gate"]["passed"])

    def test_costs_distinguish_expected_nfe_from_actual_runtime_evidence(self):
        manifest = self.read(suffix=".index.json")
        costs = manifest["cost_accounting"]
        self.assertEqual(costs["expected_cold_mask_only_model_nfe"], 2)
        self.assertEqual(costs["expected_partial_edit_model_nfe_each"], 16)
        self.assertEqual(costs["expected_cold_all_edited_model_nfe"], 114)
        self.assertEqual(costs["measurement_status"], "not_run")
        for key in ("actual_model_nfe", "wall_seconds", "peak_memory_bytes"):
            self.assertIsNone(costs[key])
        self.assertEqual(costs["suite_save_node"], "10")
        self.assertIn("cache", costs["measurement_instructions"].lower())
        self.assertIn("process", costs["memory_scope_warning"].lower())
        record = manifest["evaluation_record_template"]
        for key in ("collection_id", "suite_manifest_path", "queued_api_path", "actual_model_nfe", "wall_seconds", "peak_memory_bytes", "variant_results"):
            self.assertIn(key, record)
        self.assertEqual([r["variant"] for r in record["variant_results"]], list(VARIANTS))
        for result in record["variant_results"]:
            for key in ("pose_gate", "edit_diagnostic_json_path", "target_body_change", "protected_identity", "background_leakage", "contact_anatomy", "old_silhouette_residue"):
                self.assertIn(key, result)
            self.assertEqual(result["pose_gate"]["status"], "not_evaluated")
            self.assertIsNone(result["pose_gate"]["passed"])

    def test_output_prefixes_are_unique_and_fully_listed_per_graph(self):
        manifest = self.read(suffix=".index.json")
        for label, stem in (("mask_only", STEM), ("edited", EDITED)):
            api = self.read(stem)
            actual = {sid: n["inputs"]["filename_prefix"] for sid, n in api.items() if "filename_prefix" in n["inputs"]}
            self.assertEqual(actual, manifest["output_prefixes"][label])
            self.assertEqual(len(actual), len(set(actual.values())))
            for sid, prefix in actual.items():
                if api[sid]["class_type"] == SAVE:
                    self.assertEqual(prefix, "attn_validation_woman_front_strong_overlap_s42_t0_" + label + "_suite")
                else:
                    self.assertTrue(prefix.startswith("slider_attention_validation/woman_front_strong_overlap/seed42/trial0/"))
                self.assertNotIn("..", prefix)

    def test_ui_api_roundtrip_is_complete_typed_and_topological(self):
        for stem in (STEM, EDITED):
            api, ui = self.read(stem), self.read(stem, ".json")
            nodes = {str(n["id"]): n for n in ui["nodes"]}
            links = {n[0]: n for n in ui["links"]}
            self.assertEqual(set(api), {sid for sid, n in nodes.items() if n["type"] != "Note"})
            self.assertEqual(ui["last_node_id"], max(int(sid) for sid in nodes))
            self.assertEqual(ui["last_link_id"], max(links))
            produced, consumed = [], []
            for sid, entry in api.items():
                node = nodes[sid]
                self.assertEqual(node["type"], entry["class_type"])
                self.assertEqual(node["mode"], 0)
                self.assertEqual([out["type"] for out in node["outputs"]], OUTPUTS[node["type"]])
                named = node["widgets_values_named"]
                ordered = list(named.values())
                if node["type"] in (COLLECT, SAMPLER):
                    ordered.insert(list(named).index("seed") + 1, "fixed")
                self.assertEqual(node["widgets_values"], ordered)
                reconstructed = dict(named)
                for slot, port in enumerate(node["inputs"]):
                    if port["link"] is None:
                        continue
                    lid, origin, outslot, target, inslot, kind = links[port["link"]]
                    self.assertEqual((target, inslot), (int(sid), slot))
                    self.assertEqual(kind, port["type"])
                    self.assertEqual(kind, nodes[str(origin)]["outputs"][outslot]["type"])
                    self.assertLess(nodes[str(origin)]["order"], node["order"])
                    reconstructed[port["name"]] = [str(origin), outslot]
                    consumed.append(lid)
                self.assertEqual(reconstructed, entry["inputs"])
                for slot, output in enumerate(node["outputs"]):
                    for lid in output["links"]:
                        self.assertEqual(links[lid][1:3], [int(sid), slot])
                        produced.append(lid)
            self.assertCountEqual(consumed, links)
            self.assertCountEqual(produced, links)

    def test_generator_reproduces_all_five_artifacts_and_keeps_other_files(self):
        generator = self.generator()
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / "keep.txt"
            marker.write_text("unchanged")
            paths = generator.write_workflows(output_dir=folder)
            self.assertEqual(len(paths), 5)
            for path in map(Path, paths):
                self.assertEqual(path.read_bytes(), (ROOT / "workflows" / path.name).read_bytes())
                self.assertNotIn(b"\r\n", path.read_bytes())
                self.assertTrue(path.read_bytes().endswith(b"\n"))
            self.assertEqual(marker.read_text(), "unchanged")

    def test_generator_propagates_case_seed_trial_and_background_phrase(self):
        generator = self.generator()
        with tempfile.TemporaryDirectory() as folder:
            generator.write_workflows(output_dir=folder, case_id="park", seed=123, trial_id=9)
            manifest = self.read(suffix=".index.json", folder=folder)
            self.assertEqual(manifest["selected_case"], dict(case_id="park", seed=123, trial_id=9))
            for stem in (STEM, EDITED):
                api = self.read(stem, folder=folder)
                self.assertEqual(api["8"]["inputs"]["background_phrase"], "park path")
                self.assertIn("park path", api["4"]["inputs"]["prompt"])
                for node in api.values():
                    if node["class_type"] in (COLLECT, SAMPLER):
                        self.assertEqual(node["inputs"]["seed"], 123)
                        self.assertEqual(node["inputs"]["trial_id"], 9)
                    if "filename_prefix" in node["inputs"]:
                        expected = "park_s123_t9_" if node["class_type"] == SAVE else "park/seed123/trial9/"
                        self.assertIn(expected, node["inputs"]["filename_prefix"])

    def test_invalid_generator_arguments_fail_before_writing(self):
        generator = self.generator()
        for kwargs in ({"case_id": "unknown"}, {"seed": True}, {"seed": -1}, {"seed": 2**64},
                       {"seed": 1.5}, {"trial_id": True}, {"trial_id": -1}, {"trial_id": 1.5}, {"trial_id": 2**31}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    generator.write_workflows(output_dir=folder, **kwargs)
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_maximum_seed_and_trial_stay_within_runtime_save_prefix_limits(self):
        generator = self.generator()
        with tempfile.TemporaryDirectory() as folder:
            generator.write_workflows(output_dir=folder, seed=2**64 - 1, trial_id=2**31 - 1)
            for stem in (STEM, EDITED):
                api = self.read(stem, folder=folder)
                prefix = api["10"]["inputs"]["filename_prefix"]
                self.assertIsNotNone(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", prefix))
                self.assertEqual(api["8"]["inputs"]["seed"], 2**64 - 1)
                self.assertEqual(api["8"]["inputs"]["trial_id"], 2**31 - 1)

    def test_custom_node_inputs_and_widget_order_match_actual_source_schema(self):
        schemas = {}
        for cls in ast.parse((ROOT / "nodes.py").read_text()).body:
            if not isinstance(cls, ast.ClassDef):
                continue
            method = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "define_schema"), None)
            if method is None:
                continue
            call = next(n.value for n in method.body if isinstance(n, ast.Return))
            fields = next(k.value.elts for k in call.keywords if k.arg == "inputs")
            schemas[cls.name] = fields
        self.assertTrue({COLLECT, SELECT, SAVE} <= set(schemas))
        for stem in (STEM, EDITED):
            api, ui = self.read(stem), self.read(stem, ".json")
            nodes = {str(n["id"]): n for n in ui["nodes"]}
            for sid, entry in api.items():
                if entry["class_type"] not in schemas:
                    continue
                fields = schemas[entry["class_type"]]
                names = [f.args[0].value for f in fields]
                required = {f.args[0].value for f in fields if not any(k.arg == "optional" and k.value.value is True for k in f.keywords)}
                self.assertLessEqual(set(entry["inputs"]), set(names))
                self.assertLessEqual(required, set(entry["inputs"]))
                self.assertEqual(list(nodes[sid]["widgets_values_named"]),
                    [name for name in names if name in entry["inputs"] and not is_link(entry["inputs"][name])])

    def test_cli_runs_without_importing_model_libraries(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = subprocess.run([sys.executable, "-S", str(ROOT / "tools/build_attention_validation_workflows.py"),
                "--case", "man_front", "--seed", "777", "--trial-id", "2", "--output-dir", folder],
                capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads(proc.stdout)
            self.assertEqual(result["status"], "generated_not_gpu_validated")
            self.assertEqual(len(result["files"]), 5)


if __name__ == "__main__":
    unittest.main()
