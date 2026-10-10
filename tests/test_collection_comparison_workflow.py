"""Static contracts for the collection comparison; does not import ComfyUI.

Run from the repository root: python -m unittest discover -s tests -p
'test_collection_comparison_workflow.py' -v. For a standalone artifact staging
directory, set FREEFUSE_SOURCE_ROOT to the matching source checkout. This suite
checks graph/schema compatibility, not GPU execution or image quality.
"""

import ast
import json
import os
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(os.environ.get("FREEFUSE_SOURCE_ROOT", ROOT))
STEM = "krea2_female_slider_collection_comparison"
SAMPLER = "Krea2SliderFusePredictionMixSampler"
COLLECTORS = ("8", "11")
FINALS = ("21", "31", "42")
SAMPLERS = COLLECTORS + FINALS
WIDGETS = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "Krea2SliderFuseEncode": ["prompt"],
    "ConditioningZeroOut": [],
    "Krea2SliderFuseSubjects": ["target_phrase", "protected_phrase", "target_occurrence", "protected_occurrence"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    SAMPLER: ["lora_name", "strength", "seed", "control_after_generate", "steps", "cfg", "mix_scope", "trial_id", "diagnostic_level", "mask_mode", "collect_step", "collect_block", "top_k_ratio", "temperature", "fill_holes_max_area", "mask_dilate_radius", "selection_dilate_radius"],
    "VAELoader": ["vae_name"],
    "VAEDecode": [],
    "Krea2SliderFuseDiagnosticSave": ["filename_prefix"],
    "Krea2SliderFuseMaskPreview": [],
    "MaskToImage": [],
    "SaveImage": ["filename_prefix"],
    "MaskComposite": ["x", "y", "operation"],
}
OUTPUTS = {
    "UNETLoader": ["MODEL"], "LoraLoaderModelOnly": ["MODEL"],
    "CLIPLoader": ["CLIP"], "Krea2SliderFuseEncode": ["CONDITIONING", "KREA2_SLIDER_FUSE_PROMPT"],
    "ConditioningZeroOut": ["CONDITIONING"],
    "Krea2SliderFuseSubjects": ["KREA2_SLIDER_FUSE_SUBJECTS"],
    "EmptySD3LatentImage": ["LATENT"],
    SAMPLER: ["LATENT", "KREA2_SLIDER_FUSE_MASKS", "KREA2_SLIDER_FUSE_DIAGNOSTICS"],
    "VAELoader": ["VAE"], "VAEDecode": ["IMAGE"],
    "Krea2SliderFuseDiagnosticSave": [], "Krea2SliderFuseMaskPreview": ["MASK"] * 9,
    "MaskToImage": ["IMAGE"], "SaveImage": ["IMAGE"], "MaskComposite": ["MASK"],
}


def is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


class CollectionComparisonWorkflowTests(unittest.TestCase):
    def read_json(self, suffix):
        path = ROOT / "workflows" / (STEM + suffix)
        self.assertTrue(path.is_file(), "Missing comparison deliverable: " + path.name)
        return json.loads(path.read_text())

    def api(self):
        return self.read_json("_api.json")

    def test_deliverables_are_complete_and_only_existing_nodes_are_used(self):
        api = self.api()
        ui = self.read_json(".json")
        index = self.read_json(".index.json")
        self.assertEqual(ui["version"], 0.4)
        self.assertEqual(set(api), {str(n["id"]) for n in ui["nodes"] if n["type"] != "Note"})
        self.assertEqual({k for k, v in api.items() if v["class_type"] == SAMPLER}, set(SAMPLERS))
        self.assertEqual(index["validation"]["gpu_execution"], "not_run")
        for node in api.values():
            self.assertIn(node["class_type"], OUTPUTS)
            self.assertNotIn("collection_mode", node["inputs"])

    def test_api_graph_is_a_dag_and_every_output_slot_exists(self):
        api = self.api()
        complete, active = set(), set()

        def visit(node_id):
            self.assertNotIn(node_id, active, "Cycle in API graph")
            if node_id in complete:
                return
            active.add(node_id)
            for value in api[node_id]["inputs"].values():
                if is_link(value):
                    origin, slot = value
                    self.assertIn(origin, api)
                    self.assertGreaterEqual(slot, 0)
                    self.assertLess(slot, len(OUTPUTS[api[origin]["class_type"]]))
                    visit(origin)
            active.remove(node_id)
            complete.add(node_id)

        for node_id in api:
            visit(node_id)
        self.assertEqual(complete, set(api))

    def test_ui_links_widget_positions_and_api_are_equivalent(self):
        api = self.api()
        ui = self.read_json(".json")
        nodes = {n["id"]: n for n in ui["nodes"]}
        links = {link[0]: link for link in ui["links"]}
        self.assertEqual(len(links), len(ui["links"]))
        self.assertEqual(ui["last_node_id"], max(nodes))
        self.assertEqual(ui["last_link_id"], max(links))
        consumed, produced = [], []
        for sid, entry in api.items():
            node = nodes[int(sid)]
            self.assertEqual(node["type"], entry["class_type"])
            self.assertEqual(node["mode"], 0)
            self.assertEqual(len(node["outputs"]), len(OUTPUTS[node["type"]]))
            if node["type"] == "SaveImage":
                self.assertEqual(node["outputs"][0]["name"], "images")
            widgets = dict(zip(WIDGETS[node["type"]], node["widgets_values"], strict=True))
            if node["type"] == SAMPLER:
                self.assertEqual(widgets.pop("control_after_generate"), "fixed")
            reconstructed = dict(widgets)
            for slot, port in enumerate(node["inputs"]):
                if port["link"] is None:
                    self.assertNotIn(port["name"], entry["inputs"])
                    continue
                lid, origin, origin_slot, target, target_slot, dtype = links[port["link"]]
                self.assertEqual((target, target_slot), (int(sid), slot))
                self.assertEqual(dtype, port["type"])
                self.assertEqual(dtype, OUTPUTS[nodes[origin]["type"]][origin_slot])
                self.assertLess(nodes[origin]["order"], node["order"])
                reconstructed[port["name"]] = [str(origin), origin_slot]
                consumed.append(lid)
            for slot, port in enumerate(node["outputs"]):
                self.assertEqual(port["type"], OUTPUTS[node["type"]][slot])
                for lid in port["links"] or []:
                    self.assertEqual(links[lid][1:3], [int(sid), slot])
                    produced.append(lid)
            self.assertEqual(reconstructed, entry["inputs"], "API/UI mismatch at " + sid)
            self.assertEqual(node["widgets_values_named"], widgets)
        self.assertCountEqual(consumed, links)
        self.assertCountEqual(produced, links)

    def test_all_custom_inputs_match_existing_source_without_importing_it(self):
        api = self.api()
        source = SOURCE_ROOT / "nodes.py"
        self.assertTrue(source.is_file(), "Set FREEFUSE_SOURCE_ROOT to the matching checkout for source validation")
        tree = ast.parse(source.read_text())
        schemas = {}
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            method = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "define_schema"), None)
            if method is None:
                continue
            call = next((n.value for n in method.body if isinstance(n, ast.Return)), None)
            if not isinstance(call, ast.Call):
                continue
            inputs = next(kw.value for kw in call.keywords if kw.arg == "inputs")
            allowed, required, order = set(), set(), []
            for field in inputs.elts:
                name = field.args[0].value
                allowed.add(name)
                order.append(name)
                optional = any(k.arg == "optional" and isinstance(k.value, ast.Constant) and k.value.value is True for k in field.keywords)
                if not optional:
                    required.add(name)
            schemas[cls.name] = (allowed, required, order)
        for node in api.values():
            typ = node["class_type"]
            if not typ.startswith("Krea2SliderFuse"):
                continue
            allowed, required, schema_order = schemas[typ]
            self.assertLessEqual(set(node["inputs"]), allowed)
            self.assertLessEqual(required, set(node["inputs"]))
            schema_widgets = [n for n in schema_order if n in WIDGETS[typ]]
            self.assertEqual(schema_widgets, [n for n in WIDGETS[typ] if n != "control_after_generate"])

    def test_collectors_differ_only_in_upstream_model(self):
        api = self.api()
        off = api["8"]["inputs"]
        on = api["11"]["inputs"]
        self.assertEqual(off["model"], ["2", 0])
        self.assertEqual(on["model"], ["10", 0])
        self.assertEqual({k: v for k, v in off.items() if k != "model"}, {k: v for k, v in on.items() if k != "model"})
        self.assertEqual(api["10"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(api["10"]["inputs"], {"model": ["2", 0], "lora_name": off["lora_name"], "strength_model": 4.0})
        for collector in (off, on):
            for name, value in {"strength": 0.0, "mix_scope": "none", "mask_mode": "auto", "collect_step": 2, "collect_block": 18, "top_k_ratio": 0.2, "temperature": 10000.0, "fill_holes_max_area": 0, "mask_dilate_radius": 0, "selection_dilate_radius": 0}.items():
                self.assertEqual(collector[name], value, name)
        self.assertEqual(api["12"]["inputs"]["mask_bank"], ["8", 1])
        self.assertEqual(api["13"]["inputs"]["mask_bank"], ["11", 1])

    def test_final_sampler_baselines_do_not_double_apply_the_slider(self):
        api = self.api()
        final_keys = {"strength": 4.0, "mask_mode": "manual", "mix_scope": "target_mask", "selection_dilate_radius": 4, "diagnostic_level": "audit"}
        for node_id in FINALS:
            inputs = api[node_id]["inputs"]
            self.assertEqual(inputs["model"], ["2", 0])
            self.assertEqual(inputs["lora_name"], api["10"]["inputs"]["lora_name"])
            for key, expected in final_keys.items():
                self.assertEqual(inputs[key], expected)
        self.assertNotEqual(api["2"]["inputs"]["lora_name"], api["10"]["inputs"]["lora_name"])
        self.assertEqual(api["2"]["inputs"]["model"], ["1", 0])

    def test_initial_latent_conditioning_and_sampling_settings_are_shared(self):
        api = self.api()
        for node_id in SAMPLERS:
            inputs = api[node_id]["inputs"]
            for name, value in {"latent": ["7", 0], "positive": ["4", 0], "negative": ["5", 0], "prompt_info": ["4", 1], "seed": 42, "steps": 8, "cfg": 1.0, "trial_id": 0}.items():
                self.assertEqual(inputs[name], value, node_id + ":" + name)
        self.assertEqual(api["7"]["inputs"], {"width": 1024, "height": 1024, "batch_size": 1})
        self.assertEqual(api["5"]["inputs"]["conditioning"], ["4", 0])
        self.assertIn("fully clothed adults", api["4"]["inputs"]["prompt"])
        self.assertEqual(api["3"]["inputs"]["clip_name"], "qwen3vl_4b_bf16.safetensors")

    def test_branch_mask_sources_and_subtraction_polarity(self):
        api = self.api()
        for sampler_id, subjects_id, target, protected in (("21", "20", ["12", 0], ["12", 1]), ("31", "30", ["13", 0], ["13", 1]), ("42", "41", ["40", 0], ["12", 1])):
            self.assertEqual(api[sampler_id]["inputs"]["subjects"], [subjects_id, 0])
            inputs = api[subjects_id]["inputs"]
            self.assertEqual(inputs["target_mask"], target)
            self.assertEqual(inputs["protected_mask"], protected)
            for key in ("prompt_info", "target_phrase", "protected_phrase", "target_occurrence", "protected_occurrence"):
                self.assertEqual(inputs[key], api["6"]["inputs"][key])
        for node_id, destination, source, operation in (("40", ["13", 0], ["12", 1], "subtract"), ("50", ["13", 0], ["12", 1], "multiply"), ("51", ["12", 0], ["13", 0], "subtract")):
            self.assertEqual(api[node_id], {"class_type": "MaskComposite", "inputs": {"destination": destination, "source": source, "x": 0, "y": 0, "operation": operation}, "_meta": api[node_id]["_meta"]})
        # Old-minus-new is an observation only; no final sampler may consume it.
        self.assertEqual([n["class_type"] for n in api.values() if ["51", 0] in n["inputs"].values()], ["MaskToImage"])

    def test_terminal_diagnostic_saves_have_supported_single_style_provenance(self):
        api = self.api()
        saves = [n for n in api.values() if n["class_type"] == "Krea2SliderFuseDiagnosticSave"]
        self.assertEqual(len(saves), 3)
        saved = []
        for save in saves:
            s = save["inputs"]
            sampler_id, output = s["latent"]
            saved.append(sampler_id)
            self.assertIn(sampler_id, FINALS)
            self.assertEqual(output, 0)
            self.assertEqual(s["diagnostics"], [sampler_id, 2])
            decode_id, image_slot = s["images"]
            self.assertEqual(image_slot, 0)
            self.assertEqual(api[decode_id]["class_type"], "VAEDecode")
            self.assertEqual(api[decode_id]["inputs"], {"samples": [sampler_id, 0], "vae": ["9", 0]})
            current = api[sampler_id]["inputs"]["model"][0]
            loaders = []
            while api[current]["class_type"] == "LoraLoaderModelOnly":
                loaders.append(api[current]["inputs"])
                current = api[current]["inputs"]["model"][0]
            self.assertEqual(len(loaders), 1)
            self.assertEqual(api[current]["class_type"], "UNETLoader")
            self.assertNotEqual(loaders[0]["lora_name"], api[sampler_id]["inputs"]["lora_name"])
        self.assertCountEqual(saved, FINALS)

    def test_raw_masks_conflict_and_references_are_saved_with_unique_prefixes(self):
        api = self.api()
        sources = []
        prefixes = []
        for node in api.values():
            if node["class_type"] not in ("SaveImage", "Krea2SliderFuseDiagnosticSave"):
                continue
            prefix = node["inputs"]["filename_prefix"]
            prefixes.append(prefix)
            self.assertNotIn("..", prefix)
            self.assertFalse(prefix.startswith("/"))
            if node["class_type"] == "SaveImage":
                upstream = api[node["inputs"]["images"][0]]
                self.assertIn(upstream["class_type"], ("MaskToImage", "VAEDecode"))
                sources.append(tuple(upstream["inputs"].get("mask", upstream["inputs"].get("samples"))))
        self.assertEqual(len(prefixes), len(set(prefixes)))
        self.assertCountEqual(sources, [("12", 0), ("12", 1), ("13", 0), ("13", 1), ("40", 0), ("50", 0), ("51", 0), ("8", 0), ("11", 0)])

    def test_manifest_settings_maps_and_conditional_nfe_are_consistent(self):
        api = self.api()
        index = self.read_json(".index.json")
        for group, node_ids in (("seed", SAMPLERS), ("steps", SAMPLERS), ("trial_id", SAMPLERS)):
            locations = index["coordinated_settings"][group]["api_inputs"]
            self.assertCountEqual(locations, [[node_id, group] for node_id in node_ids])
        strength_inputs = index["coordinated_settings"]["slider_strength"]["api_inputs"]
        self.assertCountEqual(strength_inputs, [["10", "strength_model"]] + [[n, "strength"] for n in FINALS])
        for group in index["coordinated_settings"].values():
            for node_id, input_name in group["api_inputs"]:
                self.assertIn(input_name, api[node_id]["inputs"])
        collector_nfe = api["8"]["inputs"]["collect_step"] + api["8"]["inputs"]["steps"]
        final_nfe = 2 * api["21"]["inputs"]["steps"]
        self.assertEqual(index["nfe_estimate"]["collector_each"], collector_nfe)
        self.assertEqual(index["nfe_estimate"]["final_each_partial_mask"], final_nfe)
        self.assertEqual(index["nfe_estimate"]["shared_full_comparison"], 2 * collector_nfe + 3 * final_nfe)
        self.assertEqual(index["nfe_estimate"]["isolated_a_or_b"], collector_nfe + final_nfe)
        self.assertEqual(index["nfe_estimate"]["isolated_c"], 2 * collector_nfe + final_nfe)
        self.assertTrue(index["nfe_estimate"]["assumptions"])
        self.assertIn("cache", index["rerun_guidance"].lower())

    def test_ui_explains_collector_strength_and_experiment_limits(self):
        ui = self.read_json(".json")
        notes = "\n".join(n["widgets_values"][0] for n in ui["nodes"] if n["type"] == "Note")
        for text in ("strength=0", "OFF", "ON", "A:", "B:", "C:", "geometry", "trial_id", "DiagnosticSave"):
            self.assertIn(text, notes)


if __name__ == "__main__":
    unittest.main()
