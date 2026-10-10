"""Stdlib-only contracts for generated hybrid validation graphs; no GPU claims.

Every behavioral assertion names an artifact/graph mutation that must be caught:
wrong output slots, duplicate slider paths, changed controls, swapped subtraction,
missing input gates, stale generated JSON, and UI/API drift. Run with unittest
(discovery pattern test_*workflow*.py also includes the prior 12 contracts).
"""
import ast
import importlib.util
import hashlib
import shutil
import json
import struct
import zlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
STEM = "krea2_female_slider_hybrid_validation"
MANUAL = "krea2_female_slider_hybrid_manual_d"
SAMPLER = "Krea2SliderFusePredictionMixSampler"
BASE = "krea2_female_slider_collection_comparison"
WIDGETS = {
    "UNETLoader": ["unet_name", "weight_dtype"],
    "LoraLoaderModelOnly": ["lora_name", "strength_model"],
    "CLIPLoader": ["clip_name", "type", "device"],
    "Krea2SliderFuseEncode": ["prompt"], "ConditioningZeroOut": [],
    "Krea2SliderFuseSubjects": ["target_phrase", "protected_phrase", "target_occurrence", "protected_occurrence"],
    "EmptySD3LatentImage": ["width", "height", "batch_size"],
    SAMPLER: ["lora_name", "strength", "seed", "control_after_generate", "steps", "cfg", "mix_scope", "trial_id", "diagnostic_level", "mask_mode", "collect_step", "collect_block", "top_k_ratio", "temperature", "fill_holes_max_area", "mask_dilate_radius", "selection_dilate_radius"],
    "VAELoader": ["vae_name"], "VAEDecode": [],
    "Krea2SliderFuseDiagnosticSave": ["filename_prefix"],
    "Krea2SliderFuseMaskPreview": [], "MaskToImage": [],
    "SaveImage": ["filename_prefix"], "MaskComposite": ["x", "y", "operation"],
    "LoadImageMask": ["image", "channel", "upload"],
}
OUTPUTS = {
    "UNETLoader": ["MODEL"], "LoraLoaderModelOnly": ["MODEL"], "CLIPLoader": ["CLIP"],
    "Krea2SliderFuseEncode": ["CONDITIONING", "KREA2_SLIDER_FUSE_PROMPT"],
    "ConditioningZeroOut": ["CONDITIONING"], "Krea2SliderFuseSubjects": ["KREA2_SLIDER_FUSE_SUBJECTS"],
    "EmptySD3LatentImage": ["LATENT"], SAMPLER: ["LATENT", "KREA2_SLIDER_FUSE_MASKS", "KREA2_SLIDER_FUSE_DIAGNOSTICS"],
    "VAELoader": ["VAE"], "VAEDecode": ["IMAGE"], "Krea2SliderFuseDiagnosticSave": [],
    "Krea2SliderFuseMaskPreview": ["MASK"] * 9, "MaskToImage": ["IMAGE"], "SaveImage": ["IMAGE"],
    "MaskComposite": ["MASK"], "LoadImageMask": ["MASK"],
}


def link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def write_png(path, width=64, height=64, color_type=0, pixel=b"\xff"):
    """Tiny real PNG fixture encoder; no Pillow or torch in test imports."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    raw = (b"\x00" + pixel * width) * height
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
    if color_type == 3:
        data += chunk(b"PLTE", b"\x00\x00\x00\xff\xff\xff")
    path.write_bytes(data + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class HybridValidationWorkflowTests(unittest.TestCase):
    def read(self, stem=STEM, suffix="_api.json", folder=None):
        path = (Path(folder) if folder else ROOT / "workflows") / (stem + suffix)
        self.assertTrue(path.is_file(), "Missing hybrid validation deliverable: " + path.name)
        return json.loads(path.read_text())

    def generator(self):
        path = ROOT / "tools/build_hybrid_validation_workflows.py"
        self.assertTrue(path.is_file(), "Missing hybrid validation generator")
        spec = importlib.util.spec_from_file_location("hybrid_generator", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def mask_exports(self, api):
        result = {}
        for n in api.values():
            if n["class_type"] == "SaveImage":
                upstream = api[n["inputs"]["images"][0]]
                if upstream["class_type"] == "MaskToImage":
                    result[n["inputs"]["filename_prefix"].split("/")[-1]] = upstream["inputs"]["mask"]
        return result

    def assert_parity(self, api, ui):
        self.assertEqual(ui["version"], 0.4)
        nodes = {n["id"]: n for n in ui["nodes"]}
        links = {v[0]: v for v in ui["links"]}
        self.assertEqual(set(api), {str(n["id"]) for n in nodes.values() if n["type"] != "Note"})
        self.assertEqual(len(links), len(ui["links"]))
        self.assertEqual(ui["last_node_id"], max(nodes))
        self.assertEqual(ui["last_link_id"], max(links))
        consumed, produced = [], []
        for sid, entry in api.items():
            node = nodes[int(sid)]
            self.assertEqual(node["type"], entry["class_type"])
            self.assertEqual(node["mode"], 0)
            self.assertIn(node["type"], OUTPUTS)
            self.assertEqual(len(node["outputs"]), len(OUTPUTS[node["type"]]))
            widget_values = dict(zip(WIDGETS[node["type"]], node["widgets_values"], strict=True))
            if node["type"] == SAMPLER:
                self.assertEqual(widget_values.pop("control_after_generate"), "fixed")
            if node["type"] == "LoadImageMask":
                self.assertEqual(widget_values.pop("upload"), "image")
            self.assertEqual(node["widgets_values_named"], widget_values)
            reconstructed = dict(widget_values)
            for slot, port in enumerate(node["inputs"]):
                if port["link"] is None:
                    self.assertNotIn(port["name"], entry["inputs"])
                    continue
                lid, origin, outslot, target, inslot, dtype = links[port["link"]]
                self.assertEqual((target, inslot), (int(sid), slot))
                self.assertEqual(dtype, port["type"])
                self.assertEqual(dtype, OUTPUTS[nodes[origin]["type"]][outslot])
                self.assertLess(nodes[origin]["order"], node["order"])
                reconstructed[port["name"]] = [str(origin), outslot]
                consumed.append(lid)
            for slot, port in enumerate(node["outputs"]):
                self.assertEqual(port["type"], OUTPUTS[node["type"]][slot])
                for lid in port["links"] or []:
                    self.assertEqual(links[lid][1:3], [int(sid), slot])
                    produced.append(lid)
            self.assertEqual(reconstructed, entry["inputs"], "UI/API mismatch: " + sid)
        self.assertCountEqual(consumed, links)
        self.assertCountEqual(produced, links)

    def test_deliverables_ui_api_and_existing_node_schema(self):
        for stem in (STEM, MANUAL):
            self.assert_parity(self.read(stem), self.read(stem, ".json"))
        manifest = self.read(suffix=".index.json")
        self.assertEqual(manifest["validation"]["gpu_execution"], "not_run")
        self.assertEqual(manifest["validation"]["pose_gate"], "not_evaluated")

    def test_all_graphs_are_acyclic_and_use_valid_output_slots(self):
        for stem in (STEM, MANUAL):
            api = self.read(stem)
            seen, active = set(), set()
            def visit(sid):
                self.assertNotIn(sid, active, "Cycle in API workflow")
                if sid in seen:
                    return
                active.add(sid)
                for value in api[sid]["inputs"].values():
                    if link(value):
                        source, slot = value
                        self.assertIn(source, api)
                        self.assertGreaterEqual(slot, 0)
                        self.assertLess(slot, len(OUTPUTS[api[source]["class_type"]]))
                        visit(source)
                active.remove(sid)
                seen.add(sid)
            for sid in api:
                visit(sid)

    def test_custom_input_schemas_still_match_source(self):
        schemas = {}
        tree = ast.parse((ROOT / "nodes.py").read_text())
        for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
            method = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "define_schema"), None)
            if not method:
                continue
            call = next(n.value for n in method.body if isinstance(n, ast.Return))
            fields = next(k.value.elts for k in call.keywords if k.arg == "inputs")
            schemas[cls.name] = ({f.args[0].value for f in fields}, {f.args[0].value for f in fields if not any(k.arg == "optional" and k.value.value is True for k in f.keywords)})
        for stem in (STEM, MANUAL):
            for node in self.read(stem).values():
                if node["class_type"] in schemas:
                    allowed, required = schemas[node["class_type"]]
                    self.assertLessEqual(set(node["inputs"]), allowed)
                    self.assertLessEqual(required, set(node["inputs"]))

    def test_collectors_only_change_upstream_model_and_keep_own_strength_zero(self):
        api = self.read()
        off, on = api["8"]["inputs"], api["11"]["inputs"]
        self.assertEqual(off["model"], ["2", 0])
        self.assertEqual(on["model"], ["10", 0])
        self.assertEqual({k: v for k, v in off.items() if k != "model"}, {k: v for k, v in on.items() if k != "model"})
        for k, v in {"strength": 0.0, "mix_scope": "none", "mask_mode": "auto", "selection_dilate_radius": 0, "mask_dilate_radius": 0, "fill_holes_max_area": 0}.items():
            self.assertEqual(off[k], v)
        self.assertEqual(api["10"]["inputs"], {"model": ["2", 0], "lora_name": off["lora_name"], "strength_model": 4.0})

    def test_all_finals_use_single_local_slider_original_style_and_empty_latent(self):
        abc, manual = self.read(), self.read(MANUAL)
        base = self.read(BASE)
        for api, ids in ((abc, ("21", "31", "42")), (manual, ("21",))):
            for sid in ("1", "2", "3", "7", "9"):
                self.assertEqual(api[sid]["inputs"], base[sid]["inputs"])
            for sid in ids:
                inputs = api[sid]["inputs"]
                for key, value in {"model": ["2", 0], "latent": ["7", 0], "positive": ["4", 0], "negative": ["5", 0], "prompt_info": ["4", 1], "strength": 4.0, "mask_mode": "manual", "mix_scope": "target_mask", "selection_dilate_radius": 4, "seed": 42, "steps": 8, "cfg": 1.0, "trial_id": 0, "diagnostic_level": "audit"}.items():
                    self.assertEqual(inputs[key], value, sid + ":" + key)
                self.assertEqual(inputs["lora_name"], abc["10"]["inputs"]["lora_name"])
                self.assertNotEqual(inputs["lora_name"], api["2"]["inputs"]["lora_name"])

    def test_hybrid_subtracts_off_protection_from_on_target(self):
        api = self.read()
        self.assertEqual(api["40"]["inputs"], {"destination": ["13", 0], "source": ["12", 1], "x": 0, "y": 0, "operation": "subtract"})
        for sid, target, protected in (("20", ["12", 0], ["12", 1]), ("30", ["13", 0], ["13", 1]), ("41", ["40", 0], ["12", 1])):
            self.assertEqual(api[sid]["inputs"]["target_mask"], target)
            self.assertEqual(api[sid]["inputs"]["protected_mask"], protected)

    def test_abc_save_effective_and_added_margin_from_final_banks_slots_7_8(self):
        api = self.read()
        exports = self.mask_exports(api)
        for branch, sampler, preview in (("A", "21", "100"), ("B", "31", "110"), ("C", "42", "120")):
            self.assertEqual(api[preview]["inputs"]["mask_bank"], [sampler, 1])
            self.assertEqual(exports[branch + "_effective_prediction_mask"], [preview, 7])
            self.assertEqual(exports[branch + "_selection_added_mask"], [preview, 8])
            self.assertIn(branch + "_reference_target_before_dilation", exports)
            self.assertEqual(exports[branch + "_reference_target_before_dilation"], [preview, 0])
            self.assertEqual(exports[branch + "_reference_protected"], [preview, 1])
        self.assertEqual(len(exports), 19)

    def test_new_case_prompt_and_human_pose_acceptance_are_explicit(self):
        api, manifest = self.read(), self.read(suffix=".index.json")
        cases = {c["case_id"]: c for c in manifest["cases"]}
        case = cases["woman_front_strong_overlap"]
        self.assertEqual(api["4"]["inputs"]["prompt"], case["prompt"])
        for phrase in ("exactly two fully clothed adults", "woman", "foreground", "man", "behind", "torso", "visible", "shoes"):
            self.assertIn(phrase, case["prompt"])
        self.assertEqual(case["seeds"], [42, 444444])
        self.assertEqual(case["optional_seeds"], [444444])
        self.assertTrue(case["human_pose_gate"]["required"])
        self.assertEqual(case["human_pose_gate"]["status"], "not_evaluated")
        gate = " ".join(case["human_pose_gate"]["criteria"]).lower()
        for phrase in ("woman", "foreground", "man", "behind", "torso", "face", "feet"):
            self.assertIn(phrase, gate)

    def test_historical_prompts_are_exact_and_have_traceable_sources(self):
        manifest = self.read(suffix=".index.json")
        cases = {c["case_id"]: c for c in manifest["cases"]}
        self.assertEqual(set(cases), {"original", "man_front", "park", "woman_front_strong_overlap"})
        expected_hashes = {
            "original": "995c65983d46bff3f68066b5bfa4a37ed12d0dd495c3914ae6476ff2a50d59c2",
            "man_front": "b4b333085ed9f924a6ead5f77225e6b823aa120e3ad04b6a775af24061b3b8e3",
            "park": "35f7f8b8f0ed7d4e7acfc20abd70bd4b8812350fc7c0b01f46fb214ac52148e1",
        }
        for cid, digest in expected_hashes.items():
            case = cases[cid]
            self.assertEqual(hashlib.sha256(case["prompt"].encode()).hexdigest(), digest)
            self.assertTrue((ROOT / case["prompt_source"]["path"]).is_file())
            self.assertEqual(case["prompt_source"]["status"], "historical_exact")
        self.assertEqual(cases["park"]["seeds"], [444444])

    def test_manual_d_uses_two_red_channel_inputs_with_correct_polarity(self):
        api = self.read(MANUAL)
        self.assertEqual({k for k, v in api.items() if v["class_type"] == SAMPLER}, {"21"})
        self.assertEqual({k for k, v in api.items() if v["class_type"] == "LoadImageMask"}, {"200", "201"})
        self.assertEqual(api["200"]["inputs"]["channel"], "red")
        self.assertEqual(api["201"]["inputs"]["channel"], "red")
        self.assertEqual(api["202"]["inputs"], {"destination": ["200", 0], "source": ["201", 0], "x": 0, "y": 0, "operation": "subtract"})
        self.assertEqual(api["20"]["inputs"]["target_mask"], ["202", 0])
        self.assertEqual(api["20"]["inputs"]["protected_mask"], ["201", 0])
        self.assertNotIn("SolidMask", {n["class_type"] for n in api.values()})
        self.assertNotIn("InvertMask", {n["class_type"] for n in api.values()})

    def test_manual_d_requires_human_support_and_saved_off_protection_without_fallback(self):
        api, manifest = self.read(MANUAL), self.read(suffix=".index.json")
        required = manifest["manual_d"]["required_files"]
        self.assertEqual(len(required), 2)
        for requirement in required:
            self.assertTrue(requirement["required"])
            self.assertFalse(requirement["provided"])
            self.assertIn("REQUIRED_", requirement["filename"])
            self.assertEqual(api[requirement["node_id"]]["inputs"]["image"], requirement["filename"])
        self.assertEqual(manifest["manual_d"]["missing_input_behavior"], "fail_validation_no_fallback")
        notes = "\n".join(n["widgets_values"][0] for n in self.read(MANUAL, ".json")["nodes"] if n["type"] == "Note")
        for phrase in ("not provided", "white=1", "black=0", "red", "P_off", "regrow", "radius=4", "same case", "seed"):
            self.assertIn(phrase, notes)

    def test_manual_d_saves_before_subtraction_before_dilation_and_after_final_masks(self):
        api = self.read(MANUAL)
        exports = self.mask_exports(api)
        self.assertEqual(exports["D_human_support_input"], ["200", 0])
        self.assertEqual(exports["D_saved_P_off_input"], ["201", 0])
        self.assertEqual(exports["D_support_minus_P_off"], ["202", 0])
        self.assertEqual(exports["D_reference_target_before_dilation"], ["100", 0])
        self.assertIn("D_reference_protected", exports)
        self.assertEqual(exports["D_reference_protected"], ["100", 1])
        self.assertEqual(exports["D_effective_prediction_mask"], ["100", 7])
        self.assertEqual(exports["D_selection_added_mask"], ["100", 8])
        self.assertEqual(api["100"]["inputs"]["mask_bank"], ["21", 1])

    def test_output_prefixes_are_unique_and_manifest_lists_every_active_save(self):
        manifest = self.read(suffix=".index.json")
        prefixes = []
        for label, stem in (("abc", STEM), ("manual_d", MANUAL)):
            api = self.read(stem)
            actual = {sid: n["inputs"]["filename_prefix"] for sid, n in api.items() if n["class_type"] in ("SaveImage", "Krea2SliderFuseDiagnosticSave")}
            self.assertEqual(actual, manifest["output_prefixes"][label])
            prefixes.extend(actual.values())
        self.assertEqual(len(prefixes), len(set(prefixes)))
        for prefix in prefixes:
            self.assertTrue(prefix.startswith("slider_hybrid_validation/woman_front_strong_overlap/seed42/r4/"))
            self.assertNotIn("..", prefix)

    def test_nfe_includes_all_active_saves_and_manual_increment(self):
        manifest = self.read(suffix=".index.json")
        nfe = manifest["nfe_estimate"]
        self.assertEqual(nfe["collector_each"], 10)
        self.assertEqual(nfe["final_each_partial_mask"], 16)
        self.assertEqual(nfe["abc_all_active_outputs"], 68)
        self.assertEqual(nfe["manual_d_all_active_outputs"], 16)
        self.assertEqual(nfe["abc_plus_manual_d"], 84)
        self.assertIn("all", nfe["assumptions"].lower())
        self.assertIn("cache", manifest["rerun_guidance"].lower())
        self.assertEqual(len(manifest["output_prefixes"]["abc"]), 24)
        self.assertEqual(len(manifest["output_prefixes"]["manual_d"]), 8)

    def test_generator_reproduces_committed_artifacts_byte_for_byte(self):
        module = self.generator()
        with tempfile.TemporaryDirectory() as directory:
            module.write_workflows(output_dir=Path(directory))
            for stem, suffix in ((STEM, ".json"), (STEM, "_api.json"), (MANUAL, ".json"), (MANUAL, "_api.json"), (STEM, ".index.json")):
                path = stem + suffix
                self.assertEqual((Path(directory) / path).read_bytes(), (ROOT / "workflows" / path).read_bytes(), path)

    def test_generator_applies_cases_seeds_and_radii_to_every_sampler(self):
        module = self.generator()
        with tempfile.TemporaryDirectory() as directory:
            for cid in ("original", "man_front", "park", "woman_front_strong_overlap"):
                for radius in (0, 2, 4):
                    module.write_workflows(output_dir=Path(directory), case_id=cid, seed=444444, radius=radius)
                    for stem in (STEM, MANUAL):
                        api = self.read(stem, folder=directory)
                        self.assert_parity(api, self.read(stem, ".json", directory))
                        for sid, n in api.items():
                            if n["class_type"] == SAMPLER:
                                self.assertEqual(n["inputs"]["seed"], 444444)
                                self.assertEqual(n["inputs"]["selection_dilate_radius"], 0 if sid in ("8", "11") else radius)
                        for n in api.values():
                            if "filename_prefix" in n["inputs"]:
                                self.assertIn(f"/{cid}/seed444444/r{radius}/", n["inputs"]["filename_prefix"])

    def test_generator_rejects_missing_manual_inputs_when_preflight_requested(self):
        module = self.generator()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "REQUIRED_"):
                module.write_workflows(output_dir=Path(directory) / "out", input_dir=Path(directory))
            self.assertFalse((Path(directory) / "out").exists())
            for name in ("human.png", "P_off.png"):
                write_png(Path(directory) / name)
            module.write_workflows(output_dir=Path(directory) / "out", input_dir=Path(directory), support_mask="human.png", protected_mask="P_off.png")
            manifest = self.read(suffix=".index.json", folder=Path(directory) / "out")
            self.assertTrue(all(f["provided"] for f in manifest["manual_d"]["required_files"]))
            self.assertEqual(manifest["manual_d"]["input_validation"], "binary_grayscale_equal_dimensions_human_alignment_not_verified")

    def test_manual_preflight_rejects_misaligned_nonbinary_and_ambiguous_mask_files(self):
        module = self.generator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_png(root / "P_off.png")
            cases = (
                {"width": 32},
                {"pixel": b"\x80"},
                {"color_type": 2, "pixel": b"\xff\x00\x00"},
                {"color_type": 6, "pixel": b"\xff\xff\xff\xff"},
                {"color_type": 3, "pixel": b"\x01"},
            )
            for options in cases:
                write_png(root / "human.png", **options)
                with self.assertRaises(ValueError):
                    module.write_workflows(output_dir=root / "out", input_dir=root, support_mask="human.png", protected_mask="P_off.png")
                self.assertFalse((root / "out").exists())
            write_png(root / "human.png", color_type=2, pixel=b"\xff\xff\xff")
            before = (root / "human.png").read_bytes()
            module.write_workflows(output_dir=root / "out", input_dir=root, support_mask="human.png", protected_mask="P_off.png")
            self.assertEqual((root / "human.png").read_bytes(), before)

    def test_preflight_rejects_input_symlink_escape(self):
        module = self.generator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input").mkdir()
            write_png(root / "outside.png")
            write_png(root / "input/P_off.png")
            (root / "input/human.png").symlink_to(root / "outside.png")
            with self.assertRaises(ValueError):
                module.write_workflows(output_dir=root / "out", input_dir=root / "input", support_mask="human.png", protected_mask="P_off.png")
            self.assertFalse((root / "out").exists())

    def test_generator_runs_from_clean_published_dependencies_without_evidence_archives(self):
        self.generator()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tools").mkdir()
            (root / "workflows").mkdir()
            shutil.copy2(ROOT / "tools/build_hybrid_validation_workflows.py", root / "tools")
            for suffix in (".json", "_api.json", ".index.json"):
                shutil.copy2(ROOT / "workflows" / (BASE + suffix), root / "workflows")
            result = subprocess.run([sys.executable, str(root / "tools/build_hybrid_validation_workflows.py"), "--case", "park"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = self.read(suffix=".index.json", folder=root / "workflows")
            self.assertEqual(manifest["selected_case"]["seed"], 444444)
            self.assertFalse((root / "test-results").exists())

    def test_preflight_rejects_transparency_metadata_and_animated_png(self):
        module = self.generator()
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_png(root / "P_off.png")
            white, black = Image.new("L", (64, 64), 255), Image.new("L", (64, 64), 0)
            white.save(root / "human.png", transparency=0)
            with self.assertRaises(ValueError):
                module.write_workflows(output_dir=root / "out", input_dir=root, support_mask="human.png", protected_mask="P_off.png")
            white.save(root / "human.png", save_all=True, append_images=[black], duration=100, loop=0)
            with self.assertRaises(ValueError):
                module.write_workflows(output_dir=root / "out", input_dir=root, support_mask="human.png", protected_mask="P_off.png")
            self.assertFalse((root / "out").exists())

    def test_generator_cli_rejects_invalid_settings_and_unsafe_input_paths(self):
        module = self.generator()
        for kwargs in ({"radius": 3}, {"radius": True}, {"radius": 2.0}, {"case_id": "invented"}, {"seed": -1}, {"seed": True}, {"seed": 1.5}, {"seed": 2 ** 64}, {"support_mask": "../bad.png"}, {"support_mask": "..\\bad.png"}, {"support_mask": "C:\\bad.png"}, {"protected_mask": "/bad.png"}):
            with self.assertRaises(ValueError):
                module.write_workflows(output_dir=Path("never-written"), **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(ROOT / "tools/build_hybrid_validation_workflows.py"), "--case", "woman_front_strong_overlap", "--seed", "444444", "--radius", "2", "--output-dir", directory], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = self.read(suffix=".index.json", folder=directory)
            self.assertEqual(manifest["selected_case"]["seed"], 444444)
            self.assertEqual(manifest["selected_case"]["radius"], 2)


if __name__ == "__main__":
    unittest.main()
