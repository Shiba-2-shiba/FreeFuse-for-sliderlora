"""Static ComfyUI import contracts, not GPU execution or image-quality evidence."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
CASES = ("park", "strong_overlap")
MODES = ("fast", "compare")
PILOT = "Krea2SliderFusePairedPrefixPilot"
SAVE = "Krea2SliderFusePrefixPilotSave"
SAMPLER = "Krea2SliderFusePredictionMixSampler"
COLLECT = "Krea2SliderFuseExperimentalMaskCollect"
SEED_NODES = {PILOT, SAMPLER, COLLECT}
PORT_TYPES = {
    "Model": "MODEL", "Conditioning": "CONDITIONING", "Latent": "LATENT",
    "Clip": "CLIP", "Vae": "VAE", "Image": "IMAGE", "Mask": "MASK",
    "PromptType": "KREA2_SLIDER_FUSE_PROMPT", "SubjectsType": "KREA2_SLIDER_FUSE_SUBJECTS",
    "MasksType": "KREA2_SLIDER_FUSE_MASKS", "DiagnosticsType": "KREA2_SLIDER_FUSE_DIAGNOSTICS",
    "ExperimentType": "KREA2_SLIDER_FUSE_MASK_EXPERIMENT", "ProbeType": "KREA2_SLIDER_FUSE_PAIRED_PROBE",
}


def builder():
    path = ROOT / "tools/build_prefix_pilot_ui.py"
    assert path.is_file(), "Missing conventional ComfyUI prefix workflow builder"
    spec = importlib.util.spec_from_file_location("prefix_ui_test_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stem(case, mode):
    return f"krea2_prefix_pilot_{case}_{mode}"


def read(case, mode, *, ui=False, directory=None):
    path = (Path(directory) if directory else ROOT / "workflows") / (stem(case, mode) + (".json" if ui else "_api.json"))
    assert path.is_file(), f"Missing prefix workflow: {path.name}"
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if ui else value.get("prompt", value)


def is_link(value):
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int


def custom_schemas(direction="inputs"):
    """Read actual V3 declarations without importing ComfyUI or model libraries."""
    methods = {}
    for filename in ("nodes.py", "probe_nodes.py"):
        for cls in ast.parse((ROOT / filename).read_text()).body:
            if isinstance(cls, ast.ClassDef):
                method = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "define_schema"), None)
                if method is not None:
                    methods[cls.name] = method

    def fields(name):
        method = methods[name]
        calls = [n.value for n in method.body if isinstance(n, ast.Return)]
        direct = next((call for call in calls if isinstance(call, ast.Call)), None)
        if direct is not None:
            return next(k.value.elts for k in direct.keywords if k.arg == direction)
        inherited = next(n.value.func.value.id for n in method.body if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "define_schema")
        result = list(fields(inherited))
        for n in method.body:
            if (isinstance(n, ast.Assign) and isinstance(n.value, ast.List)
                    and any(isinstance(target, ast.Attribute) and target.attr == direction for target in n.targets)):
                result = list(n.value.elts)
            if (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "append"
                    and isinstance(n.value.func.value, ast.Attribute) and n.value.func.value.attr == direction):
                result.append(n.value.args[0])
        return result

    return {name: fields(name) for name in methods}


def field_kind(field):
    owner = field.func.value
    return owner.attr if isinstance(owner, ast.Attribute) else owner.id


def test_prefix_ui_builder_exists():
    assert callable(builder().write_workflows)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", MODES)
def test_ui_api_roundtrip_has_complete_typed_topological_links_and_widgets(case, mode):
    api, ui = read(case, mode), read(case, mode, ui=True)
    nodes = {str(node["id"]): node for node in ui["nodes"]}
    assert len(nodes) == len(ui["nodes"])
    assert ui["version"] == .4
    assert set(api) == {sid for sid, node in nodes.items() if node["type"] != "Note"}
    assert ui["last_node_id"] == max(node["id"] for node in nodes.values())
    links = {link[0]: link for link in ui["links"]}
    assert len(links) == len(ui["links"])
    assert ui["last_link_id"] == max(links)
    consumed, produced = [], []
    for sid, entry in api.items():
        node = nodes[sid]
        assert node["type"] == entry["class_type"]
        assert node["mode"] == 0
        assert all(isinstance(v, (int, float)) for v in node["pos"] + node["size"])
        named = node["widgets_values_named"]
        values = list(named.values())
        if node["type"] in SEED_NODES:
            values.insert(list(named).index("seed") + 1, "fixed")
        assert node["widgets_values"] == values
        reconstructed = dict(named)
        for slot, port in enumerate(node["inputs"]):
            if port["link"] is None:
                assert port["name"] not in entry["inputs"]
                continue
            lid, origin, outslot, target, inslot, kind = links[port["link"]]
            assert (target, inslot) == (int(sid), slot)
            assert kind == port["type"] == nodes[str(origin)]["outputs"][outslot]["type"]
            assert nodes[str(origin)]["order"] < node["order"]
            reconstructed[port["name"]] = [str(origin), outslot]
            consumed.append(lid)
        assert reconstructed == entry["inputs"]
        for slot, output in enumerate(node["outputs"]):
            for lid in output["links"]:
                assert links[lid][1:3] == [int(sid), slot]
                produced.append(lid)
    assert sorted(consumed) == sorted(links) == sorted(produced)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", MODES)
def test_custom_input_ports_and_widget_order_match_live_source_schema(case, mode):
    api, ui, schemas = read(case, mode), read(case, mode, ui=True), custom_schemas()
    nodes = {str(node["id"]): node for node in ui["nodes"]}
    for sid, entry in api.items():
        if entry["class_type"] not in schemas:
            continue
        fields = schemas[entry["class_type"]]
        required = {field.args[0].value for field in fields if not any(
            k.arg == "optional" and isinstance(k.value, ast.Constant) and k.value.value is True for k in field.keywords)}
        names = [field.args[0].value for field in fields]
        assert required <= set(entry["inputs"]) <= set(names)
        assert [(p["name"], p["type"]) for p in nodes[sid]["inputs"]] == [
            (field.args[0].value, PORT_TYPES[field_kind(field)]) for field in fields if field_kind(field) in PORT_TYPES]
        assert list(nodes[sid]["widgets_values_named"]) == [field.args[0].value for field in fields
            if field_kind(field) not in PORT_TYPES and field.args[0].value in entry["inputs"]]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", MODES)
def test_every_output_slot_matches_actual_custom_schema_and_builtin_contract(case, mode):
    ui, schemas = read(case, mode, ui=True), custom_schemas("outputs")
    output_types = dict(PORT_TYPES, String="STRING")
    builtins = {"UNETLoader": ["MODEL"], "LoraLoaderModelOnly": ["MODEL"], "CLIPLoader": ["CLIP"],
        "ConditioningZeroOut": ["CONDITIONING"], "EmptySD3LatentImage": ["LATENT"],
        "VAELoader": ["VAE"], "VAEDecode": ["IMAGE"]}
    for node in ui["nodes"]:
        if node["type"] in schemas:
            assert [(p["name"], p["type"]) for p in node["outputs"]] == [
                (field.args[0].value, output_types[field_kind(field)]) for field in schemas[node["type"]]]
        elif node["type"] != "Note":
            assert [p["type"] for p in node["outputs"]] == builtins[node["type"]]


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", MODES)
def test_visible_nodes_do_not_overlap_and_execution_groups_contain_them(case, mode):
    ui = read(case, mode, ui=True)
    def rectangle(node):
        x, y = node["pos"]
        width, height = node["size"]
        return x, y-30, x+width, y+height  # Include the title bar.
    for index, node in enumerate(ui["nodes"]):
        x1, y1, x2, y2 = rectangle(node)
        for other in ui["nodes"][index+1:]:
            a1, b1, a2, b2 = rectangle(other)
            assert x2 <= a1 or a2 <= x1 or y2 <= b1 or b2 <= y1, (node["id"], other["id"])
        if node["type"] != "Note":
            assert sum(x <= x1 and y <= y1 and x+width >= x2 and y+height >= y2
                       for x, y, width, height in (g["bounding"] for g in ui["groups"])) == 1


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", MODES)
def test_no_partial_latent_or_inherited_off_branch_and_all_nodes_reach_saves(case, mode):
    api, ui = read(case, mode), read(case, mode, ui=True)
    nodes = {str(node["id"]): node for node in ui["nodes"]}
    pilots = {sid: node for sid, node in api.items() if node["class_type"] == PILOT}
    assert len(pilots) == (1 if mode == "fast" else 2)
    assert {n["inputs"]["pilot_mode"] for n in pilots.values()} == ({"prefix"} if mode == "fast" else {"prefix", "full_reference"})
    assert sum(n["class_type"] == COLLECT for n in api.values()) == 1
    assert sum(n["class_type"] == "LoraLoaderModelOnly" for n in api.values()) == 1
    for sid in pilots:
        assert [(p["name"], p["type"]) for p in nodes[sid]["outputs"]] == [
            ("mask_bank", "KREA2_SLIDER_FUSE_MASKS"), ("probe", "KREA2_SLIDER_FUSE_PAIRED_PROBE")]
        assert api[sid]["inputs"]["latent"] == ["7", 0]
    seen = set()
    def visit(sid):
        if sid in seen:
            return
        seen.add(sid)
        for value in api[sid]["inputs"].values():
            if is_link(value):
                visit(value[0])
    for sid, node in api.items():
        if node["class_type"].endswith("Save"):
            visit(sid)
        if node["class_type"] == SAVE:
            assert set(node["inputs"]) == {"probe", "mask_bank", "filename_prefix"}
            assert node["inputs"]["probe"] == [node["inputs"]["mask_bank"][0], 1]
        if node["class_type"] == SAMPLER:
            assert node["inputs"]["steps"] == 8
            assert node["inputs"]["latent"] == ["7", 0]
        if node["class_type"] == "VAEDecode":
            assert api[node["inputs"]["samples"][0]]["class_type"] == SAMPLER
    assert seen == set(api)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("mode", MODES)
def test_annotations_explain_nonfinal_prefix_full_replay_and_unvalidated_costs(case, mode):
    ui = read(case, mode, ui=True)
    notes = " ".join(n["widgets_values"][0] for n in ui["nodes"] if n["type"] == "Note").lower()
    for phrase in ("non-final", "original noise", "8 steps", "same full 8-step", "expected", "nfe", "not gpu-validated", "human", "pose"):
        assert phrase in notes
    assert ("2 + 6 + 16 = 24" if mode == "fast" else "2 + 16 + 6 + 32 = 56") in notes
    if case == "strong_overlap":
        assert "6/6" in notes


def test_generator_reproduces_only_four_ui_files_with_literal_and_lf_parity(tmp_path):
    generator = builder()
    marker = tmp_path / "keep.txt"
    marker.write_text("unchanged")
    original_factory = generator.tempfile.NamedTemporaryFile
    def windows_factory(*args, **kwargs):
        kwargs.setdefault("newline", "\r\n")
        return original_factory(*args, **kwargs)
    with patch.object(generator.tempfile, "NamedTemporaryFile", side_effect=windows_factory):
        paths = generator.write_workflows(output_dir=tmp_path)
    assert len(paths) == 4
    assert {Path(p).name for p in paths} == {stem(c, m) + ".json" for c in CASES for m in MODES}
    for path in map(Path, paths):
        assert path.read_bytes() == (ROOT / "workflows" / path.name).read_bytes()
        assert b"\r\n" not in path.read_bytes()
        assert path.read_bytes().endswith(b"\n")
    assert marker.read_text() == "unchanged"


@pytest.mark.parametrize("mutation", ("missing_source", "bad_slot", "wrong_type", "unknown_literal", "cycle"))
def test_converter_rejects_invalid_graph_instead_of_silently_dropping_data(mutation):
    generator = builder()
    graph = copy.deepcopy(read("park", "fast"))
    if mutation == "missing_source":
        graph["200"]["inputs"]["model"] = ["99999", 0]
    elif mutation == "bad_slot":
        graph["200"]["inputs"]["model"] = ["2", 10]
    elif mutation == "wrong_type":
        graph["200"]["inputs"]["model"] = ["7", 0]
    elif mutation == "unknown_literal":
        graph["200"]["inputs"]["misspelled_field"] = 123
    else:
        graph["2"]["inputs"]["model"] = ["2", 0]
    with pytest.raises(ValueError):
        generator.make_ui(graph, case_id="park", mode="fast")


def test_cli_is_stdlib_only_and_preserves_the_source_api_graphs(tmp_path):
    snapshots = {p: p.read_bytes() for p in (ROOT / "workflows").glob("krea2_prefix_pilot_*_api.json")}
    proc = subprocess.run([sys.executable, "-S", str(ROOT / "tools/build_prefix_pilot_ui.py"),
        "--output-dir", str(tmp_path)], text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "generated_not_gpu_validated"
    assert len(result["files"]) == 4
    assert all(path.read_bytes() == contents for path, contents in snapshots.items())
