import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1] / "workflows"


@pytest.mark.parametrize("case,strength,scope", [("zero", 0., "target_mask"), ("none", 4., "none"),
                                              ("all", 4., "all"), ("half", 4., "target_mask")])
def test_mix_workflows_have_controlled_settings_and_consistent_links(case, strength, scope):
    p = ROOT / f"krea2_slider_mix_{case}.json"
    graph = json.loads(p.read_text(encoding="utf-8"))
    api = json.loads(p.with_name(p.stem + "_api.json").read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "krea2_slider_diag_hook_half_none_api.json").read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in graph["nodes"]}
    s = api["8"]
    assert s["class_type"] == nodes[8]["type"] == "Krea2SliderFusePredictionMixSampler"
    assert s["inputs"]["strength"] == strength and s["inputs"]["mix_scope"] == scope
    assert s["inputs"]["seed"] == 42 and s["inputs"]["steps"] == 8 and s["inputs"]["cfg"] == 1.
    assert s["inputs"]["diagnostic_level"] == "audit" and s["inputs"]["trial_id"] == 0
    assert nodes[8]["widgets_values"] == [s["inputs"]["lora_name"], strength, 42, "fixed", 8, 1., scope, 0, "audit"]
    for name, value in nodes[8]["widgets_values_named"].items():
        assert s["inputs"][name] == value
    for key in api:
        if key not in ("8", "11"):
            assert api[key] == baseline[key]
    assert api["11"]["inputs"]["latent"] == ["8", 0]
    assert api["11"]["inputs"]["diagnostics"] == ["8", 2]
    assert api["11"]["inputs"]["filename_prefix"] == p.stem
    for link_id, source, slot, target, target_slot, kind in graph["links"]:
        assert nodes[source]["outputs"][slot]["type"] == nodes[target]["inputs"][target_slot]["type"] == kind
        assert link_id in nodes[source]["outputs"][slot]["links"]
        assert nodes[target]["inputs"][target_slot]["link"] == link_id
