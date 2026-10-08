import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1] / "workflows"
CASES = {
    "native_zero": ("native", "all", "all", 0.),
    "hook_zero": ("hook", "all", "all", 0.),
    "native_global": ("native", "all", "all", 4.),
    "hook_all_all": ("hook", "all", "all", 4.),
    "hook_all_none": ("hook", "all", "none", 4.),
    "hook_all_target": ("hook", "all", "target_phrase", 4.),
    "hook_half_none": ("hook", "target_mask", "none", 4.),
    "hook_half_target": ("hook", "target_mask", "target_phrase", 4.),
    "hook_half_all": ("hook", "target_mask", "all", 4.),
}


@pytest.mark.parametrize("case", list(CASES) + ["standard_zero", "standard_global"])
def test_case_settings_links_and_api_ui_match(case):
    graph = json.loads((ROOT / f"krea2_slider_diag_{case}.json").read_text(encoding="utf-8"))
    api = json.loads((ROOT / f"krea2_slider_diag_{case}_api.json").read_text(encoding="utf-8"))
    nodes = {str(n["id"]): n for n in graph["nodes"]}
    for ident, node in api.items():
        assert nodes[ident]["type"] == node["class_type"]
        for key, value in node["inputs"].items():
            if isinstance(value, list): assert str(value[0]) in api
            else: assert nodes[ident]["widgets_values_named"][key] == value
    for ident, source, slot, target, target_slot, kind in graph["links"]:
        assert nodes[str(source)]["outputs"][slot]["type"] == nodes[str(target)]["inputs"][target_slot]["type"] == kind
        assert ident in nodes[str(source)]["outputs"][slot]["links"]
        assert nodes[str(target)]["inputs"][target_slot]["link"] == ident
    sampler = api["8"]["inputs"]
    assert sampler["seed"] == 42 and sampler["steps"] == 8 and sampler["cfg"] == 1.
    assert nodes["8"]["widgets_values"][3 if case in CASES else 1] == "fixed"
    if case in CASES:
        assert api["8"]["class_type"] == "Krea2SliderFuseDiagnosticSampler"
        assert tuple(sampler[k] for k in ("backend", "image_scope", "text_scope", "strength")) == CASES[case]
        assert sampler["trial_id"] == 0
        assert api["6"]["inputs"]["target_phrase"] == "woman"
        assert api["6"]["inputs"]["protected_phrase"] == "man"
        assert api["6"]["inputs"]["target_mask"] == ["21", 0]
        assert api["20"]["inputs"] == {"value": 1., "width": 512, "height": 1024}
        assert api["11"]["inputs"]["diagnostics"] == ["8", 2]
        assert api["11"]["inputs"]["latent"] == ["8", 0]
        assert api["11"]["inputs"]["filename_prefix"] == f"krea2_slider_diag_{case}"
        assert sampler["model"] == ["2", 0]  # No upstream global Slider.
    else:
        assert api["8"]["class_type"] == "KSampler"
        assert api["30"]["class_type"] == "SaveLatent"
        assert api["30"]["inputs"]["samples"] == ["8", 0]
        assert ("27" in api) == (case == "standard_global")


def test_diagnostic_matrix_changes_only_documented_variables():
    base = json.loads((ROOT / "krea2_slider_diag_hook_half_none_api.json").read_text())
    for case in CASES:
        api = json.loads((ROOT / f"krea2_slider_diag_{case}_api.json").read_text())
        for ident in base:
            if ident == "8":
                for key in ("backend", "image_scope", "text_scope", "strength"):
                    api[ident]["inputs"][key] = base[ident]["inputs"][key]
            elif ident == "11":
                api[ident]["inputs"]["filename_prefix"] = base[ident]["inputs"]["filename_prefix"]
        assert api == base
