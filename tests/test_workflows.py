import json
from pathlib import Path

import pytest

ROOT=Path(__file__).parents[1]/"workflows"


@pytest.mark.parametrize("mode",["manual","auto"])
def test_workflow_links_types_and_slider_scope(mode):
    workflow=json.loads((ROOT/f"krea2_female_slider_{mode}.json").read_text(encoding="utf-8"))
    nodes={n["id"]:n for n in workflow["nodes"]}
    links={item[0]:item for item in workflow["links"]}
    for ident,source,slot,target,target_slot,kind in links.values():
        assert kind==nodes[source]["outputs"][slot]["type"]==nodes[target]["inputs"][target_slot]["type"]
        assert ident in nodes[source]["outputs"][slot]["links"]
        assert nodes[target]["inputs"][target_slot]["link"]==ident
    sampler=next(n for n in nodes.values() if n["type"]=="Krea2SliderFuseSampler")
    assert sampler["widgets_values_named"]["mask_mode"]==mode
    assert sampler["widgets_values_named"]["steps"]==8
    assert sampler["widgets_values_named"]["cfg"]==1.
    assert sampler["widgets_values_named"]["lora_name"]==""
    assert [n["widgets_values_named"]["lora_name"] for n in nodes.values() if n["type"]=="LoraLoaderModelOnly"]==["krea2_darkbrush.safetensors"]
    assert not any(n["type"]=="Krea2ApplyRegionalAttention" for n in nodes.values())
    subjects=next(n for n in nodes.values() if n["type"]=="Krea2SliderFuseSubjects")
    connected=[i["name"] for i in subjects["inputs"] if i["link"] is not None]
    assert ("target_mask" in connected)==(mode=="manual")
    assert ("protected_mask" in connected)==(mode=="manual")


@pytest.mark.parametrize("mode",["manual","auto"])
def test_api_workflow_contains_same_settings_and_valid_references(mode):
    api=json.loads((ROOT/f"krea2_female_slider_{mode}_api.json").read_text(encoding="utf-8"))
    for key,node in api.items():
        for value in node["inputs"].values():
            if isinstance(value,list): assert str(value[0]) in api
    sampler=next(n for n in api.values() if n["class_type"]=="Krea2SliderFuseSampler")
    assert sampler["inputs"]["mask_mode"]==mode
    assert sampler["inputs"]["strength"]==1.
