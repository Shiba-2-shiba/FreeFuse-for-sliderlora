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


@pytest.mark.parametrize("variant",["manual","auto","auto_subject_words","auto_step4"])
def test_controlled_comparisons_share_successful_lora_and_strength(variant):
    workflow=json.loads((ROOT/f"krea2_female_slider_compare_{variant}.json").read_text(encoding="utf-8"))
    nodes=workflow["nodes"]
    sampler=next(n for n in nodes if n["type"]=="Krea2SliderFuseSampler")["widgets_values_named"]
    assert sampler["lora_name"]=="Krea2/krea2_deaging_20261001T075826Z_c06cb035.safetensors"
    assert sampler["strength"]==2. and sampler["seed"]==42 and sampler["steps"]==8
    assert sampler["collect_step"]==(4 if variant=="auto_step4" else 2)
    subjects=next(n for n in nodes if n["type"]=="Krea2SliderFuseSubjects")["widgets_values_named"]
    assert subjects["target_phrase"]==("woman" if variant=="auto_subject_words" else "adult woman in a sage-green top")
    preview=next(n for n in nodes if n["type"]=="Krea2SliderFuseMaskPreview")
    assert [o["name"] for o in preview["outputs"]][3:]==["target_similarity","protected_similarity"]
    assert all(o["links"] for o in preview["outputs"][3:])
    by_id={n["id"]:n for n in nodes}
    for ident,source,slot,target,target_slot,kind in workflow["links"]:
        assert kind==by_id[source]["outputs"][slot]["type"]==by_id[target]["inputs"][target_slot]["type"]
        assert by_id[target]["inputs"][target_slot]["link"]==ident
    api=json.loads((ROOT/f"krea2_female_slider_compare_{variant}_api.json").read_text(encoding="utf-8"))
    settings=next(n["inputs"] for n in api.values() if n["class_type"]=="Krea2SliderFuseSampler")
    assert settings["lora_name"]==sampler["lora_name"] and settings["strength"]==sampler["strength"]


@pytest.mark.parametrize("variant,area,radius",[("off",0,0),("fill4",4,0),("fill8",8,0),("fill8_dilate1",8,1)])
def test_postprocess_workflows_keep_candidate_and_preview_changes(variant,area,radius):
    path=ROOT/f"krea2_female_slider_postprocess_{variant}.json"
    graph=json.loads(path.read_text(encoding="utf-8"));api=json.loads(path.with_name(path.stem+"_api.json").read_text(encoding="utf-8"))
    nodes={n["id"]:n for n in graph["nodes"]}
    sampler=next(n for n in nodes.values() if n["type"]=="Krea2SliderFuseSampler")
    settings=sampler["widgets_values_named"]
    assert settings["strength"]==4. and settings["seed"]==42 and settings["top_k_ratio"]==.2 and settings["temperature"]==10000.
    assert settings["fill_holes_max_area"]==area and settings["mask_dilate_radius"]==radius
    assert sampler["widgets_values"][-2:]==[area,radius]
    assert api[str(sampler["id"])]["inputs"]["fill_holes_max_area"]==area
    assert api[str(sampler["id"])]["inputs"]["mask_dilate_radius"]==radius
    preview=next(n for n in nodes.values() if n["type"]=="Krea2SliderFuseMaskPreview")
    assert [o["name"] for o in preview["outputs"]][-2:]==["original_target_mask","added_target_mask"]
    assert all(out["links"] for out in preview["outputs"])
    for ident,source,slot,target,target_slot,kind in graph["links"]:
        assert kind==nodes[source]["outputs"][slot]["type"]==nodes[target]["inputs"][target_slot]["type"]
        assert nodes[target]["inputs"][target_slot]["link"]==ident
