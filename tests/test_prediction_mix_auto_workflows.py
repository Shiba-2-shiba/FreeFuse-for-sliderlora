import json
from pathlib import Path

import pytest

ROOT=Path(__file__).parents[1]/"workflows"


@pytest.mark.parametrize("mode",["manual","auto"])
def test_recommended_workflows_have_matching_widgets_and_complete_previews(mode):
    p=ROOT/f"krea2_female_slider_prediction_mix_{mode}.json"
    assert p.exists(), "Recommended Prediction Mix workflow is missing"
    ui=json.loads(p.read_text(encoding="utf-8"));api=json.loads(p.with_name(p.stem+"_api.json").read_text(encoding="utf-8"))
    nodes={n["id"]:n for n in ui["nodes"]};s=api["8"]["inputs"]
    assert api["8"]["class_type"]=="Krea2SliderFusePredictionMixSampler"
    assert s["mask_mode"]==mode and s["diagnostic_level"]=="summary" and s["strength"]==4.
    assert nodes[8]["widgets_values"]==[s["lora_name"],4.,42,"fixed",8,1.,"target_mask",0,"summary",
        mode,2,18,s["top_k_ratio"],s["temperature"],0,0,4 if mode=="auto" else 0]
    assert all(s[k]==v for k,v in nodes[8]["widgets_values_named"].items())
    assert len(nodes[12]["outputs"])==9 and all(o["links"] for o in nodes[12]["outputs"])
    masks=api["6"]["inputs"]
    assert ("target_mask" in masks)==(mode=="manual") and ("protected_mask" in masks)==(mode=="manual")
    for lid,source,slot,target,tslot,kind in ui["links"]:
        assert nodes[source]["outputs"][slot]["type"]==nodes[target]["inputs"][tslot]["type"]==kind
        assert lid in nodes[source]["outputs"][slot]["links"] and nodes[target]["inputs"][tslot]["link"]==lid
    for nid,n in api.items():
        for value in n["inputs"].values():
            if isinstance(value,list):assert str(value[0]) in api
