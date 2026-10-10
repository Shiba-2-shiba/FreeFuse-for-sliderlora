import json
from pathlib import Path
import pytest
import torch
from test_nodes import nodes

VARIANTS=["baseline","centroid_control","multi_proto","multi_proto_bg","adaln_bg","ensemble_bg","propagated_bg"]


def test_experimental_node_contract_is_additive(nodes,monkeypatch):
    assert hasattr(nodes,"Krea2SliderFuseExperimentalMaskCollect"), "Experimental collection node is missing"
    collect=nodes.Krea2SliderFuseExperimentalMaskCollect
    schema=collect.define_schema();fields={f.id:f for f in schema.inputs}
    assert [f.id for f in schema.inputs]==["model","positive","negative","prompt_info","subjects","latent","seed","steps","trial_id","config_json","background_phrase","background_occurrence","audit_tensors"]
    assert [f.id for f in schema.outputs]==["suite"]
    assert fields["audit_tensors"].default is False and fields["audit_tensors"].optional
    assert fields["steps"].default==8 and fields["config_json"].multiline
    assert json.loads(fields["config_json"].default)["collect_step"]==2
    assert nodes.ExperimentType.name=="KREA2_SLIDER_FUSE_MASK_EXPERIMENT"
    captured={};suite={"sentinel":True}
    monkeypatch.setattr(nodes,"collect_experimental_masks",lambda *a,**k:(captured.update(k) or suite))
    assert collect.execute(object(),[],[],object(),(),{},42,8,9,"{}","room",0).result==(suite,)
    assert captured["steps"]==8 and captured["trial_id"]==9 and captured["background_phrase"]=="room"
    select=nodes.Krea2SliderFuseExperimentalMaskSelect.define_schema()
    assert [f.id for f in select.outputs]==["mask_bank","report"]
    assert select.inputs[1].options==VARIANTS
    save=nodes.Krea2SliderFuseExperimentalMaskSave.define_schema()
    assert save.is_output_node and save.outputs==[]
    assert [f.id for f in save.inputs]==["suite","filename_prefix"]
    assert "vae" not in [f.id for f in save.inputs]
    source=(Path(__file__).parents[1]/"__init__.py").read_text()
    for name in ("Collect","Select","Save"):assert "Krea2SliderFuseExperimentalMask"+name in source


def test_selector_refuses_failed_candidate_and_copies_bank(nodes):
    bank={"grid":(1,2),"masks":{"target":torch.tensor([[[1.,0.]]]),"protected":torch.tensor([[[0.,1.]]]),"background":torch.zeros(1,1,2)}}
    suite={"experiment_schema_version":1,"banks":{"baseline":bank},"report":{"algorithm":{"variants":{"baseline":{"status":"ok"},"multi_proto":{"status":"failed","reason":"missing seeds"}}}}}
    output=nodes.Krea2SliderFuseExperimentalMaskSelect.execute(suite,"baseline").result
    assert len(output)==2 and json.loads(output[1])["variant"]=="baseline"
    assert torch.equal(output[0]["masks"]["target"],bank["masks"]["target"])
    output[0]["masks"]["target"].zero_()
    assert bank["masks"]["target"].sum()==1
    with pytest.raises(ValueError,match="missing seeds"):
        nodes.Krea2SliderFuseExperimentalMaskSelect.execute(suite,"multi_proto")
    with pytest.raises(ValueError):nodes.Krea2SliderFuseExperimentalMaskSelect.execute(suite,"not-a-variant")
