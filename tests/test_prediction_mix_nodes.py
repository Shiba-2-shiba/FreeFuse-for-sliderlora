from test_nodes import nodes


def test_prediction_mix_schema_and_forwarding(nodes, monkeypatch):
    captured = {}
    def sample(*args, **kwargs):
        captured.update(kwargs)
        return {}, {}, object()
    monkeypatch.setattr(nodes, "sample_krea2_prediction_mix", sample, raising=False)
    cls = nodes.Krea2SliderFusePredictionMixSampler
    fields = {v.id: v for v in cls.define_schema().inputs}
    assert fields["mix_scope"].options == ["target_mask", "none", "all"]
    assert fields["diagnostic_level"].default == "audit"
    cls.execute(object(), [], [], object(), (), {}, "slider.safetensors", 4., 42, 8, 1.,
                "target_mask", 1, "audit")
    assert captured["seed"] == 42 and captured["trial_id"] == 1
    assert captured["mix_scope"] == "target_mask"


def test_audit_is_optional_and_appended_to_existing_diagnostic_inputs(nodes):
    schema = nodes.Krea2SliderFuseDiagnosticSampler.define_schema()
    assert schema.inputs[-1].id == "diagnostic_level"
    assert schema.inputs[-1].optional and schema.inputs[-1].default == "summary"


def test_legacy_mix_omitted_options_remain_manual(nodes, monkeypatch):
    captured = {}
    monkeypatch.setattr(nodes, "sample_krea2_prediction_mix",
                        lambda *a, **kw: (captured.update(kw) or {}, {}, object()))
    nodes.Krea2SliderFusePredictionMixSampler.execute(object(), [], [], object(), (), {},
        "slider.safetensors", 4., 42, 8, 1., "target_mask")
    assert captured.get("mask_mode", "manual") == "manual"
    assert captured["diagnostic_level"] == "audit"


def test_mix_auto_inputs_are_optional_appended_and_in_main_category(nodes, monkeypatch):
    cls=nodes.Krea2SliderFusePredictionMixSampler
    schema=cls.define_schema();ids=[f.id for f in schema.inputs]
    appended=["mask_mode","collect_step","collect_block","top_k_ratio","temperature",
              "fill_holes_max_area","mask_dilate_radius","selection_dilate_radius"]
    assert ids[-8:]==appended and ids[-9]=="diagnostic_level"
    fields={f.id:f for f in schema.inputs}
    assert all(fields[n].optional for n in appended)
    assert fields["mask_mode"].default=="manual" and fields["collect_step"].default==2
    assert fields["selection_dilate_radius"].default==0 and fields["selection_dilate_radius"].max==16
    assert schema.category==nodes.CATEGORY and "automatic" in schema.description.lower()
    captured={}
    monkeypatch.setattr(nodes,"sample_krea2_prediction_mix",lambda *a,**kw:(captured.update(kw) or {},{},object()))
    cls.execute(object(),[],[],object(),(),{},"slider.safetensors",4.,42,8,1.,"target_mask",
                mask_mode="auto",collect_step=3,collect_block=5,top_k_ratio=.2,temperature=10000.,selection_dilate_radius=4)
    assert captured["mask_mode"]=="auto" and captured["collect_step"]==3 and captured["collect_block"]==5
    assert captured["selection_dilate_radius"]==4
