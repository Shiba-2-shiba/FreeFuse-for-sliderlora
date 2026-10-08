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
