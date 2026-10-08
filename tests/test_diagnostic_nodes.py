import types

from test_nodes import nodes


def test_diagnostic_schema_is_separate_and_has_trial_and_hidden_metadata(nodes):
    schema = nodes.Krea2SliderFuseDiagnosticSampler.define_schema()
    fields = {i.id: i for i in schema.inputs}
    assert fields["trial_id"].default == 0
    assert fields["backend"].options == ["hook", "native"]
    assert fields["image_scope"].options == ["target_mask", "all"]
    assert fields["text_scope"].options == ["none", "target_phrase", "all"]
    save = nodes.Krea2SliderFuseDiagnosticSave.define_schema()
    assert save.is_output_node
    assert save.hidden == ["PROMPT", "EXTRA_PNGINFO", "UNIQUE_ID"]


def test_diagnostic_sampler_forwards_trial_without_changing_seed(nodes, monkeypatch):
    captured = {}
    def sample(*args, **kwargs):
        captured.update(kwargs)
        return {}, {}, object()
    monkeypatch.setattr(nodes, "sample_krea2_diagnostic", sample)
    nodes.Krea2SliderFuseDiagnosticSampler.execute(object(), [], [], object(), (), {}, "slider.safetensors",
        4., 42, 8, 1., "hook", "target_mask", "none", 1)
    assert captured["seed"] == 42 and captured["trial_id"] == 1
    assert captured["text_scope"] == "none"
