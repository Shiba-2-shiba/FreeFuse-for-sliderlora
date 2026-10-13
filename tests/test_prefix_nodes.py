import importlib.util
from pathlib import Path
import sys

import pytest

from test_nodes import nodes


@pytest.fixture
def probe_nodes(nodes,monkeypatch):
    monkeypatch.setitem(sys.modules,'slider_node_test.nodes',nodes)
    path=Path(__file__).parents[1]/'probe_nodes.py'
    spec=importlib.util.spec_from_file_location('slider_node_test.probe_nodes',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_prefix_node_has_only_masks_and_probe_outputs(probe_nodes):
    assert hasattr(probe_nodes,'Krea2SliderFusePairedPrefixPilot'), 'prefix node missing'
    schema=probe_nodes.Krea2SliderFusePairedPrefixPilot.define_schema()
    assert [f.id for f in schema.outputs]==['mask_bank','probe']
    inputs={f.id:f for f in schema.inputs}
    assert inputs['steps'].default==8 and inputs['steps'].min==inputs['steps'].max==8
    assert inputs['pilot_mode'].options==['prefix','full_reference']
    assert inputs['mix_scope'].options==['target_mask']
    assert inputs['mask_mode'].options==['manual']
    assert inputs['diagnostic_level'].options==['audit']
    assert not {'vae','images','diagnostics'} & set(inputs)
    assert '/Experimental' in schema.category
    original=probe_nodes.Krea2SliderFusePredictionMixSampler.define_schema()
    assert [f.id for f in original.outputs]==['latent','mask_bank','diagnostics']
    assert next(f for f in original.inputs if f.id=='steps').min==1


def test_prefix_node_forwards_bounded_mode_without_exposing_latent(probe_nodes,monkeypatch):
    assert hasattr(probe_nodes,'sample_paired_pilot'), 'prefix runtime missing'
    captured={};mask=object();probe=object()
    def sample(**kwargs):
        captured.update(kwargs);return mask,probe
    monkeypatch.setattr(probe_nodes,'sample_paired_pilot',sample)
    result=probe_nodes.Krea2SliderFusePairedPrefixPilot.execute(lora_name='slider.safetensors',
        pilot_mode='prefix',background_phrase='room',strength=4.,seed=42,steps=8)
    assert result.result==(mask,probe)
    assert captured['steps']==8 and captured['pilot_mode']=='prefix'
    assert 'lora_name' not in captured


def test_prefix_artifact_saver_has_no_vae_or_latent_dependency(probe_nodes):
    assert hasattr(probe_nodes,'Krea2SliderFusePrefixPilotSave'), 'prefix saver missing'
    schema=probe_nodes.Krea2SliderFusePrefixPilotSave.define_schema()
    assert [f.id for f in schema.inputs]==['probe','mask_bank','filename_prefix']
    assert schema.is_output_node and schema.outputs==[]
    entry=(Path(__file__).parents[1]/'__init__.py').read_text()
    assert 'Krea2SliderFusePairedPrefixPilot' in entry and 'Krea2SliderFusePrefixPilotSave' in entry
