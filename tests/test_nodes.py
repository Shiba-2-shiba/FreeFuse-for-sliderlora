import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


@pytest.fixture
def nodes(monkeypatch):
    class Kind:
        @staticmethod
        def Input(name, **kwargs): return types.SimpleNamespace(id=name, **kwargs)
        @staticmethod
        def Output(name=None, **kwargs): return types.SimpleNamespace(id=name, **kwargs)
    class Custom(Kind):
        def __init__(self, name): self.name = name
    io = types.SimpleNamespace(ComfyNode=object, Custom=Custom, Schema=lambda **kwargs: types.SimpleNamespace(**kwargs),
        NodeOutput=lambda *args: types.SimpleNamespace(result=args))
    for name in ("Clip", "Model", "Conditioning", "Latent", "String", "Combo", "Float", "Int", "Mask"):
        setattr(io,name,Kind)
    latest = types.ModuleType("comfy_api.latest"); latest.io = io
    monkeypatch.setitem(sys.modules,"comfy_api",types.ModuleType("comfy_api"))
    monkeypatch.setitem(sys.modules,"comfy_api.latest",latest)
    paths = types.ModuleType("folder_paths")
    paths.get_filename_list = lambda kind: ["slider.safetensors"]
    paths.get_full_path_or_raise = lambda kind,name: str(Path(name).resolve())
    monkeypatch.setitem(sys.modules,"folder_paths",paths)
    root=Path(__file__).parents[1]
    package=types.ModuleType("slider_node_test"); package.__path__=[str(root)]
    monkeypatch.setitem(sys.modules,package.__name__,package)
    spec=importlib.util.spec_from_file_location("slider_node_test.nodes",root/"nodes.py")
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_four_schema_contracts(nodes):
    classes = (nodes.Krea2SliderFuseEncode,nodes.Krea2SliderFuseSubjects,nodes.Krea2SliderFuseSampler,nodes.Krea2SliderFuseMaskPreview)
    assert len({c.define_schema().node_id for c in classes})==4
    sampler=nodes.Krea2SliderFuseSampler.define_schema()
    defaults={i.id:getattr(i,"default",None) for i in sampler.inputs}
    assert defaults["steps"]==8 and defaults["cfg"]==1.
    assert defaults["collect_step"]==2 and defaults["collect_block"]==18
    assert [o.id for o in sampler.outputs]==["latent","mask_bank","diagnostics"]


def test_mask_preview_preserves_target_and_protected(nodes):
    masks={"target":torch.ones(1,2,2),"protected":torch.zeros(1,2,2),"background":torch.zeros(1,2,2)}
    assert nodes.Krea2SliderFuseMaskPreview.execute({"masks":masks}).result[0] is masks["target"]


def test_sampler_file_fingerprint_changes_with_file_content(nodes,tmp_path):
    file=tmp_path/"slider.safetensors"; file.write_bytes(b"first")
    a=nodes.Krea2SliderFuseSampler.fingerprint_inputs(lora_name=str(file))
    file.write_bytes(b"second")
    b=nodes.Krea2SliderFuseSampler.fingerprint_inputs(lora_name=str(file))
    assert a!=b
