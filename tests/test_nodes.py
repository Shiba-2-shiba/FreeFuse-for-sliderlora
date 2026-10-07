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


def test_raw_similarity_preview_is_normalized_independently_without_mutating_maps(nodes):
    target=torch.tensor([[10.,20.,30.,10.]])
    protected=torch.tensor([[0.,4.,0.,2.]])
    bank={"grid":(2,2),"masks":{name:torch.zeros(1,2,2) for name in ("target","protected","background")},
          "raw_maps":{"target":target,"protected":protected}}
    result=nodes.Krea2SliderFuseMaskPreview.execute(bank).result
    assert len(result)==7
    torch.testing.assert_close(result[3],torch.tensor([[[0.,.5],[1.,0.]]]))
    torch.testing.assert_close(result[4],torch.tensor([[[0.,1.],[0.,.5]]]))
    assert torch.equal(target,torch.tensor([[10.,20.,30.,10.]]))
    assert [o.id for o in nodes.Krea2SliderFuseMaskPreview.define_schema().outputs][3:5]==["target_similarity","protected_similarity"]


def test_manual_preview_marks_raw_similarity_unavailable_with_black_outputs(nodes):
    masks={"target":torch.ones(1,2,2),"protected":torch.zeros(1,2,2),"background":torch.zeros(1,2,2)}
    result=nodes.Krea2SliderFuseMaskPreview.execute({"masks":masks}).result
    assert len(result)==7 and not result[3].any() and not result[4].any()


def test_postprocess_inputs_are_optional_zero_defaults_at_end(nodes):
    schema=nodes.Krea2SliderFuseSampler.define_schema()
    assert [item.id for item in schema.inputs][-3:-1]==["fill_holes_max_area","mask_dilate_radius"]
    assert all(item.optional and item.default==0 for item in schema.inputs[-3:-1])


def test_old_api_payload_omits_new_inputs_and_executes_as_noop(nodes,monkeypatch):
    captured={}
    def sample(*args,**kwargs):
        captured.update(kwargs)
        return args[5],{}, {"ok":True}
    monkeypatch.setattr(nodes,"sample_krea2",sample)
    old=dict(model=object(),positive=[],negative=[],prompt_info=object(),subjects=(),latent={"samples":torch.zeros(1,16,4,4)},
             lora_name="slider.safetensors",strength=1.,seed=42,steps=8,cfg=1.,mask_mode="auto",
             collect_step=2,collect_block=18,top_k_ratio=.2,temperature=10000.)
    assert nodes.Krea2SliderFuseSampler.execute(**old).result[0] is old["latent"]
    assert captured["fill_holes_max_area"]==0 and captured["mask_dilate_radius"]==0
    assert captured["target_text_scale"]==0.


def test_original_and_added_preview_and_legacy_fallback(nodes):
    original=torch.tensor([[[1.,0.],[0.,0.]]]);added=torch.tensor([[[0.,1.],[0.,0.]]])
    masks={"target":original+added,"protected":torch.zeros_like(original),"background":1-original-added}
    bank={"masks":masks,"original_masks":{"target":original},"added_target_mask":added}
    result=nodes.Krea2SliderFuseMaskPreview.execute(bank).result
    assert result[5] is original and result[6] is added
    legacy=nodes.Krea2SliderFuseMaskPreview.execute({"masks":masks}).result
    assert legacy[5] is masks["target"] and not legacy[6].any()


def test_target_text_input_is_optional_zero_default_appended_after_old_widgets(nodes):
    schema=nodes.Krea2SliderFuseSampler.define_schema()
    field=schema.inputs[-1]
    assert field.id=="target_text_scale" and field.optional and field.default==0.
    assert field.min==0. and field.max==1.
