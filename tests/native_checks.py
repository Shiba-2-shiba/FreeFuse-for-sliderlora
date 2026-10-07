"""Executed only by tools/validate_comfy.py, never substituted with fixtures."""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from comfy.ldm.krea2.model import SingleStreamDiT
from comfy.model_patcher import ModelPatcher
from comfy.patcher_extension import PatcherInjection

from slider_fuse.attention import AttentionCollector
from slider_fuse.lora import Adapter, RoutingState, SliderHook
from slider_fuse.masks import patch_grid


class NativeChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        self.core = SingleStreamDiT(features=64, tdim=16, txtdim=32, heads=4, kvheads=2,
            multiplier=1, layers=2, patch=2, channels=4, txtlayers=3, txtheads=2, txtkvheads=2,
            dtype=torch.float32, device="cpu", operations=nn).eval()
        with torch.no_grad():
            for parameter in self.core.parameters(): parameter.normal_(0,.09)
        self.x = torch.randn(1,4,8,8); self.context = torch.randn(1,3,96); self.sigma = torch.tensor([.7])

    def forward(self, x=None):
        with torch.no_grad():
            return self.core(self.x if x is None else x,self.sigma,self.context,transformer_options={})

    def collector(self, grid=(4,4)):
        return AttentionCollector({"target":(0,),"protected":(2,)},grid,1,0,.3,4000.,torch.tensor([.7,0.]))

    def test_observe_only_gqa_predictions_and_rng_unchanged(self):
        expected=self.forward(); collector=self.collector(); state=RoutingState()
        collector.active=True; collector.begin_forward(self.sigma,3); collector.install(self.core,state)
        rng=torch.random.get_rng_state()
        try:
            torch.testing.assert_close(self.forward(),expected,rtol=1e-5,atol=1e-6)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
            self.assertEqual(set(collector.maps),{"target","protected"})
            self.assertEqual(collector.observation["kv_heads"],2)
        finally: collector.remove()

    def test_observe_4d_5d_odd_non_square(self):
        for value in (self.x,self.x.unsqueeze(2),torch.randn(1,4,9,7)):
            grid=patch_grid(value,self.core.patch); collector=self.collector(grid)
            collector.active=True; collector.begin_forward(self.sigma,3); collector.install(self.core,RoutingState())
            try:
                self.assertEqual(self.forward(value).shape,value.shape)
                self.assertEqual(collector.maps["target"].numel(),grid[0]*grid[1])
            finally: collector.remove()

    def test_zero_slider_masks_equal_native_output(self):
        expected=self.forward(); state=RoutingState(phase="route",cap_len=3,mask=torch.zeros(1,4,4))
        module=self.core.blocks[0].attn.wq
        hook=SliderHook(module,Adapter(torch.randn(2,64),torch.randn(64,2),2.),1.,state,"blocks.0.attn.wq")
        hook.inject()
        try: torch.testing.assert_close(self.forward(),expected,rtol=1e-5,atol=1e-6)
        finally: hook.eject()

    def test_nonzero_native_slider_and_exact_restore(self):
        expected=self.forward(); module=self.core.blocks[0].attn.wo; original=module.forward
        state=RoutingState(phase="route",cap_len=3,mask=torch.ones(1,4,4))
        hook=SliderHook(module,Adapter(torch.randn(2,64),torch.randn(64,2),2.),1.,state,"blocks.0.attn.wo")
        hook.inject()
        try: self.assertGreater(float((self.forward()-expected).abs().max()),1e-7)
        finally: hook.eject()
        self.assertEqual(module.forward,original)
        torch.testing.assert_close(self.forward(),expected,rtol=1e-5,atol=1e-6)

    def test_real_patcher_injection_ejection_and_shared_core(self):
        native=nn.Module(); native.diffusion_model=self.core
        native.model_config=SimpleNamespace(unet_config={"image_model":"krea2"})
        patcher=ModelPatcher(native,torch.device("cpu"),torch.device("cpu")); clone=patcher.clone()
        module=self.core.blocks[0].attn.wq; original=module.forward
        hook=SliderHook(module,Adapter(torch.randn(2,64),torch.randn(64,2),2.),1.,RoutingState(),"wq")
        clone.set_injections("slider_native_test",[PatcherInjection(lambda p:hook.inject(),lambda p:hook.eject())])
        self.assertIs(clone.model.diffusion_model,patcher.model.diffusion_model)
        try:
            clone.inject_model(); self.assertNotEqual(module.forward,original)
        finally: clone.eject_model()
        self.assertEqual(module.forward,original)

    def test_v3_entrypoint_and_schema_registration(self):
        root=Path(__file__).parents[1]
        spec=importlib.util.spec_from_file_location("slider_native_extension",root/"__init__.py",submodule_search_locations=[str(root)])
        package=importlib.util.module_from_spec(spec); sys.modules[spec.name]=package; spec.loader.exec_module(package)
        extension=asyncio.run(package.comfy_entrypoint()); nodes=asyncio.run(extension.get_node_list())
        self.assertEqual(len(nodes),4)
        for node in nodes:
            schema=node.define_schema(); schema.validate()
            if schema.node_id=="Krea2SliderFuseSampler":
                inputs={item.id:item for item in schema.inputs}
                for name in ("fill_holes_max_area","mask_dilate_radius","target_text_scale"):
                    self.assertTrue(inputs[name].optional)
                    self.assertEqual(inputs[name].default,0)
