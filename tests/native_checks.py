"""Executed only by tools/validate_comfy.py, never substituted with fixtures."""
import asyncio
import copy
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
from slider_fuse.native_pair import NativePairRunner
from slider_fuse.prediction_mixing import mix_predictions, prediction_mask
from slider_fuse.mask_collection import collect_auto_mask
from slider_fuse.masks import manual_masks
from slider_fuse.lora import core_guard
from slider_fuse.diagnostics import DiagnosticRecorder
from slider_fuse.native_pair import make_prediction_mix_guider


class NativeProbeModel(nn.Module):
    """Small native Krea2 core with the Comfy BaseModel boundary used by the runner."""
    def __init__(self, core):
        super().__init__()
        self.diffusion_model = core
        self.current_patcher = None
        self.model_config = SimpleNamespace(unet_config={"image_model": "krea2"})
        self.manual_cast_dtype = None
        from comfy.model_sampling import ModelSamplingFlux, CONST
        class NativeSampling(ModelSamplingFlux, CONST):
            pass
        self.model_sampling = NativeSampling()
        self.latent_format = SimpleNamespace(latent_channels=4, latent_dimensions=2,
            spacial_downscale_ratio=8, temporal_downscale_ratio=1, fix_empty_latent=lambda x:x)
    def get_dtype(self):
        return torch.float32
    def memory_required(self, input_shape, **kwargs):
        return 0
    def extra_conds_shapes(self, **kwargs):
        return {}
    def extra_conds(self, **kwargs):
        from comfy.conds import CONDRegular
        context = kwargs.get("cross_attn")
        return {} if context is None else {"c_crossattn": CONDRegular(context)}
    def process_latent_in(self, x):
        return x
    def process_latent_out(self, x):
        return x
    def apply_model(self, x, sigma, c_crossattn, **kwargs):
        velocity = self.diffusion_model(self.model_sampling.calculate_input(sigma,x), sigma, c_crossattn,
                                       transformer_options=kwargs.get("transformer_options", {}))
        return self.model_sampling.calculate_denoised(sigma,velocity,x)


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

    def test_full_sequence_hook_matches_fp32_merged_native_linear(self):
        module = self.core.blocks[0].attn.wq
        x = torch.randn(1, 19, 64)
        adapter = Adapter(torch.randn(2, 64) * .01, torch.randn(64, 2) * .01, 2.)
        before = module.weight.detach().clone()
        state = RoutingState(phase="route", cap_len=3, mask=torch.ones(1, 4, 4),
                             image_scope="all", text_scope="all")
        state.configure_target_text(1., (0,), (1,), 3)
        hook = SliderHook(module, adapter, 4., state, "blocks.0.attn.wq")
        hook.inject()
        try:
            routed = module(x)
        finally:
            hook.eject()
        try:
            with torch.no_grad(): module.weight.add_((adapter.up @ adapter.down) * adapter.scale * 4.)
            torch.testing.assert_close(module(x), routed, rtol=1e-5, atol=1e-6)
        finally:
            with torch.no_grad(): module.weight.copy_(before)

    def test_v3_entrypoint_and_schema_registration(self):
        root=Path(__file__).parents[1]
        spec=importlib.util.spec_from_file_location("slider_native_extension",root/"__init__.py",submodule_search_locations=[str(root)])
        package=importlib.util.module_from_spec(spec); sys.modules[spec.name]=package; spec.loader.exec_module(package)
        extension=asyncio.run(package.comfy_entrypoint()); nodes=asyncio.run(extension.get_node_list())
        self.assertEqual(len(nodes),7)
        for node in nodes:
            schema=node.define_schema(); schema.validate()
            if schema.node_id=="Krea2SliderFuseSampler":
                inputs={item.id:item for item in schema.inputs}
                for name in ("fill_holes_max_area","mask_dilate_radius","target_text_scale"):
                    self.assertTrue(inputs[name].optional)
                    self.assertEqual(inputs[name].default,0)
            if schema.node_id=="Krea2SliderFusePredictionMixSampler":
                inputs={item.id:item for item in schema.inputs}
                self.assertTrue(inputs["mask_mode"].optional)
                self.assertEqual(inputs["mask_mode"].default,"manual")

    def native_pair(self):
        model = NativeProbeModel(self.core)
        base = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        key = "diffusion_model.blocks.0.attn.wq.weight"
        base.add_patches({key: ("diff", (torch.full_like(self.core.blocks[0].attn.wq.weight, .003),))})
        native = base.clone()
        native.add_patches({key: ("diff", (torch.full_like(self.core.blocks[0].attn.wq.weight, .007),))})
        return base, native, NativePairRunner(base, native)

    def predict_pair(self, runner, branch):
        from comfy.conds import CONDRegular
        positive = [{"model_conds": {"c_crossattn": CONDRegular(self.context)}, "uuid": "native-probe"}]
        with torch.inference_mode():
            return runner.predict(branch, self.x, self.sigma, positive=positive, negative=[],
                                  model_options={"transformer_options": {}}, seed=42)

    def close_pair(self, base, runner):
        import comfy.model_management
        try:
            runner.close()
        finally:
            comfy.model_management.unload_model_and_clones(base, unload_additional_models=False)

    def test_native_pair_switch_and_restore(self):
        weight = self.core.blocks[0].attn.wq.weight.detach().clone()
        base, native, runner = self.native_pair()
        try:
            a = self.predict_pair(runner, "base")
            self.assertIs(base.model.current_patcher, base)
            b = self.predict_pair(runner, "slider")
            self.assertIs(base.model.current_patcher, native)
            c = self.predict_pair(runner, "base")
            self.assertTrue(torch.equal(a, c))
            self.assertGreater(float((a-b).abs().max()), 0.)
        finally:
            self.close_pair(base, runner)
        self.assertTrue(torch.equal(self.core.blocks[0].attn.wq.weight, weight))
        self.assertIsNone(base.model.current_patcher)

    def test_native_pair_prediction_matches_individual_calls(self):
        base, native, runner = self.native_pair()
        try:
            with torch.inference_mode():
                a = self.predict_pair(runner, "base")
                self.assertTrue(torch.equal(a, base.model.apply_model(self.x, self.sigma, self.context)))
                b = self.predict_pair(runner, "slider")
                self.assertTrue(torch.equal(b, native.model.apply_model(self.x, self.sigma, self.context)))
        finally:
            self.close_pair(base, runner)

    def test_native_prediction_mix_endpoints_and_partial_mask(self):
        base, _, runner = self.native_pair()
        try:
            a = self.predict_pair(runner, "base"); b = self.predict_pair(runner, "slider")
            token = torch.zeros(1, 4, 4); token[:, :, :2] = 1.
            mask = prediction_mask(token, a, patch=2)
            self.assertTrue(torch.equal(mix_predictions(a, b, torch.zeros_like(mask)), a))
            self.assertTrue(torch.equal(mix_predictions(a, b, torch.ones_like(mask)), b))
            mixed = mix_predictions(a, b, mask)
            self.assertTrue(torch.equal(mixed[..., :4], b[..., :4]))
            self.assertTrue(torch.equal(mixed[..., 4:], a[..., 4:]))
        finally:
            self.close_pair(base, runner)

    def collect_native_mask(self, base):
        info=SimpleNamespace(token_ids=(0,1,2))
        subjects=(SimpleNamespace(id="target",positions=(0,)),SimpleNamespace(id="protected",positions=(2,)))
        sigmas=torch.tensor([.7,.35,0.])
        return collect_auto_mask(base,[[self.context,{}]],[],info,subjects,
            torch.zeros_like(self.x),self.x,sigmas,grid=(4,4),seed=42,steps=2,cfg=1.,
            collect_step=1,collect_block=0,top_k_ratio=.3,temperature=4000.,
            fill_holes_max_area=0,mask_dilate_radius=0)

    def sample_native_mask(self, base, slider, mask):
        recorder=DiagnosticRecorder(set(),level="audit")
        report={"sampler_nfe":0,"grid":[4,4],"text_token_count":3}
        guider=make_prediction_mix_guider(base,slider,mask,patch=2,recorder=recorder,report=report,mode="both",restore_base=True)
        guider.set_conds([[self.context,{}]],[]);guider.set_cfg(1.)
        import comfy.samplers
        with torch.inference_mode():
            value=guider.sample(self.x.clone(),torch.zeros_like(self.x),comfy.samplers.sampler_object("euler"),
                                torch.tensor([.7,.35,0.]),seed=42)
        traces=guider.finish_report()
        return value,report,traces

    def test_native_auto_collect_then_mix(self):
        base,slider,runner=self.native_pair()
        style_key="diffusion_model.blocks.0.attn.wq.weight"
        style_before=base.patches[style_key][0][1][1][0].clone()
        options_before=copy.deepcopy(base.model_options)
        patch_counts={k:len(v) for k,v in base.patches.items()}
        try:
            with core_guard(self.core):
                expected=self.predict_pair(runner,"base");runner.close()
                bank,collection=self.collect_native_mask(base)
                self.assertEqual(collection["phase1_nfe"],1)
                self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in self.core.modules()))
                self.assertTrue(torch.equal(expected,self.predict_pair(runner,"base")))
                self.assertIs(base.model.current_patcher,base);runner.close()
                from unittest.mock import patch
                with patch.object(NativeProbeModel,"apply_model",side_effect=RuntimeError("forced native collection failure")):
                    with self.assertRaisesRegex(RuntimeError,"forced native collection"):
                        self.collect_native_mask(base)
                self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in self.core.modules()))
                self.assertTrue(torch.equal(expected,self.predict_pair(runner,"base")))
                self.assertIs(base.model.current_patcher,base);runner.close()
                self.assertEqual(base.model_options,options_before)
                self.assertEqual({k:len(v) for k,v in base.patches.items()},patch_counts)
                self.assertTrue(torch.equal(style_before,base.patches[style_key][0][1][1][0]))
                value,report,traces=self.sample_native_mask(base,slider,bank["masks"]["target"])
                self.assertTrue(torch.isfinite(value).all())
                self.assertEqual(report["branch_nfe"],{"base":2,"slider":2})
                selector=prediction_mask(bank["masks"]["target"],value,patch=2)
                self.assertTrue(torch.equal(traces["trace_predictions"],torch.where(selector,
                    traces["trace_slider_predictions"],traces["trace_base_predictions"])))
        finally:
            self.close_pair(base,runner)

    def test_native_auto_manual_same_mask(self):
        base,slider,runner=self.native_pair()
        try:
            with core_guard(self.core):
                bank,_=self.collect_native_mask(base)
                manual=manual_masks(bank["masks"]["target"],bank["masks"]["protected"],(4,4))
                a,_,ta=self.sample_native_mask(base,slider,bank["masks"]["target"])
                b,_,tb=self.sample_native_mask(base,slider,manual["masks"]["target"])
                self.assertTrue(torch.equal(a,b))
                self.assertTrue(torch.equal(ta["trace_inputs"],tb["trace_inputs"]))
                self.assertTrue(torch.equal(ta["trace_predictions"],tb["trace_predictions"]))
        finally:
            self.close_pair(base,runner)
