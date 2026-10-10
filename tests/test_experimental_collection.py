"""Real tensor collection checks; ComfyUI lifecycle is the only boundary double."""
import importlib
from dataclasses import replace
import json
import types

import pytest
import torch

from test_runtime import environment, setup_run
from slider_fuse.conditioning import build_info, make_subjects
from slider_fuse.sampling import tensor_hash


def integration():
    assert importlib.util.find_spec("slider_fuse.experimental_collection") is not None, "Experimental collection integration is missing"
    return importlib.import_module("slider_fuse.experimental_collection")


def prepared(tmp_path, environment):
    model, _, _, _, _ = setup_run(tmp_path)
    info = build_info("woman man room", [(151644,1.),(90,1.),(151644,1.),(872,1.),(198,1.),(1,1.),(2,1.),(3,1.)],
                      lambda ids: "".join({1:"woman",2:" man",3:" room"}[i] for i in ids))
    positive = [[torch.tensor([[[4.,0.,0.,0.],[0.,4.,0.,0.],[0.,0.,4.,0.]]]), {}]]
    info = replace(info, _conditioning=positive)
    subjects = make_subjects(info, "woman", "man")
    environment[0]["samplers"].KSampler = lambda *a, **kw: types.SimpleNamespace(sigmas=torch.linspace(1.,0.,9))
    config = {"collect_step":2,"collect_block":0,"selected_steps":[1,2],"selected_blocks":[0],
              "prototypes":1,"seed_confidence":.5,"seed_margin":0.,"mask_confidence":.5,"mask_margin":0.}
    return dict(model=model, positive=positive, negative=positive, prompt_info=info, subjects=subjects,
                latent={"samples":torch.zeros(1,4,4,4)}, seed=42, steps=8, trial_id=7,
                config_json=json.dumps(config), background_phrase="room", background_occurrence=0)


def test_collection_module_exists():
    assert callable(integration().collect_experimental_masks)


def test_exact_two_nfe_preserves_inputs_and_no_generation(environment,tmp_path):
    module = integration(); kw=prepared(tmp_path, environment);kw["audit_tensors"]=True
    core=kw["model"].model.diffusion_model
    image=kw["latent"]["samples"]; sigma=torch.ones(1)
    baseline=core(image,sigma,kw["positive"][0][0]).detach().clone()
    before=tensor_hash(image); rng=torch.random.get_rng_state()
    suite=module.collect_experimental_masks(**kw)
    report=suite["report"]
    assert len(environment[1])==1 and len(environment[2])==1
    assert report["phase1_nfe"]==2 and report["phase2_nfe"]==0
    assert report["completed_reference_image_generated"] is False
    assert report["steps"]==8 and report["trial_id"]==7
    assert report["used_sigmas"]==[1.,.875,.75] and len(report["full_sigmas"])==9
    assert torch.equal(environment[1][0][2],torch.tensor([1.,.875,.75]))
    assert report["initial_noise_sha256"]==tensor_hash(environment[1][0][0])
    assert before==tensor_hash(image)==report["initial_latent_sha256"]
    assert torch.equal(rng,torch.random.get_rng_state())
    assert torch.equal(baseline,core(image,sigma,kw["positive"][0][0]))
    assert report["forward_integrity"] and all(row["input_before_sha256"]==row["input_after_sha256"] for row in report["forward_integrity"])
    assert all(row["output_before_sha256"]==row["output_after_sha256"] for row in report["forward_integrity"])
    assert {(row["step"],row["block"]) for row in report["tap_manifest"]}=={(1,0),(2,0)}
    assert len(report["algorithm"]["variants"])==7
    assert report["memory"]["peak_allocated_bytes"] is None and report["memory"]["unavailable_reason"]
    assert not kw["model"].model_options and not kw["model"].injections
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in core.modules())
    assert "latent" not in suite
    def tensors(value):
        if isinstance(value,torch.Tensor):yield value
        elif isinstance(value,dict):
            for v in value.values():yield from tensors(v)
        elif isinstance(value,(tuple,list)):
            for v in value:yield from tensors(v)
    assert all(t.device.type=="cpu" and not t.requires_grad and t.ndim in (2,3) and t.shape[-1] in (2,4) for t in tensors(suite))
    assert not any(name in suite for name in ("img_q","txt_k","img_k","attention_input","attention_output"))


@pytest.mark.parametrize("failure",["sample","observe","install","cleanup"])
def test_failures_release_owned_hooks_and_keep_primary_exception(environment,tmp_path,monkeypatch,failure):
    module=integration();kw=prepared(tmp_path,environment)
    if failure=="sample":
        original=environment[0]["sample"].sample
        def fail(*a,**k):
            original(*a,**k);raise RuntimeError("primary sampling failure")
        monkeypatch.setattr(environment[0]["sample"],"sample",fail)
    elif failure=="observe":
        monkeypatch.setattr(module.ExperimentalMaskAccumulator,"observe",lambda *a,**k:(_ for _ in()).throw(RuntimeError("primary observation failure")))
    elif failure=="install":
        original=module.ExperimentalAttentionCollector.install
        def fail(self,*a):
            original(self,*a);raise RuntimeError("primary installation failure")
        monkeypatch.setattr(module.ExperimentalAttentionCollector,"install",fail)
    else:
        original=environment[0]["model_management"].unload_model_and_clones
        def fail(*a,**k):
            original(*a,**k);raise RuntimeError("primary cleanup failure")
        monkeypatch.setattr(environment[0]["model_management"],"unload_model_and_clones",fail)
    with pytest.raises(RuntimeError,match="primary"):
        module.collect_experimental_masks(**kw)
    assert environment[2]
    core=kw["model"].model.diffusion_model
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in core.modules())
    assert not kw["model"].model_options and not kw["model"].injections
    from slider_fuse.lora import core_guard
    with core_guard(core):pass


@pytest.mark.parametrize("kind",["background_empty","background_absent","background_overlap","unknown_config","schedule","hook","callback","missing_step"])
def test_invalid_scope_fails_before_sampling(environment,tmp_path,kind):
    module=integration();kw=prepared(tmp_path,environment);handle=None
    if kind=="background_empty":kw["background_phrase"]=""
    if kind=="background_absent":kw["background_phrase"]="forest"
    if kind=="background_overlap":kw["background_phrase"]="woman"
    if kind=="unknown_config":kw["config_json"]='{"unknown":1}'
    if kind=="schedule":kw["steps"]=2
    if kind=="hook":handle=kw["model"].model.diffusion_model.blocks[0].register_forward_hook(lambda *a:None)
    if kind=="callback":kw["model"].model_options["sampler_post_cfg_function"]=[lambda *a:None]
    if kind=="missing_step":
        config=json.loads(kw["config_json"]);config["collect_step"]=3;kw["config_json"]=json.dumps(config)
    try:
        with pytest.raises(ValueError):module.collect_experimental_masks(**kw)
        assert not environment[1]
    finally:
        if handle:handle.remove()


def artifact_suite():
    masks={"target":torch.tensor([[[1.,0.],[0.,0.]]]),"protected":torch.tensor([[[0.,1.],[0.,0.]]]),"background":torch.tensor([[[0.,0.],[1.,1.]]])}
    bank={"grid":(2,2),"masks":masks,"raw_maps":{name:value.reshape(1,4).clone() for name,value in masks.items()},
          "uncertain_mask":torch.zeros(1,2,2),"real_background_mask":masks["background"].clone(),
          "seed_masks":{name:value.clone() for name,value in masks.items()}}
    statuses={name:{"status":"failed","error":"missing seeds"} for name in
        ("baseline","centroid_control","multi_proto","multi_proto_bg","adaln_bg","ensemble_bg","propagated_bg")}
    statuses["baseline"]={"status":"ok"}
    return {"experiment_schema_version":1,"banks":{"baseline":bank},"report":{"run_id":"test-run","seed":42,
            "phase1_nfe":2,"phase2_nfe":0,"completed_reference_image_generated":False,
            "algorithm":{"variants":statuses}}}


def test_artifact_saver_exists():
    assert importlib.util.find_spec("slider_fuse.experimental_artifacts") is not None, "Experimental artifact saver is missing"


def test_save_all_masks_raw_floats_and_hashes_without_vae(tmp_path):
    from PIL import Image
    from safetensors.torch import load_file
    from slider_fuse.experimental_artifacts import save_experimental_artifacts
    suite=artifact_suite()
    saved=save_experimental_artifacts(tmp_path,"case",suite)
    manifest=json.loads((tmp_path/saved["manifest"]).read_text())
    assert manifest["complete"] and manifest["experiment_schema_version"]==1
    assert "diagnostic_schema_version" not in manifest
    assert manifest["report"]["phase1_nfe"]==2 and manifest["save_seconds"]>=0
    tensors=load_file(str(tmp_path/saved["tensors"]))
    for role in ("target","protected","background"):
        key=f"baseline.masks.{role}";value=suite["banks"]["baseline"]["masks"][role]
        assert torch.equal(tensors[key],value) and tensors[key].dtype==torch.float32
        png=Image.open(tmp_path/saved["mask_pngs"][key])
        assert png.size==(2,2) and list(__import__("numpy").asarray(png).flatten())==[int(v*255) for v in value.flatten()]
        assert manifest["tensor_sha256"][key]==tensor_hash(value)
    assert "baseline.uncertain_mask" in tensors and "baseline.raw_maps.target" in tensors
    assert all("latent" not in key and "qkv" not in key for key in tensors)
    second=save_experimental_artifacts(tmp_path,"case",suite)
    assert saved["manifest"]!=second["manifest"]


@pytest.mark.parametrize("prefix",["../bad","/abs","a/b","a\\b","",".","x\x00y"])
def test_artifact_paths_reject_traversal_before_writing(tmp_path,prefix):
    from slider_fuse.experimental_artifacts import save_experimental_artifacts
    with pytest.raises(ValueError):save_experimental_artifacts(tmp_path,prefix,artifact_suite())
    assert list(tmp_path.iterdir())==[]


def test_failed_artifact_save_cleans_entire_new_bundle(tmp_path,monkeypatch):
    import safetensors.torch
    from slider_fuse.experimental_artifacts import save_experimental_artifacts
    monkeypatch.setattr(safetensors.torch,"save_file",lambda *a,**kw:(_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError,match="disk full"):
        save_experimental_artifacts(tmp_path,"case",artifact_suite())
    assert list(tmp_path.iterdir())==[]


def test_saver_rejects_nonfinite_map_before_writing(tmp_path):
    from slider_fuse.experimental_artifacts import save_experimental_artifacts
    suite=artifact_suite();suite["banks"]["baseline"]["raw_maps"]["target"][0,0]=float("nan")
    with pytest.raises(ValueError,match="finite"):save_experimental_artifacts(tmp_path,"case",suite)
    assert list(tmp_path.iterdir())==[]


def test_default_collection_avoids_full_feature_hash_audit(environment,tmp_path):
    module=integration();kw=prepared(tmp_path,environment)
    suite=module.collect_experimental_masks(**kw)
    report=suite["report"]
    assert report["tensor_audit_enabled"] is False
    assert all("tensor_sha256_before" not in tap for tap in report["tap_manifest"])
    assert all("input_before_sha256" not in row for row in report["forward_integrity"])
    assert report["implementation"]["implementation_source_sha256"]


@pytest.mark.parametrize("metadata",["hooks","control"])
def test_conditioning_hooks_and_controlnet_rejected(environment,tmp_path,metadata):
    module=integration();kw=prepared(tmp_path,environment)
    kw["positive"][0][1][metadata]=object()
    with pytest.raises(ValueError,match="conditioning|ControlNet"):
        module.collect_experimental_masks(**kw)
    assert not environment[1]


def test_all_failed_suite_still_saves_failure_manifest_and_evidence(tmp_path):
    from slider_fuse.experimental_artifacts import save_experimental_artifacts
    suite=artifact_suite();suite["banks"]={}
    suite["report"]["algorithm"]["variants"]={name:{"status":"failed","error":"weak evidence"} for name in
        ("baseline","centroid_control","multi_proto","multi_proto_bg","adaln_bg","ensemble_bg","propagated_bg")}
    suite["evidence_maps"]={"step_2.block_0.target.head_scores":torch.ones(2,4)}
    saved=save_experimental_artifacts(tmp_path,"failed",suite)
    report=json.loads((tmp_path/saved["manifest"]).read_text())
    assert report["complete"] and report["artifacts"]["mask_pngs"]=={}
    assert len(report["report"]["algorithm"]["variants"])==7


def test_selector_preserves_actual_algorithm_error():
    module=integration();suite=artifact_suite();suite["report"]["algorithm"]["variants"]["multi_proto"]={"status":"failed","error":"no confident target seed"}
    with pytest.raises(ValueError,match="no confident target seed"):
        module.select_experimental_mask(suite,"multi_proto")


def test_remove_attempts_every_handle_and_can_retry_failure():
    module=integration();collector=module.ExperimentalAttentionCollector(None,{},(1,1),1,torch.ones(1))
    calls=[]
    class Handle:
        def __init__(self,name,fail=False):self.name=name;self.fail=fail
        def remove(self):
            calls.append(self.name)
            if self.fail:
                self.fail=False
                raise RuntimeError("remove failed")
    collector._handles=[Handle("first"),Handle("failing",True),Handle("last")]
    with pytest.raises(RuntimeError,match="remove failed"):collector.remove()
    assert calls==["last","failing","first"]
    assert len(collector._handles)==1
    collector.remove()
    assert collector._handles==[]


@pytest.mark.parametrize("damage",["status","raw_maps","uncertainty","background_decomposition","extra_ok","map_shape"])
def test_saver_rejects_incomplete_candidate_artifacts(tmp_path,damage):
    from slider_fuse.experimental_artifacts import save_experimental_artifacts
    suite=artifact_suite();bank=suite["banks"]["baseline"]
    if damage=="status":del suite["report"]["algorithm"]["variants"]["multi_proto"]
    if damage=="raw_maps":del bank["raw_maps"]
    if damage=="uncertainty":del bank["uncertain_mask"]
    if damage=="background_decomposition":bank["uncertain_mask"].fill_(1.)
    if damage=="extra_ok":suite["report"]["algorithm"]["variants"]["multi_proto"]={"status":"ok"}
    if damage=="map_shape":bank["raw_maps"]["target"]=torch.ones(4,1)
    with pytest.raises(ValueError):save_experimental_artifacts(tmp_path,"case",suite)
    assert list(tmp_path.iterdir())==[]


def test_collection_records_conditioning_and_model_identity_limit(environment,tmp_path):
    module=integration();kw=prepared(tmp_path,environment)
    report=module.collect_experimental_masks(**kw)["report"]
    assert report["conditioning"]["positive_sha256"]==tensor_hash(kw["positive"][0][0])
    assert report["conditioning"]["negative_used"] is False
    assert report["model"]["weights_identity_verified"] is False
    assert report["model"]["diffusion_class"].endswith("TinyKrea")


def test_real_seven_variant_algorithm_banks_roundtrip_through_saver(tmp_path):
    from test_experimental_masks import fixture, config
    from slider_fuse.experimental_masks import ExperimentalMaskAccumulator, VARIANTS
    from slider_fuse.experimental_artifacts import save_experimental_artifacts
    from safetensors.torch import load_file
    data,positions=fixture()
    accumulator=ExperimentalMaskAccumulator(config(),(3,4),positions)
    accumulator.observe(step=2,block=18,**data)
    banks,algorithm=accumulator.finalize()
    assert set(banks)==set(VARIANTS)
    suite={"experiment_schema_version":1,"banks":banks,"report":{"algorithm":algorithm},"evidence_maps":accumulator.evidence_maps}
    saved=save_experimental_artifacts(tmp_path,"seven",suite)
    values=load_file(str(tmp_path/saved["tensors"]))
    for variant in VARIANTS:
        for role in ("target","protected","background"):
            assert torch.equal(values[f"{variant}.masks.{role}"],banks[variant]["masks"][role])
    assert any(key.endswith("head_allowed_fraction") for key in values)
