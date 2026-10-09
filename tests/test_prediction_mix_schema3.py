import json

import pytest
import torch

from test_prediction_mix_artifacts import mix_payload
from test_diagnostic_tools import tool
from slider_fuse.diagnostics import save_diagnostic_artifacts, validate_audit_coverage
from slider_fuse.masks import generate_masks, postprocess_masks
from slider_fuse.sampling import tensor_hash


def v3_payload(*, auto=True, level="audit", run_id="v3"):
    g, latent, data = mix_payload()
    raw = {"target": torch.tensor([[1., 0., 1., 0.]]), "protected": torch.tensor([[0., 1., 0., 1.]])}
    bank = postprocess_masks(generate_masks(raw, (2, 2)))
    config = {"mode": "auto" if auto else "manual", "collection_branch": "base" if auto else None,
        "collect_step": 2 if auto else None, "collect_block": 0 if auto else None,
        "top_k_ratio": .3 if auto else None, "temperature": 4000. if auto else None,
        "fill_holes_max_area": 0, "mask_dilate_radius": 0}
    g["8"]["inputs"].update(mask_mode=config["mode"], collect_step=2, collect_block=0,
                             diagnostic_level=level)
    if auto:
        g["6"]["inputs"].pop("target_mask", None); g["6"]["inputs"].pop("protected_mask", None)
    if level == "summary":
        data.extra_tensors = {}; data.report.update(step_trace=None, prediction_mix_audit=[])
    data.extra_tensors.update({"reference_" + k + "_mask": v for k, v in bank["masks"].items()})
    if auto:
        data.extra_tensors.update({"raw_"+k+"_similarity": v for k, v in raw.items()})
        data.extra_tensors.update(original_target_mask=bank["original_masks"]["target"],
                                 added_target_mask=bank["added_target_mask"])
    sigmas = torch.cat([data.extra_tensors["trace_sigmas"], torch.zeros(1)]) if level == "audit" else torch.linspace(1,0,9)
    if auto:data.extra_tensors["collection_sigmas"]=sigmas.clone()
    data.report.update(diagnostic_schema_version=3, diagnostic_level=level, run_id=run_id,
        prediction_selection={"dilate_radius":0,"method":"largest_component_background_only"},
        mask_mode=config["mode"], mask_generation=config, phase1_nfe=2 if auto else 0,
        phase2_nfe=8, total_model_nfe=18 if auto else 16, grid=[2, 2], patch_size=2,
        text_token_count=2,token_positions={"target":[0],"protected":[1]},
        raw_maps_available=auto,effective_image_mask_coverage=.5,
        collection_attention_heads={"query":2,"key_value":2} if auto else None,
        initial_noise_sha256="noise", initial_latent_sha256="latent", full_sigmas_sha256=tensor_hash(sigmas),
        conditioning_sha256="cond", first_input_sha256=tensor_hash(torch.ones_like(latent)),
        first_timestep_sha256="sigma", first_input_shape=list(latent.shape), first_input_dtype=str(latent.dtype),
        lora_sha256="lora", environment={"comfy_revision":"test"}, implementation_revision="rev",
        implementation_source_sha256="source", prediction_space="comfy_cfg1_denoised",
        reference_partition_sha256={k:tensor_hash(v) for k,v in bank["masks"].items()},
        mask_postprocess=bank["postprocess_diagnostics"], masks=bank["diagnostics"],
        map_diagnostics=bank["map_diagnostics"] if auto else None,
        observation={"step":2,"block":0,"sigma":float(sigmas[1]),"grid":[2,2],"cap_len":2,
                     "positions":{"target":[0],"protected":[1]},"image_len":4,"q_heads":2,"kv_heads":2} if auto else None,
        collection_initial_noise_sha256="noise" if auto else None,
        collection_initial_latent_sha256="latent" if auto else None,
        collection_full_sigmas_sha256=tensor_hash(sigmas) if auto else None,
        collection_used_sigmas_sha256=tensor_hash(sigmas[:3]) if auto else None)
    return g, latent, data


def save(tmp_path, *, auto=True, level="audit", run_id="v3"):
    g, latent, data = v3_payload(auto=auto, level=level, run_id=run_id)
    report = save_diagnostic_artifacts(tmp_path, run_id, latent, torch.zeros(1,8,8,3), data, g, None, "11")
    return tmp_path/report["artifacts"]["report"]


def test_schema3_auto_roundtrip_has_validated_mask_source(tmp_path):
    p = save(tmp_path)
    report, tensors, _ = tool().load_report(p)
    assert report["diagnostic_schema_version"] == 3
    assert "raw_target_similarity" in tensors and report["phase1_nfe"] == 2


@pytest.mark.parametrize("level", ["summary", "audit"])
def test_auto_audit_counts_phase1_separately(level):
    _, _, data = v3_payload(level=level)
    data.report["branch_nfe"]["base"] += 2
    with pytest.raises(ValueError, match="branch|coverage|NFE"):
        validate_audit_coverage(data.report)


def test_auto_provenance_rejects_changed_collection_settings(tmp_path):
    g, latent, data = v3_payload()
    g["8"]["inputs"]["temperature"] = 10000.
    with pytest.raises(ValueError, match="provenance|mask"):
        save_diagnostic_artifacts(tmp_path,"bad",latent,torch.zeros(1,8,8,3),data,g,None,"11")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["raw_missing", "original_changed", "phase1_hash", "used_sigmas"])
def test_auto_artifact_requires_complete_mask_source(tmp_path, kind):
    g, latent, data = v3_payload()
    if kind == "raw_missing": data.extra_tensors.pop("raw_target_similarity")
    if kind == "original_changed": data.extra_tensors["original_target_mask"] = 1-data.extra_tensors["original_target_mask"]
    if kind == "phase1_hash": data.report["collection_initial_noise_sha256"] = "different"
    if kind == "used_sigmas": data.report["collection_used_sigmas_sha256"] = "different"
    with pytest.raises(ValueError):
        save_diagnostic_artifacts(tmp_path,"bad",latent,torch.zeros(1,8,8,3),data,g,None,"11")
    assert not list(tmp_path.iterdir())


def test_same_mask_reference_accepts_only_identical_partition(tmp_path):
    a=save(tmp_path,auto=True,run_id="auto");b=save(tmp_path,auto=False,run_id="manual")
    with pytest.raises(ValueError, match="mask"):
        tool().compare_reports([a,b])
    pair=tool().compare_reports([a,b],same_mask_reference=True,include_trajectories=True)["pairs"][0]
    assert pair["comparison_kind"] == "same_mask_phase2_comparison"
    assert not pair["repeat_trial"] and pair["final_latent"]["exact_equal"]


def test_manual_summary_still_validates_counts_and_loads(tmp_path):
    p=save(tmp_path,auto=False,level="summary")
    r,t,_=tool().load_report(p)
    assert "trace_predictions" not in t and r["phase1_nfe"] == 0
    r["total_model_nfe"] += 1;p.write_text(json.dumps(r),encoding="utf-8")
    with pytest.raises(ValueError, match="NFE"):
        tool().load_report(p)


def test_schema3_recomputes_prediction_selector(tmp_path):
    g, latent, data = v3_payload()
    data.extra_tensors["trace_predictions"][1] += 1
    data.report["step_trace"][1]["prediction_sha256"] = tensor_hash(data.extra_tensors["trace_predictions"][1])
    with pytest.raises(ValueError, match="selector|selection"):
        save_diagnostic_artifacts(tmp_path,"bad",latent,torch.zeros(1,8,8,3),data,g,None,"11")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["phase1_nfe", "phase2_nfe", "sampler_nfe", "total_model_nfe"])
def test_schema3_requires_integer_nfe(field):
    _, _, data = v3_payload()
    data.report[field] = float(data.report[field])
    with pytest.raises(ValueError, match="NFE"):
        validate_audit_coverage(data.report)


def test_same_mask_reference_rejects_different_partition(tmp_path):
    a=save(tmp_path,auto=True,run_id="auto")
    g,latent,data=v3_payload(auto=False,run_id="different")
    target=1-data.effective_image_mask
    data.effective_image_mask=target
    data.extra_tensors["reference_target_mask"]=target
    data.extra_tensors["reference_protected_mask"]=1-target
    data.report["effective_image_mask_sha256"]=tensor_hash(target)
    data.report["reference_partition_sha256"]={k:tensor_hash(data.extra_tensors["reference_"+k+"_mask"]) for k in ("target","protected","background")}
    bank=postprocess_masks({"masks":{k:data.extra_tensors["reference_"+k+"_mask"] for k in ("target","protected","background")},"grid":(2,2)})
    data.report.update(masks=bank["diagnostics"],mask_postprocess=bank["postprocess_diagnostics"])
    r=save_diagnostic_artifacts(tmp_path,"different",latent,torch.zeros(1,8,8,3),data,g,None,"11")
    with pytest.raises(ValueError,match="partition|mask"):
        tool().compare_reports([a,tmp_path/r["artifacts"]["report"]],same_mask_reference=True)


def test_boolean_collection_step_is_not_an_integer_setting():
    from slider_fuse.diagnostics import mask_generation_config
    with pytest.raises(ValueError, match="collect_step"):
        mask_generation_config({"mask_mode":"auto","steps":8,"collect_step":True,"collect_block":0})


@pytest.mark.parametrize("kind",["prefix_hash","observation_positions","observation_grid","cap_len",
                                "sigma","heads","masks","coverage","raw_available"])
def test_summary_reader_rejects_fabricated_collection_provenance(tmp_path,kind):
    p=save(tmp_path,level="summary");r=json.loads(p.read_text(encoding="utf-8"))
    if kind=="prefix_hash":r["collection_used_sigmas_sha256"]="fabricated"
    if kind=="observation_positions":r["observation"]["positions"]={"target":[1],"protected":[0]}
    if kind=="observation_grid":r["observation"]["grid"]=[99,99]
    if kind=="cap_len":r["observation"]["cap_len"]=999
    if kind=="sigma":r["observation"]["sigma"]=999
    if kind=="heads":r["observation"]["q_heads"]=999
    if kind=="masks":r["masks"]={}
    if kind=="coverage":r["effective_image_mask_coverage"]=1.
    if kind=="raw_available":r["raw_maps_available"]=False
    p.write_text(json.dumps(r),encoding="utf-8")
    with pytest.raises(ValueError):tool().load_report(p)


def test_selection_setting_provenance_cannot_be_changed(tmp_path):
    g,latent,data=v3_payload()
    g["8"]["inputs"]["selection_dilate_radius"]=4
    with pytest.raises(ValueError,match="selection|provenance"):
        save_diagnostic_artifacts(tmp_path,"bad",latent,torch.zeros(1,8,8,3),data,g,None,"11")


def test_same_mask_reference_requires_identical_selection_policy(tmp_path):
    g,latent,data=v3_payload(run_id="auto")
    g["8"]["inputs"]["selection_dilate_radius"]=2
    data.report["prediction_selection"]["dilate_radius"]=2
    r=save_diagnostic_artifacts(tmp_path,"auto",latent,torch.zeros(1,8,8,3),data,g,None,"11")
    manual=save(tmp_path,auto=False,run_id="manual")
    with pytest.raises(ValueError,match="selection|mask"):
        tool().compare_reports([tmp_path/r["artifacts"]["report"],manual],same_mask_reference=True)


def test_expanded_selection_roundtrip_validates_background_predictions(tmp_path):
    from slider_fuse.masks import prediction_selection_mask
    from slider_fuse.prediction_mixing import prediction_mask
    from slider_fuse.diagnostics import region_measurement,serialize_measurement
    g,latent,data=v3_payload()
    raw={"target":torch.tensor([[1.,0.,.4,0.]]),"protected":torch.tensor([[0.,1.,0.,.3]])}
    bank=postprocess_masks(generate_masks(raw,(2,2)))
    selected=prediction_selection_mask(bank,1)
    g["8"]["inputs"]["selection_dilate_radius"]=1
    data.effective_image_mask=selected
    data.report.update(prediction_selection={"dilate_radius":1,"method":"largest_component_background_only"},
        masks=bank["diagnostics"],map_diagnostics=bank["map_diagnostics"],mask_postprocess=bank["postprocess_diagnostics"],
        reference_partition_sha256={k:tensor_hash(v) for k,v in bank["masks"].items()},
        effective_image_mask_sha256=tensor_hash(selected),effective_image_mask_coverage=.75)
    data.extra_tensors.update({"reference_"+k+"_mask":v for k,v in bank["masks"].items()})
    data.extra_tensors.update({"raw_"+k+"_similarity":v for k,v in raw.items()})
    data.extra_tensors.update(original_target_mask=bank["original_masks"]["target"],added_target_mask=bank["added_target_mask"])
    mask=prediction_mask(selected,latent,patch=2).expand_as(latent)
    base=torch.zeros_like(latent);native=torch.ones_like(latent);mixed=torch.where(mask,native,base)
    data.first_prediction=mixed.clone();latent=mixed.clone()
    data.report.update(first_prediction_sha256=tensor_hash(mixed),final_latent_sha256=tensor_hash(latent))
    data.extra_tensors.update(trace_base_predictions=torch.stack([base]*8),
        trace_slider_predictions=torch.stack([native]*8),trace_predictions=torch.stack([mixed]*8))
    for i,row in enumerate(data.report["step_trace"]):
        row["prediction_sha256"]=tensor_hash(mixed)
        data.report["prediction_mix_audit"][i].update(
            base_excluded=serialize_measurement(region_measurement(base,mixed,~mask).double().tolist()),
            slider_selected=serialize_measurement(region_measurement(native,mixed,mask).double().tolist()))
    r=save_diagnostic_artifacts(tmp_path,"expanded",latent,torch.zeros(1,8,8,3),data,g,None,"11")
    p=tmp_path/r["artifacts"]["report"]
    assert tool().load_report(p)[0]["effective_image_mask_coverage"]==.75
    r["prediction_selection"]["dilate_radius"]=0;p.write_text(json.dumps(r),encoding="utf-8")
    with pytest.raises(ValueError,match="mask|scope"):
        tool().load_report(p)
