"""Sampler orchestration doubles; actual native lifecycle is tested separately."""
from dataclasses import replace

import pytest
import torch

from test_runtime import environment, setup_run
from slider_fuse import native_pair, attention
from slider_fuse.diagnostics import DiagnosticRecorder, region_measurement, serialize_measurement
from slider_fuse.prediction_mixing import prediction_mask, mix_predictions


@pytest.fixture
def mix_runtime(environment, monkeypatch):
    calls = []
    def native(base, source, strength, modules):
        slider = base.clone(); slider.strength = strength
        return slider
    monkeypatch.setattr(native_pair, "apply_native_slider", native)
    monkeypatch.setattr(environment[0]["model_management"], "intermediate_device", lambda: torch.device("cpu"), raising=False)
    monkeypatch.setattr(environment[0]["model_management"], "intermediate_dtype", lambda: torch.float32, raising=False)
    monkeypatch.setattr(environment[0]["samplers"], "sampler_object", lambda n: n, raising=False)
    monkeypatch.setattr(attention, "similarity_maps", lambda *a, **k: {
        "target": torch.tensor([[1., 0., 1., 0.]]), "protected": torch.tensor([[0., 1., 0., 1.]])})

    def make(base, slider, token_mask, *, patch, recorder, report, mode, restore_base=False):
        class Guider:
            def set_conds(self, positive, negative): self.context = positive[0][0]
            def set_cfg(self, cfg): assert cfg == 1.
            def sample(self, noise, image, sampler, sigmas, seed=None):
                calls.append((noise.clone(), image.clone(), sigmas.clone(), token_mask.clone(),restore_base))
                x = noise.clone(); self.rows = []; self.bases = []; self.sliders = []
                for i, sigma in enumerate(sigmas[:-1]):
                    recorder.begin_step(i, sigma.reshape(1))
                    # Cross-region dependence gives the later-step invariance test meaning.
                    p_base = x * .1 + x.mean()
                    p_slider = p_base + (slider.strength * .01 if slider is not None else 0)
                    mask = prediction_mask(token_mask, x, patch=patch)
                    p = mix_predictions(p_base, p_slider, mask) if mode == "both" else p_base if mode == "base" else p_slider
                    recorder.record_inputs(x, self.context, sigma.reshape(1))
                    recorder.record_prediction(p); recorder.record_step(p, x, sigma.reshape(1))
                    if mode == "both" and recorder.level == "audit":
                        self.rows.append({"eval_index": i,
                            "base_excluded": serialize_measurement(region_measurement(p_base, p, (~mask).expand_as(p)).double().tolist()),
                            "slider_selected": serialize_measurement(region_measurement(p_slider, p, mask.expand_as(p)).double().tolist())})
                        self.bases.append(p_base.clone()); self.sliders.append(p_slider.clone())
                    report["sampler_nfe"] += 1
                    x = x + (x - p) / sigma * (sigmas[i+1] - sigma)
                return x
            def finish_report(self):
                audit, tensors = recorder.finalize(); report.update(audit)
                steps = report["sampler_nfe"]
                report.update(phase2_nfe=steps, branch_nfe={"base": steps if mode != "slider" else 0,
                    "slider": steps if mode != "base" else 0}, prediction_mix_audit=self.rows)
                if self.bases:
                    tensors.update(trace_base_predictions=torch.stack(self.bases), trace_slider_predictions=torch.stack(self.sliders))
                return tensors
        return Guider()
    monkeypatch.setattr(native_pair, "make_prediction_mix_guider", make)
    return calls


def run(tmp_path, *, auto=True, scope="target_mask", strength=4., level="audit", radius=0):
    model, positive, info, subjects, file = setup_run(tmp_path)
    if auto: subjects = tuple(replace(s, manual_mask=None) for s in subjects)
    return native_pair.sample_krea2_prediction_mix(model, positive, positive, info, subjects,
        {"samples": torch.zeros(1, 4, 4, 4)}, str(file), strength=strength, seed=42, steps=2,
        cfg=1., mix_scope=scope, diagnostic_level=level, mask_mode="auto" if auto else "manual",
        collect_step=1, collect_block=0, selection_dilate_radius=radius)


def test_auto_restarts_full_schedule_with_collected_mask(mix_runtime, environment, tmp_path):
    latent, bank, payload = run(tmp_path)
    collection = environment[1][0]; generation = mix_runtime[0]
    assert torch.equal(collection[0], generation[0]) and torch.equal(collection[1], generation[1])
    assert torch.equal(generation[2], torch.tensor([1., .5, 0.]))
    assert torch.equal(generation[3], bank["masks"]["target"])
    assert payload.report["phase1_nfe"] == 1 and payload.report["phase2_nfe"] == 2
    assert payload.report["total_model_nfe"] == 5
    assert generation[4] is True


@pytest.mark.parametrize("scope,strength,counts", [
    ("none", 4., {"base": 2, "slider": 0}), ("all", 4., {"base": 0, "slider": 2}),
    ("target_mask", 0., {"base": 2, "slider": 0}), ("target_mask", -1., {"base": 2, "slider": 2})])
def test_auto_endpoint_counts(mix_runtime, tmp_path, scope, strength, counts):
    _, _, data = run(tmp_path, scope=scope, strength=strength)
    assert data.report["branch_nfe"] == counts and data.report["phase1_nfe"] == 1
    assert data.report["total_model_nfe"] == 1 + sum(counts.values())


def test_auto_matches_manual_with_identical_partition(mix_runtime, tmp_path):
    torch.manual_seed(111)
    a, bank, pa = run(tmp_path)
    # setup_run's manual masks are the same column partition as the collected maps.
    torch.manual_seed(111)
    b, _, pb = run(tmp_path, auto=False)
    assert torch.equal(a["samples"], b["samples"])
    assert torch.equal(pa.extra_tensors["trace_inputs"], pb.extra_tensors["trace_inputs"])
    assert torch.equal(pa.extra_tensors["trace_predictions"], pb.extra_tensors["trace_predictions"])
    assert pb.report["phase1_nfe"] == 0


def test_manual_postprocess_rejected_before_observation(mix_runtime, environment, tmp_path):
    model, positive, info, subjects, file = setup_run(tmp_path)
    with pytest.raises(ValueError, match="auto"):
        native_pair.sample_krea2_prediction_mix(model, positive, positive, info, subjects,
            {"samples": torch.zeros(1, 4, 4, 4)}, str(file), strength=4., seed=42,
            steps=2, fill_holes_max_area=8)
    assert not mix_runtime and not environment[1]


def test_auto_expansion_changes_only_prediction_selection(mix_runtime,tmp_path,monkeypatch):
    monkeypatch.setattr(attention,"similarity_maps",lambda *a,**k:{
        "target":torch.tensor([[1.,0.,.4,0.]]),"protected":torch.tensor([[0.,1.,0.,.3]])})
    _,bank,data=run(tmp_path,radius=1)
    assert bank["masks"]["target"].sum()==1 and bank["masks"]["background"].sum()==2
    assert torch.equal(data.effective_image_mask,torch.tensor([[[1.,0.],[1.,1.]]]))
    assert bank["selection_added_mask"].sum()==2
    assert not (data.effective_image_mask*bank["masks"]["protected"]).any()
    assert data.report["prediction_selection"]["dilate_radius"]==1
