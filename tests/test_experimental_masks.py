"""CPU contracts for bounded diagnostic-only attention mask experiments."""
import copy
import importlib
import importlib.util
import json

import pytest
import torch

from slider_fuse.attention import similarity_maps
from slider_fuse.masks import generate_masks


def algorithms():
    assert importlib.util.find_spec("slider_fuse.experimental_masks") is not None, "experimental algorithms have not been implemented"
    return importlib.import_module("slider_fuse.experimental_masks")


def fixture():
    # Three separated concepts; last token has no evidence and must remain uncertain.
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    q = torch.nn.functional.one_hot(labels, 3).float()[None, :, None, :] * 12
    q = q.repeat(1, 1, 2, 1)
    q[:, -1] = 0
    txt = torch.eye(3)[None, :, None, :].repeat(1, 1, 2, 1) * 12
    out = torch.nn.functional.one_hot(labels, 3).float()[None] * 100
    out[:, -1] = 0
    return dict(img_q=q, txt_k=txt, img_k=q.clone(), attention_output=out,
                attention_input=out.clone()), {"target": [0], "protected": [1], "background": [2]}


def config(**kwargs):
    return {"selected_steps": [2], "selected_blocks": [18], **kwargs}


def run(data=None, positions=None, **kwargs):
    default_data, default_positions = fixture()
    data = default_data if data is None else data
    positions = default_positions if positions is None else positions
    collector = algorithms().ExperimentalMaskAccumulator(config(**kwargs), (3, 4), positions)
    collector.observe(step=2, block=18, **data)
    return collector.finalize()


def test_legacy_baseline_is_exact_not_recalibrated():
    data, positions = fixture()
    banks, report = run(data, positions)
    expected = generate_masks(similarity_maps(data["img_q"], data["txt_k"], data["attention_output"],
                               {k: positions[k] for k in ("target", "protected")}, .2, 10000.), (3, 4))
    for name in expected["masks"]:
        assert torch.equal(banks["baseline"]["masks"][name], expected["masks"][name])
    for name in expected["raw_maps"]:
        assert torch.equal(banks["baseline"]["raw_maps"][name], expected["raw_maps"][name])
    assert report["variants"]["baseline"]["status"] == "ok"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("override", [
    {"prototypes": True}, {"prototypes": 0}, {"prototypes": 3.0}, {"collect_step": 0},
    {"collect_block": -1}, {"temperature": float("nan")}, {"temperature": float("inf")},
    {"temperature": True}, {"top_k_ratio": 0}, {"top_k_ratio": 1.01}, {"seed_margin": -1},
    {"mask_confidence": 1.1}, {"context_temperature": 0}, {"feature_temperature": 0},
    {"propagation_iterations": 4}, {"propagation_iterations": False}, {"affinity_chunk_size": 0},
    {"affinity_threshold": -1}, {"propagation_strength": 2}, {"propagation_heads": [True]},
    {"propagation_heads": []}, {"propagation_heads": [0, 0]}, {"selected_steps": [1]},
    {"selected_steps": [2, 3]}, {"selected_blocks": [16]}, {"selected_steps": [1, 1, 2]},
    {"selected_steps": [1, 2], "selected_blocks": [1, 2, 3, 18]}, {"unknown": 1},
])
def test_config_rejects_invalid_values(override):
    with pytest.raises(ValueError):
        algorithms().validate_experimental_config(config(**override))


def test_config_is_fresh_json_and_defaults_document_conservative_limits():
    mod = algorithms()
    first = mod.validate_experimental_config({})
    assert first["selected_steps"] == [1, 2]
    assert first["selected_blocks"] == [16, 18]
    assert first["propagation_iterations"] == 1
    assert first["top_k_ratio"] == .2
    first["selected_steps"].append(9)
    assert mod.validate_experimental_config({})["selected_steps"] == [1, 2]
    json.dumps(mod.validate_experimental_config({}), allow_nan=False)


def test_one_prototype_equality_is_a_matched_centroid_control():
    banks, _ = run(prototypes=1)
    for key in ("masks", "raw_maps", "seed_masks"):
        assert banks["centroid_control"][key].keys() == banks["multi_proto"][key].keys()
        for role in banks["centroid_control"][key]:
            torch.testing.assert_close(banks["centroid_control"][key][role], banks["multi_proto"][key][role], rtol=0, atol=0)


def test_candidate_partition_keeps_uncertain_distinct_from_evidence_background():
    banks, report = run()
    assert set(banks) == set(algorithms().VARIANTS)
    for name, bank in banks.items():
        assert all(value.device.type == "cpu" and value.grad_fn is None for value in bank["masks"].values())
        assert torch.equal(sum(bank["masks"].values()), torch.ones(1, 3, 4))
        if name == "baseline":
            continue
        assert bank["uncertain_mask"].flatten()[-1] == 1
        assert bank["real_background_mask"].flatten()[-1] == 0
        assert not (bank["uncertain_mask"] * bank["real_background_mask"]).any()
        assert report["variants"][name]["confidence_semantics"] == "heuristic_not_calibrated"
    assert banks["multi_proto_bg"]["real_background_mask"].sum() > 0
    assert banks["multi_proto"]["real_background_mask"].sum() == 0


def test_empty_background_candidate_preserves_successful_baseline_and_no_bg_variants():
    data, positions = fixture()
    positions["background"] = []
    banks, report = run(data, positions)
    assert set(banks) == {"baseline", "centroid_control", "multi_proto"}
    for name in ("multi_proto_bg", "adaln_bg", "ensemble_bg", "propagated_bg"):
        assert report["variants"][name]["status"] == "failed"
        assert report["variants"][name]["error_code"] == "missing_background_tokens"


def test_no_confident_seeds_reports_failure_without_equal_area_or_full_mask():
    data, positions = fixture()
    data["img_q"].zero_()
    banks, report = run(data, positions)
    assert not banks
    assert all(value["status"] == "failed" for value in report["variants"].values())
    assert report["variants"]["multi_proto_bg"]["error_code"] == "empty_seed_pool"


def test_seed_pool_is_only_a_cap_and_seeds_remain_anchored():
    banks, report = run(top_k_ratio=.8)
    bank = banks["propagated_bg"]
    assert bank["seed_masks"]["target"].sum() == 4  # not 80% of all image cells
    assert bank["seed_masks"]["background"].sum() == 3
    for role, seeds in bank["seed_masks"].items():
        evidence = bank["real_background_mask"] if role == "background" else bank["masks"][role]
        assert (seeds <= evidence).all()
    assert report["variants"]["propagated_bg"]["propagation"]["seed_changes"] == 0


def test_observation_is_deterministic_rng_neutral_and_holds_no_full_features():
    data, positions = fixture()
    data["attention_output"].requires_grad_(True)
    mod = algorithms()
    accumulator = mod.ExperimentalMaskAccumulator(config(), (3, 4), positions)
    before = torch.random.get_rng_state()
    accumulator.observe(step=2, block=18, **data)
    assert torch.equal(before, torch.random.get_rng_state())
    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from tensors(item)
    held = list(tensors(vars(accumulator)))
    assert held
    assert all(x.device.type == "cpu" and x.grad_fn is None and not x.requires_grad for x in held)
    assert all(x.ndim <= 3 for x in held)
    banks1, _ = accumulator.finalize()
    banks2, _ = run()
    for name in banks1:
        for role in banks1[name]["masks"]:
            assert torch.equal(banks1[name]["masks"][role], banks2[name]["masks"][role])


def test_candidate_feature_units_do_not_change_cosine_masks_or_scores():
    original, _ = run()
    data, positions = fixture()
    data["attention_output"] *= 1e-3
    data["attention_input"] *= 1e3
    changed, _ = run(data, positions)
    for name in set(original) - {"baseline"}:
        for role in original[name]["raw_maps"]:
            torch.testing.assert_close(original[name]["raw_maps"][role], changed[name]["raw_maps"][role])


def test_target_protected_permutation_swaps_candidates_and_leaves_background():
    banks, _ = run()
    data, positions = fixture()
    positions["target"], positions["protected"] = positions["protected"], positions["target"]
    swapped, _ = run(data, positions)
    for name in banks:
        assert torch.equal(banks[name]["masks"]["target"], swapped[name]["masks"]["protected"])
        assert torch.equal(banks[name]["masks"]["protected"], swapped[name]["masks"]["target"])
        assert torch.equal(banks[name]["masks"]["background"], swapped[name]["masks"]["background"])


def test_input_features_only_affect_adaln_variant():
    banks, _ = run()
    data, positions = fixture()
    data["attention_input"] = torch.randn(data["attention_input"].shape, generator=torch.Generator().manual_seed(123))
    changed, _ = run(data, positions)
    for name in set(banks) - {"adaln_bg"}:
        for role in banks[name]["raw_maps"]:
            torch.testing.assert_close(banks[name]["raw_maps"][role], changed[name]["raw_maps"][role], rtol=0, atol=0)
    assert any(not torch.equal(banks["adaln_bg"]["raw_maps"][role], changed["adaln_bg"]["raw_maps"][role]) for role in banks["adaln_bg"]["raw_maps"])


@pytest.mark.parametrize("field,replace", [
    ("img_q", lambda x: x.repeat(2, 1, 1, 1)),
    ("img_k", lambda x: x[:, :-1]),
    ("txt_k", lambda x: x[:, :, :1]),
    ("attention_output", lambda x: x[:, :-1]),
    ("attention_input", lambda x: x.long()),
    ("img_q", lambda x: x * float("nan")),
])
def test_observation_shape_dtype_and_finiteness_validation(field, replace):
    data, positions = fixture()
    data[field] = replace(data[field])
    with pytest.raises(ValueError):
        run(data, positions)


@pytest.mark.parametrize("positions", [
    {"target": [True], "protected": [1], "background": [2]},
    {"target": [0], "protected": [0], "background": [2]},
    {"target": [0], "protected": [1], "background": [99]},
    {"target": [0], "protected": [1]},
])
def test_positions_are_exact_disjoint_indices_never_clamped(positions):
    with pytest.raises(ValueError):
        run(positions=positions)


def test_non_primary_missing_taps_are_reported_not_silently_ensembled():
    data, positions = fixture()
    accumulator = algorithms().ExperimentalMaskAccumulator({}, (3, 4), positions)
    accumulator.observe(step=2, block=18, **data)
    banks, report = accumulator.finalize()
    assert "baseline" in banks
    assert "ensemble_bg" not in banks
    assert report["variants"]["ensemble_bg"]["error_code"] == "missing_taps"
    assert report["missing_taps"] == [[1, 16], [1, 18], [2, 16]]


def test_ensemble_averages_same_variant_and_is_observation_order_invariant():
    data, positions = fixture()
    first, _ = run()
    cfg = config(selected_steps=[1, 2])
    acc1 = algorithms().ExperimentalMaskAccumulator(cfg, (3, 4), positions)
    acc2 = algorithms().ExperimentalMaskAccumulator(cfg, (3, 4), positions)
    altered = copy.deepcopy(data)
    altered["attention_output"][:, 3] = torch.tensor([5., 30., 10.])
    altered_banks, _ = run(altered, positions)
    for acc, order in ((acc1, [(1, data), (2, altered)]), (acc2, [(2, altered), (1, data)])):
        for step, values in order:
            acc.observe(step=step, block=18, **values)
    banks1, _ = acc1.finalize()
    banks2, _ = acc2.finalize()
    for role in first["multi_proto_bg"]["raw_maps"]:
        expected = (first["multi_proto_bg"]["raw_maps"][role] + altered_banks["multi_proto_bg"]["raw_maps"][role]) / 2
        torch.testing.assert_close(banks1["ensemble_bg"]["raw_maps"][role], expected)
        torch.testing.assert_close(banks1["ensemble_bg"]["raw_maps"][role], banks2["ensemble_bg"]["raw_maps"][role], rtol=0, atol=0)


def test_duplicate_and_unselected_observations_are_rejected():
    data, positions = fixture()
    acc = algorithms().ExperimentalMaskAccumulator(config(), (3, 4), positions)
    with pytest.raises(ValueError, match="selected"):
        acc.observe(step=1, block=18, **data)
    acc.observe(step=2, block=18, **data)
    with pytest.raises(ValueError, match="already"):
        acc.observe(step=2, block=18, **data)


def test_permission_mask_identity_blocks_all_propagation_and_chunks_agree():
    data, positions = fixture()
    initial, _ = run(data, positions)
    data["attention_mask"] = torch.eye(12, dtype=torch.bool)
    small, report = run(data, positions, affinity_chunk_size=2, propagation_iterations=3)
    large, _ = run(data, positions, affinity_chunk_size=128, propagation_iterations=3)
    for role in initial["multi_proto_bg"]["raw_maps"]:
        torch.testing.assert_close(small["propagated_bg"]["raw_maps"][role], initial["multi_proto_bg"]["raw_maps"][role])
        torch.testing.assert_close(small["propagated_bg"]["raw_maps"][role], large["propagated_bg"]["raw_maps"][role])
    assert report["variants"]["propagated_bg"]["propagation"]["largest_affinity_chunk_shape"] == [2, 12]


def test_barriers_stop_qk_affinity_from_invading_protected_or_background_evidence():
    data, positions = fixture()
    initial, _ = run(data, positions)
    # Maliciously strong image affinity to target still cannot overwrite other evidence.
    data["img_k"].zero_()
    data["img_k"][:, :4] = 1000
    banks, report = run(data, positions, propagation_strength=1., propagation_iterations=3)
    for role in ("protected", "background"):
        barrier = initial["multi_proto_bg"]["masks"][role] if role == "protected" else initial["multi_proto_bg"]["real_background_mask"]
        assert not (banks["propagated_bg"]["masks"]["target"] * barrier).any()
    assert report["variants"]["propagated_bg"]["propagation"]["barrier_violations"] == 0


def test_bad_attention_permission_masks_fail_closed():
    for mask in (torch.ones(12), torch.ones(12, 11), torch.ones(12, 12, dtype=torch.int32), torch.full((12, 12), float("nan"))):
        data, positions = fixture()
        data["attention_mask"] = mask
        with pytest.raises(ValueError, match="mask"):
            run(data, positions)


def test_token_and_head_summary_exposes_averaging_without_raw_feature_dump():
    _, report = run()
    summary = report["observations"][0]["context_summary"]
    assert summary["head_count"] == 2
    assert summary["role_tokens"]["target"][0]["token_index"] == 0
    assert "head_spatial_std" in summary["role_tokens"]["target"][0]
    assert len(summary["role_tokens"]["target"][0]["head_spatial_std"]) == 2
    assert "heuristic" in report["limitations"]


def test_evidence_preserves_per_head_and_subword_cpu_maps():
    data, positions = fixture()
    acc = algorithms().ExperimentalMaskAccumulator(config(), (3, 4), positions)
    acc.observe(step=2, block=18, **data)
    assert acc.evidence_maps["step_2.block_18.target.head_scores"].shape == (2, 12)
    assert acc.evidence_maps["step_2.block_18.target.token_scores"].shape == (1, 12)
    assert all(x.device.type == "cpu" and not x.requires_grad for x in acc.evidence_maps.values())


def test_heterogeneous_feature_modes_are_retained_by_multiple_prototypes():
    # Synthetic mechanism check only: two target appearance axes average toward
    # a protected distractor. The centroid then abstains on unseeded appearance.
    data, positions = fixture()
    q = data["img_q"]
    q.zero_()
    q[:, 0, :, 0] = 12
    q[:, 1, :, 0] = 11
    q[:, 4:6, :, 1] = 12
    q[:, 8:10, :, 2] = 12
    q[:, 2:4, :, 0] = .005
    feature = data["attention_output"]
    feature.zero_()
    feature[:, 0] = torch.tensor([100., 0., 0.])
    feature[:, 1] = torch.tensor([0., 100., 0.])
    feature[:, 2] = feature[:, 0]
    feature[:, 3] = feature[:, 1]
    feature[:, 4:6] = torch.tensor([70., 70., 20.])
    feature[:, 8:10] = torch.tensor([0., 0., 100.])
    banks, report = run(data, positions, top_k_ratio=.5)
    assert banks["multi_proto"]["masks"]["target"].flatten()[2:4].tolist() == [1., 1.]
    assert banks["centroid_control"]["masks"]["target"].flatten()[2:4].tolist() == [0., 0.]
    assert len(report["variants"]["multi_proto"]["prototype_indices"]["target"]) == 2


def test_full_joint_mask_cannot_turn_forbidden_text_links_into_context_seeds():
    data, positions = fixture()
    joint = torch.ones(15, 15, dtype=torch.bool)
    joint[3:, 0] = False
    data["attention_mask"] = joint
    banks, report = run(data, positions)
    assert "baseline" in banks  # exact legacy math remains explicitly separate
    assert "multi_proto" not in banks
    assert report["variants"]["multi_proto"]["error_code"] == "empty_seed_pool"


def test_full_joint_additive_negative_infinity_obeys_text_permissions():
    data, positions = fixture()
    joint = torch.zeros(1, 2, 15, 15)
    joint[:, :, 3:, 0] = -torch.inf
    data["attention_mask"] = joint
    banks, _ = run(data, positions)
    assert "multi_proto_bg" not in banks


def test_all_image_edges_masked_restarts_without_inventing_self_edges():
    data, positions = fixture()
    initial, _ = run(data, positions)
    data["attention_mask"] = torch.zeros(12, 12, dtype=torch.bool)
    banks, report = run(data, positions)
    for role in initial["multi_proto_bg"]["raw_maps"]:
        torch.testing.assert_close(banks["propagated_bg"]["raw_maps"][role], initial["multi_proto_bg"]["raw_maps"][role])
    info = report["variants"]["propagated_bg"]["propagation"]
    assert info["zero_rows_across_iterations"] == 12
    assert info["empty_row_policy"] == "restart_initial_scores"


@pytest.mark.parametrize("strength", [.5, 1.])
def test_propagation_normalization_cannot_amplify_an_unsupported_class(strength):
    mod = algorithms()
    scores = torch.tensor([[1., 0., 0.], [.49, .30, .21], [0., 1., 0.], [0., 0., 1.]])
    candidate = {"scores": scores, "roles": ("target", "protected", "background"),
                 "seeds": {"target": torch.tensor([True, False, False, False]),
                           "protected": torch.tensor([False, False, True, False]),
                           "background": torch.tensor([False, False, False, True])},
                 "growth_allowed": torch.zeros(4, 3, dtype=torch.bool)}
    q = torch.ones(1, 4, 1, 1)
    k = torch.tensor([100., 0., 0., 0.]).reshape(1, 4, 1, 1)
    changed, _ = mod._propagate(candidate, q, k, None, mod.validate_experimental_config(config(propagation_strength=strength)))
    assert torch.equal(changed["scores"][1], scores[1])


def test_propagation_uses_actual_image_keys_and_changes_only_supported_unknown():
    mod = algorithms()
    scores = torch.tensor([[1., 0., 0.], [.45, .30, .25], [0., 1., 0.], [0., 0., 1.]])
    seeds = {role: torch.arange(4) == index for role, index in (("target", 0), ("protected", 2), ("background", 3))}
    candidate = {"scores": scores, "roles": ("target", "protected", "background"), "seeds": seeds,
                 "growth_allowed": torch.ones(4, 3, dtype=torch.bool)}
    q = torch.ones(1, 4, 1, 1)
    target_keys = torch.tensor([100., 0., 0., 0.]).reshape(1, 4, 1, 1)
    protected_keys = torch.tensor([0., 0., 100., 0.]).reshape(1, 4, 1, 1)
    cfg = mod.validate_experimental_config(config())
    target, _ = mod._propagate(candidate, q, target_keys, None, cfg)
    protected, _ = mod._propagate(candidate, q, protected_keys, None, cfg)
    assert target["scores"][1, 0] > .55
    assert protected["scores"][1, 1] > .55
    assert torch.equal(target["scores"][[0, 2, 3]], scores[[0, 2, 3]])
    assert torch.equal(protected["scores"][[0, 2, 3]], scores[[0, 2, 3]])


def test_zero_iterations_is_identity_and_image_token_permutation_is_equivariant():
    banks, _ = run(propagation_iterations=0)
    for role in banks["multi_proto_bg"]["raw_maps"]:
        assert torch.equal(banks["multi_proto_bg"]["raw_maps"][role], banks["propagated_bg"]["raw_maps"][role])
    data, positions = fixture()
    order = torch.tensor([7, 1, 10, 4, 2, 9, 0, 11, 3, 6, 8, 5])
    for name in ("img_q", "img_k", "attention_output", "attention_input"):
        data[name] = data[name][:, order]
    permuted, _ = run(data, positions, propagation_iterations=0)
    for name in banks:
        for role in banks[name]["masks"]:
            assert torch.equal(banks[name]["masks"][role].flatten()[order], permuted[name]["masks"][role].flatten())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA retention check requires CUDA")
def test_cuda_observation_retains_only_cpu_tensors():
    data, positions = fixture()
    data = {key: value.cuda() for key, value in data.items()}
    acc = algorithms().ExperimentalMaskAccumulator(config(), (3, 4), positions)
    acc.observe(step=2, block=18, **data)
    banks, _ = acc.finalize()
    def check(value):
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu" and value.grad_fn is None
        elif isinstance(value, dict):
            for inner in value.values():
                check(inner)
        elif isinstance(value, (tuple, list)):
            for inner in value:
                check(inner)
    check(vars(acc))
    check(banks)


@pytest.mark.parametrize("name", ["temperature", "context_temperature", "feature_temperature"])
def test_temperature_domain_rejects_numerically_unsafe_subnormal_scales(name):
    with pytest.raises(ValueError):
        algorithms().validate_experimental_config(config(**{name: 1e-300}))


def test_nonfinite_image_affinity_fails_only_propagation_and_keeps_other_candidates():
    data, positions = fixture()
    data["img_k"].fill_(1e38)  # finite input whose dot products overflow float32
    banks, report = run(data, positions)
    assert "propagated_bg" not in banks
    assert report["variants"]["propagated_bg"]["error_code"] == "nonfinite_affinity"
    for name in set(algorithms().VARIANTS) - {"propagated_bg"}:
        assert name in banks and report["variants"][name]["status"] == "ok"
