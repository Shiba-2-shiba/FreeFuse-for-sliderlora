import copy

import pytest
import torch

from slider_fuse.masks import postprocess_masks


def bank_from(target, protected=None):
    target = target.float().reshape(1, *target.shape[-2:])
    if protected is None:
        protected = torch.zeros_like(target)
        protected[:, :, -1] = 1
    else:
        protected = protected.float().reshape_as(target)
    return {"grid": tuple(target.shape[-2:]), "masks": {"target": target, "protected": protected,
            "background": 1 - target - protected}, "mode": "auto"}


def ring_bank(hole_size=1):
    target = torch.zeros(9, 12)
    target[1:8, 1:9] = 1
    cells = [(y, x) for y in range(2, 6) for x in range(2, 6)][:hole_size]
    for y, x in cells:
        target[y, x] = 0
    return bank_from(target), cells


def assert_contract(before, after):
    original, actual = before["masks"], after["masks"]
    assert torch.equal(actual["protected"], original["protected"])
    assert torch.all(actual["target"] >= original["target"])
    added = actual["target"] - original["target"]
    assert torch.all(added <= original["background"])
    assert not (actual["target"] * actual["protected"]).any()
    assert torch.equal(sum(actual.values()), torch.ones_like(actual["target"]))
    assert torch.equal(after["added_target_mask"], added)
    assert added.sum() == after["postprocess_diagnostics"]["added_token_count"]
    for name in original:
        assert actual[name].shape == original[name].shape
        assert actual[name].dtype == original[name].dtype
        assert actual[name].device == original[name].device


@pytest.mark.parametrize("size", [1, 4, 8])
def test_small_enclosed_background_holes_are_filled(size):
    bank, cells = ring_bank(size)
    before = copy.deepcopy(bank)
    after = postprocess_masks(bank, max_hole_area=size)
    assert after["postprocess_diagnostics"]["filled_token_count"] == size
    assert after["postprocess_diagnostics"]["filled_hole_count"] == 1
    for y, x in cells:
        assert after["masks"]["target"][0, y, x] == 1
    for name in bank["masks"]:
        assert torch.equal(bank["masks"][name], before["masks"][name])
    assert_contract(before, after)


def test_large_hole_is_skipped():
    bank, _ = ring_bank(8)
    after = postprocess_masks(bank, max_hole_area=4)
    assert torch.equal(bank["masks"]["target"], after["masks"]["target"])
    assert after["postprocess_diagnostics"]["oversize_holes_skipped"] == 1


@pytest.mark.parametrize("protected_cells", [1, 4])
def test_protected_or_mixed_hole_is_wholly_preserved(protected_cells):
    bank, cells = ring_bank(4)
    for y, x in cells[:protected_cells]:
        bank["masks"]["protected"][0, y, x] = 1
        bank["masks"]["background"][0, y, x] = 0
    after = postprocess_masks(bank, max_hole_area=8)
    assert not after["added_target_mask"].any()
    assert after["postprocess_diagnostics"]["protected_holes_blocked"] == 1
    assert_contract(bank, after)


def test_edge_connected_space_is_not_a_hole():
    bank, _ = ring_bank(4)
    bank["masks"]["target"][0, :3, 2] = 0
    bank["masks"]["background"] = 1 - bank["masks"]["target"] - bank["masks"]["protected"]
    after = postprocess_masks(bank, max_hole_area=64)
    assert not after["added_target_mask"].any()
    assert after["postprocess_diagnostics"]["enclosed_holes_total"] == 0


def test_diagonal_background_connection_is_closed_under_four_connectivity():
    target = torch.zeros(5, 7)
    target[1:4, 1:4] = 1
    target[1, 1] = 0; target[2, 2] = 0
    bank = bank_from(target)
    after = postprocess_masks(bank, max_hole_area=1)
    assert after["masks"]["target"][0, 2, 2] == 1
    assert after["masks"]["target"][0, 1, 1] == 0


def test_noop_exact_values_independent_snapshots_and_raw_maps():
    bank, _ = ring_bank(4)
    raw = torch.randn(1, 108)
    bank["raw_maps"] = {"target": raw}
    after = postprocess_masks(bank)
    for name, mask in bank["masks"].items():
        assert torch.equal(mask, after["masks"][name])
        assert torch.equal(mask, after["original_masks"][name])
        assert after["original_masks"][name].data_ptr() != mask.data_ptr()
    assert after["raw_maps"]["target"] is raw
    assert after["postprocess_diagnostics"]["enabled"] is False
    assert after["postprocess_diagnostics"]["enclosed_holes_total"] is None
    assert_contract(bank, after)


def test_only_largest_component_is_dilated_with_protection():
    target = torch.zeros(7, 11); target[2:4, 2:4] = 1; target[5, 8] = 1
    protected = torch.zeros_like(target); protected[:, 4] = 1
    bank = bank_from(target, protected)
    after = postprocess_masks(bank, dilate_radius=1)
    assert after["masks"]["target"][0, 1, 1] == 1
    assert not after["masks"]["target"][0, 4:7, 7:10].sum() > 1
    assert after["postprocess_diagnostics"]["dilation_protected_blocked_count"] == 4
    assert_contract(bank, after)


def test_diagonal_foreground_is_one_component_and_ties_use_first_index():
    target = torch.zeros(7, 12); target[1, 1] = 1; target[2, 2] = 1
    target[4, 8] = 1; target[5, 8] = 1
    bank = bank_from(target)
    after = postprocess_masks(bank, dilate_radius=1)
    assert after["masks"]["target"][0, 0, 0] == 1
    assert after["masks"]["target"][0, 3:7, 7:10].sum() == 2


def test_corner_dilation_does_not_wrap_canvas():
    target = torch.zeros(4, 7); target[0, 0] = 1
    after = postprocess_masks(bank_from(target), dilate_radius=1)
    assert after["masks"]["target"].sum() == 4
    assert not after["masks"]["target"][0, -1].any()


def test_fill_and_dilate_account_for_new_tokens_once():
    bank, _ = ring_bank(4)
    after = postprocess_masks(bank, max_hole_area=8, dilate_radius=1)
    report = after["postprocess_diagnostics"]
    assert report["filled_token_count"] == 4
    assert report["added_token_count"] == report["filled_token_count"] + report["dilated_token_count"]
    assert_contract(bank, after)


@pytest.mark.parametrize("settings", [dict(max_hole_area=-1), dict(max_hole_area=65), dict(max_hole_area=True),
    dict(max_hole_area=1.5), dict(dilate_radius=2), dict(dilate_radius=-1), dict(dilate_radius=True)])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        postprocess_masks(ring_bank()[0], **settings)


@pytest.mark.parametrize("kind", ["nonbinary", "nan", "overlap", "unowned", "shape", "grid", "dtype", "empty"])
def test_invalid_banks_are_rejected(kind):
    bank, _ = ring_bank()
    if kind == "nonbinary": bank["masks"]["target"][0, 2, 2] = .5
    if kind == "nan": bank["masks"]["target"][0, 2, 2] = float("nan")
    if kind == "overlap": bank["masks"]["protected"][0, 1, 1] = 1
    if kind == "unowned": bank["masks"]["background"].zero_()
    if kind == "shape": bank["masks"]["background"] = torch.zeros(1, 3, 3)
    if kind == "grid": bank["grid"] = (3, 3)
    if kind == "dtype": bank["masks"]["background"] = bank["masks"]["background"].double()
    if kind == "empty":
        bank["masks"]["target"].zero_(); bank["masks"]["background"] = 1 - bank["masks"]["protected"]
    with pytest.raises(ValueError): postprocess_masks(bank)


def test_bfloat16_token_accounting_uses_integer_counts_not_rounded_sums():
    target=torch.zeros(70,90);target[2:67,2:87]=1
    protected=torch.zeros_like(target);protected[1,87]=1
    bank=bank_from(target,protected)
    bank["masks"]={name:mask.to(torch.bfloat16) for name,mask in bank["masks"].items()}
    after=postprocess_masks(bank,dilate_radius=1)
    exact=int(torch.count_nonzero(after["added_target_mask"]))
    assert exact==303
    assert after["postprocess_diagnostics"]["added_token_count"]==exact
    assert after["postprocess_diagnostics"]["dilated_token_count"]==exact
    assert after["masks"]["target"].dtype==torch.bfloat16
