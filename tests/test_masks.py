import pytest
import torch

from slider_fuse.masks import generate_masks, manual_masks, patch_grid


def test_auto_masks_cover_canvas_and_keep_unequal_people_sizes():
    maps = {"target": torch.tensor([[0., 1., 1., 1., 1., 1., 0., 0.]]),
            "protected": torch.tensor([[0., 0., 0., 0., 0., 0., 1., 0.]])}
    bank = generate_masks(maps, (2, 4))
    masks = bank["masks"]
    assert masks["target"].sum() == 5
    assert masks["protected"].sum() == 1
    assert masks["background"].sum() == 2
    assert torch.equal(sum(masks.values()), torch.ones(1, 2, 4))
    assert not (masks["target"] * masks["protected"]).any()


def test_ties_go_to_background():
    maps = {"target": torch.tensor([[0., 1., .8, 0.]]),
            "protected": torch.tensor([[0., 0., .8, 1.]])}
    assert generate_masks(maps, (1, 4))["masks"]["background"][0, 0, 2] == 1


@pytest.mark.parametrize("bad", [torch.zeros(1, 4), torch.tensor([[0., 1., float('nan'), 0.]]), torch.ones(1, 3)])
def test_invalid_auto_map_stops_instead_of_full_canvas(bad):
    with pytest.raises(ValueError):
        generate_masks({"target": bad, "protected": torch.tensor([[0., 0., 0., 1.]])}, (2, 2))


def test_manual_masks_and_canvas_mismatch():
    target = torch.zeros(1, 8, 12); target[:, :, :4] = 1
    protected = torch.zeros_like(target); protected[:, :, 8:] = 1
    masks = manual_masks(target, protected, (2, 3))["masks"]
    assert masks["target"].sum() == 2
    with pytest.raises(ValueError, match="same canvas"):
        manual_masks(target, torch.zeros(1, 4, 4), (2, 3))


@pytest.mark.parametrize("bad", [torch.ones(1, 4, 4) * 2, torch.ones(2, 4, 4), torch.full((1, 4, 4), float('nan'))])
def test_manual_invalid_ranges_and_batches(bad):
    with pytest.raises(ValueError):
        manual_masks(bad, torch.zeros_like(bad), (2, 2))


def test_manual_overlap_and_empty_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        manual_masks(torch.ones(1, 4, 4), torch.ones(1, 4, 4), (2, 2))
    with pytest.raises(ValueError, match="empty"):
        manual_masks(torch.zeros(1, 4, 4), torch.ones(1, 4, 4), (2, 2))


def test_patch_grid_4d_5d_and_odd_non_square():
    assert patch_grid(torch.zeros(1, 16, 9, 7), 2) == (5, 4)
    assert patch_grid(torch.zeros(1, 16, 1, 9, 7), 2) == (5, 4)
    with pytest.raises(ValueError, match="single still"):
        patch_grid(torch.zeros(1, 16, 2, 8, 8), 2)
