import pytest
import torch

from slider_fuse import anchored_masks as m


def masks():
    return tuple(torch.tensor([v], dtype=torch.float32) for v in
                 ([[1, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0]],
                  [[1, 0, 0], [0, 1, 0]], [[0, 1, 1], [0, 0, 0]]))


def test_union_factor_can_remove_pilot_false_positive_without_mutation():
    base, slider, pilot, protected = masks()
    before = [x.clone() for x in (base, slider, pilot, protected)]
    solo = m.compose_paired_masks(base, slider, pilot, protected, include_pilot=False)
    union = m.compose_paired_masks(base, slider, pilot, protected, include_pilot=True)
    assert solo['target'][0, 1, 1] == 0 and union['target'][0, 1, 1] == 1
    assert solo['removed_from_pilot'].sum() == 1
    assert solo['collision'].sum() == 1
    assert torch.equal(solo['target'] + solo['protected'] + solo['background'], torch.ones_like(base))
    assert not (solo['target'] * protected).any()
    assert all(torch.equal(a, b) for a, b in zip(before, (base, slider, pilot, protected)))


@pytest.mark.parametrize('bad', ['nonbinary', 'nan', 'grid', 'rank', 'empty_protected'])
def test_composition_rejects_invalid_inputs(bad):
    base, slider, pilot, protected = masks()
    if bad == 'nonbinary': base[0, 0, 0] = .5
    elif bad == 'nan': base[0, 0, 0] = float('nan')
    elif bad == 'grid': slider = slider[:, :, :2]
    elif bad == 'rank': base = base[0]
    else: protected.zero_()
    with pytest.raises(ValueError):
        m.compose_paired_masks(base, slider, pilot, protected, include_pilot=False)


def test_empty_after_protection_is_not_replaced():
    _, _, pilot, protected = masks()
    with pytest.raises(ValueError, match='empty'):
        m.compose_paired_masks(protected, protected, pilot, protected, include_pilot=False)


def test_release_changes_only_collision_and_preserves_inputs():
    base, slider, _, protected = masks()
    raw = (base + slider).clamp(max=1)
    release = slider.clone()
    result = m.apply_diagnostic_release(raw, protected, release)
    original = raw * (1 - protected)
    assert torch.equal(result['target'] - original, release)
    assert torch.equal(protected - result['protected'], release)
    assert torch.equal(result['target'] + result['protected'] + result['background'], torch.ones_like(raw))
    assert protected.sum() == 2


def test_release_rejects_noncollision_cells():
    base, slider, _, protected = masks()
    with pytest.raises(ValueError, match='collision'):
        m.apply_diagnostic_release((base + slider).clamp(max=1), protected, base)
