import pytest
import torch

from slider_fuse.attention import similarity_maps, AttentionCollector


def test_similarity_maps_are_finite_deterministic_and_rng_neutral():
    torch.manual_seed(7)
    q = torch.randn(1, 8, 2, 4); k = torch.randn(1, 3, 2, 4); hidden = torch.randn(1, 8, 8)
    before = torch.random.get_rng_state()
    result = similarity_maps(q, k, hidden, {"target": (0,), "protected": (2,)}, .3, 4000.)
    assert set(result) == {"target", "protected"}
    assert all(x.shape == (1, 8) and torch.isfinite(x).all() for x in result.values())
    assert torch.equal(before, torch.random.get_rng_state())
    for name, value in result.items():
        torch.testing.assert_close(value.sum(-1), torch.ones(1))
        assert torch.equal(value, similarity_maps(q, k, hidden, {"target": (0,), "protected": (2,)}, .3, 4000.)[name])


def test_similarity_invalid_positions_are_not_clamped():
    with pytest.raises(ValueError, match="position"):
        similarity_maps(torch.ones(1, 8, 2, 4), torch.ones(1, 3, 2, 4), torch.ones(1, 8, 8),
                        {"target": (99,), "protected": (2,)}, .3, 4000.)


def test_collector_sigma_selection_is_not_callback_based():
    collector = AttentionCollector({"target": (0,), "protected": (1,)}, (2, 2), 2, 0, .3, 4000., torch.tensor([1., .75, 0.]))
    collector.begin_forward(torch.tensor([1.]), 2)
    assert not collector.collect_now
    collector.begin_forward(torch.tensor([.75]), 2)
    assert collector.collect_now
    assert collector.observation["sigma"] == .75
    with pytest.raises(ValueError, match="text"):
        collector.begin_forward(torch.tensor([.75]), 3)
