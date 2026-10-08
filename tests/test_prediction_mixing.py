import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_full_and_partial_preserve_values_exactly(dtype):
    from slider_fuse.prediction_mixing import mix_predictions
    base = torch.tensor([[[[1000., -.001], [3., -200.]]]], dtype=dtype)
    slider = torch.tensor([[[[.001, 200.], [-300., 1.]]]], dtype=dtype)
    before = base.clone()
    mask = torch.tensor([[[[True, False], [True, False]]]])
    assert torch.equal(mix_predictions(base, slider, torch.zeros_like(mask)), base)
    assert torch.equal(mix_predictions(base, slider, torch.ones_like(mask)), slider)
    assert torch.equal(mix_predictions(base, slider, mask),
                       torch.tensor([[[[.001, -.001], [-300., -200.]]]], dtype=dtype))
    assert torch.equal(base, before)


@pytest.mark.parametrize("shape", [(1, 2, 5, 7), (1, 2, 1, 5, 7)])
def test_grid_expansion_crops_odd_non_square(shape):
    from slider_fuse.prediction_mixing import prediction_mask
    token = torch.tensor([[[1., 0., 1., 0.], [0., 1., 0., 1.], [1., 1., 0., 0.]]])
    mask = prediction_mask(token, torch.zeros(shape), patch=2)
    expected = torch.tensor([[1, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 1, 1, 0],
                             [0, 0, 1, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 1],
                             [1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    assert torch.equal(mask.reshape(5, 7), expected)
    assert mask.shape == ((1, 1, 5, 7) if len(shape) == 4 else (1, 1, 1, 5, 7))


@pytest.mark.parametrize("value", [.5, float("nan")])
def test_non_binary_mask_rejected(value):
    from slider_fuse.prediction_mixing import prediction_mask
    with pytest.raises(ValueError):
        prediction_mask(torch.full((1, 2, 2), value), torch.zeros(1, 2, 4, 4), patch=2)


def test_incompatible_predictions_and_layouts_rejected():
    from slider_fuse.prediction_mixing import prediction_mask, mix_predictions
    base = torch.zeros(1, 2, 4, 4)
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    for slider in (torch.zeros(1, 3, 4, 4), base.double(), base + float("nan")):
        with pytest.raises(ValueError):
            mix_predictions(base, slider, mask)
    for shape in ((2, 2, 4, 4), (1, 2, 2, 4, 4)):
        with pytest.raises(ValueError):
            prediction_mask(torch.ones(1, 2, 2), torch.zeros(shape), patch=2)
