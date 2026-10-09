import importlib

import pytest
import torch

from slider_fuse import masks


def selection(bank,radius):
    assert hasattr(masks,"prediction_selection_mask"),"Selection-only expansion is missing"
    return masks.prediction_selection_mask(bank,radius)


def bank():
    target=torch.zeros(1,6,8);target[0,2,1:3]=1;target[0,5,7]=1
    protected=torch.zeros_like(target);protected[:,:,4]=1
    return {"grid":(6,8),"masks":{"target":target,"protected":protected,"background":1-target-protected}}


def test_expansion_keeps_reference_partition_and_does_not_cross_protection():
    b=bank();before={k:v.clone() for k,v in b["masks"].items()}
    result=selection(b,3)
    assert all(torch.equal(before[k],v) for k,v in b["masks"].items())
    assert bool((result>=before["target"]).all())
    assert not (result*before["protected"]).any()
    assert result[0,1,3]==1
    # The full-height protected barrier blocks propagation into its far background.
    assert result[0,:5,5:].sum()==0
    # Small isolated target is retained but is not expanded.
    assert result[0,5,7]==1 and result[0,4,7]==0


def test_zero_radius_is_exact_and_full_half_mask_has_no_background_to_expand():
    b=bank();assert torch.equal(selection(b,0),b["masks"]["target"])
    target=torch.zeros(1,4,4);target[:,:,:2]=1
    b={"grid":(4,4),"masks":{"target":target,"protected":1-target,"background":torch.zeros_like(target)}}
    assert torch.equal(selection(b,16),target)


@pytest.mark.parametrize("radius",[-1,17,True,2.5])
def test_invalid_selection_radius_is_rejected(radius):
    assert hasattr(masks,"prediction_selection_mask"),"Selection-only expansion is missing"
    with pytest.raises(ValueError):masks.prediction_selection_mask(bank(),radius)


def test_runtime_version_matches_package_metadata():
    assert importlib.util.find_spec("slider_fuse.version") is not None,"Shared version constant is missing"
    from slider_fuse.version import __version__
    import re
    from pathlib import Path
    package=(Path(__file__).parents[1]/"pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"',package,re.M).group(1)==__version__
