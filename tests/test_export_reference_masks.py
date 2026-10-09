import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from test_prediction_mix_schema3 import save


def exporter():
    p=Path(__file__).parents[1]/"tools/export_reference_masks.py"
    assert p.exists(), "Mask export tool is missing"
    spec=importlib.util.spec_from_file_location("export_masks_test",p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def test_exports_verified_reference_partition_and_never_overwrites(tmp_path):
    p=save(tmp_path)
    outputs=exporter().export_reference_masks(p,tmp_path/"masks")
    assert list(Image.open(outputs["target"]).tobytes())==[255,0,255,0]
    assert list(Image.open(outputs["protected"]).tobytes())==[0,255,0,255]
    before={k:Path(v).read_bytes() for k,v in outputs.items()}
    with pytest.raises(FileExistsError):exporter().export_reference_masks(p,tmp_path/"masks")
    assert all(Path(outputs[k]).read_bytes()==v for k,v in before.items())
