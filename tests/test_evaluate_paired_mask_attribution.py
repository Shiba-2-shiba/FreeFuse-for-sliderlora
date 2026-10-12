import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

MODULE = Path(__file__).resolve().parents[1] / 'tools' / 'evaluate_paired_mask_attribution.py'


def evaluator():
    assert MODULE.is_file(), 'paired evaluator not implemented'
    spec = importlib.util.spec_from_file_location('paired_evaluator_test', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collision_classes_partition_and_do_not_release_disagreements():
    m = evaluator()
    base = {'target_face': [1, 0, 1, 0, 0], 'protected_face': [0, 1, 0, 0, 0],
            'known_background': [0, 0, 0, 1, 0]}
    slider = {'target_face': [1, 0, 0, 0, 0], 'protected_face': [0, 1, 1, 0, 0],
              'known_background': [0, 0, 0, 1, 0]}
    classes = m.classify_collisions([1]*5, [1]*5, base, slider)
    assert classes['confirmed_target'] == [True, False, False, False, False]
    assert classes['confirmed_protected'] == [False, True, False, False, False]
    assert classes['role_disagreement'] == [False, False, True, False, False]
    assert classes['common_background'] == [False, False, False, True, False]
    assert classes['unresolved'] == [False, False, False, False, True]
    assert [sum(x) for x in zip(*classes.values())] == [1]*5


def test_partial_and_empty_annotations_are_not_full_body_precision():
    m = evaluator()
    result = m.region_metrics([1, 0, 0, 1], {'target_face': [1, 1, 0, 0], 'target_foot_left': [0]*4})
    assert result['target_face']['recall'] == .5
    assert result['target_face']['missing_cells'] == 1
    assert result['target_foot_left']['recall'] is None
    assert result['target_foot_left']['reason'] == 'empty_annotation'
    assert result['target_full']['precision'] is None
    assert result['target_full']['reason'] == 'annotation_not_supplied'


@pytest.mark.parametrize('regions', [
    {'target_face': [1, 0], 'protected_face': [1, 0]},
    {'target_face': [1, 0], 'target_full': [0, 1]},
    {'target_face': [1, .5]},
    {'unknown_role': [0, 0]},
    {'target_face': [1]},
])
def test_annotations_are_strict_and_never_repaired(regions):
    with pytest.raises(ValueError): evaluator().validate_regions(regions, 2)


def test_union_difference_accounts_for_unlabelled_cells():
    m = evaluator()
    result = m.attribute_difference([1, 0, 1, 0], [0, 1, 0, 0],
                                    {'target_face': [1, 0, 0, 0]}, {'target_face': [1, 0, 0, 0]})
    assert result['added']['total_cells'] == 2
    assert result['added']['by_region']['target_face'] == 1
    assert result['added']['by_region']['unlabelled'] == 1
    assert result['removed']['total_cells'] == 1


def test_paired_background_requires_both_views_not_one_view_background():
    m=evaluator()
    base={'target_face':[1,0,0],'known_background':[0,1,1]}
    slider={'target_face':[0,1,0],'known_background':[1,0,1]}
    metrics=m.paired_support_metrics([1,1,1],base,slider)
    assert metrics['common_background_intrusion_cells']==1
    assert metrics['common_background_annotation_cells']==1
    assert metrics['unlabelled_cells']==0


def test_independently_supplied_full_target_is_not_counted_as_unlabelled_delta():
    m=evaluator()
    base={'target_full':[1,1,0],'target_face':[1,0,0]}
    result=m.attribute_difference([1,1,0],[0,0,0],base,base)
    assert result['added']['by_region']['target_full_other']==1
    assert result['added']['by_region']['unlabelled']==0


def test_import_does_not_load_torch_or_comfy():
    assert MODULE.is_file()
    script = "import runpy,sys;runpy.run_path(sys.argv[1],run_name='import_test');assert 'torch' not in sys.modules;assert not any(x.startswith('comfy') for x in sys.modules)"
    result = subprocess.run([sys.executable, '-B', '-c', script, str(MODULE)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_mismatched_reference_identity_is_rejected():
    m = evaluator()
    with pytest.raises(ValueError, match='reference'):
        m.validate_reference({'input_sha256': 'a', 'sigma': 1, 'branch': 'base', 'step': 3},
                             {'input_sha256': 'b', 'sigma': 1, 'branch': 'base', 'step': 3})
