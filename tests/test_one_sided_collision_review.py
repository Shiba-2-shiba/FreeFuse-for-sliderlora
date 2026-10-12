import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import evaluate_paired_mask_attribution as m


def data():
    base={'target_face':[1,0,1,0,0,0,0], 'protected_face':[0,0,0,0,1,0,0], 'known_background':[0,1,0,0,0,1,0]}
    slider={'target_face':[1,1,0,0,0,0,0], 'protected_face':[0,0,1,0,0,0,0], 'known_background':[0,0,0,0,0,1,0]}
    refs={b:{'input_sha256':'input','sigma':.5,'branch':b,'step':3,'image_sha256':b+'rgb','prediction_sha256':b+'prediction'} for b in ('base','slider')}
    return base,slider,refs


def review(refs,cell=1,support='slider',opposite='known_background'):
    return {'release_review_schema':1,'references':copy.deepcopy(refs),'rows':[{'cell_index':cell,
        'support_branch':support,'target_part':'target_face','opposite_status':opposite,'reason':'RGB reviewed',
        'author_kind':'assistant','human_verified':False}]}


def test_v2_collision_partition_and_v1_compatibility():
    base,slider,_=data();old=m.classify_collisions([1]*7,[1]*7,base,slider)
    result=m.classify_collisions_v2([1]*7,[1]*7,base,slider)
    assert list(result)==['role_conflict','both_target','base_only_target','slider_only_target','protected','common_background','unresolved']
    assert all(sum(c[i] for c in result.values())==1 for i in range(7))
    assert result['role_conflict'][2] and result['slider_only_target'][1] and result['both_target'][0]
    assert m.classify_collisions([1]*7,[1]*7,base,slider)==old


def test_one_sided_collision_requires_explicit_opposite_review():
    base,slider,refs=data();classes=m.classify_collisions_v2([1]*7,[1]*7,base,slider)
    for state in ('unknown','occluded','unclear','protected'):
        with pytest.raises(ValueError):m.validate_release_review(classes,review(refs,opposite=state),refs)
    allowed=m.validate_release_review(classes,review(refs),refs)
    assert allowed==[False,True,False,False,False,False,False]
    with pytest.raises(ValueError):m.validate_release_review(classes,review(refs,cell=2,support='base'),refs)


@pytest.mark.parametrize('bad',['duplicate','outside','not_collision','rgb','branch','author','reason'])
def test_review_rejects_invalid_evidence(bad):
    base,slider,refs=data();classes=m.classify_collisions_v2([1,1,1,0,1,1,1],[1]*7,base,slider);r=review(refs)
    if bad=='duplicate':r['rows']*=2
    elif bad=='outside':r['rows'][0]['cell_index']=7
    elif bad=='not_collision':r['rows'][0]['cell_index']=3
    elif bad=='rgb':r['references']['base']['image_sha256']='changed'
    elif bad=='branch':r['rows'][0]['support_branch']='base'
    elif bad=='author':r['rows'][0]['human_verified']='false'
    else:r['rows'][0]['reason']=''
    with pytest.raises(ValueError):m.validate_release_review(classes,r,refs)


def test_base_only_target_and_unknown_empty_review():
    base,slider,refs=data();slider['protected_face'][2]=0
    classes=m.classify_collisions_v2([1]*7,[1]*7,base,slider)
    assert classes['base_only_target'][2]
    r=review(refs);r['rows']=[]
    assert not any(m.validate_release_review(classes,r,refs))
