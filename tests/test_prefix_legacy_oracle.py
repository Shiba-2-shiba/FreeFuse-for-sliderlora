"""Cross-check the new in-memory R0 against the actual historical scorer.

Only legacy artifact reads are substituted. All anchor, prototype, seed-lock,
scoring, mask, and composition calculations run through their real code paths.
No live runner or machine-specific helper is imported.
"""
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

from slider_fuse.anchored_masks import compose_paired_masks
from slider_fuse.experimental_masks import validate_experimental_config
from slider_fuse.paired_prefix import build_r0_masks


@pytest.mark.parametrize('seed', [7, 19, 31])
def test_r0_matches_actual_historical_scorer_on_nontrivial_4096_cell_grid(seed,tmp_path,monkeypatch):
    docs=Path(__file__).parents[1]/'docs'
    monkeypatch.syspath_prepend(str(docs))
    spec=importlib.util.spec_from_file_location('prefix_historical_scorer',docs/'build_paired_mask_candidates.py')
    historical=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    roles=('target','protected','background');n=4096
    cfg=validate_experimental_config({'collect_step':3,'collect_block':18,'selected_steps':[3],
                                      'selected_blocks':[18],'prototypes':3})
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        labels=torch.arange(n)%3
        base_logits=torch.nn.functional.one_hot(labels,3).float()*7
        features=torch.cat((torch.nn.functional.one_hot(labels,3).float(),torch.randn(n,5)*.06),-1)
        target=(labels==0).float().reshape(1,64,64)
        protected=(labels==1).float().reshape(1,64,64)
        target.flatten()[2]=1  # Deliberate bootstrap-only target false positive.
        bootstrap={'target':target,'protected':protected,'background':1-target-protected}
        branch_taps={};store={}
        for bi,branch in enumerate(('base','slider')):
            taps={}
            for block in (8,10,18):
                logits=base_logits+torch.randn(n,3)*(.2 if block!=18 else .7)+bi*.04
                if block==10 and bi==1:
                    logits[30:42]=logits[30:42].roll(1,-1)  # Non-unanimous early anchors.
                raw=features+torch.randn(n,8)*bi*.025
                maps={}
                for ri,role in enumerate(roles):
                    maps[f'step_3.block_{block}.{role}.token_scores']=torch.stack((logits[:,ri]+.1,logits[:,ri]-.1))
                    maps[f'step_3.block_{block}.{role}.token_allowed_fraction']=torch.ones(2,n)
                if block==18:
                    maps['capture.attention_output']=raw
                config={**cfg,'collect_block':block,'selected_blocks':[block]}
                taps[f'step_3.block_{block}']={'algorithm':{'grid':[64,64],'config':config},'evidence_maps':maps}
                store[branch,block]=({'report':{'algorithm':{'config':config}}},
                                     {'evidence.'+key:value for key,value in maps.items()})
            branch_taps[branch]=taps
    probe={'report':{'grid':[64,64]},'branches':branch_taps}
    new,evidence=build_r0_masks(probe,{'grid':(64,64),'masks':bootstrap})
    lookup={};outer={'taps':{}}
    for (branch,block),value in store.items():
        path=tmp_path/f'{branch}_{block}.json';path.write_text('{}')
        outer['taps'][f'{branch}.step_3.block_{block}']={
            'manifest':path.name,'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        lookup[str(path)]=value
    monkeypatch.setattr(historical,'read',lambda path:outer)
    monkeypatch.setattr(historical,'safe_tap',lambda root,item:root/item['manifest'])
    monkeypatch.setattr(historical,'verify_suite',lambda path:lookup[str(path)])
    old=historical.build_seed_control_banks(tmp_path,bootstrap)
    for branch in ('base','slider'):
        result=old['variants']['R0'][branch]
        assert result['status']=='ok'
        assert torch.equal(result['candidate']['scores'],evidence[branch+'.scores'])
        assert torch.equal(result['bank']['masks']['target'],evidence[branch+'.target_mask'])
        for role in roles:
            assert torch.equal(result['candidate']['seeds'][role],evidence[branch+'.seeds.'+role])
    assert torch.equal(old['anchor'],evidence['anchor'])
    assert torch.equal(evidence['anchor'][30:42],torch.full((12,3),1/3))
    expected=compose_paired_masks(old['variants']['R0']['base']['bank']['masks']['target'],
        old['variants']['R0']['slider']['bank']['masks']['target'],bootstrap['target'],
        bootstrap['protected'],include_pilot=False)
    for role in roles:
        assert torch.equal(expected[role],new['masks'][role])
    assert torch.equal(new['masks']['protected'],protected)
    assert new['masks']['target'].flatten()[2]==0
