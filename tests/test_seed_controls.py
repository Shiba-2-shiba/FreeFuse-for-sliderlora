import copy

import pytest
import torch

from slider_fuse import anchored_masks as m
from slider_fuse.experimental_masks import _candidate_bank, _DEFAULTS


def fixture():
    torch.manual_seed(17)
    anchor = torch.tensor([[.8, .1, .1]] * 4 + [[.1, .8, .1]] * 4 + [[.1, .1, .8]] * 4)
    features = torch.randn(1, 12, 5)
    late = torch.randn(12, 3)
    config = dict(_DEFAULTS, top_k_ratio=1.)
    bootstrap = {'target': torch.tensor([[[1] * 4 + [0] * 8]], dtype=torch.float32),
                 'protected': torch.tensor([[[0] * 4 + [1] * 4 + [0] * 4]], dtype=torch.float32)}
    return features, late, anchor, bootstrap, config


def compute(features, late, anchor, pools, cfg, name):
    pool_kind = 'supported' if name in ('R3', 'R4') else 'full'
    bank = m.fit_role_prototypes(features, pools[pool_kind], cfg)
    locks = (pools['locks'] if name in ('R2', 'R4') else
             {r: torch.zeros(len(anchor), dtype=torch.bool) for r in ('target', 'protected', 'background')})
    if name == 'R0':
        locks = {r: torch.zeros(len(anchor), dtype=torch.bool).scatter_(0, p, True) for r, p in pools['full'].items()}
    return m.score_with_seed_controls(features, late, anchor, bank, locks, cfg)


def test_r0_matches_legacy_candidate_and_bank():
    f, late, anchor, bootstrap, cfg = fixture()
    pools = m.build_seed_pools(anchor, bootstrap, cfg)
    old = m.anchored_candidate(f, late, anchor, cfg)
    new = compute(f, late, anchor, pools, cfg, 'R0')
    for key in ('scores', 'growth_allowed'):
        assert torch.equal(new[key], old[key])
    for role in old['roles']:
        assert torch.equal(new['seeds'][role], old['seeds'][role])
    assert new['prototype_indices'] == old['prototype_indices']
    assert new['seed_counts'] == old['seed_counts']
    a, b = _candidate_bank(new, (1, 12), cfg), _candidate_bank(old, (1, 12), cfg)
    for role in old['roles']:
        assert torch.equal(a['masks'][role], b['masks'][role])
    for key in ('real_background_mask', 'uncertain_mask'):
        assert torch.equal(a[key], b[key])


def test_lock_only_change_preserves_prelock_scores():
    f, late, anchor, bootstrap, cfg = fixture()
    pools = m.build_seed_pools(anchor, bootstrap, cfg)
    results = [compute(f, late, anchor, pools, cfg, r) for r in ('R0', 'R1', 'R2')]
    assert all(torch.equal(results[0]['prelock_scores'], r['prelock_scores']) for r in results)
    assert sum(results[1]['lock_counts'].values()) == 0
    assert sum(results[2]['lock_counts'].values()) == 9
    assert torch.equal(results[1]['scores'], results[1]['prelock_scores'])
    for role in pools['locks']:
        assert torch.equal(results[2]['seeds'][role], pools['locks'][role])


def test_supported_prototype_change_preserves_locks():
    f, late, anchor, bootstrap, cfg = fixture()
    bootstrap['target'][0, 0, 2:] = 0
    pools = m.build_seed_pools(anchor, bootstrap, cfg)
    a, b = [compute(f, late, anchor, pools, cfg, r) for r in ('R2', 'R4')]
    assert pools['supported']['target'].tolist() == [0, 1]
    for role in pools['locks']:
        assert torch.equal(a['lock_masks'][role], b['lock_masks'][role])
    assert a['prototype_pool_counts']['target'] == 4
    assert b['prototype_pool_counts']['target'] == 2


def test_empty_supported_pool_does_not_block_full_pool_controls():
    f, late, anchor, bootstrap, cfg = fixture()
    bootstrap['target'].zero_()
    pools = m.build_seed_pools(anchor, bootstrap, cfg)
    assert pools['supported_failures'] == {'target': 'empty_supported_pool'}
    for name in ('R0', 'R1'):
        assert torch.isfinite(compute(f, late, anchor, pools, cfg, name)['scores']).all()
    with pytest.raises(ValueError, match='empty'):
        m.fit_role_prototypes(f, pools['supported'], cfg)


def test_seed_controls_do_not_mutate_inputs_or_prelock_evidence():
    f, late, anchor, bootstrap, cfg = fixture()
    before = (f.clone(), late.clone(), anchor.clone(), copy.deepcopy(bootstrap))
    pools = m.build_seed_pools(anchor, bootstrap, cfg)
    result = compute(f, late, anchor, pools, cfg, 'R0')
    saved = result['prelock_scores'].clone()
    result['scores'].zero_()
    assert torch.equal(result['prelock_scores'], saved)
    assert torch.equal(f, before[0]) and torch.equal(late, before[1]) and torch.equal(anchor, before[2])
    assert all(torch.equal(bootstrap[k], before[3][k]) for k in bootstrap)


@pytest.mark.parametrize('bad', ['nan', 'probabilities', 'grid', 'role_conflict'])
def test_seed_pool_rejects_invalid_input(bad):
    _, _, anchor, bootstrap, cfg = fixture()
    if bad == 'nan': anchor[0, 0] = float('nan')
    elif bad == 'probabilities': anchor[0] = 0
    elif bad == 'grid': bootstrap['target'] = bootstrap['target'][:, :, :2]
    else: bootstrap['protected'][0, 0, 0] = 1
    with pytest.raises(ValueError):
        m.build_seed_pools(anchor, bootstrap, cfg)


def test_equal_margin_uses_row_major_supported_order():
    _, _, anchor, bootstrap, cfg = fixture()
    pools = m.build_seed_pools(anchor, bootstrap, cfg)
    assert pools['full']['target'].tolist() == [0, 1, 2, 3]
    assert pools['locks']['target'].nonzero().flatten().tolist() == [0, 1, 2]


def test_scoring_rejects_overlapping_locks_and_nonfinite_features():
    f, late, anchor, bootstrap, cfg = fixture()
    pools = m.build_seed_pools(anchor, bootstrap, cfg)
    bank = m.fit_role_prototypes(f, pools['full'], cfg)
    locks = copy.deepcopy(pools['locks'])
    locks['protected'][0] = True
    with pytest.raises(ValueError, match='overlap'):
        m.score_with_seed_controls(f, late, anchor, bank, locks, cfg)
    f[0, 0, 0] = float('inf')
    with pytest.raises(ValueError, match='finite'):
        m.fit_role_prototypes(f, pools['full'], cfg)
