"""Bounded experimental mask pilot; its partial latent is never a public output.

The ordinary sampler remains unchanged unless its private pilot option is used.
This module deliberately owns no live-runtime or author-specific script imports.
"""
from __future__ import annotations

import copy
import math
import json

import torch

from .anchored_masks import anchored_candidate, compose_paired_masks
from .experimental_masks import _candidate_bank
from .masks import manual_masks, postprocess_masks, prediction_selection_mask, _validate_partition
from .sampling import tensor_hash

ROLES = ('target', 'protected', 'background')
PILOT_BLOCKS = (8, 10, 18, 23, 26)  # Identical instrumentation to the existing reference.


def validate_pilot_options(*, steps=8, cfg=1., strength=4., mix_scope='target_mask',
                           mask_mode='manual', diagnostic_level='audit',
                           fill_holes_max_area=0, mask_dilate_radius=0,
                           selection_dilate_radius=0, pilot_mode='prefix'):
    if pilot_mode not in ('prefix', 'full_reference'):
        raise ValueError('Unknown pilot_mode')
    if type(steps) is not int or steps != 8 or isinstance(cfg, bool) or cfg != 1.:
        raise ValueError('Paired pilot requires the original eight-step CFG1 schedule')
    if isinstance(strength, bool) or not isinstance(strength, (int,float)) or not math.isfinite(strength) or strength == 0:
        raise ValueError('Paired pilot requires finite nonzero Slider strength')
    if mix_scope != 'target_mask' or mask_mode != 'manual' or diagnostic_level != 'audit':
        raise ValueError('Paired pilot requires target_mask manual audit mode')
    if any(type(v) is not int or v != 0 for v in (fill_holes_max_area,mask_dilate_radius,selection_dilate_radius)):
        raise ValueError('Paired pilot requires zero postprocess and selection radii')


class PrefixEulerSampler:
    """Slice only at the sampler boundary, after CFGGuider records full sigmas.

    The delegate retains native noise scaling, max_denoise and Euler behavior.
    Its terminal inverse scaling is allowed to run, but the caller discards the
    resulting partial latent. It must never be resumed or decoded as final RGB.
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if (not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) != 9
                or not torch.isfinite(sigmas).all() or sigmas[-1] != 0
                or not torch.all(sigmas[:-1] > sigmas[1:]) or sigmas[3] <= 0):
            raise ValueError('Prefix pilot requires a finite descending eight-step schedule ending at zero')
        return self.sampler.sample(model_wrap, sigmas[:4].clone(), extra_args, callback, noise,
                                   latent_image, denoise_mask, disable_pbar)


def _tap_logits(tap, block):
    maps = tap['evidence_maps']
    prefix = f'step_3.block_{block}.'
    if not all(torch.all(maps[prefix+r+'.token_allowed_fraction'] == 1) for r in ROLES):
        raise ValueError('Masked text links need a separate anchor policy')
    logits = torch.stack([maps[prefix+r+'.token_scores'].mean(0) for r in ROLES], -1)
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite paired anchor logits')
    return logits


def build_r0_masks(probe, bootstrap_bank):
    """The existing A-R0 oracle: four-view anchors and paired union minus B.P.

    No propagation, supported-pool variant, human release, historical D mask, or
    union with the bootstrap target is introduced here.
    """
    if set(probe['branches']) != {'base', 'slider'}:
        raise ValueError('R0 requires both paired branches')
    _validate_partition(bootstrap_bank)
    grid = tuple(probe['report']['grid'])
    if grid != tuple(bootstrap_bank['grid']):
        raise ValueError('Pilot and bootstrap mask grids differ')
    contexts = []
    for branch in ('base','slider'):
        for block in (8,10):
            contexts.append(_tap_logits(probe['branches'][branch][f'step_3.block_{block}'],block).softmax(-1))
    views = torch.stack(contexts)
    anchor = views.mean(0)
    anchor[(views.argmax(-1) != anchor.argmax(-1)[None]).any(0)] = 1/3
    evidence = {'anchor': anchor.detach().clone()}
    branches = {}
    for branch in ('base','slider'):
        tap = probe['branches'][branch]['step_3.block_18']
        config = copy.deepcopy(tap['algorithm']['config'])
        features = tap['evidence_maps']['capture.attention_output'][None]
        if features.shape[1] != grid[0]*grid[1] or not torch.isfinite(features).all():
            raise ValueError('Raw late features do not match a finite pilot grid')
        candidate = anchored_candidate(features, _tap_logits(tap,18), anchor, config)
        branches[branch] = _candidate_bank(candidate, grid, config)
        evidence[branch+'.scores'] = candidate['scores'].detach().clone()
        evidence[branch+'.target_mask'] = branches[branch]['masks']['target'].detach().clone()
        for role in ROLES:
            evidence[branch+'.seeds.'+role] = candidate['seeds'][role].detach().clone()
    values = compose_paired_masks(branches['base']['masks']['target'], branches['slider']['masks']['target'],
        bootstrap_bank['masks']['target'], bootstrap_bank['masks']['protected'], include_pilot=False)
    bank = postprocess_masks(manual_masks(values['target'],values['protected'],grid))
    bank['selection_target_mask'] = prediction_selection_mask(bank,0)
    bank['selection_added_mask'] = bank['selection_target_mask'] - bank['masks']['target']
    bank['effective_image_mask'] = bank['selection_target_mask'].clone()
    evidence.update({'composition.'+k:v.detach().clone() for k,v in values.items()})
    return bank, evidence


def sample_paired_pilot(model, positive, negative, prompt_info, subjects, latent, lora_path, *,
                        background_phrase, pilot_mode='prefix', source_identity_json='{}', **options):
    """Return only an R0 mask bank and an owned validation probe, never LATENT."""
    def unique_pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate source identity key')
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError('Nonfinite source identity')
    if not isinstance(source_identity_json, str):
        raise ValueError('source_identity_json must be a JSON object string')
    source_identity = json.loads(source_identity_json, object_pairs_hook=unique_pairs, parse_constant=reject_constant)
    if not isinstance(source_identity, dict):
        raise ValueError('source_identity_json must contain an object')
    from .conditioning import resolve_phrase
    from .native_pair import sample_krea2_prediction_mix
    from .paired_observation import PairedObserver
    controls = {key:options[key] for key in ('steps','cfg','strength','mix_scope','mask_mode','diagnostic_level',
        'fill_holes_max_area','mask_dilate_radius','selection_dilate_radius') if key in options}
    validate_pilot_options(pilot_mode=pilot_mode, **controls)
    positions = {s.id:list(s.positions) for s in subjects}
    positions['background'] = list(resolve_phrase(prompt_info,background_phrase,0))
    observer = PairedObserver(positions,blocks=PILOT_BLOCKS,keep_raw=True)
    discarded, bootstrap_bank, payload = sample_krea2_prediction_mix(model,positive,negative,prompt_info,
        subjects,latent,lora_path,_observer=observer,_pilot_steps=3 if pilot_mode=='prefix' else 8,**options)
    if discarded is not None:
        raise RuntimeError('Experimental pilot unexpectedly exposed a latent')
    count = 3 if pilot_mode == 'prefix' else 8
    if payload.report['branch_nfe'] != {'base':count,'slider':count}:
        raise RuntimeError('Incomplete paired pilot branch counts')
    extra = payload.extra_tensors
    predictions = {branch:extra[name] for branch,name in (('base','trace_base_predictions'),
                   ('slider','trace_slider_predictions'),('mix','trace_predictions'))}
    probe = observer.result(payload.report,predictions)
    if set(probe['branches']) != {'base','slider'} or len(probe['frames']) != 6:
        raise RuntimeError('Incomplete paired pilot observation')
    validation_keys = ('initial_noise','initial_latent','full_sigmas','used_sigmas','trace_inputs','trace_sigmas',
                       'trace_base_predictions','trace_slider_predictions','trace_predictions')
    probe['validation_tensors'] = {k:extra[k].detach().cpu().clone() for k in validation_keys}
    probe['bootstrap_masks'] = {r:bootstrap_bank['masks'][r].detach().cpu().clone() for r in ROLES}
    probe['report']['source_identity'] = source_identity
    probe['report']['source_identity_verification'] = 'supplied_frozen_provenance_not_runtime_verified'
    probe['report']['r0_cpu_runtime'] = {'torch_num_threads':torch.get_num_threads(),
                                        'torch_num_interop_threads':torch.get_num_interop_threads()}
    bank, evidence = build_r0_masks(probe,bootstrap_bank)
    probe['r0_tensors'] = evidence
    probe['report']['r0_partition_sha256'] = {r:tensor_hash(bank['masks'][r]) for r in ROLES}
    probe['report']['observer_blocks'] = list(PILOT_BLOCKS)
    probe['report']['observer_steps'] = [1,2,3]
    return bank, probe
