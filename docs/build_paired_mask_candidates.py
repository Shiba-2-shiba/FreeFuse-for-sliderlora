"""Offline seed / shape ablations on the same observed Mix inputs."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from safetensors.torch import save_file, load_file

from audit_attention_ablation_live import read as _read,verify_suite
from audit_attention_step_block_sweep import safe_tap
from slider_fuse.anchored_masks import anchored_candidate, compose_paired_masks, apply_diagnostic_release
from slider_fuse.experimental_masks import _candidate,_candidate_bank,_propagate,_CandidateFailure
from slider_fuse.paired_observation import checked_diagnostic

ROLES=("target","protected","background")


def build_seed_control_banks(probe_root, bootstrap_masks):
    """R0–R4 on one retained pilot; no historical target union or annotation input."""
    from slider_fuse.anchored_masks import build_seed_pools, fit_role_prototypes, score_with_seed_controls
    probe_root = Path(probe_root)
    outer = read(probe_root/'manifest.json')
    cache = {}
    def tap(branch, block):
        item = outer['taps'][f'{branch}.step_3.block_{block}']
        path = safe_tap(probe_root, item)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['manifest_sha256']:
            raise ValueError('Changed seed source tap')
        key = branch, block
        if key not in cache:
            cache[key] = verify_suite(path)
        return cache[key]
    contexts = [logits(tap(branch, block)[1],3,block).softmax(-1)
                for branch in ('base','slider') for block in (8,10)]
    views = torch.stack(contexts)
    anchor = views.mean(0)
    anchor[(views.argmax(-1) != anchor.argmax(-1)[None]).any(0)] = 1/3
    result = {'branches':{}, 'anchor':anchor, 'variants':{r:{} for r in ('R0','R1','R2','R3','R4')}}
    for branch in ('base','slider'):
        manifest, tensors = tap(branch,18)
        config = copy.deepcopy(manifest['report']['algorithm']['config'])
        features = tensors['evidence.capture.attention_output'][None]
        late = logits(tensors,3,18)
        pools = build_seed_pools(anchor,bootstrap_masks,config)
        result['branches'][branch] = {'pools':pools,'config':config}
        for variant in result['variants']:
            if variant in ('R2','R3','R4') and pools['supported_failures']:
                result['variants'][variant][branch] = {'status':'failed','reason':pools['supported_failures']}
                continue
            try:
                kind = 'supported' if variant in ('R3','R4') else 'full'
                prototypes = fit_role_prototypes(features,pools[kind],config)
                locks = {r:torch.zeros(len(anchor),dtype=torch.bool) for r in ROLES}
                if variant=='R0':
                    for role in ROLES:locks[role][pools['full'][role]]=True
                elif variant in ('R2','R4'):
                    locks = pools['locks']
                candidate = score_with_seed_controls(features,late,anchor,prototypes,locks,config)
                bank = _candidate_bank(candidate,(64,64),config)
                if variant=='R0':
                    legacy = anchored_candidate(features,late,anchor,config)
                    old_bank = _candidate_bank(legacy,(64,64),config)
                    if (not all(torch.equal(candidate[k],legacy[k]) for k in ('scores','growth_allowed'))
                            or candidate['prototype_indices']!=legacy['prototype_indices']
                            or candidate['seed_counts']!=legacy['seed_counts']
                            or not all(torch.equal(candidate['seeds'][r],legacy['seeds'][r]) for r in ROLES)
                            or not all(torch.equal(bank['masks'][r],old_bank['masks'][r]) for r in ROLES)
                            or not all(torch.equal(bank[k],old_bank[k]) for k in ('uncertain_mask','real_background_mask'))):
                        raise RuntimeError('R0 differs from the independent legacy oracle')
                result['variants'][variant][branch] = {'status':'ok','candidate':candidate,'bank':bank,'prototypes':prototypes}
            except _CandidateFailure as exc:
                result['variants'][variant][branch] = {'status':'failed','reason':exc.code}
    return result


def read(path):
    return _read(Path(path))


def logits(tensors,step,block):
    prefix=f"evidence.step_{step}.block_{block}."
    if not all(torch.all(tensors[prefix+r+".token_allowed_fraction"]==1) for r in ROLES):raise ValueError("Masked text links need a separate anchor policy")
    return torch.stack([tensors[prefix+r+".token_scores"].mean(0) for r in ROLES],-1)


def legacy_main():
    torch.set_num_threads(4);root=Path(sys.argv[1]).resolve()
    summary=read(root/"probe.summary.json")
    for case in summary:
        cid=case["case"];folder=root/cid;probe=Path(case["probes"]["mix"]["local"]);outer=read(probe/"manifest.json")
        cache={}
        def tap(branch,block):
            key=f"{branch}.step_3.block_{block}"
            if key not in cache:cache[key]=verify_suite(safe_tap(probe,outer["taps"][key]))
            return cache[key]
        contexts=[]
        for branch in ("base","slider"):
            for block in (8,10):
                manifest,tensors=tap(branch,block);contexts.append(logits(tensors,3,block).softmax(-1))
        views=torch.stack(contexts);anchor=views.mean(0)
        consistent=(views.argmax(-1)==anchor.argmax(-1)[None]).all(0)
        anchor[~consistent]=1/3
        pilot=next(r for r in case["diagnostics"] if r["label"]=="probe_mix")
        pilot_report,pilot_tensors=checked_diagnostic(pilot["local_report"])
        old_target=pilot_tensors["reference_target_mask"];protected=pilot_tensors["reference_protected_mask"]
        candidate_plan=[];ablations=[]
        for block in (18,23):
            banks={};failures=[]
            for branch in ("base","slider"):
                manifest,tensors=tap(branch,block);cfg=copy.deepcopy(manifest["report"]["algorithm"]["config"])
                f=tensors["evidence.capture.attention_output"][None];q=tensors["evidence.capture.img_q"][None];k=tensors["evidence.capture.img_k"][None]
                late=logits(tensors,3,block);cfg.update(propagation_heads=list(range(q.shape[2])),propagation_strength=.5)
                try:
                    native=_candidate(f,late,ROLES,cfg,3)
                    anchored=anchored_candidate(f,late,anchor,cfg)
                    variants={"baseline":native,"seed_only":anchored}
                    variants["shape_only"],shape_info=_propagate(native,q,k,None,cfg,lock_confident=False)
                    variants["combined"],combined_info=_propagate(anchored,q,k,None,cfg,lock_confident=False)
                    for name,candidate in variants.items():
                        bank=_candidate_bank(candidate,(64,64),cfg);dest=folder/"ablations"/f"{branch}_s3_b{block}_{name}";dest.mkdir(parents=True,exist_ok=True)
                        for role in ROLES:Image.fromarray((bank["masks"][role][0].numpy()*255).astype("uint8")).save(dest/(role+".png"))
                        stats={"branch":branch,"step":3,"block":block,"variant":name,"diagnostics":bank["diagnostics"],
                               "seed_counts":candidate["seed_counts"],"anchor_consistency_cells":int(consistent.sum()),
                               "source_manifest":str((probe/outer["taps"][f"{branch}.step_3.block_{block}"]["manifest"]).resolve()),
                               "lock_confident":False if name in ("shape_only","combined") else None,
                               "graph_heads":cfg["propagation_heads"] if name in ("shape_only","combined") else []}
                        (dest/"statistics.json").write_text(json.dumps(stats,indent=2),encoding="utf-8");ablations.append(stats)
                        if name=="combined":banks[branch]=bank
                except _CandidateFailure as error:failures.append({"branch":branch,"reason":error.code,"error":str(error)})
            label=f"anchored_s3_b{block}"
            if failures or set(banks)!={"base","slider"}:
                candidate_plan.append({"label":label,"status":"failed","failures":failures});continue
            target=((old_target+banks["base"]["masks"]["target"]+banks["slider"]["masks"]["target"])>0).float()
            collision=int((target*protected).sum());target=target*(1-protected)
            if not target.any():candidate_plan.append({"label":label,"status":"failed","reason":"empty_after_fixed_protection"});continue
            masks={"target":target,"protected":protected,"background":1-target-protected}
            dest=folder/"candidates"/label;dest.mkdir(parents=True,exist_ok=True)
            save_file(masks,str(dest/"masks.safetensors"))
            for role,value in masks.items():Image.fromarray((value[0].numpy()*255).astype("uint8")).save(dest/(role+".png"))
            record={"anchored_mask_schema":1,"complete":True,"tensors":"masks.safetensors","tensors_sha256":hashlib.sha256((dest/"masks.safetensors").read_bytes()).hexdigest(),
                    "label":label,"grid":[64,64],"target_cells":int(target.sum()),"protected_cells":int(protected.sum()),
                    "added_vs_pilot_cells":int(((target>0)&(old_target==0)).sum()),"collision_before_protection_cells":collision,
                    "method":"same-input early 8/10 four-view consensus seeds, late features and all-head QK refinement; old/paired target union minus unchanged D protection",
                    "source_probe_manifest_sha256":hashlib.sha256((probe/"manifest.json").read_bytes()).hexdigest(),"pilot_report":pilot["local_report"],
                    "selection_radius":0,"protection_policy":"fixed pilot D for this first controlled target-mask comparison; collisions retained"}
            (dest/"manifest.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
            candidate_plan.append({"label":label,"status":"ok","manifest":str((dest/"manifest.json").resolve()),"target_cells":int(target.sum())})
            print("CANDIDATE",cid,label,int(target.sum()),"added",record["added_vs_pilot_cells"],flush=True)
        (folder/"candidate-plan.json").write_text(json.dumps(candidate_plan,indent=2),encoding="utf-8")
        (folder/"ablation-summary.json").write_text(json.dumps(ablations,indent=2),encoding="utf-8")
        print("CASE_DONE",cid,flush=True)


def save_attribution_candidate(dest, label, values, metadata):
    from run_mask_attribution_validation import write_new
    dest.mkdir(parents=True, exist_ok=False)
    masks={role:values[role].detach().cpu().contiguous() for role in ROLES}
    save_file(masks,str(dest/'masks.safetensors'))
    evidence={key:value.detach().cpu().contiguous() for key,value in values.items() if key not in ROLES}
    if evidence:save_file(evidence,str(dest/'composition.safetensors'))
    png_hashes={}
    for name,value in values.items():
        if value.shape!=(1,64,64):raise ValueError('Unexpected attribution mask grid')
        image=dest/(name+'.png')
        Image.fromarray((value[0].cpu().numpy()*255).astype('uint8')).save(image)
        png_hashes[name]=hashlib.sha256(image.read_bytes()).hexdigest()
    meta={'anchored_mask_schema':1,'complete':True,'label':label,'grid':[64,64],
          'tensors':'masks.safetensors','tensors_sha256':hashlib.sha256((dest/'masks.safetensors').read_bytes()).hexdigest(),
          'png_sha256':png_hashes,'target_cells':int(masks['target'].sum()),'protected_cells':int(masks['protected'].sum()),
          'selection_radius':0,**metadata}
    if evidence:meta['composition_sha256']=hashlib.sha256((dest/'composition.safetensors').read_bytes()).hexdigest()
    write_new(dest/'manifest.json',meta)
    return {'label':label,'status':'ok','manifest':str((dest/'manifest.json').resolve()),'target_cells':meta['target_cells']}


def build_attribution_candidates(experiment_path, output_root):
    from run_mask_attribution_validation import write_new, validate_frozen_sources
    experiment_path=Path(experiment_path).resolve();experiment=read(experiment_path)
    root=Path(output_root).resolve()
    if root!=Path(experiment['output_root']):raise ValueError('Candidate output differs from frozen experiment')
    if (root/'candidate-index.json').exists():raise ValueError('Candidates already exist')
    validate_frozen_sources(experiment);torch.set_num_threads(4)
    experiment_hash=hashlib.sha256(experiment_path.read_bytes()).hexdigest()
    index={'experiment_sha256':experiment_hash,'cases':{}}
    for case in experiment['cases']:
        cid=case['case'];folder=root/cid;folder.mkdir(exist_ok=True)
        probe=Path(case['probe_root']);outer=read(probe/'manifest.json');cache={}
        def tap(branch,block):
            key=f'{branch}.step_3.block_{block}'
            if key not in cache:cache[key]=verify_suite(safe_tap(probe,outer['taps'][key]))
            return cache[key]
        contexts=[]
        for branch in ('base','slider'):
            for block in (8,10):contexts.append(logits(tap(branch,block)[1],3,block).softmax(-1))
        views=torch.stack(contexts);anchor=views.mean(0)
        consistent=(views.argmax(-1)==anchor.argmax(-1)[None]).all(0);anchor[~consistent]=1/3
        _,pilot=checked_diagnostic(case['pilot_report'])
        old=pilot['reference_target_mask'];protected=pilot['reference_protected_mask']
        dvalues={r:pilot['reference_'+r+'_mask'] for r in ROLES}
        dvalues.update(raw_target=old,collision=torch.zeros_like(old),added_to_pilot=torch.zeros_like(old),removed_from_pilot=torch.zeros_like(old))
        rows=[save_attribution_candidate(folder/'candidates'/'D','D',dvalues,
            {'experiment_sha256':experiment_hash,'method':'verified pilot D control','pilot_report':case['pilot_report']})]
        primary={};ablation_rows=[]
        for block in (18,23):
            for branch in ('base','slider'):
                manifest,tensors=tap(branch,block);cfg=copy.deepcopy(manifest['report']['algorithm']['config'])
                f=tensors['evidence.capture.attention_output'][None];q=tensors['evidence.capture.img_q'][None];k=tensors['evidence.capture.img_k'][None]
                cfg.update(propagation_heads=list(range(q.shape[2])),propagation_strength=.5)
                native=_candidate(f,logits(tensors,3,block),ROLES,cfg,3)
                anchored=anchored_candidate(f,logits(tensors,3,block),anchor,cfg)
                shape,shape_info=_propagate(native,q,k,None,cfg,lock_confident=False)
                combined,combined_info=_propagate(anchored,q,k,None,cfg,lock_confident=False)
                for name,candidate in (('baseline',native),('seed_only',anchored),('shape_only',shape),('combined',combined)):
                    bank=_candidate_bank(candidate,(64,64),cfg)
                    dest=folder/'ablations'/f'{branch}_s3_b{block}_{name}';dest.mkdir(parents=True,exist_ok=False)
                    data={'scores':candidate['scores'].contiguous(),
                          **{'seeds.'+r:candidate['seeds'][r].contiguous() for r in ROLES},
                          **{'masks.'+r:bank['masks'][r].contiguous() for r in ROLES}}
                    save_file(data,str(dest/'evidence.safetensors'))
                    for r in ROLES:Image.fromarray((bank['masks'][r][0].numpy()*255).astype('uint8')).save(dest/(r+'.png'))
                    stats={'branch':branch,'block':block,'variant':name,'seed_counts':candidate['seed_counts'],
                        'prototype_indices':candidate['prototype_indices'],'config':cfg,'anchor_consistency_cells':int(consistent.sum()),
                        'diagnostics':bank['diagnostics'],'experiment_sha256':experiment_hash,
                        'evidence_sha256':hashlib.sha256((dest/'evidence.safetensors').read_bytes()).hexdigest(),
                        'png_sha256':{r:hashlib.sha256((dest/(r+'.png')).read_bytes()).hexdigest() for r in ROLES}}
                    if name in ('shape_only','combined'):stats['propagation']=shape_info if name=='shape_only' else combined_info
                    write_new(dest/'statistics.json',stats);ablation_rows.append(stats)
                    if block==18 and name in ('seed_only','combined'):primary[branch,name]=bank['masks']['target']
                print('RAW_RECOMPUTED',cid,block,branch,flush=True)
        for label,variant,union in (('S0','seed_only',False),('S1','seed_only',True),('C0','combined',False),('C1','combined',True)):
            values=compose_paired_masks(primary['base',variant],primary['slider',variant],old,protected,include_pilot=union)
            row=save_attribution_candidate(folder/'candidates'/label,label,values,
                {'experiment_sha256':experiment_hash,'variant':variant,'include_pilot':union,
                 'method':'paired same-input target union; optional D support; fixed D protection',
                 'pilot_report':case['pilot_report'],'anchor_blocks':[8,10],'step':3,'block':18})
            if label=='C1':
                previous=Path(experiment['source_root'])/cid/'candidates'/'anchored_s3_b18'/'masks.safetensors'
                masks=load_file(str(previous))
                if not all(torch.equal(masks[r],values[r]) for r in ROLES):raise RuntimeError('C1 differs from old combined union candidate')
            rows.append(row);print('CANDIDATE',cid,label,row['target_cells'],int(values['added_to_pilot'].sum()),int(values['removed_from_pilot'].sum()),flush=True)
        index['cases'][cid]=rows;write_new(folder/'candidate-plan.json',rows);write_new(folder/'ablation-summary.json',ablation_rows)
    write_new(root/'candidate-index.json',index)


def build_diagnostic_releases(experiment_path,evaluation_path):
    from run_mask_attribution_validation import write_new
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tools'))
    from evaluate_paired_mask_attribution import load_annotation_views,classify_collisions
    experiment=read(experiment_path);root=Path(experiment['output_root']);index=read(root/'candidate-index.json')
    evaluation=read(evaluation_path);eh=hashlib.sha256(Path(experiment_path).read_bytes()).hexdigest()
    if evaluation['experiment_sha256']!=eh:raise ValueError('Release evaluation belongs to a different experiment')
    annotations=read(root/'annotations'/'manifest.json')
    if hashlib.sha256((root/'annotations'/'manifest.json').read_bytes()).hexdigest()!=evaluation['annotation_manifest_sha256']:
        raise ValueError('Release annotations changed')
    experiment['_path']=str(experiment_path)
    for case in experiment['cases']:
        cid=case['case'];s0=next(r for r in index['cases'][cid] if r['label']=='S0')
        dest=Path(s0['manifest']).parent;meta=read(s0['manifest'])
        if hashlib.sha256((dest/'composition.safetensors').read_bytes()).hexdigest()!=meta['composition_sha256']:raise ValueError('Changed S0 composition')
        values=load_file(str(dest/'composition.safetensors'));masks=load_file(str(dest/'masks.safetensors'))
        views=load_annotation_views(experiment,annotations,case)
        classes=classify_collisions(values['raw_target'].flatten().tolist(),masks['protected'].flatten().tolist(),views['base'],views['slider'])
        release=torch.tensor(classes['confirmed_target'],dtype=torch.float32).reshape(1,64,64)
        proof=next(c['collisions'] for c in evaluation['cases'] if c['case']==cid)
        if proof['class_cells']['confirmed_target'] != [i for i,v in enumerate(classes['confirmed_target']) if v]:raise ValueError('Release cell evidence mismatch')
        for name,vector in classes.items():
            path=root/cid/'collisions';path.mkdir(exist_ok=True)
            Image.fromarray((torch.tensor(vector).reshape(64,64).numpy()*255).astype('uint8')).save(path/(name+'.png'))
        if not release.any():row={'label':'S0-release','status':'not_applicable','reason':'no_confirmed_target_collision_cells'}
        else:
            result=apply_diagnostic_release(values['raw_target'],masks['protected'],release)
            result.update(raw_target=values['raw_target'],release_mask=release,
                          collision=values['collision'],added_to_pilot=(result['target']>masks['target']).float(),removed_from_pilot=torch.zeros_like(release))
            row=save_attribution_candidate(root/cid/'candidates'/'S0-release','S0-release',result,
                {'experiment_sha256':eh,'method':'annotation_dependent_diagnostic','human_verified':annotations['human_verified'],
                 'evaluation_sha256':hashlib.sha256(Path(evaluation_path).read_bytes()).hexdigest(),
                 'release_cells':int(release.sum()),'source_candidate_manifest_sha256':hashlib.sha256(Path(s0['manifest']).read_bytes()).hexdigest()})
        index['cases'][cid].append(row);print('RELEASE',cid,row['status'],int(release.sum()),flush=True)
    write_new(root/'candidate-index-release.json',index)


def main():
    if '--experiment-plan' not in sys.argv:
        return legacy_main()
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-plan',type=Path,required=True)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--release-evaluation',type=Path)
    args=parser.parse_args()
    if args.release_evaluation:build_diagnostic_releases(args.experiment_plan,args.release_evaluation)
    else:
        if args.output_root is None:parser.error('--output-root is required')
        build_attribution_candidates(args.experiment_plan,args.output_root)


if __name__=="__main__":main()
