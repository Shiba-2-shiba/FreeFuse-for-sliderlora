"""Reconstruct saved candidate composition and independently audit each replay."""
import argparse
from pathlib import Path
import sys

import torch
from PIL import Image
from safetensors.torch import load_file

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'tools'))
from run_mask_attribution_validation import read,sha,write_new,validate_frozen_sources,validate_generation_controls
from audit_attention_ablation_live import verify_edit
from slider_fuse.anchored_masks import compose_paired_masks,apply_diagnostic_release
from slider_fuse.paired_observation import checked_diagnostic,read_retained_masks
from slider_fuse.diagnostics import diagnostic_provenance
from slider_fuse.experimental_masks import _labels
from evaluate_paired_mask_attribution import evaluate_attribution


def png_equals(path,tensor):
    from audit_mask_support import load_binary_png
    size,values=load_binary_png(path)
    if size!=tuple(reversed(tensor.shape[-2:])):raise ValueError('PNG/tensor grid mismatch')
    decoded=torch.tensor(values,dtype=torch.float32).reshape(tensor.shape)
    if not torch.equal(decoded,tensor.float()):raise ValueError('PNG/tensor values mismatch')


def audit(experiment_path, evaluation_path=None):
    torch.set_num_threads(4)
    experiment=read(experiment_path);validate_frozen_sources(experiment)
    root=Path(experiment['output_root']);index=read(root/'candidate-index-release.json')
    if index['experiment_sha256']!=sha(experiment_path):raise ValueError('Candidate experiment mismatch')
    evaluation_path=Path(evaluation_path) if evaluation_path else root/'attribution.json'
    recalculated=evaluate_attribution(experiment_path,root/'annotations'/'manifest.json',root/'candidate-index.json')
    if recalculated!=read(evaluation_path):raise ValueError('Saved attribution differs from independently recalculated metrics')
    result={'mask_attribution_verification_schema':1,'experiment_sha256':sha(experiment_path),
        'source_integrity':True,'annotation_human_verified':recalculated['human_verified'],'attribution_recalculated_equal':True,
        'evaluation_path':str(evaluation_path.resolve()),'evaluation_sha256':sha(evaluation_path),
        'cases':[],'diffusion_model_nfe':0,'verified_raw_variants':0,'quality_approval':'separate assistant visual observations; no human approval',
        'effective_source_sha256':{str(p.relative_to(HERE.parent)):sha(p) for p in
            (HERE/'run_mask_attribution_validation.py',HERE/'build_paired_mask_candidates.py',
             HERE/'prepare_mask_attribution_annotations.py',HERE.parent/'tools'/'evaluate_paired_mask_attribution.py',
             HERE.parent/'slider_fuse'/'anchored_masks.py',Path(__file__))}}
    for case in experiment['cases']:
        cid=case['case'];folder=root/cid;rows=index['cases'][cid]
        _,pilot=checked_diagnostic(case['pilot_report']);d=pilot['reference_target_mask'];p=pilot['reference_protected_mask']
        raw={}
        for block in (18,23):
            for branch in ('base','slider'):
                for variant in ('baseline','seed_only','shape_only','combined'):
                    dest=folder/'ablations'/f'{branch}_s3_b{block}_{variant}'
                    stats=read(dest/'statistics.json')
                    if sha(dest/'evidence.safetensors')!=stats['evidence_sha256']:raise ValueError('Changed raw variant evidence')
                    t=load_file(str(dest/'evidence.safetensors'));labels=_labels(t['scores'],stats['config'])
                    for r,k in (('target',0),('protected',1),('background',2)):
                        if sha(dest/(r+'.png'))!=stats['png_sha256'][r]:raise ValueError('Changed raw mask PNG')
                        expected=(labels==k).reshape(1,64,64).float() if r!='background' else 1-t['masks.target']-t['masks.protected']
                        if not torch.equal(expected,t['masks.'+r]):raise ValueError('Raw scores do not reconstruct mask')
                        seeds=t['seeds.'+r]
                        if seeds.any() and not torch.equal(t['scores'][seeds],torch.nn.functional.one_hot(torch.full((int(seeds.sum()),),k),3).float()):
                            raise ValueError('Anchor seeds lost their one-hot role')
                        png_equals(dest/(r+'.png'),t['masks.'+r])
                    raw[block,branch,variant]=t['masks.target'];result['verified_raw_variants']+=1
        for row in rows:
            if row['status']!='ok':continue
            meta=read(row['manifest']);dest=Path(row['manifest']).parent
            if meta.get('experiment_sha256')!=sha(experiment_path):raise ValueError('Wrong candidate source experiment')
            masks=read_retained_masks(row['manifest'])
            for role in ('target','protected','background'):
                if sha(dest/(role+'.png'))!=meta['png_sha256'][role]:raise ValueError('Changed final mask PNG')
                png_equals(dest/(role+'.png'),masks[role])
            if row['label']=='D':expected={'target':d,'protected':p,'background':1-d-p}
            elif row['label']!='S0-release':
                variant='seed_only' if row['label'].startswith('S') else 'combined'
                expected=compose_paired_masks(raw[18,'base',variant],raw[18,'slider',variant],d,p,include_pilot=row['label'].endswith('1'))
                composition=load_file(str(dest/'composition.safetensors'))
                if sha(dest/'composition.safetensors')!=meta['composition_sha256']:raise ValueError('Changed composition')
                for key,value in composition.items():
                    if not torch.equal(value,expected[key]):raise ValueError('Wrong recorded composition')
                    png_equals(dest/(key+'.png'),value)
            else:
                s0=next(x for x in rows if x['label']=='S0');source=Path(s0['manifest']).parent
                if meta['source_candidate_manifest_sha256']!=sha(s0['manifest']):raise ValueError('Changed release source')
                data=load_file(str(source/'composition.safetensors'));prior_masks=read_retained_masks(s0['manifest'])
                composition=load_file(str(dest/'composition.safetensors'))
                release=composition['release_mask']
                confirmed=next(x for x in recalculated['cases'] if x['case']==cid)['collisions']['class_cells']['confirmed_target']
                expected_release=torch.zeros(4096);expected_release[confirmed]=1
                if not torch.equal(release.flatten(),expected_release):raise ValueError('Released cells lack independent paired target support')
                expected=apply_diagnostic_release(data['raw_target'],prior_masks['protected'],release)
                if not torch.equal(expected['target']-prior_masks['target'],release):raise ValueError('Release target delta differs')
                if not torch.equal(prior_masks['protected']-expected['protected'],release):raise ValueError('Release protection delta differs')
            if not all(torch.equal(masks[r],expected[r]) for r in masks):raise ValueError('Candidate composition mismatch')
        case_result={'case':cid,'control_exact_parity':True,'candidates':[], 'release_status':next(x['status'] for x in rows if x['label']=='S0-release'),'nfe':0}
        for stage in ('control','factorial','release'):
            artifacts=read(folder/(stage+'.artifacts.json'))
            case_result['nfe']+=artifacts['generation_nfe']
            if artifacts['status']=='not_applicable':continue
            graph=read(folder/(stage+'.queued_api.json'))['prompt'];history=read(folder/(stage+'.history.json'))
            if history['status']['status_str']!='success' or history['prompt'][2]!=graph:raise ValueError('History/graph mismatch')
            cached={str(n) for kind,msg in history['status']['messages'] if kind=='execution_cached' for n in msg.get('nodes',[])}
            if sorted(cached)!=artifacts['cached_nodes']:raise ValueError('Cache accounting mismatch')
            calculated_nfe=0
            for record in artifacts['diagnostics']:
                path=Path(record['local_report']);r=verify_edit(path);_,t=checked_diagnostic(path)
                validate_generation_controls(r)
                label=record['label'].removeprefix(stage+'_')
                candidate=next(x for x in rows if x['label']==label)
                masks=read_retained_masks(candidate['manifest'])
                if r['generation']!=diagnostic_provenance(graph,record['save_node'],r):raise ValueError('Diagnostic generation provenance mismatch')
                if not r['owned_hooks_removed'] or r['total_model_nfe']!=16 or r['effective_slider_strength']!=4:raise ValueError('Runtime control mismatch')
                if r['trial_id']!=experiment['trial_id']:raise ValueError('Stale trial')
                if not all(torch.equal(t['reference_'+role+'_mask'],masks[role]) for role in masks):raise ValueError('Sampler used different masks')
                for key in ('initial_noise_sha256','conditioning_sha256','full_sigmas_sha256','lora_sha256'):
                    if r[key]!=case[key]:raise ValueError('Changed generation input: '+key)
                for key in ('trace_base_predictions','trace_slider_predictions'):
                    if not torch.equal(t[key][0],pilot[key][0]):raise ValueError('Initial branch prediction differs')
                if label=='D':
                    for key in ('trace_inputs','trace_predictions','trace_base_predictions','trace_slider_predictions','final_latent'):
                        if not torch.equal(t[key],pilot[key]):raise ValueError('D parity failed')
                is_cached=record['sampler_node'] in cached
                if not is_cached:calculated_nfe+=r['total_model_nfe']
                case_result['candidates'].append({'label':label,'image':str(path.parent/r['artifacts']['image']),
                    'diagnostic':str(path),'mask_manifest':candidate['manifest'],'target_cells':int(masks['target'].sum()),
                    'protected_control_equal':torch.equal(t['reference_protected_mask'],p),
                    'all_step_routing_verified':True,'cached':is_cached,'image_sha256':r['artifact_sha256']['image']})
            if calculated_nfe!=artifacts['generation_nfe']:raise ValueError('NFE accounting mismatch')
        traces=read(folder/'traces.artifacts.json')
        for label,record in traces['bundles'].items():
            trace_root=Path(record['local']);manifest=read(trace_root/'manifest.json')
            if sha(trace_root/'manifest.json')!=record['manifest_sha256']:raise ValueError('Changed decoded trace manifest')
            if not manifest['complete'] or not manifest['final_decoded_rgb_equal'] or manifest['diffusion_model_nfe']!=0:raise ValueError('Trace decoding failed')
            source=next(c for c in case_result['candidates'] if c['label']==label)
            if manifest['source_report_sha256']!=sha(source['diagnostic']):raise ValueError('Trace references different replay')
            for name,expected in manifest['file_sha256'].items():
                if sha(trace_root/name)!=expected:raise ValueError('Changed decoded trace image')
            with Image.open(trace_root/'final.png') as a,Image.open(source['image']) as b:
                if a.convert('RGB').tobytes()!=b.convert('RGB').tobytes():raise ValueError('Trace final RGB parity failed')
        case_result['verified_trace_bundles']=len(traces['bundles'])
        result['diffusion_model_nfe']+=case_result['nfe'];result['cases'].append(case_result)
        print('AUDITED',cid,case_result['nfe'],'NFE',len(case_result['candidates']),'images',flush=True)
    if result['diffusion_model_nfe']>experiment['planned_max_diffusion_nfe']:raise ValueError('Budget exceeded without recorded plan amendment')
    result['output_file_sha256']={str(p.relative_to(root)):sha(p) for p in root.rglob('*')
        if p.is_file() and 'pytest' not in p.parts and p.suffix in ('.json','.png','.safetensors')}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment-plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path)
    args=parser.parse_args();write_new(args.output,audit(args.experiment_plan,args.evaluation));print(args.output)
