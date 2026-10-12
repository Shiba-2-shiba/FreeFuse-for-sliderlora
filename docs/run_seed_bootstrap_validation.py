"""Frozen seed-lock controls and base-only bootstrap on the current local ComfyUI."""
from __future__ import annotations

import argparse
import copy
import datetime
import json
from pathlib import Path
import sys
import time
import uuid
import urllib.error

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'tools'))
from run_mask_attribution_validation import read, sha, write_new, validate_frozen_sources, validate_generation_controls
from run_mask_source_comparison_live import node, rt, OUTPUT
from run_attention_step_block_sweep import retain_edits
from run_paired_mix_validation import retain_bundle
from audit_attention_ablation_live import verify_edit, verify_suite
from build_paired_mask_candidates import build_seed_control_banks, save_attribution_candidate
from evaluate_paired_mask_attribution import (load_annotation_views, region_metrics, PARTS, PROTECTED,
    validate_regions, validate_reference, classify_collisions_v2, validate_release_review)
from slider_fuse.anchored_masks import compose_paired_masks, apply_diagnostic_release
from slider_fuse.paired_observation import checked_diagnostic, read_retained_masks
from slider_fuse.experimental_artifacts import validate_experimental_prefix
from slider_fuse.sampling import tensor_hash

ROLES = ('target','protected','background')
DIAGNOSTICS = ('park__seed_444444','woman_front_strong_overlap__seed_42')
PARITY_KEYS = ('trace_inputs','trace_base_predictions','trace_slider_predictions','trace_predictions','final_latent')


def case_identity(prompt_case, seed):
    if not isinstance(prompt_case,str) or not prompt_case.replace('_','').isalnum() or type(seed) is not int or seed<0:
        raise ValueError('Invalid case identity')
    return f'{prompt_case}__seed_{seed}'


def _validate_run_inputs(trial_id,prefix):
    if type(trial_id) is not int or not 0<=trial_id<=0x7fffffff:
        raise ValueError('trial_id must be a nonnegative 31-bit integer')
    validate_experimental_prefix(prefix)


def base_graph(case):
    old = case['base_graph']
    return {sid:copy.deepcopy(old[sid]) for sid in ('1','2','3','4','5','7','9')}


def build_bootstrap_graph(case_config:dict, *, trial_id:int, prefix:str)->dict:
    _validate_run_inputs(trial_id,prefix)
    old=case_config['base_graph'];graph=base_graph(case_config)
    sub=copy.deepcopy(old['6']['inputs'])
    for field in ('target_mask','protected_mask'):sub.pop(field,None)
    node(graph,'6','Krea2SliderFuseSubjects',sub,'Text roles only; no historical masks')
    inputs=copy.deepcopy(old['8']['inputs'])
    inputs.update(model=['2',0],subjects=['6',0],seed=case_config['seed'],trial_id=trial_id,
                  steps=8,config_json=json.dumps(case_config['collector_config']),audit_tensors=True)
    node(graph,'8','Krea2SliderFuseExperimentalMaskCollect',inputs,'Base-only two forwards')
    node(graph,'18','Krea2SliderFuseExperimentalMaskSave',{'suite':['8',0],'filename_prefix':prefix},'Retain bootstrap')
    return graph


def build_pilot_graph(case_config:dict, initial_mask_manifest:Path, *, trial_id:int, prefix:str)->dict:
    _validate_run_inputs(trial_id,prefix)
    old=case_config['base_graph'];graph=base_graph(case_config)
    node(graph,'60','Krea2SliderFuseRetainedMask',{'report_path':str(initial_mask_manifest)},'Explicit initial masks')
    sub=copy.deepcopy(old['6']['inputs']);sub.update(target_mask=['60',0],protected_mask=['60',1])
    node(graph,'6','Krea2SliderFuseSubjects',sub,'Initial paired routing')
    inputs=copy.deepcopy(old['21']['inputs'])
    inputs.update(model=['2',0],subjects=['6',0],seed=case_config['seed'],steps=8,cfg=1.,strength=4.,
                  lora_name=case_config['slider_path'],trial_id=trial_id,mask_mode='manual',mix_scope='target_mask',
                  diagnostic_level='audit',selection_dilate_radius=0,fill_holes_max_area=0,mask_dilate_radius=0,
                  background_phrase=case_config['background_phrase'])
    node(graph,'100','Krea2SliderFuseObservedPredictionMix',inputs,'Passive observed Mix')
    node(graph,'101','VAEDecode',{'samples':['100',0],'vae':['9',0]},'Pilot RGB')
    node(graph,'102','Krea2SliderFuseDiagnosticSave',{'images':['101',0],'latent':['100',0],
         'diagnostics':['100',2],'filename_prefix':f'seed_bootstrap/{prefix}/pilot'},'Pilot audit')
    node(graph,'105','Krea2SliderFusePairedProbeSave',{'probe':['100',3],'model':['2',0],
         'vae':['9',0],'filename_prefix':prefix},'Retain raw paired observations and RGB')
    return graph


def evaluate_holdout_eligibility(selection:dict,gates:dict)->dict:
    selected=selection.get('selected_variant');variant=selected or 'R0';reasons=[]
    if selected not in (None,'R1','R2','R3','R4'):
        raise ValueError('Unknown selected variant')
    for cid in DIAGNOSTICS:
        row=gates.get('cases',{}).get(cid,{}).get('A-'+variant,{})
        for gate in ('G0','G1','G2','G3'):
            if row.get('gates',{}).get(gate)!='pass':reasons.append(f'{cid}: {gate} not pass')
        if row.get('major_anatomy_defect') is not False:reasons.append(f'{cid}: anatomy unconfirmed or defective')
    return {'eligible':not reasons,'mode':('selected_method' if selected else 'path_only') if not reasons else 'not_eligible',
            'required_variant':variant,'reasons':reasons}


def verified_resume(path,identity):
    path=Path(path)
    if not path.exists():return None
    saved=read(path)
    if saved.get('identity')!=identity:raise ValueError('Run identity mismatch')
    if saved.get('status')!='complete':raise ValueError('Run is not complete; inspect retained history before retry')
    for item,digest in saved['file_sha256'].items():
        if sha(item)!=digest:raise ValueError('Changed completed run artifact: '+item)
    return saved


def freeze_experiment(source_root,output_root,trial_id,server):
    source_root=Path(source_root).resolve();root=Path(output_root).resolve()
    if root.exists() or root.is_symlink() or root.is_relative_to(source_root):
        raise ValueError('Refuse existing/source output directory')
    _validate_run_inputs(trial_id,'freeze')
    prior=read(source_root/'experiment.json');validate_frozen_sources(prior)
    verification=read(source_root/'verification.json')
    if verification['experiment_sha256']!=sha(source_root/'experiment.json') or not verification['source_integrity']:
        raise ValueError('Unverified attribution source')
    source_files=dict(prior['source_files'])
    for p in source_root.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.png','.safetensors') and 'pytest' not in p.parts:
            source_files[str(p)]=sha(p)
    cases=[]
    for old_case in prior['cases']:
        cid=old_case['case'];graph_path=Path(old_case['prior_graph']);old=read(graph_path)['prompt']
        report_a=next((graph_path.parent/'edits'/'A_off').glob('*.json'))
        a,at=checked_diagnostic(report_a);validate_generation_controls(a)
        for name in a['artifacts'].values():
            p=report_a.parent/name;source_files[str(p)]=sha(p)
        source_files[str(report_a)]=sha(report_a)
        cases.append({'case_id':case_identity(cid,old_case['seed']),'prompt_case':cid,'seed':old_case['seed'],
          'base_graph':old,'collector_config':json.loads(old['8']['inputs']['config_json']),
          'slider_path':old['21']['inputs']['lora_name'],'asset_sha256':{'slider':old_case['lora_sha256']},
          'role_positions':copy.deepcopy(a['token_positions']),'background_phrase':old['8']['inputs']['background_phrase'],
          'reference_A_report':str(report_a),'reference_D_report':old_case['pilot_report'],
          'reference_D_probe':old_case['probe_root'],'reference_D_images':old_case['references'],
          'source_case':old_case})
    root.mkdir(parents=True)
    plan={'seed_bootstrap_schema':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'source_root':str(source_root),'output_root':str(root),'trial_id':trial_id,'server':server,'cases':cases,
          'source_files':source_files,'planned_max_images':26,'planned_max_diffusion_nfe':428,
          'human_verified':False,'holdout_cases':[{'prompt_case':c['prompt_case'],'seed':s,'case_id':case_identity(c['prompt_case'],s)} for c in cases for s in (123,777)],
          'code_sha256':{str(p.relative_to(HERE.parent)):sha(p) for p in (Path(__file__),HERE/'build_paired_mask_candidates.py',HERE.parent/'slider_fuse/anchored_masks.py')}}
    write_new(root/'experiment.json',plan)
    return plan


def annotation_views(plan,case):
    old=read(Path(plan['source_root'])/'experiment.json');old['_path']=str(Path(plan['source_root'])/'experiment.json')
    return load_annotation_views(old,read(Path(plan['source_root'])/'annotations/manifest.json'),case['source_case'])


def _metrics(mask,views):
    values=mask.flatten().bool().tolist()
    result={b:region_metrics(values,v) for b,v in views.items()}
    result['common_background_cells']=sum(c and a and b for c,a,b in zip(values,views['base']['known_background'],views['slider']['known_background']))
    return result


def save_scoring(root,case,pilot_label,probe,bootstrap,views=None):
    result=build_seed_control_banks(probe,bootstrap);folder=root/case['case_id']/('scoring_'+pilot_label)
    folder.mkdir(parents=True,exist_ok=False);rows={}
    for variant,branches in result['variants'].items():
        row={'status':'ok','branches':{}}
        for branch,value in branches.items():
            if value['status']!='ok':row['status']='failed';row['branches'][branch]=value;continue
            dest=folder/variant/branch;dest.mkdir(parents=True)
            candidate=value['candidate'];bank=value['bank'];prototypes=value['prototypes'];pools=result['branches'][branch]['pools']
            tensors={k:candidate[k].contiguous().clone() for k in ('scores','prelock_scores','growth_allowed')}
            for role in ROLES:
                tensors.update({f'locks.{role}':candidate['lock_masks'][role].clone(),f'full_pool.{role}':pools['full'][role].clone(),
                    f'supported_pool.{role}':pools['supported'][role].clone(),f'centers.{role}':prototypes['centers'][role].clone(),
                    f'masks.{role}':bank['masks'][role].clone()})
            for key in ('uncertain_mask','real_background_mask'):tensors[key]=bank[key].clone()
            save_file(tensors,str(dest/'evidence.safetensors'))
            metadata={'evidence_sha256':sha(dest/'evidence.safetensors'),'config':result['branches'][branch]['config'],
                'prototype_indices':candidate['prototype_indices'],'prototype_pool_counts':candidate['prototype_pool_counts'],
                'lock_counts':candidate['lock_counts'],'prelock_sha256':tensor_hash(candidate['prelock_scores']),
                'prototype_sha256':{r:tensor_hash(prototypes['centers'][r]) for r in ROLES},
                'lock_sha256':{r:tensor_hash(candidate['lock_masks'][r]) for r in ROLES}}
            write_new(dest/'statistics.json',metadata)
            entry={'status':'ok','evidence':str(dest/'evidence.safetensors'),'statistics':str(dest/'statistics.json')}
            if views:entry['metrics']=region_metrics(bank['masks']['target'].flatten().bool().tolist(),views[branch])
            row['branches'][branch]=entry
        if row['status']=='ok':
            b,s=(branches[x]['bank']['masks']['target'] for x in ('base','slider'))
            try:
                values=compose_paired_masks(b,s,bootstrap['target'],bootstrap['protected'],include_pilot=False)
                entry=save_attribution_candidate(folder/variant/'candidate',pilot_label+'-'+variant,values,
                    {'pilot':pilot_label,'variant':variant,'source_probe_manifest_sha256':sha(Path(probe)/'manifest.json'),
                     'method':'paired target union minus shared base-only protection; no historical target added'})
                row['manifest']=entry['manifest']
                if views:row['metrics']=_metrics(values['target'],views)
                common=torch.tensor([a and b for a,b in zip(views['base']['known_background'],views['slider']['known_background'])]) if views else None
                lock=branches['base']['candidate']['lock_masks']['target']
                row['fixed_background_errors']=int((lock & common).sum()) if views else None
                row['fixed_cells']=sum(branches['base']['candidate']['lock_counts'].values())
            except ValueError as exc:row.update(status='failed',reason=str(exc))
        rows[variant]=row
    # Causal identities must hold even when a mask fails confidence thresholds.
    for branch in ('base','slider'):
        raw=result['variants']
        for left,right in (('R0','R1'),('R1','R2')):
            if raw[left][branch]['status']==raw[right][branch]['status']=='ok':
                a,b=raw[left][branch],raw[right][branch]
                if not torch.equal(a['candidate']['prelock_scores'],b['candidate']['prelock_scores']):raise ValueError('Lock comparison changed scores')
                if not all(torch.equal(a['prototypes']['centers'][r],b['prototypes']['centers'][r]) for r in ROLES):raise ValueError('Lock comparison changed prototypes')
        if raw['R2'][branch]['status']==raw['R4'][branch]['status']=='ok':
            if not all(torch.equal(raw['R2'][branch]['candidate']['lock_masks'][r],raw['R4'][branch]['candidate']['lock_masks'][r]) for r in ROLES):raise ValueError('Prototype comparison changed locks')
    write_new(folder/'index.json',{'case_id':case['case_id'],'pilot':pilot_label,'variants':rows})
    return rows,result


def select_variant(all_rows):
    eligible=[];rejected={}
    for variant in ('R1','R2','R3','R4'):
        reasons=[];strict=False;bg=errors=locks=0
        for cid,rows in all_rows.items():
            row,baseline=rows[variant],rows['R0']
            if row['status']!='ok' or baseline['status']!='ok':reasons.append(cid+': invalid candidate');continue
            for branch in ('base','slider'):
                for metrics_old,metrics_new in ((baseline['branches'][branch]['metrics'],row['branches'][branch]['metrics']),
                                               (baseline['metrics'][branch],row['metrics'][branch])):
                    for part in PARTS+PROTECTED:
                        field='missing_cells' if part in PARTS else 'intrusion_cells'
                        old,new=metrics_old[part][field],metrics_new[part][field]
                        if part=='target_face' and (old is None or new is None):reasons.append(cid+': missing face annotation')
                        elif old is not None and (new is None or new>old):reasons.append(f'{cid}/{branch}: {part} worsened')
            old,new=baseline['metrics']['common_background_cells'],row['metrics']['common_background_cells']
            if new>old:reasons.append(cid+': common background worsened')
            strict |= new<old or row['fixed_background_errors']<baseline['fixed_background_errors']
            bg+=new;errors+=row['fixed_background_errors'];locks+=row['fixed_cells']
        if not strict:reasons.append('no strict background/lock error improvement')
        if reasons:rejected[variant]=sorted(set(reasons))
        else:eligible.append((bg,errors,locks,variant))
    selected=min(eligible)[-1] if eligible else None
    return {'selected_variant':selected,'eligible_ranked':[r[-1] for r in sorted(eligible)],'rejected':rejected,
            'basis':'frozen sparse assistant labels; not human ground truth'}


def run_offline(experiment_path):
    plan=read(experiment_path);validate_frozen_sources(plan);root=Path(plan['output_root']);all_rows={};baseline=[]
    for case in plan['cases']:
        _,a=checked_diagnostic(case['reference_A_report']);bootstrap={r:a['reference_'+r+'_mask'] for r in ROLES}
        views=annotation_views(plan,case)
        rows,result=save_scoring(root,case,'D',case['reference_D_probe'],bootstrap,views)
        if rows['R0']['status']!='ok':raise ValueError('R0 invalid; stop comparisons')
        for branch in ('base','slider'):
            saved=load_file(str(Path(plan['source_root'])/case['prompt_case']/'ablations'/f'{branch}_s3_b18_seed_only'/'evidence.safetensors'))
            fresh=result['variants']['R0'][branch]['candidate']
            if not torch.equal(fresh['scores'],saved['scores']):raise ValueError('R0 differs from retained source scores')
            for role in ROLES:
                if not torch.equal(fresh['seeds'][role],saved['seeds.'+role]) or not torch.equal(result['variants']['R0'][branch]['bank']['masks'][role],saved['masks.'+role]):raise ValueError('R0 differs from retained bank')
        common=torch.tensor([x and y for x,y in zip(views['base']['known_background'],views['slider']['known_background'])])
        pool=result['branches']['base']['pools']['full']['target']
        raw=result['variants']['R0']['base']['bank']['masks']['target']
        face=torch.tensor(views['base']['target_face']).reshape(1,64,64).bool()
        final=read_retained_masks(rows['R0']['manifest'])['target']
        baseline.append({'case_id':case['case_id'],'background_target_seeds':int(common[pool].sum()),'target_seed_count':len(pool),
                         'base_face_raw':int((raw.bool()&face).sum()),'base_face_final':int((final.bool()&face).sum()),
                         'removed_face_xy':[(int(i)%64,int(i)//64) for i in (raw.bool()&~final.bool()&face).flatten().nonzero().flatten()]})
        all_rows[case['case_id']]=rows
        print('OFFLINE',case['case_id'],{k:r['status'] for k,r in rows.items()},flush=True)
    expected={'park__seed_444444':14,'woman_front_strong_overlap__seed_42':126}
    if any(r['background_target_seeds']!=expected[r['case_id']] for r in baseline):raise ValueError('Background seed baseline changed')
    park=baseline[0]
    if park['base_face_raw']!=9 or park['base_face_final']!=8 or park['removed_face_xy']!=[(23,15)]:raise ValueError('Face collision baseline changed')
    write_new(root/'baseline-recheck.json',{'R0_exact_saved_parity':True,'cases':baseline})
    selection=select_variant(all_rows)
    selection.update(experiment_sha256=sha(experiment_path),annotation_sha256=sha(Path(plan['source_root'])/'annotations/manifest.json'))
    write_new(root/'selection.json',selection)
    print('SELECTION',selection['selected_variant'],flush=True)


def execute_graph(plan,folder,graph):
    """Preflight once; retain the submission before polling, never resubmit unknown jobs."""
    folder.mkdir(parents=True,exist_ok=True)
    queued=folder/'queued_api.json'
    if queued.exists():
        if read(queued)!= {'prompt':graph}:raise ValueError('Changed queued graph identity')
    else:write_new(queued,{'prompt':graph})
    preflight=rt.preflight(plan['server'],str(queued));issues=rt.format_preflight_issues(preflight)
    if not (folder/'preflight.json').exists():write_new(folder/'preflight.json',{'issues':issues,'output_node_count':preflight['output_node_count']})
    if issues:raise ValueError(issues)
    submission=folder/'submission.json'
    if submission.exists():
        response=read(submission)
        if response['queued_api_sha256']!=sha(queued):raise ValueError('Changed submission graph')
    else:
        intent_path=folder/'submission-intent.json'
        if intent_path.exists():
            intent=read(intent_path)
            if intent['queued_api_sha256']!=sha(queued):raise ValueError('Changed submission intent')
            pid=intent['prompt_id'];known=rt.request_json(plan['server']+'/history/'+pid,timeout=30).get(pid)
            if known:
                if known['prompt'][2]!=graph:raise ValueError('Recovered history has different graph')
            else:
                queue=rt.request_json(plan['server']+'/queue',timeout=30)
                matches=[row for row in queue.get('queue_running',[])+queue.get('queue_pending',[]) if row[1]==pid]
                if not matches:raise ValueError('Unresolved submission intent; inspect the reserved job ID before any new attempt')
                if any(row[2]!=graph for row in matches):raise ValueError('Recovered queue has different graph')
            response={'prompt_id':pid,'recovered_from_intent':True}
        else:
            intent={'prompt_id':str(uuid.uuid4()),'queued_api_sha256':sha(queued),
                    'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            write_new(intent_path,intent)
            try:
                response=rt.request_json(plan['server']+'/prompt',method='POST',
                    payload={'prompt':graph,'prompt_id':intent['prompt_id']},timeout=30)
            except urllib.error.HTTPError as exc:
                write_new(folder/'submission-error.json',rt.parse_http_error(exc))
                raise
        write_new(submission,{**response,'queued_api_sha256':sha(queued),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    pid=response['prompt_id'];print('SUBMITTED',folder.name,pid,flush=True)
    started=time.perf_counter();notice=0
    if (folder/'history.json').exists():entry=read(folder/'history.json')
    else:
        while True:
            history=rt.request_json(plan['server']+'/history/'+pid,timeout=30);entry=history.get(pid)
            if entry and (entry.get('status',{}).get('completed') or entry.get('status',{}).get('status_str') in ('success','error')):break
            elapsed=time.perf_counter()-started
            if elapsed>3600:raise TimeoutError('Unresolved submitted job; do not resubmit '+pid)
            if elapsed-notice>=25:print('WAITING',folder.name,round(elapsed,1),flush=True);notice=elapsed
            time.sleep(3)
        write_new(folder/'history.json',entry)
    if entry['prompt'][2]!=graph:raise ValueError('History graph identity mismatch')
    if entry['status']['status_str']!='success':
        if not (folder/'failure.json').exists():write_new(folder/'failure.json',{'status':'failed','prompt_id':pid,'nfe':None,'reason':entry['status']})
        raise RuntimeError('Live execution failed; history retained '+pid)
    if not (folder/'timing.json').exists():write_new(folder/'timing.json',{'poll_wall_seconds':time.perf_counter()-started,
        'scope':'one polling period; not summed as inference time; decode costs belong to this queue'})
    return entry


def _prefix(plan,case,label):
    value='sb_'+Path(plan['output_root']).name.replace('-','_')+'_'+case['case_id'].replace('__','_')+'_'+label
    validate_experimental_prefix(value)
    return value


def _identity(case,phase,variant):
    return {'case_id':case['case_id'],'phase':phase,'variant':variant,'attempt_id':1}


def _cache(history):
    return sorted({str(n) for kind,msg in history['status']['messages'] if kind=='execution_cached' for n in msg.get('nodes',[])})


def _complete(folder,row):
    files={str(p.resolve()):sha(p) for p in folder.rglob('*') if p.is_file() and p.name!='artifacts.json'}
    row.update(status='complete',file_sha256=files)
    write_new(folder/'artifacts.json',row)
    return row


def run_bootstrap_case(plan,case,offset):
    root=Path(plan['output_root']);folder=root/case['case_id']/'bootstrap';identity=_identity(case,'bootstrap','base')
    resumed=verified_resume(folder/'artifacts.json',identity)
    if resumed:return resumed
    prefix=_prefix(plan,case,'bootstrap');trial=(plan['trial_id']+offset)%2147483648
    graph=build_bootstrap_graph(case,trial_id=trial,prefix=prefix)
    history=execute_graph(plan,folder,graph)
    bundle=retain_bundle(folder/'suite',prefix);manifest,tensors=verify_suite(folder/'suite/manifest.json')
    report=manifest['report'];algorithm=report['algorithm']['variants']['multi_proto_bg']
    if algorithm['status']!='ok' or report['phase1_nfe']!=2 or not report['owned_hooks_removed'] or report['slider_loaded']:
        raise ValueError('Invalid base-only bootstrap')
    masks={r:tensors['multi_proto_bg.masks.'+r] for r in ROLES}
    record=save_attribution_candidate(folder/'mask','B',masks,{'source_bootstrap_manifest_sha256':sha(folder/'suite/manifest.json'),
        'method':'style base-only step2/block18 multi_proto_bg; no global ON or D inputs'})
    parity={}
    if 'reference_A_report' in case:
        _,a=checked_diagnostic(case['reference_A_report']);_,d=checked_diagnostic(case['reference_D_report'])
        parity={r:torch.equal(masks[r],a['reference_'+r+'_mask']) for r in ('target','protected')}
        parity['D_protected']=torch.equal(masks['protected'],d['reference_protected_mask'])
        if not all(parity.values()):raise ValueError('Fresh base bootstrap differs from A/D control')
    nfe=0 if '8' in _cache(history) else report['phase1_nfe']
    return _complete(folder,{'identity':identity,'bundle':bundle,'mask_manifest':record['manifest'],
        'parity':parity,'cached_nodes':_cache(history),'generation_nfe':nfe,'trial_id':trial,'prompt_id':history['prompt'][1]})


def run_pilot_case(plan,case,label,offset):
    root=Path(plan['output_root']);folder=root/case['case_id']/('pilot_'+label);identity=_identity(case,'pilot',label)
    resumed=verified_resume(folder/'artifacts.json',identity)
    if resumed:return resumed
    initial=case['reference_D_report'] if label=='D' else read(root/case['case_id']/'bootstrap/artifacts.json')['mask_manifest']
    prefix=_prefix(plan,case,'pilot_'+label);trial=(plan['trial_id']+offset)%2147483648
    graph=build_pilot_graph(case,initial,trial_id=trial,prefix=prefix);history=execute_graph(plan,folder,graph)
    records=retain_edits(folder,[{'label':label,'status':'queued','sampler_node':'100','save_node':'102'}],history)
    bundle=retain_bundle(folder/'probe',prefix);report,tensors=checked_diagnostic(records[0]['local_report']);verify_edit(Path(records[0]['local_report']))
    validate_generation_controls(report)
    if not report['owned_hooks_removed'] or report['total_model_nfe']!=16:raise ValueError('Pilot inference/observer integrity failure')
    parity={}
    if 'reference_A_report' in case:
        source=case['reference_D_report'] if label=='D' else case['reference_A_report'];_,old=checked_diagnostic(source)
        parity={k:torch.equal(tensors[k],old[k]) for k in PARITY_KEYS}
        if not all(parity.values()):raise ValueError('Pilot '+label+' trace differs from source control')
        if label=='D':
            prior=Path(case['reference_D_probe']);outer=read(prior/'manifest.json');fresh=read(folder/'probe/manifest.json')
            if fresh['frames']!=outer['frames']:raise ValueError('D paired frame/input identity differs')
            # Frame tap metadata is deterministic; verify every retained map independently.
            for key,item in outer['taps'].items():
                old_tap,_=verify_suite(prior/item['manifest']);new_tap,_=verify_suite(folder/'probe'/fresh['taps'][key]['manifest'])
                if old_tap['tensor_sha256']!=new_tap['tensor_sha256']:raise ValueError('D observed tap differs')
    refs=probe_references(folder/'probe')
    row={'identity':identity,'initial_mask_manifest':str(initial),'diagnostics':records,'probe':bundle,'references':refs,
        'parity':parity,'cached_nodes':_cache(history),'generation_nfe':0 if '100' in _cache(history) else report['total_model_nfe'],
        'trial_id':trial,'prompt_id':history['prompt'][1]}
    return _complete(folder,row)


def probe_references(probe):
    probe=Path(probe);outer=read(probe/'manifest.json');previews=read(probe/'previews.json');refs={}
    for branch in ('base','slider'):
        frame=next(f for f in outer['frames'] if f['branch']==branch and f['step']==3)
        rgb=next(p for p in previews if p['branch']==branch and p['step']==3)
        refs[branch]={k:frame[k] for k in ('input_sha256','sigma','branch','step')}
        refs[branch].update(image=str(probe/rgb['image']),image_sha256=rgb['image_sha256'],prediction_sha256=rgb['prediction_sha256'])
        if sha(refs[branch]['image'])!=refs[branch]['image_sha256']:raise ValueError('Changed branch RGB')
    if refs['base']['input_sha256']!=refs['slider']['input_sha256']:raise ValueError('Pilot branches have different inputs')
    return refs


def regions_from_rectangles(reference,spec):
    if spec.get('image_sha256')!=reference['image_sha256'] or sha(reference['image'])!=reference['image_sha256']:
        raise ValueError('Annotation specification belongs to different RGB')
    regions={}
    if set(spec['regions'])-set(PARTS+PROTECTED+('known_background','target_full')):raise ValueError('Unknown annotation role')
    for role in PARTS+PROTECTED+('known_background','target_full'):
        rectangles=spec['regions'].get(role)
        if rectangles is None:regions[role]=None;continue
        values=[False]*4096
        for rectangle in rectangles:
            if len(rectangle)!=4 or any(type(v) is not int for v in rectangle):raise ValueError('Invalid annotation rectangle')
            x0,y0,x1,y1=rectangle
            if not 0<=x0<x1<=64 or not 0<=y0<y1<=64:raise ValueError('Annotation rectangle outside grid')
            for y in range(y0,y1):
                for x in range(x0,x1):values[y*64+x]=True
        regions[role]=values
    validate_regions(regions,4096)
    return regions


def freeze_pilot_annotations(plan,case,label):
    root=Path(plan['output_root']);folder=root/case['case_id'];dest=folder/('annotations_'+label+'.json')
    pilot=read(folder/('pilot_'+label)/'artifacts.json');references=pilot['references']
    if dest.exists():
        saved=read(dest)
        for branch in ('base','slider'):validate_reference(references[branch],saved['references'][branch])
        return saved['regions']
    if label=='D':
        for branch in ('base','slider'):validate_reference(case['reference_D_images'][branch],references[branch])
        views=annotation_views(plan,case)
        basis='RGB hash-identical D references; reused frozen source labels'
        source_sha=sha(Path(plan['source_root'])/'annotations/manifest.json')
    else:
        spec_path=root/('assistant-label-spec.json' if 'reference_A_report' in case else 'holdout-label-spec.json')
        if not spec_path.exists():raise ValueError('awaiting_annotation: inspect new A RGB and save label specification')
        specs=read(spec_path)['cases'][case['case_id']]['A']
        views={b:regions_from_rectangles(references[b],specs[b]) for b in ('base','slider')}
        basis='Conservative assistant RGB interiors recorded before candidate overlays; sparse, not full segmentation'
        source_sha=sha(spec_path)
    write_new(dest,{'annotation_schema':1,'case_id':case['case_id'],'pilot':label,'human_verified':False,'revision':1,
        'references':references,'regions':views,'source_spec_sha256':source_sha,'label_basis':basis,
        'limitations':'Unknown/occluded areas stay null; no full-body precision or calibrated confidence claimed'})
    return views


def evaluate_case_candidates(plan,case,label,rows,views):
    root=Path(plan['output_root']);folder=root/case['case_id'];references=read(folder/('pilot_'+label)/'artifacts.json')['references']
    evaluated={}
    for variant,row in rows.items():
        if row['status']!='ok':evaluated[variant]={'status':'failed','reason':row.get('reason',row['branches'])};continue
        masks=read_retained_masks(row['manifest']);raw=load_file(str(Path(row['manifest']).parent/'composition.safetensors'))['raw_target']
        classes=classify_collisions_v2(raw.flatten().bool().tolist(),masks['protected'].flatten().bool().tolist(),views['base'],views['slider'])
        branches={}
        for branch in ('base','slider'):
            data=load_file(row['branches'][branch]['evidence']);target=data['masks.target']
            branches[branch]={'raw':region_metrics(target.flatten().bool().tolist(),views[branch]),
                'after_protection':region_metrics((target.bool()&~masks['protected'].bool()).flatten().tolist(),views[branch]),
                'uncertain_cells':int(data['uncertain_mask'].sum())}
        evaluated[variant]={'status':'ok','metrics':_metrics(masks['target'],views),'branches':branches,
            'collision_schema':2,'class_masks':classes,'class_cells':{k:[i for i,v in enumerate(x) if v] for k,x in classes.items()},
            'class_counts':{k:sum(x) for k,x in classes.items()},'references':references,'candidate_manifest_sha256':sha(row['manifest'])}
    dest=folder/('evaluation_'+label+'.json')
    if dest.exists():
        if read(dest)!=evaluated:raise ValueError('Saved evaluation differs from current evidence')
    else:write_new(dest,evaluated)
    return evaluated


def run_replay_case(plan,case,label,variant,row,offset,*,phase='replay',extra=None):
    if phase not in ('replay','release'):raise ValueError('Unsupported fixed replay phase')
    root=Path(plan['output_root']);name=label+'-'+variant;folder=root/case['case_id']/(phase+'_'+name);identity=_identity(case,phase,name)
    resumed=verified_resume(folder/'artifacts.json',identity)
    if resumed:return resumed
    prefix=_prefix(plan,case,phase+'_'+name);trial=(plan['trial_id']+offset)%2147483648
    graph=build_pilot_graph(case,Path(row['manifest']),trial_id=trial,prefix=prefix)
    graph.pop('105');graph['100']['class_type']='Krea2SliderFusePredictionMixSampler';graph['100']['inputs'].pop('background_phrase')
    history=execute_graph(plan,folder,graph)
    records=retain_edits(folder,[{'label':name,'status':'queued','sampler_node':'100','save_node':'102'}],history)
    r,t=checked_diagnostic(records[0]['local_report']);verify_edit(Path(records[0]['local_report']));validate_generation_controls(r)
    masks=read_retained_masks(row['manifest']);pilot=read(root/case['case_id']/('pilot_'+label)/'artifacts.json')
    pr,pt=checked_diagnostic(pilot['diagnostics'][0]['local_report'])
    same={k:r[k]==pr[k] for k in ('initial_noise_sha256','conditioning_sha256','full_sigmas_sha256','lora_sha256')}
    same.update({role:torch.equal(t['reference_'+role+'_mask'],masks[role]) for role in ROLES})
    same.update({k:torch.equal(t[k][0],pt[k][0]) for k in ('trace_base_predictions','trace_slider_predictions')})
    if not all(same.values()) or not r['owned_hooks_removed'] or r['total_model_nfe']!=16:raise ValueError('Replay input/routing integrity failure')
    return _complete(folder,{'identity':identity,'mask_manifest':row['manifest'],'candidate_manifest_sha256':sha(row['manifest']),
        'diagnostics':records,'numeric_checks':same,'trial_id':trial,'cached_nodes':_cache(history),
        'generation_nfe':0 if '100' in _cache(history) else r['total_model_nfe'],'prompt_id':history['prompt'][1],**(extra or {})})


def run_replays(plan):
    root=Path(plan['output_root']);selected=read(root/'selection.json')['selected_variant'];output=[]
    for i,case in enumerate(plan['cases']):
        folder=root/case['case_id'];bootstrap=read_retained_masks(read(folder/'bootstrap/artifacts.json')['mask_manifest'])
        for j,label in enumerate(('D','A')):
            views=freeze_pilot_annotations(plan,case,label)
            index=folder/('scoring_'+label)/'index.json'
            if index.exists():rows=read(index)['variants']
            else:
                pilot=read(folder/('pilot_'+label)/'artifacts.json')
                rows,_=save_scoring(root,case,label,pilot['probe']['local'],bootstrap,views)
            evaluate_case_candidates(plan,case,label,rows,views)
            for k,variant in enumerate(['R0']+([selected] if selected else [])):
                if rows[variant]['status']!='ok':
                    output.append({'case_id':case['case_id'],'label':label+'-'+variant,'status':'failed','generation_nfe':0,'reason':'candidate invalid'});continue
                output.append(run_replay_case(plan,case,label,variant,rows[variant],30+i*8+j*2+k))
    return output


def combine_gate_review(review,image_sha256,numeric_pass):
    if review.get('image_sha256')!=image_sha256:raise ValueError('Gate review image identity mismatch')
    if not isinstance(review.get('notes'),str) or not review['notes'].strip():raise ValueError('Explicit visual notes required')
    gates=review.get('gates',{})
    if set(gates)!=set('G'+str(i) for i in range(1,6)) or any(v not in ('pass','fail','unknown','not_applicable') for v in gates.values()):raise ValueError('Explicit G1–G5 verdicts required')
    if review.get('major_anatomy_defect') not in (True,False,None):raise ValueError('Anatomy defect must be bool/null')
    return {**review,'gates':{'G0':'pass' if numeric_pass else 'fail',**gates},'human_verified':False}


def run_gates(experiment_path,visual_review):
    plan=read(experiment_path);root=Path(plan['output_root'])
    if visual_review is None:raise ValueError('awaiting_visual_review: review file required')
    review=read(visual_review)
    if review.get('experiment_sha256')!=sha(experiment_path):raise ValueError('Visual review belongs to different experiment')
    selection=read(root/'selection.json');gates={'experiment_sha256':sha(experiment_path),'selection_sha256':sha(root/'selection.json'),
        'visual_review_sha256':sha(visual_review),'cases':{}}
    for case in plan['cases']:
        cid=case['case_id'];gates['cases'][cid]={}
        for label,row in review['cases'].get(cid,{}).items():
            artifact=read(root/cid/('replay_'+label)/'artifacts.json');report=read(artifact['diagnostics'][0]['local_report'])
            pilot_label,variant=label.split('-');pilot=read(root/cid/('pilot_'+pilot_label)/'artifacts.json')
            for phase in ('pilot_'+pilot_label,'replay_'+label):
                trace=read(root/cid/('trace_'+phase)/'artifacts.json')
                if trace.get('status')!='complete' or trace.get('generation_nfe')!=0:raise ValueError('Trace evidence must precede gate review')
            numeric=all(artifact['numeric_checks'].values()) and all(pilot['parity'].values())
            gates['cases'][cid][label]=combine_gate_review(row,report['artifact_sha256']['image'],numeric)
    eligibility=evaluate_holdout_eligibility(selection,gates);gates['holdout']=eligibility
    write_new(root/'gates.json',gates)
    return gates


def run_trace_case(plan,case):
    root=Path(plan['output_root']);case_folder=root/case['case_id'];output=[]
    for phase in sorted([p for p in case_folder.iterdir() if p.is_dir() and (p.name.startswith('pilot_') or p.name.startswith('replay_') or p.name=='release_D-R0')]):
        artifact_path=phase/'artifacts.json'
        if not artifact_path.exists():continue
        artifact=read(artifact_path)
        if artifact.get('status')!='complete':continue
        folder=case_folder/('trace_'+phase.name);identity=_identity(case,'traces',phase.name)
        resumed=verified_resume(folder/'artifacts.json',identity)
        if resumed:output.append(resumed);continue
        prefix=_prefix(plan,case,'trace_'+phase.name);graph={s:copy.deepcopy(case['base_graph'][s]) for s in ('1','2','9')}
        node(graph,'100','Krea2SliderFuseTraceDecode',{'model':['2',0],'vae':['9',0],
            'report_path':artifact['diagnostics'][0]['local_report'],'filename_prefix':prefix},'Decode retained trace; zero diffusion NFE')
        history=execute_graph(plan,folder,graph);bundle=retain_bundle(folder/'bundle',prefix);manifest=read(folder/'bundle/manifest.json')
        if not manifest['complete'] or not manifest['final_decoded_rgb_equal'] or manifest['diffusion_model_nfe']!=0:raise ValueError('Trace final RGB mismatch')
        output.append(_complete(folder,{'identity':identity,'bundle':bundle,'source_report':artifact['diagnostics'][0]['local_report'],
            'generation_nfe':0,'cached_nodes':_cache(history),'prompt_id':history['prompt'][1]}))
    return output


def run_release_case(plan,case,offset):
    root=Path(plan['output_root']);folder=root/case['case_id'];dest=folder/'release_D-R0'
    evaluation=read(folder/'evaluation_D.json')['R0'];review_path=folder/'release-review.json'
    if not review_path.exists():raise ValueError('awaiting_release_review: inspect each D-R0 collision on both RGB')
    release=validate_release_review(evaluation['class_masks'],read(review_path),evaluation['references'])
    if not any(release):
        dest.mkdir(exist_ok=True)
        row={'status':'not_applicable','reason':'no_individually_confirmed_releasable_collision','generation_nfe':0,
            'release_review_sha256':sha(review_path),'class_counts':evaluation['class_counts']}
        if not (dest/'artifacts.json').exists():write_new(dest/'artifacts.json',row)
        return row
    identity=_identity(case,'release','D-R0');resumed=verified_resume(dest/'artifacts.json',identity)
    if resumed:return resumed
    candidate=read(folder/'scoring_D/index.json')['variants']['R0'];prior=read_retained_masks(candidate['manifest'])
    raw=load_file(str(Path(candidate['manifest']).parent/'composition.safetensors'))['raw_target']
    L=torch.tensor(release,dtype=torch.float32).reshape(1,64,64);values=apply_diagnostic_release(raw,prior['protected'],L)
    if not torch.equal(values['target']-prior['target'],L) or not torch.equal(prior['protected']-values['protected'],L):raise ValueError('Release changed cells outside L')
    mask_folder=folder/'release_candidate'
    if (mask_folder/'manifest.json').exists():entry={'manifest':str(mask_folder/'manifest.json')}
    else:
        entry=save_attribution_candidate(mask_folder,'D-R0-release',{**values,'release_mask':L,'raw_target':raw},
            {'release_review_sha256':sha(review_path),'source_candidate_manifest_sha256':sha(candidate['manifest']),
             'method':'individually reviewed diagnostic only; excluded from method selection and holdout'})
    # A separate phase stores exactly one generation and one NFE accounting row.
    return run_replay_case(plan,case,'D','R0',entry,offset,phase='release',extra={
        'release_review_sha256':sha(review_path),'released_cells':[i for i,v in enumerate(release) if v]})


def run_holdout(experiment_path):
    plan=read(experiment_path);root=Path(plan['output_root']);selection=read(root/'selection.json');gates=read(root/'gates.json')
    if gates['selection_sha256']!=sha(root/'selection.json'):raise ValueError('Selection changed after diagnostic gates')
    eligibility=evaluate_holdout_eligibility(selection,gates)
    frozen={'selection_sha256':sha(root/'selection.json'),'gates_sha256':sha(root/'gates.json'),
        'cases':plan['holdout_cases'],'eligibility':eligibility}
    conditions=root/'holdout-conditions.json'
    if conditions.exists():
        if read(conditions)!=frozen:raise ValueError('Holdout conditions changed; no reselection allowed')
    else:write_new(conditions,frozen)
    if (root/'holdout.json').exists():
        saved=read(root/'holdout.json')
        if saved['conditions_sha256']!=sha(conditions):raise ValueError('Changed held-out generation identity')
        for row in saved['cases']:
            if row['status']!='generated':continue
            phases=[('bootstrap','bootstrap','base'),('pilot_A','pilot','A'),('trace_pilot_A','traces','pilot_A')]
            for variant in ['R0']+([selection['selected_variant']] if selection['selected_variant'] else []):
                phases.extend([('replay_A-'+variant,'replay','A-'+variant),('trace_replay_A-'+variant,'traces','replay_A-'+variant)])
            for folder,phase,variant in phases:
                if verified_resume(root/row['case_id']/folder/'artifacts.json',_identity(row,phase,variant)) is None:
                    raise ValueError('Missing completed holdout evidence; never resubmit it implicitly')
        return saved
    if not eligibility['eligible']:
        result={'status':'not_applicable','reason':'diagnostic_gate_not_passed','eligibility':eligibility,
            'conditions_sha256':sha(conditions),'generation_nfe':0,'cases':[{**c,'status':'not_applicable','reason':'diagnostic_gate_not_passed'} for c in plan['holdout_cases']]}
        if not (root/'holdout.json').exists():write_new(root/'holdout.json',result)
        return result
    template={c['prompt_case']:c for c in plan['cases']};results=[]
    for i,row in enumerate(plan['holdout_cases']):
        case=copy.deepcopy(template[row['prompt_case']]);case.update(row)
        for key in list(case):
            if key.startswith('reference_') or key=='source_case':del case[key]
        # Search retained completed diagnostic reports, not future planned seed lists.
        used=[];prompt=case['base_graph']['4']['inputs'].get('text')
        for report_path in HERE.glob('**/edits/**/*.json'):
            if report_path.resolve().is_relative_to(root):continue
            report=read(report_path)
            if (report.get('complete') is True and report.get('seed')==case['seed']
                    and report.get('effective_slider_strength')==4 and report.get('prompt')==prompt
                    and report.get('lora_sha256')==case['asset_sha256']['slider']):used.append(str(report_path))
        if used:
            results.append({**row,'status':'not_applicable','reason':'previously_used','matched_reports':used});continue
        try:
            boot=run_bootstrap_case(plan,case,100+i*8);pilot=run_pilot_case(plan,case,'A',101+i*8)
            folder=root/case['case_id'];bootstrap=read_retained_masks(boot['mask_manifest']);index=folder/'scoring_A/index.json'
            if index.exists():rows=read(index)['variants']
            else:rows,_=save_scoring(root,case,'A',pilot['probe']['local'],bootstrap,None)
            variants=['R0']+([selection['selected_variant']] if selection['selected_variant'] else [])
            replays=[]
            for j,variant in enumerate(variants):
                if rows[variant]['status']!='ok':replays.append({'status':'failed','variant':variant,'generation_nfe':0});continue
                replays.append(run_replay_case(plan,case,'A',variant,rows[variant],102+i*8+j))
            traces=run_trace_case(plan,case)
            # Missing labels retain a pending assessment without repeating generation on resume.
            try:
                views=freeze_pilot_annotations(plan,case,'A');evaluate_case_candidates(plan,case,'A',rows,views)
                assessment='awaiting_visual_review'
            except (ValueError,KeyError) as exc:
                if 'awaiting_annotation' not in str(exc) and not isinstance(exc,KeyError):raise
                assessment='awaiting_annotation'
            results.append({**row,'status':'generated','assessment':assessment,'generation_nfe':boot['generation_nfe']+pilot['generation_nfe']+sum(r['generation_nfe'] for r in replays)})
        except (ValueError,RuntimeError) as exc:
            results.append({**row,'status':'failed','reason':str(exc),'generation_nfe':None})
    result={'status':'executed','conditions_sha256':sha(conditions),'cases':results,
        'generation_nfe':sum(r.get('generation_nfe') or 0 for r in results),'unknown_failure_nfe':any(r.get('generation_nfe') is None and r['status']=='failed' for r in results)}
    if not (root/'holdout.json').exists():write_new(root/'holdout.json',result)
    elif read(root/'holdout.json')!=result:raise ValueError('Holdout result changed; use a new assessment revision, never overwrite')
    return result


def assess_holdout(experiment_path,visual_review_path):
    """Finish held-out assessment without changing conditions or rerunning inference."""
    plan=read(experiment_path);root=Path(plan['output_root']);generation=read(root/'holdout.json');review=read(visual_review_path)
    if review.get('experiment_sha256')!=sha(experiment_path):raise ValueError('Holdout review experiment mismatch')
    templates={c['prompt_case']:c for c in plan['cases']};cases={}
    for row in generation['cases']:
        cid=row['case_id']
        if row['status']!='generated':cases[cid]={'status':row['status'],'reason':row.get('reason')};continue
        case=copy.deepcopy(templates[row['prompt_case']]);case.update(prompt_case=row['prompt_case'],case_id=cid,seed=row['seed'])
        for key in list(case):
            if key.startswith('reference_') or key=='source_case':del case[key]
        views=freeze_pilot_annotations(plan,case,'A');index=read(root/cid/'scoring_A/index.json')['variants']
        evaluation=evaluate_case_candidates(plan,case,'A',index,views)
        labels={}
        for label,visual in review['cases'].get(cid,{}).items():
            artifact=read(root/cid/('replay_'+label)/'artifacts.json');report=read(artifact['diagnostics'][0]['local_report'])
            pilot=read(root/cid/'pilot_A/artifacts.json')
            numeric=all(artifact['numeric_checks'].values()) and pilot['status']=='complete'
            labels[label]=combine_gate_review(visual,report['artifact_sha256']['image'],numeric)
        required='A-'+read(root/'holdout-conditions.json')['eligibility']['required_variant']
        if required not in labels:raise ValueError('awaiting_visual_review: required holdout variant missing')
        assessment={'status':'assessed','case_id':cid,'human_verified':False,'labels':labels,
            'evaluation_sha256':sha(root/cid/'evaluation_A.json'),'visual_review_sha256':sha(visual_review_path)}
        write_new(root/cid/'holdout-assessment.json',assessment);cases[cid]=assessment
    result={'status':'assessed','experiment_sha256':sha(experiment_path),'conditions_sha256':sha(root/'holdout-conditions.json'),
        'generation_sha256':sha(root/'holdout.json'),'visual_review_sha256':sha(visual_review_path),'human_verified':False,'cases':cases,
        'mode':read(root/'holdout-conditions.json')['eligibility']['mode']}
    write_new(root/'holdout-assessment.json',result)
    return result


def run_live_stage(experiment_path,stage,visual_review=None):
    plan=read(experiment_path);validate_frozen_sources(plan);root=Path(plan['output_root'])
    if not (root/'selection.json').exists():raise ValueError('Offline selection must be frozen before live execution')
    if read(root/'selection.json')['experiment_sha256']!=sha(experiment_path):raise ValueError('Frozen experiment identity changed')
    if stage=='bootstrap':
        rows=[run_bootstrap_case(plan,c,i+1) for i,c in enumerate(plan['cases'])]
    elif stage=='pilots':
        rows=[]
        for i,c in enumerate(plan['cases']):
            for j,label in enumerate(('D','A')):rows.append(run_pilot_case(plan,c,label,10+i*2+j))
    elif stage=='replay':rows=run_replays(plan)
    elif stage=='traces':rows=[r for c in plan['cases'] for r in run_trace_case(plan,c)]
    elif stage=='gates':return run_gates(experiment_path,visual_review)
    elif stage=='release':rows=[run_release_case(plan,c,70+i) for i,c in enumerate(plan['cases'])]
    elif stage=='holdout':return run_holdout(experiment_path)
    else:raise ValueError('Unknown stage: '+stage)
    summary=root/(stage+'.summary.json')
    if not summary.exists():write_new(summary,rows)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('freeze','offline','bootstrap','pilots','replay','gates','release','holdout','traces'))
    parser.add_argument('--source-root',type=Path);parser.add_argument('--output-root',type=Path);parser.add_argument('--trial-id',type=int)
    parser.add_argument('--server',default='http://127.0.0.1:8000');parser.add_argument('--experiment-plan',type=Path)
    parser.add_argument('--visual-review',type=Path)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.stage=='freeze':
        if args.source_root is None or args.output_root is None or args.trial_id is None:parser.error('freeze needs source/output/trial')
        result=freeze_experiment(args.source_root,args.output_root,args.trial_id,args.server);print(Path(result['output_root'])/'experiment.json')
    elif args.experiment_plan is None:parser.error('experiment plan required')
    elif args.stage=='offline':run_offline(args.experiment_plan)
    else:run_live_stage(args.experiment_plan,args.stage,args.visual_review)


if __name__=='__main__':main()
