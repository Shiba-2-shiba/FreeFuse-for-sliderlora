"""Reconstruct saved seed controls, routing, release proofs and inference costs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from PIL import Image
from safetensors.torch import load_file

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'tools'))
from run_seed_bootstrap_validation import (read,sha,write_new,validate_frozen_sources,validate_generation_controls,
    annotation_views,_metrics,select_variant,evaluate_holdout_eligibility,regions_from_rectangles,ROLES,PARITY_KEYS)
from audit_attention_ablation_live import verify_edit,verify_suite
from slider_fuse.paired_observation import checked_diagnostic,read_retained_masks
from slider_fuse.anchored_masks import anchored_candidate,apply_diagnostic_release
from slider_fuse.experimental_masks import _labels
from slider_fuse.diagnostics import diagnostic_provenance
from slider_fuse.sampling import tensor_hash
from build_paired_mask_candidates import logits
from audit_attention_step_block_sweep import safe_tap
from evaluate_paired_mask_attribution import region_metrics,validate_release_review,validate_reference


def validate_run_files(row):
    if row.get('status')!='complete':raise ValueError('Missing complete run evidence')
    for path,expected in row.get('file_sha256',{}).items():
        if sha(path)!=expected:raise ValueError('Changed run evidence: '+path)
    if not row.get('file_sha256'):raise ValueError('Missing run file identities')
    if not all(row.get('numeric_checks',{}).values()):raise ValueError('Run numeric checks failed')
    nfe=row.get('generation_nfe')
    if type(nfe) is not int or nfe<0:raise ValueError('Missing measured NFE')
    return nfe


def validate_holdout_record(record,eligibility,planned_cases):
    if len(planned_cases)!=len(set(planned_cases)):raise ValueError('Duplicate planned case')
    rows=record.get('cases',[])
    if sorted(r['case_id'] for r in rows)!=sorted(planned_cases):raise ValueError('Holdout case coverage differs')
    if record.get('status')=='not_applicable':
        if eligibility['eligible'] or record.get('reason')!='diagnostic_gate_not_passed':raise ValueError('Holdout eligibility contradicts skip')
        if record.get('generation_nfe')!=0 or any(r.get('status')!='not_applicable' for r in rows):raise ValueError('Skipped holdout has generation')
        return 'not_applicable'
    if not eligibility['eligible']:raise ValueError('Ineligible holdout executed')
    if record.get('status')!='executed':raise ValueError('Holdout status missing')
    for row in rows:
        if row.get('status') not in ('generated','failed','not_applicable'):raise ValueError('Holdout terminal case status missing')
        if row['status'] in ('failed','not_applicable') and not row.get('reason'):raise ValueError('Holdout reason missing')
    return 'executed_with_failure' if any(r['status']=='failed' for r in rows) else 'executed'


def audit_scoring(plan,case,label):
    root=Path(plan['output_root']);folder=root/case['case_id'];index=read(folder/('scoring_'+label)/'index.json');rows=index['variants']
    views=read(folder/('annotations_'+label+'.json'))['regions']
    if label=='D':probe=Path(case['reference_D_probe'])
    else:probe=Path(read(folder/'pilot_A/artifacts.json')['probe']['local'])
    outer=read(probe/'manifest.json');contexts=[]
    for branch in ('base','slider'):
        for block in (8,10):
            _,maps=verify_suite(safe_tap(probe,outer['taps'][f'{branch}.step_3.block_{block}']))
            contexts.append(logits(maps,3,block).softmax(-1))
    stacked=torch.stack(contexts);anchor=stacked.mean(0);anchor[(stacked.argmax(-1)!=anchor.argmax(-1)[None]).any(0)]=1/3
    bootstrap=read_retained_masks(read(folder/'bootstrap/artifacts.json')['mask_manifest'])
    count=0
    for variant,row in rows.items():
        if row['status']!='ok':continue
        targets=[]
        for branch in ('base','slider'):
            entry=row['branches'][branch];stats=read(entry['statistics']);data=load_file(entry['evidence'])
            if sha(entry['evidence'])!=stats['evidence_sha256']:raise ValueError('Changed raw scores')
            _,maps=verify_suite(safe_tap(probe,outer['taps'][f'{branch}.step_3.block_18']))
            features=maps['evidence.capture.attention_output'][None];late=logits(maps,3,18);cfg=stats['config']
            normalized=torch.nn.functional.normalize(features.float()[0],dim=-1)
            scores=torch.stack([(normalized@data['centers.'+r].T).amax(-1) for r in ROLES],-1)
            expected=(scores/cfg['feature_temperature']+(late/cfg['context_temperature']).softmax(-1).clamp_min(1e-12).log()).softmax(-1)
            if not torch.equal(expected,data['prelock_scores']):raise ValueError('Prelock scores do not reconstruct')
            locks=[]
            for i,role in enumerate(ROLES):
                lock=data['locks.'+role];locks.append(lock)
                expected[lock]=torch.nn.functional.one_hot(torch.tensor(i),3).to(expected)
                if stats['lock_sha256'][role]!=tensor_hash(lock):raise ValueError('Lock identity changed')
                if stats['prototype_sha256'][role]!=tensor_hash(data['centers.'+role]):raise ValueError('Prototype identity changed')
            if torch.stack(locks).sum(0).max()>1 or not torch.equal(expected,data['scores']):raise ValueError('Lock clamping failed')
            labels=_labels(expected,cfg)
            for i,role in enumerate(ROLES):
                mask=(labels==i).reshape(1,64,64).float() if role!='background' else 1-data['masks.target']-data['masks.protected']
                if not torch.equal(mask,data['masks.'+role]):raise ValueError('Raw masks do not reconstruct from scores')
            if not torch.equal((labels==-1).reshape(1,64,64).float(),data['uncertain_mask']):raise ValueError('Uncertain cells changed')
            if variant=='R0':
                legacy=anchored_candidate(features,late,anchor,cfg)
                if not torch.equal(legacy['scores'],data['scores']) or legacy['prototype_indices']!=stats['prototype_indices']:raise ValueError('Independent R0 parity failed')
                if not all(torch.equal(legacy['seeds'][r],data['locks.'+r]) for r in ROLES):raise ValueError('R0 seed parity failed')
            if 'metrics' in entry and region_metrics(data['masks.target'].flatten().bool().tolist(),views[branch])!=entry['metrics']:raise ValueError('Saved raw metrics differ')
            targets.append(data['masks.target'].bool());count+=1
        raw=targets[0]|targets[1];target=raw&~bootstrap['protected'].bool();masks=read_retained_masks(row['manifest'])
        if not torch.equal(target.float(),masks['target']) or not torch.equal(bootstrap['protected'],masks['protected']):raise ValueError('Final paired composition differs')
        composition=load_file(str(Path(row['manifest']).parent/'composition.safetensors'))
        if not torch.equal(raw.float(),composition['raw_target']):raise ValueError('Historical target support was added')
        if 'metrics' in row and _metrics(masks['target'],views)!=row['metrics']:raise ValueError('Saved final metrics differ')
        if not torch.equal(masks['target']+masks['protected']+masks['background'],torch.ones_like(masks['target'])):raise ValueError('Partition failure')
    return rows,count


def audit(experiment_path):
    torch.set_num_threads(4);experiment_path=Path(experiment_path);plan=read(experiment_path)
    if plan.get('seed_bootstrap_schema')!=1:raise ValueError('Unsupported seed experiment')
    validate_frozen_sources(plan);root=Path(plan['output_root']);baseline=read(root/'baseline-recheck.json')
    if not baseline.get('R0_exact_saved_parity'):raise ValueError('Baseline R0 parity evidence missing')
    selection=read(root/'selection.json');gates=read(root/'gates.json');holdout=read(root/'holdout.json')
    if selection['experiment_sha256']!=sha(experiment_path) or gates['experiment_sha256']!=sha(experiment_path):raise ValueError('Experiment identity mismatch')
    if gates['selection_sha256']!=sha(root/'selection.json'):raise ValueError('Selection was changed')
    eligibility=evaluate_holdout_eligibility(selection,gates)
    if eligibility!=gates['holdout']:raise ValueError('Eligibility does not independently reconstruct')
    holdout_status=validate_holdout_record(holdout,eligibility,[c['case_id'] for c in plan['holdout_cases']])
    if holdout['conditions_sha256']!=sha(root/'holdout-conditions.json'):raise ValueError('Holdout condition identity mismatch')
    assets=read(root/'asset-verification.json')
    if assets['experiment_sha256']!=sha(experiment_path) or sha(assets['prior_path'])!=assets['prior_sha256']:raise ValueError('Asset verification source mismatch')
    prior_assets=read(assets['prior_path'])['assets']
    for role,asset in assets['assets'].items():
        stat=Path(asset['path']).stat()
        if (asset['sha256']!=prior_assets[role]['sha256'] or stat.st_size!=asset['size_bytes']
                or stat.st_mtime_ns!=asset['mtime_ns']):raise ValueError('Runtime asset changed after fresh hash: '+role)
    result={'seed_bootstrap_verification_schema':1,'experiment_sha256':sha(experiment_path),'source_integrity':True,
        'human_verified':False,'cases':[],'diffusion_model_nfe':0,'verified_raw_banks':0,'final_generation_images':0,
        'verified_trace_bundles':0,'holdout_status':holdout_status,'holdout_eligibility':eligibility,'selected_variant':selection['selected_variant'],
        'asset_sha256':{k:v['sha256'] for k,v in assets['assets'].items()},
        'quality_approval':'Separate gate verdicts and assistant observations; no default adoption'}
    drows={};all_cases=list(plan['cases']);templates={c['prompt_case']:c for c in plan['cases']}
    if holdout_status!='not_applicable':
        assessment=read(root/'holdout-assessment.json')
        if (assessment['conditions_sha256']!=sha(root/'holdout-conditions.json')
                or assessment['generation_sha256']!=sha(root/'holdout.json') or assessment['status']!='assessed'):
            raise ValueError('Held-out assessment identity differs')
        result['holdout_assessment']=assessment
        for row in holdout['cases']:
            if not (root/row['case_id']).is_dir():continue
            case={**templates[row['prompt_case']],**row}
            for key in list(case):
                if key.startswith('reference_') or key=='source_case':del case[key]
            all_cases.append(case)
    for case in all_cases:
        cid=case['case_id'];folder=root/cid;case_result={'case_id':cid,'runs':[],'images':[],'nfe':0}
        boot=read(folder/'bootstrap/artifacts.json');validate_run_files(boot)
        bm=read_retained_masks(boot['mask_manifest']);diagnostic='reference_A_report' in case
        if diagnostic:
            _,a=checked_diagnostic(case['reference_A_report']);_,d=checked_diagnostic(case['reference_D_report'])
            if not all(torch.equal(bm[r],a['reference_'+r+'_mask']) for r in ROLES[:2]) or not torch.equal(bm['protected'],d['reference_protected_mask']):raise ValueError('Bootstrap A/D parity failed')
        bootstrap_graph=read(folder/'bootstrap/queued_api.json')['prompt']
        if (bootstrap_graph['8']['inputs']['model']!=['2',0] or 'target_mask' in bootstrap_graph['6']['inputs']
                or any(n['class_type']=='Krea2SliderFuseRetainedMask' for n in bootstrap_graph.values())
                or set(bootstrap_graph)!=set(('1','2','3','4','5','6','7','8','9','18'))):raise ValueError('Bootstrap has forbidden upstream dependency')
        for label in (('D','A') if diagnostic else ('A',)):
            rows,count=audit_scoring(plan,case,label);result['verified_raw_banks']+=count
            if label=='D':drows[cid]=rows
            annotation=read(folder/('annotations_'+label+'.json'));pilot=read(folder/('pilot_'+label)/'artifacts.json')
            for branch in ('base','slider'):validate_reference(pilot['references'][branch],annotation['references'][branch])
            if label=='D':expected_views=annotation_views(plan,case)
            else:
                spec_path=root/('assistant-label-spec.json' if diagnostic else 'holdout-label-spec.json');specs=read(spec_path)['cases'][cid]['A']
                if annotation['source_spec_sha256']!=sha(spec_path):raise ValueError('RGB annotation specification changed')
                expected_views={b:regions_from_rectangles(pilot['references'][b],specs[b]) for b in ('base','slider')}
            if expected_views!=annotation['regions'] or annotation['human_verified'] is not False:raise ValueError('Annotations do not reproduce from frozen RGB specification')
        for run_folder in sorted(folder.iterdir()):
            artifact_path=run_folder/'artifacts.json'
            if not run_folder.is_dir() or not artifact_path.exists():continue
            row=read(artifact_path)
            if row['status']=='not_applicable':
                if row.get('generation_nfe')!=0:raise ValueError('Skipped phase has nonzero NFE')
                case_result['runs'].append({'phase':run_folder.name,'status':'not_applicable','reason':row['reason']});continue
            nfe=validate_run_files(row);case_result['nfe']+=nfe
            history=read(run_folder/'history.json');graph=read(run_folder/'queued_api.json')['prompt']
            if history['status']['status_str']!='success' or history['prompt'][2]!=graph:raise ValueError('History differs from submitted graph')
            cached={str(n) for kind,msg in history['status']['messages'] if kind=='execution_cached' for n in msg.get('nodes',[])}
            if sorted(cached)!=row['cached_nodes']:raise ValueError('Cache accounting differs')
            phase=row['identity']['phase'];variant=row['identity']['variant']
            if phase in ('pilot','replay','release'):
                record=row['diagnostics'][0];path=Path(record['local_report']);report=verify_edit(path);_,t=checked_diagnostic(path)
                validate_generation_controls(report)
                if report['generation']!=diagnostic_provenance(graph,record['save_node'],report):raise ValueError('Routing provenance differs')
                if not report['owned_hooks_removed'] or report['total_model_nfe']!=16 or report['trial_id']!=row['trial_id']:raise ValueError('Inference control mismatch')
                if nfe!=(0 if record['sampler_node'] in cached else 16):raise ValueError('Inference NFE differs')
                if phase=='pilot':
                    if diagnostic:
                        old=d if variant=='D' else a
                        if not all(torch.equal(t[k],old[k]) for k in PARITY_KEYS):raise ValueError('Pilot exact source parity failed')
                    else:
                        outer=read(Path(row['probe']['local'])/'manifest.json')
                        for frame in outer['frames']:
                            step=frame['step']-1
                            if frame['input_sha256']!=tensor_hash(t['trace_inputs'][step]) or frame['sigma']!=float(t['trace_sigmas'][step]):raise ValueError('Passive holdout observer input changed')
                            for tap in frame['taps']:
                                if tap['tensor_sha256_before']!=tap['tensor_sha256_after'] or tap['block_input_before_sha256']!=tap['block_input_after_sha256']:raise ValueError('Passive observer mutated tap input')
                    if variant=='A' and graph['60']['inputs']['report_path']!=boot['mask_manifest']:raise ValueError('A pilot initial mask has D dependency')
                else:
                    masks=read_retained_masks(row['mask_manifest'])
                    if sha(row['mask_manifest'])!=row['candidate_manifest_sha256']:raise ValueError('Replay candidate changed')
                    if not all(torch.equal(t['reference_'+r+'_mask'],masks[r]) for r in ROLES):raise ValueError('Sampler used different masks')
                    pilot_label=variant.split('-')[0];pilot=read(folder/('pilot_'+pilot_label)/'artifacts.json');pr,pt=checked_diagnostic(pilot['diagnostics'][0]['local_report'])
                    if any(report[k]!=pr[k] for k in ('initial_noise_sha256','conditioning_sha256','full_sigmas_sha256','lora_sha256')):raise ValueError('Replay input conditions changed')
                    if not all(torch.equal(t[k][0],pt[k][0]) for k in ('trace_base_predictions','trace_slider_predictions')):raise ValueError('Replay first branch predictions changed')
                if phase=='release':
                    evaluation=read(folder/'evaluation_D.json')['R0'];review=read(folder/'release-review.json')
                    L=torch.tensor(validate_release_review(evaluation['class_masks'],review,evaluation['references']),dtype=torch.float32).reshape(1,64,64)
                    source=read(folder/'scoring_D/index.json')['variants']['R0'];old_masks=read_retained_masks(source['manifest'])
                    raw=load_file(str(Path(source['manifest']).parent/'composition.safetensors'))['raw_target']
                    expected=apply_diagnostic_release(raw,old_masks['protected'],L)
                    if not all(torch.equal(expected[r],t['reference_'+r+'_mask']) for r in ROLES):raise ValueError('Diagnostic release proof differs')
                    if row['release_review_sha256']!=sha(folder/'release-review.json'):raise ValueError('Release review changed')
                case_result['images'].append({'phase':phase,'label':variant,'diagnostic':str(path),
                    'image':str(path.parent/report['artifacts']['image']),'image_sha256':report['artifact_sha256']['image']})
                result['final_generation_images']+=1
            elif phase=='traces':
                trace_root=Path(row['bundle']['local']);manifest=read(trace_root/'manifest.json');source=Path(row['source_report']);report=read(source)
                if manifest['source_report_sha256']!=sha(source) or not manifest['final_decoded_rgb_equal'] or nfe!=0:raise ValueError('Trace decoding identity mismatch')
                with Image.open(trace_root/'final.png') as x,Image.open(source.parent/report['artifacts']['image']) as y:
                    if x.convert('RGB').tobytes()!=y.convert('RGB').tobytes():raise ValueError('Decoded final RGB differs')
                result['verified_trace_bundles']+=1
            elif phase=='bootstrap':
                manifest,tensors=verify_suite(Path(row['bundle']['local'])/'manifest.json')
                if manifest['report']['phase1_nfe']!=2 or nfe!=(0 if '8' in cached else 2):raise ValueError('Bootstrap NFE differs')
            else:raise ValueError('Unknown inference phase')
            case_result['runs'].append({'phase':run_folder.name,'status':'verified','nfe':nfe,'prompt_id':row['prompt_id']})
        result['diffusion_model_nfe']+=case_result['nfe'];result['cases'].append(case_result)
        print('AUDITED',cid,case_result['nfe'],'NFE',flush=True)
    recalculated=select_variant(drows)
    for key in ('selected_variant','eligible_ranked','rejected'):
        if recalculated[key]!=selection[key]:raise ValueError('Offline selection differs from reconstructed metrics')
    # Held-out runs were independently audited above; do not count their aggregate twice.
    measured_holdout=sum(c['nfe'] for c in result['cases'] if c['case_id'] not in {x['case_id'] for x in plan['cases']})
    if measured_holdout!=holdout['generation_nfe']:raise ValueError('Holdout aggregate NFE differs from audited histories')
    result['holdout_unknown_failure_nfe']=holdout.get('unknown_failure_nfe',False)
    if result['diffusion_model_nfe']>plan['planned_max_diffusion_nfe']:raise ValueError('NFE budget exceeded without amendment')
    result['effective_code_sha256']={str(p.relative_to(HERE.parent)):sha(p) for p in (Path(__file__),HERE/'run_seed_bootstrap_validation.py',HERE/'build_paired_mask_candidates.py',HERE.parent/'slider_fuse/anchored_masks.py',HERE.parent/'tools/evaluate_paired_mask_attribution.py')}
    result['output_file_sha256']={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.png','.safetensors')}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment-plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();write_new(args.output,audit(args.experiment_plan));print(args.output)
