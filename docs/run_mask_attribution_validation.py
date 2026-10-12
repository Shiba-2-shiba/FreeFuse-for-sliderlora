"""Freeze retained evidence and run mask attribution on the active local ComfyUI."""
from __future__ import annotations

import argparse
import copy
import datetime
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from audit_attention_ablation_live import read as _read, sha as _sha, verify_edit, verify_suite
from audit_attention_step_block_sweep import safe_tap
from run_attention_step_block_sweep import run_stage, retain_edits
from run_mask_source_comparison_live import node, rt, BASE
from slider_fuse.paired_observation import checked_diagnostic


def read(path):
    return _read(Path(path))


def sha(path):
    return _sha(Path(path))


def validate_generation_controls(report):
    if (report.get('sampler')!='euler' or report.get('scheduler')!='simple'
            or report.get('cfg')!=1 or report.get('steps')!=8
            or report.get('effective_slider_strength')!=4 or report.get('grid')!=[64,64]
            or report.get('prediction_selection',{}).get('dilate_radius')!=0
            or report.get('mask_generation',{}).get('fill_holes_max_area')!=0
            or report.get('mask_generation',{}).get('mask_dilate_radius')!=0):
        raise ValueError('Generation controls differ from frozen Euler/simple/CFG1/radius0 experiment')


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def freeze_experiment(source_root: Path, output_root: Path, trial_id: int) -> dict:
    source_root = Path(source_root).resolve(); output_root = Path(output_root).resolve()
    if output_root.exists() or output_root.is_symlink() or output_root.is_relative_to(source_root):
        raise ValueError('Output must be a new, non-source experiment directory; existing output refused')
    if type(trial_id) is not int or not 0 <= trial_id <= 0x7fffffff:
        raise ValueError('trial_id must be a nonnegative 31-bit integer')
    summary = read(source_root/'probe.summary.json')
    expected_cases = {'park':444444,'woman_front_strong_overlap':42}
    if {row['case'] for row in summary} != set(expected_cases) or len(summary)!=2:
        raise ValueError('Expected the two frozen diagnostic cases')
    source_files = {}; cases = []; baseline = []
    def retain_hash(path):
        path = Path(path).resolve(); source_files[str(path)] = sha(path)
    retain_hash(source_root/'probe.summary.json')
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    hole = read(HERE/'mask-source-comparison-results'/'20261011T104616Z'/'park'/'head-hole-probe.json')['grid_indices']
    for row in summary:
        cid = row['case']; probe = Path(row['probes']['mix']['local'])
        outer = read(probe/'manifest.json')
        if not outer.get('complete') or sha(probe/'manifest.json') != row['probes']['mix']['manifest_sha256']:
            raise ValueError('Incomplete or changed paired probe')
        expected = {f'{b}.step_{s}.block_{k}' for b in ('base','slider') for s in (1,2,3) for k in (8,10,18,23,26)}
        if set(outer['taps']) != expected: raise ValueError('Missing or extra paired tap')
        retain_hash(probe/'manifest.json')
        for key, item in outer['taps'].items():
            path = safe_tap(probe,item)
            if sha(path) != item['manifest_sha256']: raise ValueError('Changed tap manifest')
            manifest, _ = verify_suite(path)
            cfg = manifest['report']['algorithm']['config']
            if (cfg['prototypes'] != 3 or cfg['context_temperature'] != 1.
                    or cfg['propagation_iterations'] != 1):
                raise ValueError('Unsupported source algorithm config')
            retain_hash(path)
            for name in manifest['file_sha256']: retain_hash(path.parent/name)
        diagnostic = next(r for r in row['diagnostics'] if r['label']=='probe_mix')
        path = Path(diagnostic['local_report']); report = verify_edit(path)
        validate_generation_controls(report)
        _, tensors = checked_diagnostic(path)
        if (report['seed'] != expected_cases[cid] or report['steps']!=8 or report['cfg']!=1
                or report['effective_slider_strength']!=4 or report['grid']!=[64,64]):
            raise ValueError('Source generation controls differ')
        retain_hash(path)
        for name in report['artifacts'].values():
            if name != path.name: retain_hash(path.parent/name)
        previews = read(probe/'previews.json'); retain_hash(probe/'previews.json')
        references = {}
        for branch in ('base','slider'):
            frame = next(f for f in outer['frames'] if f['branch']==branch and f['step']==3)
            preview = next(p for p in previews if p['branch']==branch and p['step']==3)
            image = probe/preview['image']
            if sha(image) != preview['image_sha256']: raise ValueError('Changed preview')
            retain_hash(image)
            references[branch] = {k:frame[k] for k in ('input_sha256','sigma','branch','step')}
            references[branch].update(image=str(image.resolve()),image_sha256=preview['image_sha256'],
                                      prediction_sha256=preview['prediction_sha256'])
        if references['base']['input_sha256'] != references['slider']['input_sha256']:
            raise ValueError('Source branches do not share Mix input')
        prior_graph = HERE/'mask-source-comparison-results'/'20261011T104616Z'/cid/'queued_api.json'
        retain_hash(prior_graph)
        case = {'case':cid,'seed':expected_cases[cid], 'probe_root':str(probe.resolve()),
                'pilot_report':str(path.resolve()),'prior_graph':str(prior_graph.resolve()),'references':references,
                'lora_sha256':report['lora_sha256'],'conditioning_sha256':report['conditioning_sha256'],
                'initial_noise_sha256':report['initial_noise_sha256'],'full_sigmas_sha256':report['full_sigmas_sha256']}
        cases.append(case)
        evidence = {'case':cid,'D_target_cells':int(tensors['reference_target_mask'].sum()),'old_candidates':[]}
        for block in (18,23):
            folder = source_root/cid/'candidates'/f'anchored_s3_b{block}'
            meta = read(folder/'manifest.json'); retain_hash(folder/'manifest.json')
            if sha(folder/meta['tensors']) != meta['tensors_sha256']: raise ValueError('Changed old candidate')
            retain_hash(folder/meta['tensors']); mask = load_file(str(folder/meta['tensors']))['target']
            old = tensors['reference_target_mask']
            entry = {'block':block,'added_cells':int(((mask>0)&(old==0)).sum()),
                     'removed_cells':int(((mask==0)&(old>0)).sum())}
            if cid=='park': entry.update(old_hole_D=int(old.flatten()[hole].sum()),old_hole_candidate=int(mask.flatten()[hole].sum()))
            evidence['old_candidates'].append(entry)
        if cid=='park':
            evidence['old_hole_raw_counts'] = []
            for block in (18,23):
                for branch in ('base','slider'):
                    counts = {}
                    for variant in ('baseline','seed_only','shape_only','combined'):
                        png=source_root/cid/'ablations'/f'{branch}_s3_b{block}_{variant}'/'target.png'
                        retain_hash(png)
                        with Image.open(png) as image: pixels=list(image.getdata())
                        counts[variant]=sum(pixels[i]==255 for i in hole)
                    evidence['old_hole_raw_counts'].append({'block':block,'branch':branch,'counts':counts})
        baseline.append(evidence)
        print('SOURCE_VERIFIED',cid,len(expected),'taps',flush=True)
    output_root.mkdir(parents=True)
    experiment = {'mask_attribution_experiment_schema':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_root':str(source_root),'output_root':str(output_root),'trial_id':trial_id,'server':BASE,
        'cases':cases,'source_files':source_files,'primary_block':18,'step':3,
        'variants':['D','S0','S1','C0','C1'],'planned_max_images':12,'planned_max_diffusion_nfe':192,
        'radius':0,'quality_approval':'not_evaluated','annotation_human_verified':False,
        'source_code_sha256':{str(path.relative_to(HERE.parent)):sha(path) for path in
            (HERE.parent/'slider_fuse'/'anchored_masks.py',HERE/'build_paired_mask_candidates.py',Path(__file__))}}
    write_new(output_root/'baseline-evidence.json',baseline)
    write_new(output_root/'experiment.json',experiment)
    return experiment


def validate_frozen_sources(experiment):
    for path, expected in experiment['source_files'].items():
        if sha(Path(path))!=expected: raise ValueError('Changed frozen source: '+path)


def build_stage_graph(graph, old, entries, *, case, trial_id, stamp, stage):
    labels=[r['label'] for r in entries]
    if not entries or len(labels)!=len(set(labels)) or any(r.get('status')!='ok' for r in entries):
        raise ValueError('Expected unique successful candidates')
    prompt=copy.deepcopy(graph); candidates=[]
    for i,row in enumerate(entries):
        base=100+i*100; sampler,decode,save,loader,subjects=map(str,(base,base+1,base+2,base+3,base+4))
        label=row['label']
        node(prompt,loader,'Krea2SliderFuseRetainedMask',{'report_path':row['manifest']},label+' checked masks')
        sub=copy.deepcopy(old['6']['inputs']);sub.update(target_mask=[loader,0],protected_mask=[loader,1])
        node(prompt,subjects,'Krea2SliderFuseSubjects',sub,label+' subjects')
        inputs=copy.deepcopy(old['21']['inputs'])
        inputs.update(model=['2',0],subjects=[subjects,0],mask_mode='manual',mix_scope='target_mask',
                      trial_id=trial_id,diagnostic_level='audit',selection_dilate_radius=0,
                      fill_holes_max_area=0,mask_dilate_radius=0)
        node(prompt,sampler,'Krea2SliderFusePredictionMixSampler',inputs,label)
        node(prompt,decode,'VAEDecode',{'samples':[sampler,0],'vae':['9',0]},label)
        node(prompt,save,'Krea2SliderFuseDiagnosticSave',{'images':[decode,0],'latent':[sampler,0],
             'diagnostics':[sampler,2],'filename_prefix':f'mask_attribution/{stamp}/{case}/{stage}_{label}'},label)
        candidates.append({'label':stage+'_'+label,'candidate_id':label,'status':'queued',
                           'sampler_node':sampler,'save_node':save})
    return prompt,candidates


def run_experiment_stage(experiment_path, stage):
    experiment=read(experiment_path); validate_frozen_sources(experiment)
    root=Path(experiment['output_root']); index=read(root/('candidate-index-release.json' if stage=='release' else 'candidate-index.json'))
    if index['experiment_sha256']!=sha(experiment_path): raise ValueError('Changed experiment plan')
    summary=[]
    for case in experiment['cases']:
        cid=case['case'];folder=root/cid; artifact=folder/(stage+'.artifacts.json')
        if artifact.exists():
            summary.append(read(artifact)); print('ALREADY_RETAINED',cid,stage,flush=True);continue
        desired={'control':['D'],'factorial':['S0','S1','C0','C1'],'release':['S0-release']}[stage]
        rows=[r for r in index['cases'][cid] if r['label'] in desired and r['status']=='ok']
        if stage=='release' and not rows:
            result={'case':cid,'status':'not_applicable','diagnostics':[],'generation_nfe':0}
            write_new(artifact,result);summary.append(result);continue
        if [r['label'] for r in rows]!=desired: raise ValueError('Missing stage candidates')
        if stage!='control' and not (folder/'control.parity.json').is_file():
            raise ValueError('Fresh D parity must pass before downstream generation')
        old=read(case['prior_graph'])['prompt']
        graph={sid:copy.deepcopy(old[sid]) for sid in ('1','2','3','4','5','7','9')}
        prompt,candidates=build_stage_graph(graph,old,rows,case=cid,trial_id=experiment['trial_id'],stamp=root.name,stage=stage)
        prompt,history=run_stage(folder,stage,prompt)
        kept=retain_edits(folder,candidates,history)
        cached={str(n) for kind,msg in history['status']['messages'] if kind=='execution_cached' for n in msg.get('nodes',[])}
        nfe=sum(read(r['local_report'])['total_model_nfe'] for r in kept if r['sampler_node'] not in cached)
        result={'case':cid,'status':'generated','diagnostics':kept,'cached_nodes':sorted(cached),'generation_nfe':nfe}
        if stage=='control':
            import torch
            _,fresh=checked_diagnostic(kept[0]['local_report']);_,prior=checked_diagnostic(case['pilot_report'])
            keys=('trace_inputs','trace_predictions','trace_base_predictions','trace_slider_predictions','final_latent')
            equal={k:torch.equal(fresh[k],prior[k]) for k in keys}
            if not all(equal.values()):
                write_new(folder/'control.parity-failed.json',equal)
                raise RuntimeError('Fresh D numerical parity failed')
            write_new(folder/'control.parity.json',equal)
        write_new(artifact,result);summary.append(result)
        print('CASE_COMPLETE',cid,stage,nfe,'NFE',flush=True)
    write_new(root/(stage+'.summary.json'),summary)


def run_trace_stage(experiment_path):
    from run_paired_mix_validation import retain_bundle
    experiment=read(experiment_path);root=Path(experiment['output_root']);summary=[]
    for case in experiment['cases']:
        cid=case['case'];folder=root/cid
        old=read(case['prior_graph'])['prompt']
        graph={sid:copy.deepcopy(old[sid]) for sid in ('1','2','9')}
        records=[]
        for stage in ('control','factorial','release'):
            artifact=read(folder/(stage+'.artifacts.json'))
            for record in artifact['diagnostics']:
                label=record['label'].removeprefix(stage+'_');sid=str(100+len(records))
                prefix=f'ma_trace_{root.name.replace("-","_")}_{cid}_{label}'
                node(graph,sid,'Krea2SliderFuseTraceDecode',{'model':['2',0],'vae':['9',0],
                     'report_path':record['local_report'],'filename_prefix':prefix},label+' retained predictions')
                records.append((label,prefix,record['local_report']))
        _,history=run_stage(folder,'traces',graph)
        bundles={label:retain_bundle(folder/'traces'/label,prefix) for label,prefix,_ in records}
        row={'case':cid,'bundles':bundles,'diffusion_model_nfe':0,'status':history['status']['status_str']}
        write_new(folder/'traces.artifacts.json',row);summary.append(row)
        print('TRACES_DECODED',cid,len(bundles),'bundles; 0 diffusion NFE',flush=True)
    write_new(root/'traces.summary.json',summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path);parser.add_argument('--output-root',type=Path)
    parser.add_argument('--trial-id',type=int);parser.add_argument('--experiment-plan',type=Path)
    parser.add_argument('--stage',choices=('freeze','control','factorial','release','traces'),required=True)
    args=parser.parse_args()
    if args.stage=='freeze':
        if args.source_root is None or args.output_root is None or args.trial_id is None:parser.error('freeze requires source/output/trial')
        result=freeze_experiment(args.source_root,args.output_root,args.trial_id)
        print(Path(result['output_root'])/'experiment.json')
    else:
        if args.experiment_plan is None:parser.error('execution requires --experiment-plan')
        if args.stage=='traces':run_trace_stage(args.experiment_plan)
        else:run_experiment_stage(args.experiment_plan,args.stage)


if __name__=='__main__':main()
