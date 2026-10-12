"""State-bound, descriptive paired-mask evaluation without inference imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_mask_support import load_binary_png, summarize
from evaluate_attention_masks import evaluate_metrics

PARTS = ('target_face', 'target_torso', 'target_arm_left', 'target_arm_right',
         'target_leg_left', 'target_leg_right', 'target_foot_left', 'target_foot_right')
PROTECTED = ('protected_face', 'protected_body')
ROLES = PARTS + PROTECTED + ('known_background', 'target_full')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def _binary(values, count):
    if len(values) != count or any(x not in (0, 1) for x in values):
        raise ValueError('Invalid binary region grid')


def _union(regions, names, count):
    result = [False] * count
    for name in names:
        values = regions.get(name)
        if values is not None:
            result = [a or bool(b) for a, b in zip(result, values)]
    return result


def validate_regions(regions, count):
    if set(regions) - set(ROLES):
        raise ValueError('Unknown annotation role')
    occupied = [False] * count
    for name, values in regions.items():
        if values is None:
            continue
        _binary(values, count)
        if name == 'target_full':
            continue
        if any(a and b for a, b in zip(occupied, values)):
            raise ValueError('Annotation regions overlap')
        occupied = [a or bool(b) for a, b in zip(occupied, values)]
    full = regions.get('target_full')
    if full is not None:
        for name in PARTS:
            if any(a and not b for a, b in zip(regions.get(name) or [0]*count, full)):
                raise ValueError('Partial target outside full annotation')
        forbidden = _union(regions, PROTECTED + ('known_background',), count)
        if any(a and b for a, b in zip(full, forbidden)):
            raise ValueError('Full target overlaps another role')


def validate_reference(expected, actual):
    for key in ('input_sha256', 'sigma', 'branch', 'step', 'image_sha256', 'prediction_sha256'):
        if key not in expected or actual.get(key) != expected[key]:
            raise ValueError('Annotation reference identity mismatch: ' + key)


def region_metrics(candidate, regions):
    count = len(candidate)
    _binary(candidate, count)
    validate_regions(regions, count)
    result = {}
    for name in ROLES:
        values = regions.get(name)
        cells = sum(values) if values is not None else None
        reason = 'annotation_not_supplied' if values is None else 'empty_annotation' if not cells else None
        overlap = sum(a and b for a, b in zip(candidate, values)) if reason is None else None
        row = {'annotation_cells': cells, 'overlap_cells': overlap, 'reason': reason}
        if name.startswith('target_'):
            row.update(recall=overlap/cells if reason is None else None,
                       missing_cells=cells-overlap if reason is None else None)
        else:
            row.update(intrusion_cells=overlap, intrusion_fraction=overlap/cells if reason is None else None)
        result[name] = row
    full_metrics = evaluate_metrics(candidate, {'target_full': regions['target_full']} if regions.get('target_full') is not None else {})
    result['target_full'].update(precision=full_metrics['target_full_precision']['value'],
                                 iou=full_metrics['target_full_iou']['value'])
    labelled = _union(regions, ROLES, count)
    unknown = count - sum(labelled)
    result['unknown'] = {'cells': unknown, 'fraction': unknown/count if count else None,
                         'selected_cells': sum(c and not a for c, a in zip(candidate, labelled))}
    return result


def classify_collisions(raw_target, protected, base_regions, slider_regions):
    count = len(raw_target)
    _binary(raw_target, count); _binary(protected, count)
    validate_regions(base_regions, count); validate_regions(slider_regions, count)
    targets = [_union(r, PARTS + ('target_full',), count) for r in (base_regions, slider_regions)]
    protections = [_union(r, PROTECTED, count) for r in (base_regions, slider_regions)]
    backgrounds = [_union(r, ('known_background',), count) for r in (base_regions, slider_regions)]
    result = {name: [False]*count for name in ('role_disagreement', 'confirmed_target', 'confirmed_protected', 'common_background', 'unresolved')}
    for i, (raw, p) in enumerate(zip(raw_target, protected)):
        if not raw or not p:
            continue
        if (targets[0][i] and protections[1][i]) or (targets[1][i] and protections[0][i]):
            name = 'role_disagreement'
        elif targets[0][i] and targets[1][i]: name = 'confirmed_target'
        elif protections[0][i] or protections[1][i]: name = 'confirmed_protected'
        elif backgrounds[0][i] and backgrounds[1][i]: name = 'common_background'
        else: name = 'unresolved'
        result[name][i] = True
    return result


def classify_collisions_v2(raw_target, protected, base_regions, slider_regions):
    """Partition collisions without discarding target support from only one view."""
    count=len(raw_target)
    _binary(raw_target,count);_binary(protected,count)
    validate_regions(base_regions,count);validate_regions(slider_regions,count)
    targets=[_union(r,PARTS+('target_full',),count) for r in (base_regions,slider_regions)]
    protections=[_union(r,PROTECTED,count) for r in (base_regions,slider_regions)]
    backgrounds=[_union(r,('known_background',),count) for r in (base_regions,slider_regions)]
    names=('role_conflict','both_target','base_only_target','slider_only_target','protected','common_background','unresolved')
    result={name:[False]*count for name in names}
    for i,(raw,p) in enumerate(zip(raw_target,protected)):
        if not raw or not p:continue
        if (targets[0][i] and protections[1][i]) or (targets[1][i] and protections[0][i]):name='role_conflict'
        elif targets[0][i] and targets[1][i]:name='both_target'
        elif targets[0][i]:name='base_only_target'
        elif targets[1][i]:name='slider_only_target'
        elif protections[0][i] or protections[1][i]:name='protected'
        elif backgrounds[0][i] and backgrounds[1][i]:name='common_background'
        else:name='unresolved'
        result[name][i]=True
    return result


def validate_release_review(collision_classes, review_manifest, reference_identity):
    """Accept only individually reviewed target collisions with visible opposite support."""
    names={'role_conflict','both_target','base_only_target','slider_only_target','protected','common_background','unresolved'}
    if set(collision_classes)!=names or review_manifest.get('release_review_schema')!=1:
        raise ValueError('Unsupported collision/release schema')
    count=len(collision_classes['both_target'])
    for values in collision_classes.values():_binary(values,count)
    if any(sum(values[i] for values in collision_classes.values())>1 for i in range(count)):
        raise ValueError('Collision classes overlap')
    for branch in ('base','slider'):
        expected=reference_identity[branch]
        if expected.get('branch')!=branch:raise ValueError('Reference branch mismatch')
        validate_reference(expected,review_manifest.get('references',{}).get(branch,{}))
    rows=review_manifest.get('rows')
    if not isinstance(rows,list):raise ValueError('Release review requires explicit rows')
    release=[False]*count;seen=set()
    for row in rows:
        i=row.get('cell_index');branch=row.get('support_branch')
        if type(i) is not int or not 0<=i<count or i in seen:raise ValueError('Duplicate or outside release cell')
        seen.add(i)
        allowed=collision_classes['both_target'][i] or (
            collision_classes['base_only_target'][i] if branch=='base' else
            collision_classes['slider_only_target'][i] if branch=='slider' else False)
        if (branch not in ('base','slider') or not allowed or row.get('target_part') not in PARTS
                or row.get('opposite_status') not in ('target','known_background')
                or row.get('author_kind') not in ('assistant','human') or type(row.get('human_verified')) is not bool
                or not isinstance(row.get('reason'),str) or not row['reason'].strip()):
            raise ValueError('Release lacks explicit target/nonprotected review')
        if row['human_verified'] and row['author_kind']!='human':raise ValueError('Assistant cannot claim human verification')
        release[i]=True
    return release


def attribute_difference(candidate, pilot, base_regions, slider_regions):
    count = len(candidate)
    _binary(candidate, count); _binary(pilot, count)
    validate_regions(base_regions, count); validate_regions(slider_regions, count)
    # The same precedence is used for every variant; overlapping branch labels
    # are retained separately in per-branch metrics, not counted twice here.
    disagreement = classify_collisions([1]*count, [1]*count, base_regions, slider_regions)['role_disagreement']
    regions = {'role_disagreement': disagreement}
    for name in PARTS:
        regions[name] = _union(base_regions, (name,), count)
        regions[name] = [a or b for a, b in zip(regions[name], _union(slider_regions, (name,), count))]
    if base_regions.get('target_full') is not None or slider_regions.get('target_full') is not None:
        regions['target_full_other']=[a or b for a,b in zip(_union(base_regions,('target_full',),count),_union(slider_regions,('target_full',),count))]
    for name in PROTECTED:
        regions[name]=[a or b for a,b in zip(_union(base_regions,(name,),count),_union(slider_regions,(name,),count))]
    regions['common_background'] = [a and b for a, b in zip(_union(base_regions, ('known_background',), count), _union(slider_regions, ('known_background',), count))]
    result = {}
    for name, selected in (('added', [a and not b for a,b in zip(candidate,pilot)]),
                           ('removed', [b and not a for a,b in zip(candidate,pilot)])):
        by_region = {key: 0 for key in regions}; by_region['unlabelled'] = 0
        for i, active in enumerate(selected):
            if active:
                key = next((key for key, values in regions.items() if values[i]), 'unlabelled')
                by_region[key] += 1
        result[name] = {'total_cells': sum(selected), 'by_region': by_region}
    return result


def paired_support_metrics(candidate, base_regions, slider_regions):
    count=len(candidate)
    _binary(candidate,count);validate_regions(base_regions,count);validate_regions(slider_regions,count)
    common=[a and b for a,b in zip(_union(base_regions,('known_background',),count),_union(slider_regions,('known_background',),count))]
    supported=[a or b for a,b in zip(_union(base_regions,ROLES,count),_union(slider_regions,ROLES,count))]
    cells=sum(common);intrusion=sum(a and b for a,b in zip(candidate,common))
    return {'common_background_annotation_cells':cells,'common_background_intrusion_cells':intrusion,
            'common_background_intrusion_fraction':intrusion/cells if cells else None,
            'unlabelled_cells':count-sum(supported),'unlabelled_selected_cells':sum(a and not b for a,b in zip(candidate,supported))}


def load_annotation_views(experiment, annotation_manifest, case):
    if annotation_manifest.get('experiment_sha256') != sha(experiment['_path']):
        raise ValueError('Annotation experiment hash mismatch')
    if type(annotation_manifest.get('human_verified')) is not bool:
        raise ValueError('Explicit human_verified boolean required')
    views = {}
    for branch in ('base', 'slider'):
        entry = annotation_manifest['cases'][case['case']][branch]
        reference = case['references'][branch]
        validate_reference(reference, entry['reference'])
        if sha(reference['image']) != reference['image_sha256']:
            raise ValueError('Changed reference image')
        regions = {}
        for name, item in entry['regions'].items():
            if item is None:
                regions[name] = None
                continue
            if sha(item['path']) != item['sha256']:
                raise ValueError('Changed annotation PNG')
            size, values = load_binary_png(item['path'])
            if size != (64,64): raise ValueError('Annotation grid mismatch')
            regions[name] = list(values)
        validate_regions(regions, 4096)
        views[branch] = regions
    return views


def evaluate_attribution(experiment_path, annotation_manifest, candidate_index):
    experiment = read(experiment_path); experiment['_path'] = str(experiment_path)
    annotations = read(annotation_manifest); index = read(candidate_index)
    if index.get('experiment_sha256') != sha(experiment_path):
        raise ValueError('Candidate index experiment hash mismatch')
    report = {'mask_attribution_schema': 1, 'experiment_sha256': sha(experiment_path),
              'annotation_manifest_sha256': sha(annotation_manifest), 'human_verified': annotations['human_verified'],
              'basis': 'human_declared_labels' if annotations['human_verified'] else 'assistant_unverified_visible_interiors',
              'cases': [], 'quality_approval': 'not_established_by_mask_metrics'}
    for case in experiment['cases']:
        cid = case['case']; views = load_annotation_views(experiment, annotations, case)
        rows = index['cases'][cid]
        d = next(r for r in rows if r['label']=='D')
        _, pilot = load_binary_png(Path(d['manifest']).parent/'target.png')
        result = {'case':cid, 'candidates':[], 'raw_variants':[]}
        for row in rows:
            if row['status'] != 'ok':
                result['candidates'].append(row); continue
            meta = read(row['manifest']); folder = Path(row['manifest']).parent
            expected_folder=Path(experiment['output_root'])/cid/'candidates'/row['label']
            if folder.resolve()!=expected_folder.resolve() or meta['label']!=row['label'] or meta['experiment_sha256']!=sha(experiment_path):
                raise ValueError('Candidate identity differs from experiment/case')
            for role in ('target','protected','background'):
                if sha(folder/(role+'.png')) != meta['png_sha256'][role]: raise ValueError('Changed candidate PNG')
            size, target = load_binary_png(folder/'target.png')
            if size != (64,64): raise ValueError('Candidate grid mismatch')
            result['candidates'].append({'label':row['label'], 'support':summarize(target,64,64),
                'per_branch':{b:region_metrics(target,views[b]) for b in views},
                'difference_from_D':attribute_difference(target,pilot,views['base'],views['slider']),
                'paired_support':paired_support_metrics(target,views['base'],views['slider'])})
        for block in (18,23):
            for branch in ('base','slider'):
                for variant in ('baseline','seed_only','shape_only','combined'):
                    path = Path(experiment['output_root'])/cid/'ablations'/f'{branch}_s3_b{block}_{variant}'/'target.png'
                    if not path.is_file():
                        result['raw_variants'].append({'branch':branch,'block':block,'variant':variant,'status':'failed'}); continue
                    stats=read(path.parent/'statistics.json')
                    if (stats['branch']!=branch or stats['block']!=block or stats['variant']!=variant
                            or stats['experiment_sha256']!=sha(experiment_path) or sha(path)!=stats['png_sha256']['target']):
                        raise ValueError('Raw candidate identity/hash mismatch')
                    _, target = load_binary_png(path)
                    result['raw_variants'].append({'branch':branch,'block':block,'variant':variant,
                        'target_png_sha256':sha(path),'metrics':region_metrics(target,views[branch])})
        # Release is based on the preregistered S0 raw target only.
        s0 = next(r for r in rows if r['label']=='S0' and r['status']=='ok')
        folder = Path(s0['manifest']).parent
        _, raw = load_binary_png(folder/'raw_target.png'); _, protected = load_binary_png(folder/'protected.png')
        classes = classify_collisions(raw,protected,views['base'],views['slider'])
        result['collisions'] = {'counts':{name:sum(value) for name,value in classes.items()},
                                'class_cells':{name:[i for i,v in enumerate(values) if v] for name,values in classes.items()},
                                'source_candidate_manifest_sha256':sha(s0['manifest']), 'human_verified':annotations['human_verified']}
        report['cases'].append(result)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-plan',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=evaluate_attribution(args.experiment_plan,args.annotations,args.candidates)
    write_new(args.output,report)
    print(args.output)


if __name__ == '__main__': main()
