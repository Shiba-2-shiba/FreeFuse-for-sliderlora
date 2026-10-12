"""Record conservative assistant-labelled interiors on four inspected previews.

Coordinates are half-open native-grid rectangles, selected from RGB only.
These labels are specific to the retained source images below, not a segmenter.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tools'))
from evaluate_paired_mask_attribution import PARTS, PROTECTED, read, sha, write_new, validate_regions
from PIL import Image, ImageDraw

SPEC = {
 ('park','base'): {
  'rgb_sha256':'e9db97dd18618ad92f6fcc1919d3d9d1f191707c5c28786a296d670471648345',
  'target_face':[(21,13,24,16)], 'target_torso':[(19,22,27,33)],
  'target_arm_right':[(16,30,17,37)], 'target_arm_left':[(28,29,29,36)],
  'target_leg_right':[(19,40,21,54)], 'target_leg_left':[(23,40,25,54)],
  'target_foot_right':[(19,59,21,60)], 'target_foot_left':[(24,59,26,60)],
  'protected_face':[(39,11,42,14)],
  'protected_body':[(37,20,46,32),(37,39,40,53),(43,39,45,53),(33,27,34,35),(47,27,48,35)],
  'known_background':[(0,0,64,4),(2,4,9,58),(56,4,63,58),(30,42,33,58),(0,62,64,64)]},
 ('park','slider'): {
  'rgb_sha256':'04f90b000f1f348e5a4d8129a06704ffae770507b001033e58b8b2a8f70dd006',
  'target_face':[(21,19,26,23)], 'target_torso':[(19,28,27,36)],
  'target_arm_right':[(16,34,17,40)], 'target_arm_left':[(29,34,30,39)],
  'target_leg_right':[(19,41,21,53)], 'target_leg_left':[(24,41,26,53)],
  'target_foot_right':[(19,57,21,59)], 'target_foot_left':[(25,57,27,59)],
  'protected_face':[(38,14,42,18)],
  'protected_body':[(37,24,45,35),(37,39,39,52),(42,39,44,52),(34,30,35,39),(47,30,48,39)],
  'known_background':[(0,0,64,4),(2,4,9,58),(56,4,63,58),(31,43,33,54),(0,62,64,64)]},
 ('woman_front_strong_overlap','base'): {
  'rgb_sha256':'cef7c9831c909a52abbcc9eef368eb1877dd7a02f632d5ae5ae77f85cf163296',
  'target_face':[(25,10,28,13)], 'target_torso':[(24,19,31,30)],
  'target_arm_right':[(20,26,21,35)], 'target_arm_left':[(33,26,34,35)],
  'target_leg_right':[(24,38,26,55)], 'target_leg_left':[(29,38,31,55)],
  'target_foot_right':[(25,58,27,60)], 'target_foot_left':[(29,58,31,60)],
  'protected_face':[(36,8,39,11)],
  'protected_body':[(36,18,44,31),(35,37,37,52),(41,37,43,52),(44,24,45,32)],
  'known_background':[(0,0,64,3),(0,3,15,58),(53,3,64,58),(0,62,64,64)]},
 ('woman_front_strong_overlap','slider'): {
  'rgb_sha256':'f90e3cf122bbe6003500c0fc582789882637248860b714180fc12973676fcfc4',
  'target_face':[(24,12,29,17)], 'target_torso':[(23,24,32,32)],
  'target_arm_right':[(18,31,19,38)], 'target_arm_left':None,
  'target_leg_right':[(23,39,25,52)], 'target_leg_left':[(29,39,31,52)],
  'target_foot_right':[(23,57,25,59)], 'target_foot_left':[(30,57,32,59)],
  'protected_face':[(38,12,41,16)], 'protected_body':[(40,22,45,27),(40,38,42,51)],
  'known_background':[(0,0,64,3),(0,3,13,58),(54,3,64,58),(0,62,64,64)]},
}


def prepare(experiment_path):
    experiment=read(experiment_path);root=Path(experiment['output_root']);dest=root/'annotations'
    if (dest/'manifest.json').exists():raise ValueError('Annotations already frozen')
    manifest={'annotation_schema':1,'experiment_sha256':sha(experiment_path),'human_verified':False,
        'author_kind':'assistant','revision':1,'label_basis':'conservative RGB-visible interiors, inspected before candidate quality evaluation',
        'grid':[64,64],'cell_to_rgb':'16x16 native pixels; half-open x/y rectangles',
        'limitations':'Sparse interiors only; not full segmentation. Blur, boundaries and occluded contact are unlabelled.',
        'source_spec_sha256':sha(__file__),'cases':{}}
    for case in experiment['cases']:
        cid=case['case'];manifest['cases'][cid]={}
        for branch,reference in case['references'].items():
            spec=SPEC[cid,branch]
            if reference['image_sha256']!=spec['rgb_sha256'] or sha(reference['image'])!=spec['rgb_sha256']:
                raise ValueError('Annotation coordinates belong to different source RGB')
            folder=dest/cid/branch;folder.mkdir(parents=True,exist_ok=False)
            regions={};records={};missing={}
            image=Image.open(reference['image']).convert('RGBA');layer=Image.new('RGBA',image.size);draw=ImageDraw.Draw(layer)
            for name in PARTS+PROTECTED+('known_background','target_full'):
                rectangles=spec.get(name)
                if rectangles is None:
                    records[name]=None
                    missing[name]='not_fully_traced' if name=='target_full' else 'occluded_or_ambiguous_contact'
                    continue
                values=[0]*4096
                for x0,y0,x1,y1 in rectangles:
                    if not 0<=x0<x1<=64 or not 0<=y0<y1<=64:raise ValueError('Invalid native-grid rectangle')
                    for y in range(y0,y1):
                        for x in range(x0,x1):values[y*64+x]=1
                    color=(255,220,0,90) if name.startswith('target') else (0,120,255,90) if name.startswith('protected') else (0,180,80,65)
                    draw.rectangle((x0*16,y0*16,x1*16-1,y1*16-1),fill=color,outline=color[:3]+(255,))
                png=folder/(name+'.png');Image.frombytes('L',(64,64),bytes(v*255 for v in values)).save(png)
                records[name]={'path':str(png.resolve()),'sha256':sha(png),'rectangles_xyxy':rectangles,'cells':sum(values)};regions[name]=values
            validate_regions(regions,4096)
            Image.alpha_composite(image,layer).convert('RGB').save(folder/'label-overlay.png')
            manifest['cases'][cid][branch]={'reference':reference,'regions':records,'missing_reasons':missing}
    write_new(dest/'manifest.json',manifest)
    return dest/'manifest.json'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment-plan',type=Path,required=True)
    print(prepare(parser.parse_args().experiment_plan))
