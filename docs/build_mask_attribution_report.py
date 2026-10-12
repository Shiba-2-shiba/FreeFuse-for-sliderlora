"""Render the factorial experiment, labelled regions, and protection collisions."""
import argparse
import html
import json
import os
from pathlib import Path
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_mask_attribution_validation import read, sha


def comparison_sheet(images, dest, *, crop=None, width=320):
    height = width if crop is None else max(120, int(width*(crop[3]-crop[1])/(crop[2]-crop[0])))
    sheet = Image.new('RGB', (width*len(images), height+24), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(images):
        with Image.open(path) as image:
            view = image.convert('RGB')
            if crop: view = view.crop(crop)
            view = view.resize((width, height))
        sheet.paste(view, (width*i, 24)); draw.text((width*i+4, 5), label, fill='black')
    if dest.exists(): raise FileExistsError(dest)
    sheet.save(dest)


def collision_overlays(experiment, attribution):
    root = Path(experiment['output_root'])
    colors = {'confirmed_target': (255,210,0,150), 'confirmed_protected': (0,120,255,150),
              'role_disagreement': (255,0,180,150), 'common_background': (0,200,90,150),
              'unresolved': (255,90,0,150)}
    for case in experiment['cases']:
        cid=case['case'];collisions=next(c['collisions'] for c in attribution['cases'] if c['case']==cid)
        dest=root/cid/'collision-overlays';dest.mkdir(exist_ok=False)
        refs=[(b,Path(r['image'])) for b,r in case['references'].items()]
        for stage in ('control','factorial'):
            for record in read(root/cid/(stage+'.artifacts.json'))['diagnostics']:
                label=record['label'].removeprefix(stage+'_')
                if label in ('D','S0'):
                    report=read(record['local_report'])
                    refs.append((label,Path(record['local_report']).parent/report['artifacts']['image']))
        for label,path in refs:
            image=Image.open(path).convert('RGBA');layer=Image.new('RGBA',image.size);draw=ImageDraw.Draw(layer)
            for name,cells in collisions['class_cells'].items():
                for cell in cells:
                    x,y=cell%64,cell//64
                    draw.rectangle((x*16,y*16,x*16+15,y*16+15),fill=colors[name])
            Image.alpha_composite(image,layer).convert('RGB').save(dest/(label+'.png'))


def build_report(experiment_path, verification_path, output_path):
    experiment=read(experiment_path);root=Path(experiment['output_root']);verified=read(verification_path)
    if verified['experiment_sha256']!=sha(experiment_path): raise ValueError('Wrong experiment verification')
    for name,expected in verified['output_file_sha256'].items():
        if sha(root/name)!=expected: raise ValueError('Verified artifact changed before reporting')
    notes=read(root/'assistant-visual-observations.json');attribution=read(verified['evaluation_path'])
    esc=lambda value:html.escape(str(value),quote=True)
    rel=lambda path:esc(Path(os.path.relpath(path,root)).as_posix())
    def img(path): return f"<a href='{rel(path)}'><img src='{rel(path)}'></a>"
    collision_overlays(experiment,attribution)
    parts=["<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
      "<title>D union・部位・固定保護の検証</title><style>body{font:15px/1.7 system-ui;background:#f2f4f7;color:#182838;margin:0}main{max-width:1700px;margin:auto;padding:24px}section,article{background:white;padding:18px;margin:18px 0;border-radius:8px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}pre{white-space:pre-wrap}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><main>",
      '<h1>Dとの和集合・部位・固定保護の分離検証</h1>',
      f"<section><strong>{esc(notes['conclusion'])}</strong><p>現在起動中のComfyUI {esc(experiment['server'])}。再起動なし。公園seed444444、女性手前の重なりseed42。Slider+4、1024×1024、Euler/simple、8steps、CFG1、全radius0。</p>",
      f"<p>実測 {verified['diffusion_model_nfe']} diffusion NFE。D対照2画像と4条件×2構図。Dは全step入力・base/LoRA/混合予測・最終latent完全一致。{verified['verified_raw_variants']} raw bankを再計算・検査。全画像の予測選択とmask由来を監査しました。</p>",
      '<p>部位labelは各branchのpilot RGBから確認した内部セル。human_verified=false、全身正解labelではありません。重なりのLoRA側左腕は判別不能、full-targetは未供給です。数値と目視は記述的診断であり、人の画質承認・holdout汎化を示しません。</p>',
      f"<p><a href='{rel(verification_path)}'>数値監査</a> · <a href='{rel(verified['evaluation_path'])}'>部位評価</a> · <a href='annotations/manifest.json'>凍結label</a> · <a href='experiment.json'>実験条件</a> · <a href='assistant-visual-observations.json'>目視所見</a> · <a href='cpu-final.log'>CPU検証</a></p></section>",
      '<table><tr><th>ID</th><th>分類</th><th>D対象の和集合</th><th>保護</th></tr><tr><td>D</td><td>旧D対照</td><td>対照</td><td>固定D</td></tr><tr><td>S0/S1</td><td>seed修正＋prototype再推定</td><td>なし／あり</td><td>固定D</td></tr><tr><td>C0/C1</td><td>seed修正＋全head輪郭伝播</td><td>なし／あり</td><td>固定D</td></tr></table>',
      '<p>D unionなしでも、同入力のbase/LoRA対象の和集合は使用します。rawの4 ablationはoffline比較で、完成画像の2×2比較とは区別します。</p>']
    for case in verified['cases']:
        cid=case['case'];folder=root/cid;ev=next(c for c in attribution['cases'] if c['case']==cid)
        images=[(row['label'],Path(row['image'])) for row in case['candidates']]
        for filename,crop in (('final-comparison.png',None),('face-comparison.png',(240,64,576,464)),
                              ('feet-comparison.png',(200,760,840,1024)),('arms-comparison.png',(240,340,720,700))):
            comparison_sheet(images,folder/filename,crop=crop)
        parts.append(f"<section><h2>{esc(cid)}</h2><p>{esc(notes['cases'][cid]['interpretation'])}</p>{img(folder/'final-comparison.png')}")
        parts.append('<details><summary>顔・腕・足元</summary>'+''.join(img(folder/name) for name in ('face-comparison.png','arms-comparison.png','feet-comparison.png'))+'</details>')
        parts.append('<table><tr><th>ID</th><th>対象セル</th><th>D比追加</th><th>D比削除</th><th>共通背景選択</th><th>顔recall base/LoRA</th><th>目視判定</th></tr>')
        for row in ev['candidates']:
            if row.get('status') and row['status']!='ok':continue
            label=row['label'];delta=row['difference_from_D']
            face=' / '.join(f"{row['per_branch'][b]['target_face']['recall']:.3f}" if row['per_branch'][b]['target_face']['recall'] is not None else 'unknown' for b in ('base','slider'))
            parts.append(f"<tr><td>{esc(label)}</td><td>{row['support']['cells']}</td><td>{delta['added']['total_cells']}</td><td>{delta['removed']['total_cells']}</td><td>{row['paired_support']['common_background_intrusion_cells']}</td><td>{face}</td><td>{esc(notes['cases'][cid]['routes'][label])}</td></tr>")
        parts.append('</table><details><summary>部位・保護侵入・D差分の全指標</summary><pre>'+esc(json.dumps(ev['candidates'],ensure_ascii=False,indent=2))+'</pre></details>')
        parts.append('<h3>固定保護との衝突</h3><p>'+esc(ev['collisions']['counts'])+'</p><p>黄色=両branchで対象、青=保護、緑=共通背景、ピンク=役割不一致、橙=未確定。完成像へのoverlayもpilot座標の診断表示で、新軌道の正解labelではありません。確認対象セル0のためS0-releaseは対象外です。</p><div class="grid">')
        for label in ('base','slider','D','S0'):parts.append('<article>'+esc(label)+img(folder/'collision-overlays'/(label+'.png'))+'</article>')
        parts.append('</div><details><summary>凍結label</summary><div class="grid">')
        for branch in ('base','slider'):parts.append(img(root/'annotations'/cid/branch/'label-overlay.png'))
        parts.append('</div></details><details><summary>新軌道のstep3予測（label転用なし）</summary>')
        traces=read(folder/'traces.artifacts.json')
        for label in ('D','S0','S1','C0','C1'):
            bundle=Path(traces['bundles'][label]['local']);names=('trace_base_predictions','trace_slider_predictions','trace_predictions')
            path=folder/('step3-'+label+'.png');comparison_sheet([(name,bundle/(name+'_s3.png')) for name in names],path)
            parts.append('<p>'+esc(label)+'</p>'+img(path))
        parts.append('</details></section>')
    parts.append('<section><h2>提案1〜3への回答</h2>')
    for key,value in notes['answers'].items():parts.append('<p><strong>'+esc(key)+'</strong> '+esc(value)+'</p>')
    parts.append('<p>'+esc(notes['quality_gate']['reason'])+'</p><p>既定値、動的mask、holdoutは今回変更・実施していません。</p></section></main></html>')
    path=Path(output_path)
    if path.exists():raise FileExistsError(path)
    path.write_text(''.join(parts),encoding='utf-8');return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment-plan',type=Path,required=True)
    parser.add_argument('--verification',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(build_report(args.experiment_plan,args.verification,args.output))
