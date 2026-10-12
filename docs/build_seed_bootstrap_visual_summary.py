"""Build a visual guide from retained, verified images and masks; no inference."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from safetensors.numpy import load_file

GREEN=(30,180,105); BLUE=(55,115,225); GREY=(54,61,74)
RED=(227,65,67); YELLOW=(246,194,65)
COLORS={'target':GREEN,'protected':BLUE,'background':GREY}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def font(size,bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/'+('meiryob.ttc' if bold else 'meiryo.ttc'),size)


def image_record(root,cid,phase,label):
    folder=root/cid/(phase+'_'+label)
    a=read(folder/'artifacts.json');p=Path(a['diagnostics'][0]['local_report']);r=read(p)
    return p.parent/r['artifacts']['image']


def source_rgb(root,cid,initial,branch='base'):
    return Path(read(root/cid/('pilot_'+initial)/'artifacts.json')['references'][branch]['image'])


def partition(masks):
    a=np.empty((*masks['target'].shape[-2:],3),dtype=np.uint8)
    for role,color in COLORS.items():a[masks[role][0].astype(bool)]=color
    return Image.fromarray(a).resize((1024,1024),Image.Resampling.NEAREST)


def binary_mask(mask,color):
    a=np.full((*mask.shape[-2:],3),GREY,dtype=np.uint8)
    a[mask[0].astype(bool)]=color
    return Image.fromarray(a).resize((1024,1024),Image.Resampling.NEAREST)


def overlay(rgb,masks,*,uncertain=None,collision=None):
    color=partition(masks).convert('RGBA')
    alpha=np.where(masks['background'][0].astype(bool),35,118).astype(np.uint8)
    color.putalpha(Image.fromarray(alpha).resize((1024,1024),Image.Resampling.NEAREST))
    result=Image.alpha_composite(rgb.convert('RGBA'),color)
    for mask,tint in ((uncertain,YELLOW),(collision,RED)):
        if mask is None:continue
        mask=mask[0].astype(bool)
        layer=Image.new('RGBA',rgb.size,tint+(0,))
        layer.putalpha(Image.fromarray((mask*150).astype(np.uint8)).resize(rgb.size,Image.Resampling.NEAREST))
        result=Image.alpha_composite(result,layer)
    return result.convert('RGB')


def sheet(dest,entries,*,columns=3,size=440,crop=None,resample=Image.Resampling.LANCZOS):
    header,footer,gap=66,58,12;rows=(len(entries)+columns-1)//columns
    tile_h=size+header+footer;canvas=Image.new('RGB',(columns*(size+gap)+gap,rows*(tile_h+gap)+gap),'#edf1f5')
    draw=ImageDraw.Draw(canvas)
    for n,(label,view,note,attention) in enumerate(entries):
        x=gap+(n%columns)*(size+gap);y=gap+(n//columns)*(tile_h+gap)
        draw.rounded_rectangle((x,y,x+size,y+tile_h),radius=8,fill='white',outline=RED if attention else '#ccd6e0',width=3 if attention else 1)
        draw.text((x+12,y+12),label,font=font(20,True),fill='#17304b')
        if crop is not None:view=view.crop(crop)
        view=ImageOps.contain(view,(size,size),resample)
        canvas.paste(view,(x+(size-view.width)//2,y+header+(size-view.height)//2))
        draw.text((x+10,y+header+size+10),note,font=font(16),fill=RED if attention else '#34465b')
    canvas.save(dest)


def build(root,output):
    root=Path(root).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    verified=read(root/'verification-r2.json')
    original_hashes={p:sha(p) for case in verified['cases'] for r in case['images'] for p in (Path(r['image']),)}
    for case in verified['cases']:
        for r in case['images']:
            if sha(r['image'])!=r['image_sha256']:raise ValueError('Changed generation image')
    checked=set()
    def checked_data(p):
        p=Path(p).resolve();key=str(p.relative_to(root))
        if p not in checked:
            if sha(p)!=verified['output_file_sha256'][key]:raise ValueError('Changed mask/reference evidence: '+key)
            checked.add(p)
        return p
    def raw(cid,initial,variant,branch):
        idx=read(checked_data(root/cid/('scoring_'+initial)/'index.json'))['variants'][variant]
        record=idx['branches'][branch]
        return load_file(checked_data(record['evidence']))
    def final(cid,initial):
        row=read(checked_data(root/cid/('scoring_'+initial)/'index.json'))['variants']['R0']
        meta=read(checked_data(row['manifest']));folder=Path(row['manifest']).parent
        return load_file(checked_data(folder/meta['tensors'])),load_file(checked_data(folder/'composition.safetensors'))
    def rgb(cid,initial,branch='base'):
        return Image.open(checked_data(source_rgb(root,cid,initial,branch))).convert('RGB')
    order=[('park__seed_444444','公園｜seed 444444','診断条件'),('park__seed_123','公園｜seed 123','別seedの確認'),('park__seed_777','公園｜seed 777','別seedの確認'),
           ('woman_front_strong_overlap__seed_42','重なり｜seed 42','診断条件'),('woman_front_strong_overlap__seed_123','重なり｜seed 123','注意：旧身体状の残片'),('woman_front_strong_overlap__seed_777','重なり｜seed 777','別seedの確認')]
    entries=[];mask_entries=[];cards=[]
    for cid,label,note in order:
        path=image_record(root,cid,'replay','A-R0');view=Image.open(path).convert('RGB');bad=cid.endswith('overlap__seed_123')
        entries.append((label,view,note,bad))
        masks,composition=final(cid,'A')
        p=output/(cid+'_mask.png');partition(masks).save(p)
        over=output/(cid+'_overlay.png');overlay(rgb(cid,'A'),masks,collision=composition['collision']).save(over)
        mask_entries.append((label,Image.open(p).convert('RGB'),'緑：対象 ／ 青：固定保護',bad))
        cards.append({'cid':cid,'label':label,'note':note,'attention':bad,'image':path,'mask':p,'overlay':over})
    sheet(output/'01-results-overview.png',entries)
    sheet(output/'02-effective-mask-overview.png',mask_entries,resample=Image.Resampling.NEAREST)
    face_entries=[]
    for cid,label,note in order:
        crop=(175,110,545,480) if cid.startswith('park') else (260,80,620,450)
        view=Image.open(image_record(root,cid,'replay','A-R0')).convert('RGB').crop(crop)
        face_entries.append((label,view,'完成画像の顔を拡大',cid.endswith('overlap__seed_123')))
    sheet(output/'03-faces-overview.png',face_entries,size=350)
    routes=[];mask_routes=[];seed_panels=[];construction=[]
    for cid,name,crop in [('park__seed_444444','公園',(256,128,544,464)),('woman_front_strong_overlap__seed_42','重なり',(288,80,576,384))]:
        e=[];m=[]
        for initial,display in [('D','従来経路 D'),('A','baseのみ A')]:
            e.append((display+'｜初期mask',Image.open(image_record(root,cid,'pilot',initial)).convert('RGB'),'候補評価元の生成（Slider +4）',False))
            e.append((display+'｜再分類 R0',Image.open(image_record(root,cid,'replay',initial+'-R0')).convert('RGB'),'再分類maskによる生成',False))
            masks,composition=final(cid,initial);base=rgb(cid,initial);slider=rgb(cid,initial,'slider')
            m.extend([(display+'｜baseに重ねる',overlay(base,masks,collision=composition['collision']),'赤：保護と衝突して除かれた対象',False),
                      (display+'｜Sliderに重ねる',overlay(slider,masks,collision=composition['collision']),'同じmask・同じpilot入力の別branch',False)])
        dest=output/(cid+'_routes.png');sheet(dest,e,columns=2,size=520);routes.append((name,dest,cid))
        dest=output/(cid+'_mask-routes.png');sheet(dest,m,columns=2,size=520);mask_routes.append((name,dest))
        base=raw(cid,'A','R0','base')['masks.target'];slider=raw(cid,'A','R0','slider')['masks.target']
        masks,composition=final(cid,'A')
        stages=[('① baseの対象支持',base,GREEN),('② Sliderの対象支持',slider,GREEN),
                ('③ raw対象：①と②の和',composition['raw_target'],GREEN),('④ 固定保護 BP',masks['protected'],BLUE),
                ('⑤ 衝突：③と④の重なり',composition['collision'],RED),('⑥ 実効対象：③から④を除く',masks['target'],GREEN)]
        e=[(title,binary_mask(mask,color),'選択セル数：'+str(int(mask.sum())),False) for title,mask,color in stages]
        dest=output/(cid+'_mask-construction.png');sheet(dest,e,columns=3,size=350,resample=Image.Resampling.NEAREST);construction.append((name,dest))
        e=[]
        for branch,display in [('base','base予測'),('slider','Slider予測')]:
            source=rgb(cid,'D',branch)
            for variant in ('R0','R1','R2','R3','R4'):
                data=raw(cid,'D',variant,branch);masks={r:data['masks.'+r] for r in COLORS}
                view=overlay(source,masks,uncertain=data['uncertain_mask']).crop(crop)
                e.append((variant+'｜'+display,view,'緑：対象 ／ 青：保護役割',False))
        dest=output/(cid+'_seed-face-masks.png');sheet(dest,e,columns=5,size=280);seed_panels.append((name,dest))
    park='park__seed_444444'
    before=image_record(root,park,'replay','D-R0');after=image_record(root,park,'release','D-R0')
    sheet(output/'04-one-cell-release.png',[('解除なし',Image.open(before).convert('RGB'),'完成顔は既に保持',False),('1セルだけ解除',Image.open(after).convert('RGB'),'明瞭な追加画質改善は未確認',False)],columns=2,size=520)
    release_root=root/park/'release_candidate';release_meta=read(checked_data(release_root/'manifest.json'));released=load_file(checked_data(release_root/release_meta['tensors']))
    normal,comp=final(park,'D');delta=load_file(checked_data(release_root/'composition.safetensors'))['release_mask']
    e=[]
    for title,masks in [('解除前の実効mask',normal),('解除後の実効mask',released)]:
        view=overlay(rgb(park,'D'),masks)
        draw=ImageDraw.Draw(view);draw.rectangle((23*16-2,15*16-2,24*16+2,16*16+2),outline=RED,width=3)
        e.append((title,view.crop((304,176,432,304)),'赤枠：(23,15) の1セル',False))
    sheet(output/'05-one-cell-mask-detail.png',e,columns=2,size=400)
    bad='woman_front_strong_overlap__seed_123';bad_path=image_record(root,bad,'replay','A-R0');bad_view=Image.open(bad_path).convert('RGB')
    marked=bad_view.copy();ImageDraw.Draw(marked).rectangle((256,246,338,386),outline=RED,width=5)
    sheet(output/'06-remnant-detail.png',[('完成像：注意箇所',marked,'赤枠：左肩後方の白い残片',True),('同じ画像を拡大',bad_view.crop((230,210,395,415)),'追加頭部ではなく旧身体状の残片',True)],columns=2,size=500)
    rows=[]
    for case in verified['cases']:
        for rec in case['images']:rows.append({'cid':case['case_id'],**rec})
    esc=lambda v:html.escape(str(v),quote=True)
    rel=lambda p:esc(Path(os.path.relpath(p,output)).as_posix())
    img=lambda p,title:f'<a href="{rel(p)}" target="_blank"><img loading="lazy" src="{rel(p)}" alt="{esc(title)}"></a>'
    scene=lambda cid:'公園' if cid.startswith('park') else '重なり'
    parts=['<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>検証画像とmaskの見比べ</title>',
      '<style>body{margin:0;background:#edf1f5;color:#17304b;font:16px/1.75 system-ui}main{max-width:1450px;margin:auto;padding:24px}section{background:white;border-radius:12px;padding:22px;margin:20px 0}h1{font-size:30px}h2{font-size:23px}h3{font-size:18px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}.card{border:1px solid #d8e0e8;border-radius:9px;padding:12px}.attention{border:2px solid #e34143;background:#fff7f5}.note{background:#f4f7fa;padding:12px;border-radius:6px}.legend{display:flex;flex-wrap:wrap;gap:18px}.swatch{display:inline-block;width:16px;height:16px;vertical-align:middle;border-radius:3px;margin-right:6px}img{width:100%;height:auto;display:block}a{color:#135b9b}nav{display:flex;gap:18px;flex-wrap:wrap}summary{cursor:pointer;font-weight:600}small{color:#536579}.compare{position:relative;aspect-ratio:1;overflow:hidden;border-radius:6px}.compare img{position:absolute;inset:0;width:100%;height:100%}.compare .top{clip-path:inset(0 calc(100% - var(--split,50%)) 0 0)}.compare:after{content:"";position:absolute;inset:0 auto 0 var(--split,50%);width:2px;background:#fff}input[type=range]{width:100%}@media(max-width:850px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:580px){.grid,.pair{grid-template-columns:1fr}main{padding:12px}section{padding:15px}}</style><main>',
      '<h1>検証画像とmaskの見比べ</h1><p>まず完成結果、その後にmaskの理由を確認できます。クリックで原寸画像、比較シートはPNGとして保存できます。</p>',
      '<p class="note">顔は別seedの4条件でも保持。重なりseed123は左肩後方に白い残片があり要注意。背景・部位には課題が残るため既定採用は保留です。</p>',
      '<nav><a href="#results">完成結果6条件</a><a href="#masks">実効mask</a><a href="#routes">D/A経路比較</a><a href="#seed">R0〜R4の顔分類</a><a href="#release">1セル解除</a><a href="#remnant">残片の拡大</a><a href="#all">全17画像</a></nav>',
      '<p><small>全条件：Slider +4、1024×1024、8 steps、CFG1。既存の検証結果のみを使用。新しいdiffusion生成は0 NFE。</small></p>',
      '<section id="results"><h2>1. base-only＋R0の完成結果：6条件</h2>'+img(output/'01-results-overview.png','完成結果6条件の比較シート')+'<p><a href="03-faces-overview.png" target="_blank">顔だけを拡大した6条件の比較</a></p><div class="grid">']
    for card in cards:parts.append('<article class="card '+('attention' if card['attention'] else '')+'"><h3>'+esc(card['label'])+'</h3>'+img(card['image'],card['label'])+'<p>'+esc(card['note'])+'</p></article>')
    parts.append('</div></section><section id="masks"><h2>2. 完成画像に使った実効mask</h2><p>最終targetは、同じpilot入力のbase／Slider対象支持を合わせ、固定保護を引いたものです。</p><div class="legend">')
    for color,label in [(GREEN,'緑：編集対象（Slider予測を使う）'),(BLUE,'青：固定保護'),(GREY,'灰：その他（正解背景とは限らない）'),(RED,'赤：raw対象と固定保護の衝突')]:parts.append('<span><i class="swatch" style="background:rgb'+str(color)+'"></i>'+label+'</span>')
    parts.append('</div>'+img(output/'02-effective-mask-overview.png','実効mask6条件')+'<p class="note">overlayの参照RGBはpilotのstep3予測です。maskを選んだ入力と位置を揃えて表示しています。完成画像を人物segmentationした正解maskではありません。</p><div class="grid">')
    for card in cards:parts.append('<article class="card"><h3>'+esc(card['label'])+'</h3>'+img(card['overlay'],card['label']+' 実効mask overlay')+'<p><a href="'+rel(card['mask'])+'" target="_blank">色分けmask単体</a></p></article>')
    parts.append('</div><h3>D/Aとbase／Sliderの位置を揃えて見る</h3>')
    for name,path in mask_routes:parts.append('<details><summary>'+name+'：同じmaskを両branchのRGBに重ねる</summary>'+img(path,name+' mask経路比較')+'</details>')
    parts.append('<h3>実効maskを作る流れ：base-only経路 A</h3><p>上段はbase／Sliderの対象支持と和集合、下段は固定保護・衝突・保護減算後の実効対象です。赤い衝突セルは⑥から除かれます。</p>')
    for name,path in construction:parts.append('<details open><summary>'+name+'：raw対象 → 保護減算 → 実効対象</summary>'+img(path,name+' mask構成の6段階')+'</details>')
    parts.append('</section><section id="routes"><h2>3. D/A経路：同じ構図・seedで比較</h2><p>D＝従来由来の初期mask、A＝baseのみで収集した初期mask。「初期mask」は比較元のMix生成、「再分類R0」はmaskを作り直して元noiseから生成した結果です。全てSlider +4です。</p>')
    for name,path,cid in routes:
        parts.append('<h3>'+name+'</h3>'+img(path,name+' 初期maskと再分類mask')+'<details><summary>D-R0とA-R0を重ねて比較する</summary><p>左側がD、右側がA。境界を動かすと細部の差を見られます。</p><div class="compare" style="--split:50%"><img src="'+rel(image_record(root,cid,'replay','A-R0'))+'" alt="A-R0"><img class="top" src="'+rel(image_record(root,cid,'replay','D-R0'))+'" alt="D-R0"></div><input type="range" min="0" max="100" value="50" aria-label="D/A比較の境界" oninput="this.previousElementSibling.style.setProperty(\'--split\',this.value+\'%\')"></details>')
    parts.append('</section><section id="seed"><h2>4. R0〜R4：顔の分類がどう変わるか</h2><p>同じD pilotの途中予測を、5方式のraw分類で比較します。これは完成画像の比較ではありません。上段base、下段Slider。青い顔領域は「保護役割への誤割当」、黄色は未確定です。</p><p>R0：全seed固定 ／ R1：固定なし ／ R2：最大9固定 ／ R3：prototype poolをbase支持で絞る・固定なし ／ R4：同じpool絞り込み＋R2と同じ固定。</p><p class="note">緑＝対象、青＝保護役割に分類、黄色＝未確定。R1〜R4の完成画像は選定条件不成立のため生成していません。</p>')
    for name,path in seed_panels:parts.append('<h3>'+name+'：顔の周囲だけ拡大</h3>'+img(path,name+' R0からR4のraw顔分類'))
    parts.extend(['</section><section id="release"><h2>5. 公園：固定保護の1セルだけ解除</h2><p>同じD-R0のtargetを1セル増やし、protectedを同じ1セルだけ減らした診断対照です。</p>'+img(output/'05-one-cell-mask-detail.png','解除前後の顔mask拡大')+'<p>base顔の確認済み内部セルは8/9→9/9。これは疎な9セルのlabelだけの数値で、顔全体の解除ではありません。解除後も他の青い部分は保護されたままです。完成顔は解除なしでも保持され、明瞭な追加改善は未確認です。</p>'+img(output/'04-one-cell-release.png','1セル解除前後の完成像')+'</section>',
      '<section id="remnant"><h2>6. 要注意：重なりseed123の白い残片</h2><p>2つの頭部は保たれていますが、左肩後方に旧身体状の白い残片が残りました。人数・残像gateは不合格、接触境界の保護は判定不能です。</p>'+img(output/'06-remnant-detail.png','旧身体状残片の位置と拡大')+'</section>',
      '<section id="all"><h2>7. 完成画像17枚の全一覧</h2><p>先頭の6条件と重複する画像も含め、実際に生成した全画像をphase別に掲載しています。</p><div class="grid">'])
    names={'pilot':'初期maskによる比較元','replay':'再分類maskによる生成','release':'1セル解除の診断'}
    for row in rows:
        title=scene(row['cid'])+' seed '+row['cid'].rsplit('_',1)[1]+'｜'+row['label']
        parts.append('<article class="card"><h3>'+esc(title)+'</h3><p>'+names[row['phase']]+'</p>'+img(row['image'],title)+'</article>')
    parts.append('</div></section><p>目視所見・部位annotationはassistant、human_verified=false。全身精度や保護人物の画素完全保持を示すものではありません。<a href="'+rel(root/'report-r2.html')+'">詳しい数値レポート</a> ／ <a href="'+rel(root/'verification-r2.json')+'">検証JSON</a></p></main></html>')
    (output/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    if any(sha(p)!=digest for p,digest in original_hashes.items()):raise ValueError('An original image changed')
    (output/'manifest.json').write_text(json.dumps({'source_verification_sha256':sha(root/'verification-r2.json'),
        'source_images':{str(p):d for p,d in original_hashes.items()},'checked_mask_inputs':[str(p) for p in sorted(checked)],
        'new_diffusion_nfe':0,'generated_files':{p.name:sha(p) for p in output.iterdir() if p.is_file()}},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return output/'index.html'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--result-root',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();print(build(args.result_root,args.output_dir))
