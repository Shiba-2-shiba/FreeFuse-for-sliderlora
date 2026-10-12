"""Render reproducible seed/initialization/collision/holdout evidence as local HTML."""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw

from run_seed_bootstrap_validation import read,sha


def build_report(experiment_path,verification_path,output_path):
    plan=read(experiment_path);root=Path(plan['output_root']);verified=read(verification_path);output_path=Path(output_path)
    if output_path.exists():raise FileExistsError(output_path)
    if verified['experiment_sha256']!=sha(experiment_path):raise ValueError('Verification belongs to different experiment')
    for name,digest in verified['output_file_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Verified output changed before report: '+name)
    selection=read(root/'selection.json');gates=read(root/'gates.json');holdout=read(root/'holdout.json')
    esc=lambda x:html.escape(str(x),quote=True)
    rel=lambda p:esc(Path(os.path.relpath(p,output_path.parent)).as_posix())
    link=lambda p,t:f'<a href="{rel(p)}">{esc(t)}</a>'
    image=lambda p,t:f'<figure>{link(p,t)}<a href="{rel(p)}"><img src="{rel(p)}" alt="{esc(t)}"></a></figure>'
    parts=['<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
      '<title>Seed固定・base-only初期化の検証</title><style>body{font:15px/1.65 system-ui;background:#f3f5f8;color:#14283b;margin:0}main{max-width:1600px;margin:auto;padding:24px}section{background:white;padding:20px;border-radius:10px;margin:18px 0}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0}figure a{display:block}pre{white-space:pre-wrap;overflow-wrap:anywhere}.warning{background:#fff2d7;padding:12px}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}</style><main>',
      '<h1>Seed固定・base-only初期化の検証</h1>',
      f'<p>実測 {verified["diffusion_model_nfe"]} diffusion NFE、完成画像 {verified["final_generation_images"]}、監査済みraw bank {verified["verified_raw_banks"]}。Trace復号 {verified["verified_trace_bundles"]} bundle は0 diffusion NFE、VAE／queue時間は各timing記録。</p>',
      '<p class="warning">seed固定を緩めたR1〜R4は顔・手足・保護の非悪化条件を満たさず、R*は未選定。base-only経路はR0対照で検証する。別seedの経路確認を新方式の成功や既定採用と混同しない。全annotation／目視はassistant、human_verified=false。</p>',
      '<p>'+link(experiment_path,'実験条件')+' / '+link(verification_path,'数値監査')+' / '+link(root/'selection.json','候補選定と不適格理由')+' / '+link(root/'gates.json','問題別gate')+' / '+link(root/'holdout.json','別seed検証')+'</p>',
      '<section><h2>保存入力でのseed比較</h2><p>R0=P_full＋全seed固定、R1=P_full＋固定なし、R2=P_full＋最大9固定、R3=P_supported＋固定なし、R4=P_supported＋R2と同じ固定。prototype用poolと固定セルを分離し、同じ4branchで比較。</p>',
      '<table><tr><th>構図</th><th>方式</th><th>base顔の欠落</th><th>Slider顔の欠落</th><th>共通背景選択</th><th>背景誤固定</th><th>固定セル数</th></tr>']
    for case in plan['cases']:
        rows=read(root/case['case_id']/'scoring_D/index.json')['variants']
        for variant,row in rows.items():
            if row['status']!='ok':parts.append(f'<tr><td>{esc(case["prompt_case"])}</td><td>{esc(variant)}</td><td colspan="5">failed</td></tr>');continue
            face=[row['branches'][b]['metrics']['target_face']['missing_cells'] for b in ('base','slider')]
            parts.append('<tr>'+''.join(f'<td>{esc(v)}</td>' for v in (case['prompt_case'],variant,*face,row['metrics']['common_background_cells'],row['fixed_background_errors'],row['fixed_cells']))+'</tr>')
    parts.extend(['</table><p>疎なpilot RGB内部ラベルでのセル数。全顔・全身体精度、生成後のprecision/IoU、校正済みconfidenceではない。</p><details><summary>不適格理由</summary><pre>'+esc(json.dumps(selection['rejected'],ensure_ascii=False,indent=2))+'</pre></details></section>'])
    for case in plan['cases']:
        cid=case['case_id'];folder=root/cid;parts.append('<section><h2>'+esc(cid)+'</h2><p>base収集は2 forward。B_T/B_Pと旧A、B_Pと旧Dは完全一致。I_D/I_Aは保存済みD/Aと全stepの入力・branch／Mix予測・最終latentが完全一致。</p><div class="grid">')
        for label in ('D','A'):
            pilot=read(folder/('pilot_'+label)/'artifacts.json');record=pilot['diagnostics'][0];p=Path(record['local_report']);r=read(p)
            parts.append(image(p.parent/r['artifacts']['image'],'I_'+label+' pilot'))
            artifact=read(folder/('replay_'+label+'-R0')/'artifacts.json');p=Path(artifact['diagnostics'][0]['local_report']);r=read(p)
            parts.append(image(p.parent/r['artifacts']['image'],label+'-R0 fixed replay'))
        parts.append('</div><h3>同入力のbranch RGBとraw／保護減算</h3><div class="grid">')
        for label in ('D','A'):
            pilot=read(folder/('pilot_'+label)/'artifacts.json')
            for branch in ('base','slider'):parts.append(image(pilot['references'][branch]['image'],label+' '+branch+' step3'))
        parts.append('</div><table><tr><th>pilot</th><th>branch</th><th>顔raw欠落</th><th>顔保護後欠落</th><th>unknown率</th></tr>')
        for label in ('D','A'):
            evaluation=read(folder/('evaluation_'+label+'.json'))['R0']
            for branch in ('base','slider'):
                b=evaluation['branches'][branch]
                parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in (label,branch,b['raw']['target_face']['missing_cells'],b['after_protection']['target_face']['missing_cells'],b['raw']['unknown']['fraction']))+'</tr>')
        parts.append('</table><h3>7分類の保護衝突</h3><table><tr><th>pilot</th><th>分類</th><th>セル数</th></tr>')
        for label in ('D','A'):
            counts=read(folder/('evaluation_'+label+'.json'))['R0']['class_counts']
            for name,n in counts.items():parts.append('<tr><td>'+label+'</td><td>'+esc(name)+'</td><td>'+str(n)+'</td></tr>')
        parts.append('</table><p>'+link(folder/'release-review.json','セル別の解除確認')+'</p>')
        release=read(folder/'release_D-R0/artifacts.json')
        if release['status']=='complete':
            p=Path(release['diagnostics'][0]['local_report']);r=read(p)
            parts.append('<p>診断解除：'+esc(release['released_cells'])+'。target増分とprotected減分がLだけに一致。方式選定／holdoutには不使用。</p>'+image(p.parent/r['artifacts']['image'],'D-R0 individual diagnostic release'))
        else:parts.append('<p>診断解除：not_applicable。'+esc(release['reason'])+'</p>')
        parts.append('<h3>問題別gate</h3><pre>'+esc(json.dumps(gates['cases'][cid],ensure_ascii=False,indent=2))+'</pre><details><summary>途中予測・trace</summary><ul>')
        for trace in sorted(folder.glob('trace_*/artifacts.json')):
            row=read(trace);bundle=Path(row['bundle']['local']);manifest=read(bundle/'manifest.json')
            parts.append('<li>'+link(bundle/'manifest.json',trace.parent.name)+'<div class="grid">')
            for item in manifest['rows']:
                if item['branch']=='trace_predictions':parts.append(image(bundle/item['image'],'Mix step '+str(item['step'])))
            parts.append('</div></li>')
        parts.append('</ul></details></section>')
    final_assessment=read(root/'holdout-assessment.json') if (root/'holdout-assessment.json').exists() else None
    current_holdout={**holdout,'cases':[dict(c,assessment=final_assessment['cases'][c['case_id']]['status']) if final_assessment else c for c in holdout['cases']]}
    parts.append('<section><h2>別seedの確認</h2><p>下は最終評価を反映した状態。生成時点の記録は'+link(root/'holdout.json','holdout.json')+'、最終評価は'+(link(root/'holdout-assessment.json','holdout-assessment.json') if final_assessment else '未完了')+'。</p><pre>'+esc(json.dumps(current_holdout,ensure_ascii=False,indent=2))+'</pre>')
    for entry in holdout['cases']:
        folder=root/entry['case_id']
        if folder.exists():
            parts.append('<h3>'+esc(entry['case_id'])+'</h3><div class="grid">')
            for phase in ('pilot_A','replay_A-R0'):
                p=folder/phase/'artifacts.json'
                if p.exists():
                    row=read(p);report_path=Path(row['diagnostics'][0]['local_report']);report=read(report_path)
                    parts.append(image(report_path.parent/report['artifacts']['image'],phase))
            pilot=folder/'pilot_A/artifacts.json'
            if pilot.exists():
                for branch,reference in read(pilot)['references'].items():parts.append(image(reference['image'],branch+' step3 RGB label source'))
            parts.append('</div>')
            assessment=folder/'holdout-assessment.json'
            if assessment.exists():parts.append('<pre>'+esc(json.dumps(read(assessment),ensure_ascii=False,indent=2))+'</pre>')
    parts.append('</section><section><h2>制約・費用・再現</h2><p>未知・遮蔽・blur領域を背景／非保護へ読み替えない。完成像の顔・人数・保護・背景・体型は別判定。局所解除は原因診断で自動運用しない。新R*は未選定で既定auto／公開node schemaは変更していない。</p><pre>'+esc(json.dumps(verified,ensure_ascii=False,indent=2))+'</pre></section></main></html>')
    output_path.write_text('\n'.join(parts),encoding='utf-8')
    return output_path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment-plan',type=Path,required=True);parser.add_argument('--verification',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(build_report(args.experiment_plan,args.verification,args.output))
