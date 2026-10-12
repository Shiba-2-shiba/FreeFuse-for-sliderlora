"""Render verified paired-input experiments and fixed mask replay outcomes."""
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def read(path):return json.loads(path.read_text(encoding="utf-8"))


def main():
    root=Path(sys.argv[1]).resolve();v=read(root/"verification.json");notes=read(root/"assistant-visual-observations.json")
    esc=lambda value:html.escape(str(value),quote=True)
    rel=lambda path:esc(Path(path).relative_to(root).as_posix())
    parts=["<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>同じMix入力によるmask検証</title>",
           "<style>body{margin:0;background:#f2f4f7;color:#182838;font:15px/1.7 system-ui}main{max-width:1700px;margin:auto;padding:24px}article,.box{background:white;margin:18px 0;padding:18px;border-radius:8px}img{max-width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.grid article{margin:0}a{color:#1766a8}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #dde1e7;text-align:left}pre{white-space:pre-wrap}.mask{image-rendering:pixelated;max-width:240px}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style><main>",
           "<h1>Prediction Mixと同じ入力からmaskを作る追加検証</h1>",
           f"<div class='box'><strong>{esc(notes['conclusion'])}</strong><p>公園seed444444・女性手前の重なりseed42 / 同じモデル・style・Slider +4 / Euler simple 8 steps CFG1 / 全radius0。</p>",
           f"<p>追加diffusion評価 {v['diffusion_model_nfe']} NFE。3軌道の6完成画像＋修正maskの4画像を生成。{v['verified_taps']} tapを検査。保存済み6条件のtrace復号は0 diffusion NFEで、最終RGBを元画像と完全一致で再現しました。</p>",
           "<p>両構図の観測付きDは、元Dと全stepの入力・base/LoRA/混合予測・最終latentが完全一致。交差再評価では元重みの予測とtap全tensor hashも一致。4修正mask生成は全stepのinside=LoRA/outside=baseと固定保護を監査しました。</p>",
           "<p>補助目視はCodexによる観察で、人の承認ではありません。独立人物labelとholdoutによる精度・汎化は未検証です。</p>",
           "<p><a href='verification.json'>最終数値監査</a> · <a href='verification.probe.json'>交差観測監査</a> · <a href='assistant-visual-observations.json'>補助目視・条件付きgate</a> · <a href='plan.json'>凍結条件</a> · <a href='cpu-final-green.log'>CPU 713 passed / 2 skipped / 85 subtests</a> · <a href='../../../.omx/plans/2026-10-12-prediction-mix-mask-validation.md'>実装計画</a></p></div>",
           "<article><h2>三人状配置の切り分け結果</h2><p>同じstep3の入力をbase/LoRA両方で評価しました。重なり構図の全体ON入力は、LoRAを外して評価しても三人に見え、LoRA側も三人です。OFF入力・実際のMix入力は両重みで二人の配置でした。三人状maskは、独立全体ON軌道の配置変化を捉えていたことと整合します。</p><p>この2構図・seedの証拠から、全体ONの独立軌道を実際のMixのmask収集元にする方法は採用せず、Mix入力上の観測へ移す方向を支持します。全LoRA・全seedで同じ原因という主張はしません。</p></article>",
           "<table><tr><th>構図</th><th>交差観測NFE</th><th>修正mask再生NFE</th><th>総NFE</th><th>Dの非侵入・交差再現</th></tr>"]
    for c in v["cases"]:parts.append(f"<tr><td>{esc(c['case'])}</td><td>{c['probe_nfe']}</td><td>{c['nfe']-c['probe_nfe']}</td><td>{c['nfe']}</td><td>全予測・latent・tap hash一致</td></tr>")
    parts.append("</table><p>交差観測80 NFEは計画の72に、元重みを再評価して完全一致を確かめる8 NFEを追加したものです。VAE・CPU後処理・backend再読込の費用はNFEに含めません。</p>")
    for c in v["cases"]:
        cid=c["case"];case_notes=notes["cases"][cid];probe=read(root/cid/"probe.artifacts.json")
        pilot=next(r for r in probe["diagnostics"] if r["label"]=="probe_mix");p=Path(pilot["local_report"]);r=read(p)
        parts.append(f"<h2>{esc(cid)}</h2><p>{esc(case_notes['interpretation'])}</p><div class='grid'><article><h3>D対照・観測付き</h3><a href='{rel(p.parent/r['artifacts']['image'])}'><img src='{rel(p.parent/r['artifacts']['image'])}'></a><p>元Dと全予測・latent一致</p></article>")
        for candidate in c["replays"]:
            label=candidate["label"];manifest=read(root/cid/"candidates"/label/"manifest.json")
            parts.append(f"<article><h3>{esc(label)}</h3><a href='{rel(candidate['image'])}'><img src='{rel(candidate['image'])}'></a><p>{esc(case_notes['routes'][label])}</p><p>対象{candidate['target_cells']}セル / D比追加{manifest['added_vs_pilot_cells']}セル</p><details><summary>実効target mask</summary><img class='mask' src='{cid}/candidates/{label}/target.png'></details><p><a href='{rel(candidate['diagnostic'])}'>予測監査</a> · <a href='{cid}/candidates/{label}/manifest.json'>mask由来</a></p></article>")
        parts.append(f"</div><details><summary>拡大した身体境界</summary><img src='{cid}/boundary-crops.png'></details><details open><summary>同じ入力で重みを変える6条件：step3/block26</summary><a href='{cid}/causal-step3-b26.png'><img src='{cid}/causal-step3-b26.png'></a></details><details><summary>同じ6条件：block18</summary><a href='{cid}/causal-step3-b18.png'><img src='{cid}/causal-step3-b18.png'></a></details><details><summary>既存A/C/Dのbranch予測と最終画像</summary><img src='{cid}/trace-step3-comparison.png'></details>")
        if (root/cid/"trajectory-final-comparison.png").is_file():parts.append(f"<details><summary>OFF・全体ON・Mixの完成画像</summary><img src='{cid}/trajectory-final-comparison.png'></details>")
        fig,axes=plt.subplots(4,4,figsize=(11,11),constrained_layout=True)
        for row,name in enumerate(("baseline","seed_only","shape_only","combined")):
            for column,(branch,role) in enumerate((b,r) for b in ("base","slider") for r in ("target","protected")):
                path=root/cid/"ablations"/f"{branch}_s3_b18_{name}"/(role+".png")
                if path.exists():axes[row,column].imshow(Image.open(path),cmap="gray",vmin=0,vmax=255,interpolation="nearest")
                axes[row,column].set_title(name+" / "+branch+" "+role,fontsize=9);axes[row,column].axis("off")
        fig.savefig(root/cid/"ablation-b18.png",dpi=120);plt.close(fig)
        parts.append(f"<details><summary>同じMix入力の4 ablation：block18</summary><img src='{cid}/ablation-b18.png'><p>baseline / seedのみ修正 / 非seedの輪郭再分類のみ / 両方。最終候補は旧D対象とbase/LoRAのcombined対象を合わせ、同じD保護を引きます。</p><a href='{cid}/ablation-summary.json'>全block18/23の条件・統計</a></details>")
    parts.append("<article><h2>採用と次の段階</h2><p>"+esc(notes["quality_gate"]["reason"])+"</p><p>holdoutと動的maskの条件付き段階には進んでいません。観測・再分類・再生の実装と数値検証は完了していますが、今回の候補を本番の既定maskとして採用する判定ではありません。</p><p>最終レビューは、設定されたnative reviewerモデルが利用できなかったため、チェックリストを使った著者の別パスによる自己レビューです。未完了maskの拒否、内容変更時cache、Windowsファイル保持を修正し、CPU回帰を実施しました。</p></article></main></html>")
    (root/"report.html").write_text("".join(parts),encoding="utf-8");print(root/"report.html")


if __name__=="__main__":main()
