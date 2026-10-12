"""Render verified step/block masks, fixed-ROI statistics and edited images."""
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PIL import Image, ImageDraw
import torch
from safetensors.torch import load_file


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def figures(root, case, blocks=None):
    cid = case["case"]
    blocks = list(range(14,28)) if blocks is None else blocks
    for branch in ("off", "on"):
        folder = root/cid/"sweeps"/"mask"/branch
        outer = read(folder/"manifest.json")
        fig, axes = plt.subplots(3,len(blocks),figsize=(1.8*len(blocks),6.6),constrained_layout=True,squeeze=False)
        for step in (1,2,3):
            for column,block in enumerate(blocks):
                ax=axes[step-1,column]
                item=outer["taps"][f"step_{step}.block_{block}"]
                ax.set_title(f"s{step} b{block}",fontsize=9)
                if item["status"]["status"] != "ok":
                    ax.text(.5,.5,"FAILED\n"+item["status"]["error_code"],ha="center",va="center",fontsize=7,transform=ax.transAxes)
                else:
                    path=folder/item["manifest"]
                    tensors=load_file(str(path.parent/"maps.safetensors"))
                    labels=torch.zeros(64,64,dtype=torch.int64)
                    labels[tensors["multi_proto_bg.masks.target"][0].bool()]=1
                    labels[tensors["multi_proto_bg.masks.protected"][0].bool()]=2
                    labels[tensors["multi_proto_bg.uncertain_mask"][0].bool()]=3
                    ax.imshow(labels.numpy(),vmin=0,vmax=3,cmap=ListedColormap(["#242b34","#f4c14c","#4598dc","#b374c7"]),interpolation="nearest")
                ax.axis("off")
        fig.suptitle(f"{cid} | {branch.upper()} | k3 multi_proto_bg | yellow: target, blue: protected, purple: uncertain, dark: real background",fontsize=12)
        fig.savefig(root/cid/f"{branch}-routing-grid.png",dpi=120)
        plt.close(fig)
    if "edits" not in case:
        return
    rows=case["edits"]
    sheet=Image.new("RGB",(3*512,3*548),"#edf0f4")
    draw=ImageDraw.Draw(sheet)
    for index,edit in enumerate(rows):
        col=edit["step"]-1; row=(18,22,27).index(edit["block"])
        image=Image.open(edit["image"]).convert("RGB");image.thumbnail((512,512))
        sheet.paste(image,(col*512,row*548+36))
        draw.text((col*512+12,row*548+8),f"ON step {edit['step']} / block {edit['block']} | {edit['target_cells']} target cells",fill="#172331")
    for edit in case["not_generated"]:
        col=edit["step"]-1; row=(18,22,27).index(edit["block"])
        draw.text((col*512+12,row*548+8),f"step {edit['step']} / block {edit['block']}: NOT GENERATED",fill="#172331")
    sheet.save(root/cid/"edit-contact-sheet.png")
    if "hole_probe" in case:
        fig,axes=plt.subplots(1,3,figsize=(18,3),constrained_layout=True)
        metrics=(("roi_effective_target_cells",70,"Old hole ROI: effective target cells / 70"),
                 ("qk_role_mean",1,"Q/K target role score (heuristic)"),
                 ("heads_target_wins_on_mean_roi",48,"Heads with target winning ROI mean / 48"))
        for ax,(key,maximum,title) in zip(axes,metrics):
            array=np.full((3,14),np.nan)
            for item in case["hole_probe"]["on"]:
                value=item.get(key,np.nan)
                if isinstance(value,dict): value=value["target"]
                array[item["step"]-1,item["block"]-14]=value
            ax.imshow(array,vmin=0,vmax=maximum,cmap="viridis",aspect="auto")
            for (row,col),value in np.ndenumerate(array):
                if np.isfinite(value): ax.text(col,row,f"{value:.2f}" if maximum==1 else f"{int(value)}",ha="center",va="center",color="white",fontsize=8)
                else: ax.text(col,row,"failed",ha="center",va="center",fontsize=7)
            ax.set_xticks(range(14),range(14,28));ax.set_yticks(range(3),[1,2,3]);ax.set_xlabel("block");ax.set_ylabel("step");ax.set_title(title,fontsize=10)
        fig.savefig(root/cid/"fixed-hole-summary.png",dpi=140);plt.close(fig)


def main():
    torch.set_num_threads(4)
    root=Path(sys.argv[1]).resolve()
    verification=read(root/"verification.json")
    if "--figures-only" in sys.argv:
        for case in verification["cases"]: figures(root,case)
        return
    observations=read(root/"assistant-visual-observations.json")
    esc=lambda text:html.escape(str(text),quote=True)
    rel=lambda path:esc(Path(path).relative_to(root).as_posix())
    parts=["<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Attention step / block sweep</title>",
           "<style>body{background:#f2f4f7;color:#192836;font:15px/1.7 system-ui;margin:0}main{max-width:1600px;margin:auto;padding:24px}article,.box{background:white;padding:18px;margin:18px 0;border-radius:8px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border-bottom:1px solid #dde1e7;text-align:left}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.grid article{margin:0}.mask{max-width:240px;image-rendering:pixelated}a{color:#1766a8}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><main>",
           "<h1>Attention step / block sweep：中断検証の続行結果</h1>",
           "<p>起動済みComfyUI :8000 / 公園seed 444444・女性手前の重なりseed 42 / 1024×1024 / Slider +4 / Euler simple 8 steps CFG 1 / k3 multi_proto_bg / 全radius 0</p>",
           f"<div class='box'><strong>{esc(observations['conclusion'])}</strong><p>収集はstep 1/2/3 × block 14〜27 × LoRAなし/あり × 2構図。画像生成は事前指定block 18/22/27の成功条件だけ。各生成はON対象から固定OFF step2/block18保護を引き、同じ保護maskを使います。</p>",
           f"<p>{verification['verified_mask_taps']} tapを検査、{verification['verified_edit_bundles']}編集bundleを保存・検証。今回の実測{verification['total_model_nfe']} NFE。中断済み試行の既知3 NFEと不明分は含めません。hash、元スコアからのmask再構成、固定保護、全stepのinside=Slider/outside=base、初回base/Slider予測の候補間一致を確認しました。</p>",
           "<p>mask被覆は独立した顔labelに対する精度ではありません。補助目視はCodexの観察であり、人の承認・未見seedの汎化・未編集OFF画像への保持を証明しません。</p>",
           "<p><a href='plan.json'>凍結条件</a> · <a href='continuation.json'>中断からの引継ぎ</a> · <a href='summary.json'>実行一覧</a> · <a href='verification.json'>機械検証と70セル診断</a> · <a href='assistant-visual-observations.json'>補助目視</a> · <a href='../../mask-source-comparison-results/20261011T104616Z/report.html'>前回比較</a></p></div>"]
    if (root/"range-supplement.json").is_file():
        supplement=read(root/"range-supplement.json")
        target=Path(supplement["root"])
        parts.insert(4,f"<div class='box'><strong>block 1〜13の追加検証が完了しました。</strong> <a href='../{esc(target.name)}/report.html'>block 1〜27の統合比較と追加mask・attentionを見る</a><p>このページは当初のblock14〜27・11編集画像の記録です。</p></div>")
    for case in verification["cases"]:
        cid=case["case"];note=observations["cases"][cid]
        parts.extend([f"<h2>{esc(cid)} / seed {case['seed']}</h2><p>{esc(note['interpretation'])}</p>",
                      f"<p>{len(case['edits'])}画像 / {case['nfe']} NFE / 生成対象外{len(case['not_generated'])}条件。前回ON step2/block18 partition一致：{esc(case['prior_s2_b18_on_mask_equal'])}</p>"])
        for name,title in (("face-comparison-crops.png","前回4経路と今回6候補の顔・首crop"),
                           ("prior-comparison.png","前回4経路の全身比較")):
            if (root/cid/name).is_file():
                parts.append(f"<details><summary>{esc(title)}</summary><a href='{cid}/{name}'><img src='{cid}/{name}'></a></details>")
        if "hole_probe" in case:
            parts.append(f"<article><h3>同じ70セルの抜けを全step/blockで比較</h3><p>前回公園ON step2/block18の閉じた穴を固定座標で評価。白率や役割スコアの改善は生成画質の合格とは別です。</p><a href='{cid}/fixed-hole-summary.png'><img src='{cid}/fixed-hole-summary.png'></a><table><tr><th>step / block</th><th>対象 / 保護 / 背景</th><th>実効対象 / 70</th><th>Q/K対象 / 保護</th><th>対象優勢head</th></tr>")
            for row in case["hole_probe"]["on"]:
                if row["block"] not in (14,18,22,27): continue
                routing=row.get("roi_routing")
                counts="failed" if routing is None else " / ".join(str(routing[role]) for role in ("target","protected","background"))
                scores=row["qk_role_mean"]
                parts.append(f"<tr><td>{row['step']} / {row['block']}</td><td>{counts}</td><td>{row.get('roi_effective_target_cells','failed')}</td><td>{scores['target']:.4f} / {scores['protected']:.4f}</td><td>{row['heads_target_wins_on_mean_roi']} / {row['head_count']}</td></tr>")
            parts.append("</table><p>block 14はmask観測のみで、画像生成の事前候補に含まれていません。今回の数値だけで採用しません。</p></article>")
        parts.append("<div class='grid'>")
        for edit in case["edits"]:
            parts.append(f"<article><h3>step {edit['step']} / block {edit['block']}</h3><a href='{rel(edit['image'])}'><img src='{rel(edit['image'])}'></a><p>{esc(note['routes'][edit['label']])}</p><p>実効対象{edit['target_cells']}セル · <a href='{rel(edit['local_report'])}'>診断JSON</a></p><details><summary>実効mask</summary><img class='mask' src='{rel(edit['effective_mask'])}'></details></article>")
        parts.append("</div><details><summary>全42条件の対象／保護／不確定分類（OFF / ON）</summary>")
        for branch in ("off","on"):
            parts.append(f"<h3>{branch.upper()}</h3><a href='{cid}/{branch}-routing-grid.png'><img src='{cid}/{branch}-routing-grid.png'></a>")
        parts.append("</details><details><summary>候補失敗と生成対象外</summary><pre>"+esc(json.dumps({"failed_taps":case["failed_taps"],"not_generated":case["not_generated"]},ensure_ascii=False,indent=2))+"</pre></details>")
    parts.append("<article><h2>次の判断</h2><p>"+esc(observations["next_decision"])+"</p><p>本番の既定step/blockやmask算法は変更していません。今回の結果は既存k3方式の収集時期・blockを切り分ける検証です。</p></article></main></html>")
    with (root/"report.html").open("w",encoding="utf-8") as f:f.write("".join(parts))
    print(root/"report.html")


if __name__ == "__main__":main()
