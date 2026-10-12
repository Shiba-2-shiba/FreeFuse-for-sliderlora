"""Render early-block supplement alongside the retained late-block experiment."""
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from build_attention_step_block_sweep_report import figures, read


def main():
    torch.set_num_threads(4)
    root=Path(sys.argv[1]).resolve()
    plan=read(root/"plan.json");v=read(root/"verification.json")
    prior=Path(plan["prior_root"]);old=read(prior/"verification.json")
    park=next(c for c in v["cases"] if c["case"]=="park")
    old_park=next(c for c in old["cases"] if c["case"]=="park")
    rows=sorted(park["hole_probe"]["on"]+old_park["hole_probe"]["on"],key=lambda r:(r["block"],r["step"]))
    assert {(r["block"],r["step"]) for r in rows}=={(b,s) for b in range(1,28) for s in (1,2,3)}
    fig,axes=plt.subplots(1,3,figsize=(27,3.4),constrained_layout=True)
    for ax,key,maximum,title in zip(axes,("roi_effective_target_cells","qk_role_mean","heads_target_wins_on_mean_roi"),(70,1,48),
                                    ("Effective target cells in old 70-cell hole","Q/K target role score (heuristic)","Heads with target winning ROI mean")):
        array=np.full((3,27),np.nan)
        for row in rows:
            value=row.get(key,np.nan)
            if isinstance(value,dict):value=value["target"]
            array[row["step"]-1,row["block"]-1]=value
        ax.imshow(array,vmin=0,vmax=maximum,cmap="viridis",aspect="auto")
        for (s,b),value in np.ndenumerate(array):
            ax.text(b,s,(f"{value:.2f}" if maximum==1 else str(int(value))) if np.isfinite(value) else "×",ha="center",va="center",fontsize=6,color="white" if np.isfinite(value) else "black")
        ax.set_xticks(range(27),range(1,28));ax.set_yticks(range(3),[1,2,3]);ax.tick_params(labelsize=7)
        ax.set_xlabel("actual block index");ax.set_ylabel("step");ax.set_title(title,fontsize=10)
        ax.axvline(12.5,color="white",linewidth=1)
    fig.savefig(root/"park"/"blocks1-27-fixed-hole-summary.png",dpi=140);plt.close(fig)
    analysis={"scope":"Fixed old park 70-cell hole coordinates; same controls verified; not independent face labels or image quality",
              "block_indices":"Actual module indices 1..27, zero-based; block0 outside this request",
              "rows":rows,"prior_root":str(prior),"supplement_root":str(root),
              "best_observed_roi_conditions":sorted([r for r in rows if r["status"]["status"]=="ok"],key=lambda r:r["roi_effective_target_cells"],reverse=True)[:6]}
    with (root/"blocks1-27-analysis.json").open("w",encoding="utf-8") as f:json.dump(analysis,f,indent=2,ensure_ascii=False,allow_nan=False)
    esc=lambda text:html.escape(str(text),quote=True)
    total_success=sum(sum(c["successful_new_masks"].values()) for c in v["cases"])
    best=analysis["best_observed_roi_conditions"]
    best_labels="、".join(f"step{row['step']}/block{row['block']}" for row in best if row["roi_effective_target_cells"]==best[0]["roi_effective_target_cells"])
    parts=["<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>block 1〜13 追加検証</title>",
           "<style>body{margin:0;font:15px/1.7 system-ui;background:#f2f4f7;color:#192836}main{max-width:1600px;margin:auto;padding:24px}article{background:white;padding:18px;margin:20px 0;border-radius:8px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border-bottom:1px solid #dde1e7;text-align:left}a{color:#1766a8}summary{cursor:pointer}pre{white-space:pre-wrap}</style><main>",
           "<h1>不足していたblock 1〜13の追加検証</h1>",
           f"<article><p>起動中の同じComfyUI :8000 / step 1・2・3 × block 1〜13 × LoRAなし／あり × 2構図 = <strong>156条件追加</strong>。{total_success}条件でmask成立、{156-total_success}条件は理由付きで不成立を保存。前回block 14〜27と合わせ、block 1〜27の324条件を網羅しました。</p>",
           "<p>block18を全step・両branch・両構図で再収集し、12対照tapの全保存tensor hashと候補statusが前回に完全一致しました。収集は各branchで共有3 forward、今回12 NFE、前回188と合わせ200 NFE（元の中断試行は別）。prompt、seed、trial、LoRA強度+4、k3方式・閾値、保護対照、radius 0を維持しています。</p>",
           "<p>新しいblockの編集画像は生成していません。mask・attentionの成立と、顔・全身・保護人物の画質合格は別です。番号は前回と同じ実装上のblock indexで、block0は今回の指定範囲外です。</p>",
           f"<p><strong>公園の旧70セルの穴は、{esc(best_labels)}で{best[0]['roi_effective_target_cells']}/70セルが実効対象へ戻りました。</strong> 前回のstep1/block18の49/70より高い被覆です。早いblockのmaskは周囲の散在成分もあり、顔の被覆だけで採用せず、全身・保護側・生成画像を確認する候補として扱います。</p>",
           f"<p><a href='../{esc(prior.name)}/report.html'>前回11画像のレポート</a> · <a href='plan.json'>追加条件</a> · <a href='verification.json'>保存監査と対照一致</a> · <a href='blocks1-27-analysis.json'>block 1〜27の全数値</a></p></article>",
           "<h2>公園：旧70セルの穴を同じ座標で比較</h2><article><p>左から実効対象セル数、Q/K対象役割スコア、ROI平均で対象が最大のhead数。縦線は今回block1〜13と前回14〜27の境目です。Q/Kスコアは3役割内のheuristicで、校正済みconfidenceではありません。</p><a href='park/blocks1-27-fixed-hole-summary.png'><img src='park/blocks1-27-fixed-hole-summary.png'></a></article>",
           "<table><tr><th>ON step / block</th><th>対象 / 保護 / 背景（70セル）</th><th>実効対象 / 70</th><th>Q/K 対象 / 保護</th><th>対象優勢head / 48</th></tr>"]
    for row in rows:
        routing=row.get("roi_routing")
        counts="failed" if routing is None else " / ".join(str(routing[r]) for r in ("target","protected","background"))
        scores=row["qk_role_mean"]
        parts.append(f"<tr><td>{row['step']} / {row['block']}</td><td>{counts}</td><td>{row.get('roi_effective_target_cells','failed')}</td><td>{scores['target']:.4f} / {scores['protected']:.4f}</td><td>{row['heads_target_wins_on_mean_roi']}</td></tr>")
    parts.append("</table>")
    review_path=root/"mask-shape-review"/"observations.json"
    if review_path.is_file():
        review=read(review_path)
        review_parts=["<article><h2>人物構図・輪郭・役割割当の再評価</h2><p>"+esc(review["conclusion"])+"</p>",
                      "<p><a href='mask-shape-review/observations.json'>ユーザー所見と保存maskの確認</a> · <a href='mask-shape-review/selected-mask-evidence.json'>選択maskのhash検査・形状統計</a></p>"]
        for cid,notes in review["cases"].items():
            review_parts.append(f"<h3>{esc(cid)}</h3><p>{esc(notes['assistant_verified'])}</p><p>{esc(notes['interpretation'])}</p><a href='mask-shape-review/{cid}-shape-comparison.png'><img src='mask-shape-review/{cid}-shape-comparison.png'></a>")
            for key in ("geometry_counterexample","unknown","generation_mask_control"):
                if key in notes:review_parts.append("<p>"+esc(notes[key])+"</p>")
        review_parts.append("<p>"+esc(review["next_discriminating_probe"])+"</p></article>")
        parts.insert(2,"".join(review_parts))
    for case in v["cases"]:
        cid=case["case"];figures(root,case,blocks=plan["new_blocks"])
        parts.append(f"<h2>{esc(cid)} / seed {case['seed']}</h2><p>追加39条件中、OFF {case['successful_new_masks']['off']}成立・ON {case['successful_new_masks']['on']}成立。block18対照は全3step・両branchで一致。</p>")
        for branch in ("off","on"):
            parts.append(f"<article><h3>{branch.upper()} block1〜13</h3><p>黄色：対象、青：保護、紫：不確定、暗色：実背景。成立は正確な人物labelの保証ではありません。</p><a href='{cid}/{branch}-routing-grid.png'><img src='{cid}/{branch}-routing-grid.png'></a><details><summary>前回block14〜27</summary><a href='../{esc(prior.name)}/{cid}/{branch}-routing-grid.png'><img src='../{esc(prior.name)}/{cid}/{branch}-routing-grid.png'></a></details></article>")
        parts.append("<details><summary>追加blockの不成立条件</summary><pre>"+esc(json.dumps(case["failed_new_taps"],ensure_ascii=False,indent=2))+"</pre></details>")
    parts.append("</main></html>")
    (root/"report.html").write_text("".join(parts),encoding="utf-8")
    print("SUPPLEMENT",total_success,"successful masks out of 156")
    print("ROI top",[(r["step"],r["block"],r["roi_effective_target_cells"]) for r in analysis["best_observed_roi_conditions"]])
    print(root/"report.html")


if __name__=="__main__":main()
