"""Build a local comparison report from verified live experiment evidence."""
import html
import json
from pathlib import Path
import sys


def main():
    root = Path(sys.argv[1]).resolve()
    verification = json.loads((root / "verification.json").read_text(encoding="utf-8"))
    observations = json.loads((root / "assistant-visual-observations.json").read_text(encoding="utf-8"))
    def esc(value):
        return html.escape(str(value), quote=True)
    def link(path):
        return esc(Path(path).relative_to(root).as_posix())
    labels = {
        "A_off": ("A：Sliderなし収集", "対象・保護ともSliderなしで収集"),
        "B_on": ("B：Sliderあり収集", "対象・保護ともSlider +4で収集"),
        "C_on_base_protected": ("C：ON対象＋OFF保護", "ON対象からOFF保護を除外"),
        "D_union_base_protected": ("D：対象の和集合＋OFF保護", "OFF/ON対象の和集合からOFF保護を除外"),
    }
    style = """
    body{margin:0;background:#f3f5f7;color:#172331;font:15px/1.7 system-ui,sans-serif}
    main{max-width:1680px;margin:auto;padding:24px}h1{font-size:26px}h2{margin-top:38px}
    .box,article{background:#fff;border-radius:10px;padding:18px;margin:16px 0}
    .grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}
    article{margin:0}h3{font-size:17px}img{width:100%;height:auto;display:block}
    .mask img{image-rendering:pixelated;background:#222}.mask{max-width:260px}
    table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #d9dfe7;text-align:left}
    a{color:#1766a8}code{overflow-wrap:anywhere}.alert{border-left:4px solid #b66522;padding-left:12px}
    .muted{color:#56677b}details{margin-top:12px}summary{cursor:pointer}
    .observations td:first-child{width:170px}button{padding:8px 14px;margin:10px 8px 10px 0;cursor:pointer}
    .focus .grid{grid-template-columns:repeat(2,minmax(0,1fr))}
    @media(max-width:950px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
    @media(max-width:550px){.grid,.focus .grid{grid-template-columns:1fr}main{padding:12px}}
    """
    parts = ["<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Sliderの有無によるmask収集比較</title><style>" + style + "</style><main>",
             "<h1>Sliderの有無によるmask収集比較</h1>",
             "<p>起動中のComfyUI / 公園 seed 444444・女性手前の重なり seed 42 / INT8 / Slider +4 / 1024×1024 / Euler・simple・8 steps・CFG 1</p>",
             "<div class='box'><p><strong>" + esc(observations["conclusion"]) + "</strong></p>",
             "<p>全4経路でk3 multi_proto_bg・収集step 2/block 18を固定。観測step [1,2]×block [16,18]。穴埋め・mask膨張・選択余白は0。変更点は収集時のSlider有無と、宣言したmask合成だけです。</p>",
             "<p>全4suite・8編集bundleのfile/tensor hashを検査。元suiteから各編集のpartitionを再構成し、全stepのinside=Slider/outside=base選択に完全一致。各ケースの全経路で初回base予測・Slider予測も一致。<strong>実測136 NFE。</strong></p>",
             "<p class='alert'>観察はCodexによる補助目視で、人の承認ではありません。独立labelによるmask精度、未見seed、独立no-hook parityは未検証です。画像の自然さと、目的の体格変化を分けて評価しています。</p>",
             "<p><a href='plan.json'>凍結条件</a> · <a href='verification.json'>機械検証</a> · <a href='summary.json'>実行一覧</a> · <a href='assistant-visual-observations.json'>補助目視</a> · <a href='current-assets.json'>実ファイルSHA-256</a></p></div>",
             "<table><tr><th>ケース</th><th>OFF対象</th><th>ON対象</th><th>対象IoU</th><th>ON対象/OFF保護の衝突</th><th>DのOFF比追加</th><th>NFE / wall</th></tr>"]
    for case in verification["cases"]:
        m = case["target_mask_comparison"]
        parts.append(f"<tr><td>{esc(case['case'])}</td><td>{m['off_cells']}</td><td>{m['on_cells']}</td><td>{m['iou']:.6f}</td><td>{case['on_target_off_protected_collision_cells']}</td><td>{case['union_added_vs_off_cells']}</td><td>{case['nfe']} / {case['wall_seconds']:.1f}s</td></tr>")
    parts.append("</table><p class='muted'>セル数とIoUは記述統計で、maskの正確さを表しません。wallはqueue単位で一度だけ計測しpolling遅延を含みます。</p><button onclick=\"document.body.classList.toggle('focus')\">4列／2列を切り替え</button>")
    for case in verification["cases"]:
        cid = case["case"]
        parts.append(f"<h2>{esc(cid)} / seed {case['seed']}</h2><p><code>{esc(case['prompt_id'])}</code> · <a href='{cid}/queued_api.json'>実投入API</a> · <a href='{cid}/history.json'>history</a></p><div class='grid'>")
        for route in case["routes"]:
            label = route["label"]
            title, definition = labels[label]
            notes = observations["cases"][cid]["routes"][label]
            parts.append(f"<article><h3>{esc(title)}</h3><p>{esc(definition)}</p><a href='{link(route['image'])}'><img src='{link(route['image'])}' alt='{esc(cid + ' ' + label)}'></a><p>{esc(notes['overall'])}</p><p>対象 {route['target_cells']} / 保護 {route['protected_cells']} cells</p><p><a href='{link(route['diagnostic'])}'>診断JSON</a></p><details><summary>実効maskを表示</summary><div class='mask'><a href='{link(route['effective_mask'])}'><img src='{link(route['effective_mask'])}' alt='effective mask'></a></div></details></article>")
        parts.append("</div><div class='box'><strong>補助目視：評価軸を分けた所見</strong><table class='observations'><tr><th>経路</th><th>対象の編集量・体格</th><th>破綻・輪郭</th><th>男性への影響</th></tr>")
        for route in case["routes"]:
            label = route["label"]
            notes = observations["cases"][cid]["routes"][label]
            parts.append(f"<tr><td>{esc(labels[label][0])}</td><td>{esc(notes['target_edit'])}</td><td>{esc(notes['anatomy'])}</td><td>{esc(notes['protected'])}</td></tr>")
        parts.append("</table><p>" + esc(observations["cases"][cid]["interpretation"]) + "</p><details><summary>元suiteと比較用mask</summary>")
        for branch in ("off", "on"):
            parts.append(f"<p><a href='{cid}/suites/{branch}/manifest.json'>{branch} suite manifest</a> · <a href='{cid}/suites/{branch}/multi_proto_bg.masks.target.png'>対象mask</a> · <a href='{cid}/suites/{branch}/multi_proto_bg.masks.protected.png'>保護mask</a></p>")
        parts.append("</details><details><summary>保存済みattentionスコアを比較</summary>"
                     f"<p><a href='{cid}/attention-score-comparison-v2.png'><img src='{cid}/attention-score-comparison-v2.png' alt='Q/K scores and final mask'></a></p>"
                     "<p>左4列は3役割で正規化したQ/Kスコア（全48 head平均）です。全tokenに対する本来のattention確率ではありません。5列目は特徴量prototypeを含めた最終候補スコア、6列目は実効routing、7列目は既存ensemble maskです。ensemble画像の生成は追加していません。</p></details></div>")
    probe = json.loads((root / "park" / "head-hole-probe.json").read_text(encoding="utf-8"))
    parts.append("<div class='box'><h2>顔中心付近の抜け：blockだけでなくstep・head・分類も関係</h2>"
                 "<p>今回のmulti_proto_bgはstep 2 / block 18だけからmaskを作ります。保存された4tap全体は平均されていません。公園ONの対象maskの最大の閉じた穴は70セル（grid bbox [17,17,28,28]）で、62セルがprotected、8セルがuncertain backgroundです。独立した顔labelではなく、mask内の穴についての記述です。</p>"
                 "<table><tr><th>step / block</th><th>対象Q/Kスコア平均</th><th>保護Q/Kスコア平均</th><th>穴領域平均で対象が最大のhead</th></tr>")
    for row in probe["rows"]:
        scores = row["qk_role_probability_mean_on_hole"]
        parts.append(f"<tr><td>{row['step']} / {row['block']}</td><td>{scores['target']:.4f}</td><td>{scores['protected']:.4f}</td><td>{row['heads_target_wins_on_mean_hole']} / 48</td></tr>")
    parts.append("</table><p>同じblock 18でもstep 1では対象優勢、step 2では保護優勢に変わります。step 2/block 18にも対象を強く捉えるheadは残っています。最終prototypeスコアでは穴領域の対象平均0.1538、保護平均0.8455となり、62セルが保護側へ分類されます。単なる閾値不足や『blockに顔情報がない』だけでは説明しきれません。</p>"
                 "<p>次の切り分け候補はstep選択、head集約、prototypeによる人物間競合です。この2 seedの結果からheadやblockを選ぶとtuningになるため、選んだ設定の一般化は別seedで確認が必要です。</p>"
                 "<p><a href='park/head-hole-probe.json'>穴領域の全head診断</a> · <a href='closed-hole-analysis.json'>閉じた穴の分類</a></p></div>")
    parts.append("<div class='box'><p>既存配布workflowのclassic mask方式／選択余白4をそのまま実行した結果ではありません。前回attention ablationとの比較を保つため、既存nodeでk3方式／余白0のAPIコピーを組みました。元workflow・Sampler実装は変更していません。</p><p>Slider ONの収集は、入力MODELに通常LoaderでSlider +4を適用しています。collector自身が新たなSliderを足す処理はありません。本生成の入力は全経路でstyleのみのMODELです。</p><p>収集suiteに独立した完成OFF/all参照画像はありません。男性への所見は候補間の比較であり、未編集画像への完全保持を証明しません。</p></div></main></html>")
    with (root / "report.html").open("x", encoding="utf-8") as handle:
        handle.write("".join(parts))
    print(root / "report.html")


if __name__ == "__main__":
    main()
