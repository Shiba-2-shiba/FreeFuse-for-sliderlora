"""Build a local image comparison report from retained execution evidence."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def link(path):
    return html.escape(Path(path).relative_to(ROOT).as_posix(), quote=True)


observations = {
    "observer": "Codex multimodal assistant",
    "human_approval": False,
    "scope": "Auxiliary visual observations of the six generated images; not human pose or quality approval, not a comparison against OFF.",
    "park": {
        "composition": "全3枚で2人、別々の頭、全身と4つの足先が見える。横並びの公園構図。人の構図gateは未判定。",
        "duplicate_anatomy": "このプレビューでは明らかな余分な顔・腕・脚は見当たらない。細部の人による確認は未実施。",
        "target_edit_achievement": "+4は+2より対象の頭部が大きく、顔の若返り表現が強い。+2でも若返り表現が見える。OFFとの差や数値年齢は判定しない。",
        "natural_body_proportions": "+4は頭部が胴に対して大きく、+2は相対的に控えめ。編集達成と自然さを単一の点数にはしない。",
        "male_face": "男性の顔・目元には+2と+4で見た目の差がある。元画像のidentity保持とは判定しない。",
        "male_eyeglasses": "+4の2枚には男性の眼鏡が見えるが、k3/+2では眼鏡が見えない。強度比較での保護対象変化として要確認。",
        "male_height_build": "男性の全身配置・主要な体格は近く見えるが、同一性や画素不変は認定しない。",
        "small_arm_hand_adjustment": "対象の腕・手の形状と位置に候補間の差がある。接触はなく、小さな自然な調整の許容判定は未実施。",
        "clothing_shoes": "k1/+4は対象が閉じた靴、k3/+4と+2はサンダル。衣服・靴の候補間変化として記録する。",
    },
    "woman_front_strong_overlap": {
        "composition": "全3枚で対象が前、男性が後ろに位置し、対象が男性の胴・片脚の一部を遮蔽する。2つの頭と4つの足先が見える。『胴を大きく遮蔽』する構図gateの人の最終判定は未実施。",
        "duplicate_anatomy": "このプレビューでは明らかな余分な顔・頭髪・脚は見当たらない。重なった腕・手と輪郭は細部確認が必要。",
        "target_edit_achievement": "+4の2枚は+2より頭部が大きく、若返り表現が強い。+2は相対的に控えめ。",
        "natural_body_proportions": "+4は頭部が大きく、+2では頭身の見た目が相対的に自然。編集量が弱まった効果と品質改善は区別する。",
        "male_face_eyeglasses": "男性の顔は全3枚で見える。このプレビューでは眼鏡は見えず、顔の見た目は近い。OFFへの保持は不明。",
        "male_height_build": "男性の立ち位置・身長感・主要体格は近く見える。部分的な遮蔽があり、同一性の認定はしない。",
        "small_arm_hand_adjustment": "+4では男性の手が対象の肩付近、+2では上腕付近にある。接触に関連する腕・手の調整として別記する。",
        "clothing_shoes": "+4では対象が黒い靴、+2では白い靴。候補間の差として記録する。",
    },
    "preservation_vs_off": {"value": None, "reason": "No verified matching existing OFF was supplied; no additional OFF generated."},
    "mask_accuracy": {"value": None, "reason": "No independent same-grid human-verified visible-region labels."},
}
with (ROOT / "assistant-visual-observations.json").open("x", encoding="utf-8") as f:
    json.dump(observations, f, ensure_ascii=False, indent=2)

verification = read(ROOT / "verification.json")
summary = read(ROOT / "summary.json")
style = """
body{margin:0;background:#f4f5f7;color:#1c2430;font:15px/1.65 system-ui,sans-serif}main{max-width:1400px;margin:auto;padding:28px}h1{font-size:26px}h2{margin-top:40px}.box,article{background:white;padding:20px;border-radius:12px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}article{margin:0}img{width:100%;height:auto;display:block}a{color:#1465ad}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #dae1e8;text-align:left}.muted{color:#536271}.masks{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.masks img{image-rendering:pixelated}code{overflow-wrap:anywhere}details{margin-top:14px}summary{cursor:pointer}dt{font-weight:600;margin-top:12px}dd{margin-left:0}.alert{border-left:4px solid #c47821;padding-left:14px}@media(max-width:850px){.grid{grid-template-columns:1fr}main{padding:16px}table{font-size:13px}}
"""
parts = ["<!doctype html><html lang='ja'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>Attention ablation 実機検証 2026-10-11</title><style>"+style+"</style><main>",
         "<h1>Attention ablation：起動中ComfyUIでの検証</h1>",
         "<p>2026-10-11 JST / ComfyUI 0.37.0 / AMD Radeon AI PRO R9700 / INT8 checkpoint / Euler・simple・8 steps・CFG 1</p>",
         "<div class='box'><p><strong>2ケース・6画像の生成成功。実測104 NFE。</strong>k1/k3の正規化設定（prototypes以外）、初期noise・latent・sigma・conditioning metadataと全4観測tapが一致。k3/+2と+4の全partition・実効mask・初期入力が一致。</p>",
         "<p>全4suiteと全6編集bundleのfile/tensor hash、mask PNGとtensorの整合、全stepのtrace・prediction routingを検査済み。k1は各role 1、k3は各role 3のprototypeを実現。</p>",
         "<p class='alert'>補助目視：parkのk3/+2では男性の眼鏡が見えなくなる。+2の自然な頭身だけから編集品質改善を認定しない。人による構図gateと品質評価は全件 <code>not_evaluated / passed=null</code>。</p>",
         "<p>CPU回帰：<strong>80 passed、1 failed、8 subtests passed</strong>。唯一の失敗は <code>test_dangling_output_is_refused_before_platform_exclusive_open</code> の末尾で、Windowsの <code>//?/C:/...</code> と <code>C:/...</code> のsymlink target表記を直接比較している。そこに至る出力拒否・target未作成・input不変のassertは通過している。テストは修正していない。</p>",
         "<p><a href='verification.json'>機械検証JSON</a> · <a href='summary.json'>prompt ID・出力一覧</a> · <a href='current-assets.json'>現在の5 asset SHA-256</a> · <a href='assistant-visual-observations.json'>補助目視JSON</a> · <a href='cpu-tests.log'>CPUテストlog</a></p></div>",
         "<table><tr><th>ケース / seed</th><th>NFE / wall</th><th>k1対象セル</th><th>k3対象セル</th><th>対象mask IoU</th></tr>"]
for row in verification["cases"]:
    stats = row["mask_statistics"]["target"]
    seed = 444444 if row["case"] == "park" else 42
    parts.append(f"<tr><td>{html.escape(row['case'])} / {seed}</td><td>{row['nfe']} / {row['wall_seconds']:.1f}s</td><td>{stats['k1_cells']}</td><td>{stats['k3_cells']}</td><td>{stats['iou']:.6f}</td></tr>")
parts.append("</table><p class='muted'>IoUと面積は記述統計で、maskの精度ではない。wallは各queueを一度だけ計測し、polling遅延を含む。VRAMはcollectionごとのmemory.scopeとpeakをJSONに保持し、全queue peakは欠測null。peakの合算はしていない。</p>")
for case in summary["cases"]:
    cid = case["case"]
    evaluation = read(ROOT / cid / "evaluation.json")
    parts.append(f"<h2>{html.escape(cid)}</h2><p><code>{case['prompt_id']}</code> · <a href='{cid}/evaluation.json'>評価記録</a> · <a href='{cid}/queued_api.json'>実投入API</a> · <a href='{cid}/history.json'>history</a></p><div class='grid'>")
    for item in evaluation["variant_results"]:
        parts.append(f"<article><h3>{item['label']}</h3><a href='{link(item['image_path'])}'><img src='{link(item['image_path'])}' alt='{cid} {item['label']}'></a><p>strength +{item['strength']:g} / prototype {item['achieved_prototypes_by_role']}</p><p><a href='{link(item['edit_diagnostic_json_path'])}'>診断JSON</a> · human gate: 未評価</p>")
        k = "k1" if item["requested_prototypes"] == 1 else "k3"
        manifest_path = Path(evaluation["suite_manifest_paths"][k])
        manifest = read(manifest_path)
        parts.append("<details><summary>初期mask grid（target / protected / background）</summary><p class='muted'>生成後の輪郭を正解labelにしていない。</p><div class='masks'>")
        for role in ("target", "protected", "background"):
            name = Path(manifest["artifacts"]["mask_pngs"]["multi_proto_bg.masks."+role]).name
            parts.append(f"<div>{role}<img src='{link(manifest_path.parent/name)}' alt='{role} mask'></div>")
        parts.append("</div></details></article>")
    parts.append("</div><div class='box'><strong>Codexの補助目視（人の承認ではない）</strong><dl>")
    for axis, note in observations[cid].items():
        parts.append(f"<dt>{html.escape(axis)}</dt><dd>{html.escape(note)}</dd>")
    parts.append("</dl></div>")
parts.append("<div class='box'><p>Sliderは実環境の <code>archive\\krea2_deaging_20261001T075826Z_c06cb035.safetensors</code> にパスだけ合わせた。変更は全samplerのSliderパスと保存prefixのみ。元workflow・配布indexは変更していない。</p><p>今回の証拠は診断seed 444444/42の2ケース。未見seed holdout、UI再読み込み、独立no-hook parity、人の構図・画質判定、OFFとの保持評価、独立labelを使うmask精度は未検証。元bundle内の自己申告フラグ（weights_identity_verified等）は書き換えていない。</p></div></main></html>")
with (ROOT / "report.html").open("x", encoding="utf-8") as f:
    f.write("".join(parts))
print(ROOT / "report.html")
