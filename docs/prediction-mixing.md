# Prediction Mix：manual / autoと診断保存（0.2.0）

Prediction Mix Samplerを通常カテゴリへ移し、autoのマスク収集を接続しました。既存node ID・旧入力順・Samplerの3出力型を維持し、auto関連7入力と選択拡張1入力を末尾に追加しています。省略時はmanual・選択拡張0で、旧workflowは以前と同じ範囲を使います。

## 推奨workflow

| 入口 | 設定・状態 |
|---|---|
| [manual](../workflows/krea2_female_slider_prediction_mix_manual.json) | 左半分target、右半分protected、strength4、summary。実人物位置を確認 |
| [auto](../workflows/krea2_female_slider_prediction_mix_auto.json) | 両MASK未接続、collect_step2/block18、top_k0.2、temperature10000、fill0/dilate0、選択拡張4、summary。画質は実機確認待ち |

各workflowの同名`_api.json`も同梱しています。収集値は過去の候補を使用したもので、最適値ではありません。

autoはstyleのみのbaseで短い予備生成を行い、既存の類似度map→排他mask→保護付き後処理を使います。観測用hookを外してから、元の初期noise・latent・全sigma列でPrediction Mix生成を開始します。予備生成の途中latentは引き継ぎません。

## 参照マスクと予測選択の余白

`selection_dilate_radius`（0〜16、既定0）は、Prediction Mixで採用する範囲だけの拡張です。最大の対象成分から背景へ1 gridずつ広げ、保護領域を追加せず、保護領域を飛び越えません。小さな孤立targetは保持しますが広げません。参照target/protected/backgroundは変えないため、旧autoとの比較を維持できます。

従来の`mask_dilate_radius`（0〜1）は参照targetの後処理で、別の設定です。新autoの選択拡張4は通常Krea2で約64pxの比較開始候補です。0/4/8を比較し、頭身・位置が変わった部分のはみ出しと背景境界を評価してください。適切な余白や継ぎ目の解消は実機未確認です。

Mask Previewは先頭7出力を保ち、第8に実効予測選択mask、第9に選択余白の追加領域を出します。先頭のtargetは参照マスクであり、選択拡張後の全適用範囲ではありません。Saveの`_effective_mask.png`も実際の選択範囲です。

8 steps・collect_step2・部分maskなら、phase1_nfe=2、phase2_nfe=sampler_nfe=8、branch_nfe={base:8,slider:8}、total_model_nfe=18です。none/all/強度0でもautoのmask収集は実行し、Phase 2だけ必要な経路へ省略します。空・拡散・非有限mapでは停止します。

## schema 3の保存・比較

新Prediction Mixの保存はschema 3です。旧hook/nativeのschema 1/2は読込を維持します。summaryでもmask_generation、収集時の入力hash、Phase別の実行回数を検証します。autoはraw_target/protected_similarity、original_target_mask、added_target_mask、収集時の小さな全sigma列と参照partitionを保存します。summaryでも収集prefixのhash・観測したsigma・文章位置・grid・head数・mask統計を照合し、保存・読込時にraw mapからmaskを再生成します。

auditでは従来のPhase 2 traceに加え、保存されたbase/native予測から二値選択を再計算して一致を確認します。Phase 1のtraceはPhase 2へ混ぜません。schemaの既定は互換性のためaudit、新推奨workflowはsummaryを明示しています。

autoとmanualの同じmaskを比較するには、[実機確認手順](real-machine-checklist.md)に従い、参照target/protected maskをexport_reference_masks.pyで書き出してmanualへ再入力します。

```powershell
python -B tools/compare_slider_diagnostics.py "<auto.json>" "<same-mask-manual.json>" --same-mask-reference --include-trajectories
```

このオプションはschema 3のauto/manual一組に限り、同じscope・strength・実効mask・参照partitionを要求します。環境、実装、conditioningなどの検査は緩めません。通常比較はmask生成設定が異なると拒否しますが、同条件でscopeだけを変えるnone/all/half比較は可能です。

同一mask比較では`selection_dilate_radius`も揃えてください。参照partitionが同じで、選択半径だけを変えた通常比較は`selection_policy_comparison`となり、同条件反復とは扱いません。軌道の`boundary`は参照targetの境界を指すため、余白込みの実効境界は第8Preview/実効mask画像で確認します。

## 0.1.4からの比較機能

対象の顔・頭身・体格へのLoRA効果と、保護人物の同属性の維持を比較する機能です。実INT8/GPU・UI保存/再読込・画像品質はユーザーの実機で検証します。

## 方式

同じlatent・sigma・conditioningから、styleだけのbase予測と、style＋標準LoaderでSliderを適用したnative予測を直列に計算します。

```text
prediction = where(binary_target_mask, native_prediction, base_prediction)
```

Slider強度は標準Loaderで一度だけ適用します。2branchは1つのEulerループを共有し、同じcoreのpatcherをcleanup→load→pre_runで1経路ずつ切り替えます。forward前にcurrent_patcher identityを検査します。

部分maskの8stepではbase/native各8回、合計16回のモデル評価と切替費用が必要です。全黒・全白・強度0では必要な経路だけ8回実行します。マスク外の予測は同入力のbaseと一致しますが、次stepも編集後latentを読むため、保護人物・背景の最終画素は固定されません。

## 実機確認の順序

1. 更新してComfyUIを再起動し、Prediction Mix Samplerを含む7ノードの登録を確認します。必要ならComfyUIのPythonでnative小型CPU/schema検証を行います。

```powershell
python -B tools/validate_comfy.py C:/path/to/ComfyUI
```

validatorは12件を期待し、skip/import不足を合格にしません。依存を自動変更しません。この成功と実INT8生成は別の証拠です。

2. 既存の`native_global`、`hook_all_all`、`hook_half_none`で`diagnostic_level=audit`、trial_id0/1を各1回実行します。計6回。旧workflowにoptional入力が表示されなければ診断Samplerを置き直して接続を維持し、保存/再読込を確認します。2回とも同じ0.1.4コードで採取し、seedを変更しません。Saveだけの再実行は反復試験ではありません。

3. 以下を順に実行します。UI形式へのリンクで、同名`_api.json`もあります。native_zeroも同じ新版コードで採取します。

| workflow | strength / scope | 比較先 | 期待branch NFE |
|---|---|---|---|
| [mix_zero](../workflows/krea2_slider_mix_zero.json) | 0 / target_mask | native_zero | base8/native0 |
| [mix_none](../workflows/krea2_slider_mix_none.json) | 4 / none | native_zero | base8/native0 |
| [mix_all](../workflows/krea2_slider_mix_all.json) | 4 / all | native_global | base0/native8 |
| [mix_half](../workflows/krea2_slider_mix_half.json) | 4 / target_mask | hook_half_none、native_global | base8/native8 |

端点に説明できない差や切替/復元エラーがあれば、その原因を切り分けてからmix_halfを評価します。同条件反復の差を併記し、観測後に都合よく許容値を緩めません。

共通条件はseed42、1024×1024、batch1、8steps、Euler/simple、CFG1、denoise1です。モデル`intorealismAsian_k2JAVFLASHV1.safetensors`、CLIP`wen3vl_4b_bf16.safetensors`/krea2、VAE`qwen_image_vae.safetensors`、style`krea2_darkbrush.safetensors`0.8、Slider`Krea2/krea2_deaging_20261001T075826Z_c06cb035.safetensors`4を使用します。prompt、woman/man、左512px target/右512px protectedを維持します。モデル/LoRA検索パスは実機へ合わせます。

新Sampler上流はstyleのみです。同じSliderを通常Loaderで重ねません。Subjectsの両maskは全黒/全白のmix条件でも接続したままにし、`mix_scope`だけで実効出力maskを変えます。新workflowはauditが既定です。

## 保存とログ

新Samplerのlatent/diagnosticsと、そのlatentのVAEDecode画像を既存DiagnosticSaveへ接続します。1実行につきPNG、`_effective_mask.png`、safetensors、JSONの4ファイルを保存します。**同じフォルダー・元のファイル名で一組にしてください。** JSONは最後に保存され、途中成果物を完成扱いしません。

safetensorsはfirst_prediction/final_latentに加えてv2の参照target/protected/background maskを保存します。auditでは`trace_inputs`、`trace_predictions`、`trace_sigmas`、部分mixでは`trace_base_predictions`と`trace_slider_predictions`も保存します。1024px/8stepのtraceは数十MBとなり、CPU転送/保存時間も加わります。

JSONには条件、環境、実装revision/source hash、入力hash、branch NFE、artifact ID/hash、tensor_manifestを保存します。first_predictionはSamplerへ渡した最初の予測です。native branchの内部全文LoRAと、予測出力の領域制限を別の項目に記録します。

旧hookのauditは全層/全stepの`linear_audit`を追加します。imageはtarget/protected/background/selected/unselected、textはtarget_phrase/protected_phrase/other/selected/unselected。指標はelement_count、rms、max_abs、changed_elements、nonfinite_count、computed_delta_rms、rounded_away_elementsです。同じLinear入力のresult-baseをFP32で測り、生成演算を変更しません。空群はnull/empty regionです。

従来の`adapter_stats`は最初のblockの5投影・最初の呼び出しのままです。summaryでは全層/traceを採取しません。native/mixのLinear auditは未計測です。部分mixの`prediction_mix_audit`は、同入力baseとの選択外一致とnativeとの選択内一致を各stepで実測します。

`int8_real_machine_validated`/`image_quality_validated`は機能の検証状態を表す固定falseで、実行失敗を意味しません。実行した環境/量子化形式は別の実測項目です。完成manifestだけで画質合格を宣言しません。

## 比較コマンド

```powershell
python -B tools/compare_slider_diagnostics.py "<hook_half_none.json>" "<mix_half.json>" --include-trajectories --output "comparison.json"
python -B tools/compare_slider_diagnostics.py "<native_global.json>" "<mix_all.json>" --include-trajectories
python -B tools/compare_slider_diagnostics.py "<native_zero.json>" "<mix_none.json>" --endpoint-reference --include-trajectories
python -B tools/compare_slider_diagnostics.py "<native_zero.json>" "<mix_zero.json>" --include-trajectories
```

通常比較はstrength差を拒否します。`--endpoint-reference`はmix none/zeroのeffective_slider_strength0・slider_nfe0を確認できた場合だけnative_zeroとのstrength差を扱います。その他の条件を緩めません。新旧schemaは共通項を読み込みますが、旧reportの未観測の予測表現/audit/traceを検証済みとしません。

同じstepでも入力が違えば`trajectory_difference`です。同入力・sigma・conditioning・実装の予測だけ`same_input_prediction_difference`と表記します。参照partitionのtarget/protected/backgroundと内外1tokenの境界帯について差を測ります。RGBのMAEは年齢・同一性のスコアではありません。

許容値なしは`measured_only`。根拠付きの`--atol`/`--rtol`を両方指定した場合だけ許容判定します。非選択行に直接差分を実測した場合は`direct_routing_violation`で終了コード1、未計測はunavailableです。hookは全step×全対象moduleの一意な記録、部分mixは全stepの記録を要求し、欠落や重複があれば保存・比較を拒否します。

## 画像評価票

| 条件・run_id | 対象の顔 | 頭身・胴/脚 | 保護人物の年齢/体格 | 境界・構図 | 採用理由/問題 |
|---|---|---|---|---|---|
| native_zero | | | | | |
| native_global | | | | | |
| hook_half_none | | | | | |
| mix_half | | | | | |

保護人物の年齢/体格と、服のしわ等の細部変化を別に記録します。seed42で改善候補が見つかった場合だけstrength2、seed123/777へ展開し、各seedのbase/native参照も保存します。人物が左右maskから外れるseedはmask不適合です。自動maskの評価は方式を絞ってから行います。

## 互換性と検証の限界

照合したローカルComfyUI sourceは`b26625f23a888367b92153b28d93e159e83e677b`。既存実機`52f98af2e2e42c421070a3e147c161c47cdeaf22`との同等性は未実証です。CFGGuider/helper API、current_patcher identity、conditioning/grid、同一core/device、出力shape/dtype/deviceを検査します。未知wrapper/callback/conditioning hooks/追加モデルは拒否し、別方式へ自動代替しません。

こちらのCPU suiteはComfyUI境界doubleを含みます。native小型CPU12件、実INT8切替・端点、UI保存/再読込、実画像品質はユーザーの実機での検証項目です。
