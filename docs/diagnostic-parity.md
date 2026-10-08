# 通常LoRAと局所Sliderの診断

manual左半分を使っても全体適用ほど体格が変わらなかったため、通常Loaderとの演算差、画像の適用範囲、文章の適用範囲を分けて調べます。診断ノードは画質改善機能ではありません。実INT8・UI・画像品質の評価はユーザーの実機で行います。

## 共通条件

ComfyUIを再起動し、同じモデル・TE・VAE・LoRAファイルで以下を実行してください。11条件のUI JSONと同名の`_api.json`を`workflows/`へ追加しています。

- seed42、1024×1024、batch1、8 steps、Euler/simple、CFG1、denoise1。
- `intorealismAsian_k2JAVFLASHV1.safetensors`、`wen3vl_4b_bf16.safetensors`（krea2）、`qwen_image_vae.safetensors`。
- global style `krea2_darkbrush.safetensors` =0.8。Slider `Krea2/krea2_deaging_20261001T075826Z_c06cb035.safetensors` =4、ゼロ基準のみ0。
- 同じ成人男女のprompt、target=`woman`、protected=`man`。参照マスクは左512px/右512px。診断は予備生成をせず全8ステップを実行。
- 診断Samplerに入れるMODELにはstyleだけを適用します。同じSliderの通常Loaderを上流に追加しないでください。native backendがSliderを一度だけ標準APIで適用します。

診断Samplerの`backend`は`native`（通常Loaderと同じComfyUI API）または`hook`（この拡張の加算演算）。`image_scope=all`は画像全体、`target_mask`は左半分への直接差分です。`text_scope=none/target_phrase/all`は文章行への直接差分なし/対象語だけ/全文です。全文は男性を表す語にも作用し、CLIP encoderの変更ではありません。全面/全文条件は人物保護を保証しません。

## 実行順

| 段階 | ワークフロー | 確認すること |
|---|---|---|
| 1 | `krea2_slider_diag_standard_zero.json`、`native_zero.json`、`hook_zero.json`（後二つも先頭に`krea2_slider_diag_`） | Sliderなしで診断処理自体が生成を変えていないか |
| 2 | `krea2_slider_diag_standard_global.json`、`krea2_slider_diag_native_global.json` | 標準KSamplerと診断Samplerの通常LoRA経路の差 |
| 3 | `krea2_slider_diag_native_global.json`、`krea2_slider_diag_hook_all_all.json` | 画像全面・全文という同じ範囲での演算差 |
| 4 | `krea2_slider_diag_hook_all_all.json`、`krea2_slider_diag_hook_half_all.json` | 文章を全文に固定した画像範囲の寄与 |
| 5 | `krea2_slider_diag_hook_half_none.json`、`hook_half_target.json`、`hook_half_all.json`（全て同じ先頭） | 左半分に固定した文章範囲の寄与 |
| 補助 | `krea2_slider_diag_hook_all_none.json`、`hook_all_target.json`（同じ先頭） | 画像全面時の文章範囲との相互作用 |

最初の段階で説明できない差が出たら、その経路を先に調べます。全体適用との演算差が残っている段階で、文章やmaskだけが原因とは判断しません。まずseed42で比較し、候補を絞ってから重要ペアだけをseed123/777で再確認します。人物が左半分を外れるseedは、このmanual maskの評価対象として不適合です。

同一条件を反復する場合は、診断Samplerの **trial_idだけを0→1** と変更します。seedは変更しません。新しいrun_idと`phase2_nfe=8`のログが出ることを確認してください。Saveノードだけ再実行して同じSampler結果を保存したものは、反復試験ではありません。標準KSamplerの反復はComfyUI側で実計算が行われたことをログで確認します。

## 保存される結果

診断9条件は標準outputディレクトリへ、同じbasenameで以下を保存します。保存prefixにはcase名、生成したbasenameにはseed・連番を含みます。

- `.png`: 生成画像。prompt/workflow（ある場合）とartifact_id/run_idを埋め込みます。
- `_effective_mask.png`: **実際に適用した画像mask**。全面条件は全白。workflowのMaskPreviewは左右の参照partitionで、全面条件でも左/右のままです。
- `.safetensors`: `first_prediction`（初回apply_modelの返却値）、`final_latent`。初回出力はモデルwrapperの出力であり、raw velocityと同一とは限りません。
- `.json`: backend/scope、LoRA SHA256、noise/initial latent/sigmas/conditioning/inputのhash、環境、適用行数、代表5 projectionの統計、各ファイルの対応とhash。

JSONは全保存が終わってから作られます。JSONのない途中成果物は比較に使いません。画像・JSON・safetensors・maskの4ファイルを一組として保持してください。標準2条件は標準SaveImageとSaveLatentでPNGと`.latent`を保存し、初回モデル出力やnoise/sigma hashを観測しません。

代表projectionの統計には、rank/alpha、計算dtype、選択行数、base RMS、計算した差分RMS、実際の加算後差分RMSがあります。BF16で微小差分が加算時に消えているかも調べられます。native backendには独自adapter統計を付けず、モデル出力を比較します。

## 保存結果の比較

ComfyUIなしの環境でも、既存のtorch/Pillow/safetensorsで診断結果を比較できます。まず同じstrengthの2条件を指定します。

```powershell
python -B tools/compare_slider_diagnostics.py "<native_globalのJSON>" "<hook_all_allのJSON>" --output "comparison.json"
```

初回prediction・最終latentのMAE/RMSE/max abs/relative L2とRGB MAEを出します。反復試験の2 JSONも一緒に渡すと全ペアを比較し、同一backend/scopeのペアに`repeat_trial=true`を付けます。同じrun_idの二重保存や異なる生成条件は拒否します。strength0と4の差はこの同条件比較に含めず、画像の効果を目視する基準として使います。

許容値未指定は`measured_only`です。数値一致の合格を意味しません。事前に定めた根拠付きの値がある場合だけ`--atol <値> --rtol <値>`を両方付けます。許容値を超えると終了コード1です。結果を見て都合よく閾値を緩めないでください。

比較にはJSONに記録されたComfyUIのGit revisionも必要です。取得できない場合は理由を保存し、unknown同士を同じビルドとして比較しません。Git checkoutのComfyUIから採取した結果を使用してください。モデル/TE/VAE/styleの同一性はworkflowのファイル名・設定による照合で、未取得の重みSHA256一致を保証しません。同じ実機・変更していないファイルで実行し、追加probeのasset SHA256も併せて保持してください。

標準KSamplerとの比較は最終latentとRGBだけです。同じ実行で作られたPNG/latentを明示的に指定します。ComfyUIのmetadata無効化設定は使わないでください。

```powershell
python -B tools/compare_slider_diagnostics.py --standard-latent "<standard_globalの.latent>" --standard-png "<対応するPNG>" --standard-case standard_global --against "<native_globalのJSON>" --output "standard-comparison.json"
```

標準zeroでは`--standard-case standard_zero --against <native_zeroのJSON>`に変えます。標準側は埋め込みpromptと共通Samplerを照合しますが、診断結果と同じrun_idの保証や未観測の初回出力を付けません。対応形式はComfyUI SaveLatent version_0のraw latent_tensor（倍率1）です。古い形式への推測変換はしません。

## 実INT8の追加probe（必要な場合）

実機テストはユーザー側で実行します。ComfyUIが使っているPythonで、まず既存の式検証probe、次に標準Loader比較probeを実行できます。後者は代表5 familyの同一固定入力を使用し、順序反転・反復誤差・元の量子化データの復元も記録します。

```powershell
python -B tools/probe_krea2_slider.py --comfy-root "<ComfyUIルート>" --model "<checkpoint絶対パス>" --lora "<Slider絶対パス>"
python -B tools/probe_krea2_slider_parity.py --comfy-root "<ComfyUIルート>" --model "<checkpoint絶対パス>" --lora "<Slider絶対パス>" --style-lora "<darkbrush絶対パス>" --style-strength 0.8 --strength 4 --repeats 2 --output "parity-probe.json"
```

新probeの`measured_only`は測定終了であり、通常Loaderとの同等性・画質合格ではありません。単層呼出しで実機量子化処理を再現できない場合は、失敗を記録して診断Samplerの初回prediction比較を使用します。checkpoint/Slider/styleのSHA256とComfyUI commitも記録します。

## 評価結果の記録

| 比較 | 初回prediction差 | 最終latent差 | 女性の顔/頭身 | 男性の同属性/構図 | 次に調べる対象 |
|---|---|---|---|---|---|
| native_zero ↔ hook_zero |  |  |  |  | 観測/復元の副作用 |
| standard_global ↔ native_global | 観測なし |  |  |  | サンプリング/通常API |
| native_global ↔ hook_all_all |  |  |  |  | 演算/量子化/precision |
| hook_all_all ↔ hook_half_all |  |  |  |  | 画像scope |
| hook_half_none ↔ hook_half_target ↔ hook_half_all |  |  |  |  | 文章scope |

RGB差や自動年齢推定を若返り成功判定にせず、顔だけでなく頭身・胴/脚の比率、右人物の変化、服や輪郭の破綻を併せて評価します。通常globalでは二人とも変化するため、画像全体の差を女性だけへの効果不足と解釈しません。
