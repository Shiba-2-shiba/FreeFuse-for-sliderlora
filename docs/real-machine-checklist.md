# Prediction Mix 0.2.0：ユーザーが手動で行う実機確認

実装担当はCPU検証と以下のツール・workflowを整備します。実ComfyUIでのnative検証、GPU生成、UI保存・再読込、画質評価はユーザーが手動で行います。実装完了と実機評価完了は別に記録してください。

## U1：環境とnative小型モデル

ComfyUIが使っているPythonで、拡張のルートから実行します。

```powershell
python -B tools/validate_comfy.py "<ComfyUIソースのルート>"
```

- [ ] native12件、skip0、status=passed。import失敗や途中停止は未合格として記録。
- [ ] 7ノードが登録され、Prediction Mix Samplerが通常カテゴリにある。
- [ ] checkpoint、Krea2 CLIP、VAE、darkbrush、Sliderのファイル名を環境に合わせた。同じSliderを上流へ二重適用していない。

## U2：既存manualの回帰確認

基準版はcommit `fb3cd592e7d115809ab125ef524612b7bfc62e19`です。同じComfyUI/Python/PyTorch/GPU・変更していないモデルファイルで、旧版と新版の`krea2_slider_mix_zero / none / all / half`を実行します。既存の保存結果を使う場合も環境・生成条件・入力hashを照合してください。

- [ ] 既存4条件のseed42、8steps、CFG1、Euler/simple、1024×1024、style0.8、Slider4（zeroのみ0）を維持した。
- [ ] 旧workflowの省略入力でmanual/auditとして動き、widget値がずれない。
- [ ] zero/noneでbase8・slider0、allでbase0・slider8、halfでbase8・slider8。phase1_nfe=0。
- [ ] 旧版と新版の初回予測・最終latentを比較した。新版はsource hashが異なるため、通常の同条件比較CLIの検査を緩めず、版の違う回帰確認として記録する。

同じ新版内でzeroとnoneを比較する場合：

```powershell
python -B tools/compare_slider_diagnostics.py "<mix_zero.json>" "<mix_none.json>" --endpoint-reference --include-trajectories
```

## U3：autoと同じmaskのmanual比較

1. [auto workflow](../workflows/krea2_female_slider_prediction_mix_auto.json)で、両MASK入力を未接続、diagnostic_level=auditにして生成します。
2. 保存されたautoのJSONを使い、参照target/protected maskを書き出します。

```powershell
python -B tools/export_reference_masks.py "<auto.json>" --output-dir "<マスク保存先>"
```

3. [manual workflow](../workflows/krea2_female_slider_prediction_mix_manual.json)に、書き出した2枚を読み込みます。それぞれ **LoadImageのIMAGE出力 → ImageToMask（channel=red）→ Subjectsのtarget_mask / protected_mask** と接続し、元の左右固定mask接続を置き換えます。LoadImageのMASK出力は透過情報を使うため、今回の不透明な白黒PNGの読み込みには使いません。
4. autoと同じprompt・モデル・seed・strength・steps・mix_scope・**selection_dilate_radius**で、manualもauditとして生成します。新autoの候補値は4なので、その比較ではmanualも4へ合わせます。書き出すのは参照partitionで、余白込みの実効maskではありません。収集専用値はmanualでは使われません。
5. 同じマスクのPhase 2を比較します。

```powershell
python -B tools/compare_slider_diagnostics.py "<auto.json>" "<same-mask-manual.json>" --same-mask-reference --include-trajectories --output "same-mask-comparison.json"
```

- [ ] 実効maskと参照target/protected/backgroundのhashが一致し、Phase 2入力・予測・final latentが一致した。
- [ ] autoのstrength0/4で同じ収集maskになる。収集値、seed、promptは固定した。
- [ ] autoのcollect_step2 / 8steps / 部分maskでphase1_nfe=2、phase2_nfe=sampler_nfe=8、branch_nfe={base:8,slider:8}、total_model_nfe=18。
- [ ] autoのnone/all/strength0でも収集2回を維持し、Phase 2の不要経路だけ省略する。
- [ ] maskの顔・全身被覆、raw map、処理前mask、追加領域を確認した。顔欠落を「LoRAの演算が弱い」と混同しない。
- [ ] 同じautoの収集設定でselection_dilate_radiusを0/4/8に変え、参照partitionが同じで実効予測maskだけが広がることを確認した。
- [ ] 第8Previewの実効予測maskと第9の追加余白を確認し、頭・手足の越境、二重化・欠け、輪郭沿いの背景段差を評価した。推定protectedの未捕捉部への影響も実画像で確認した。

比較のstatus=measured_onlyは測定終了を示し、画質合格を意味しません。数値のexact_equalとdirect_routingを確認してください。観測後に都合よく許容値を緩めないでください。

## U4〜U6：UI・画質・復帰

- [ ] 新manual/autoと旧workflowをUIで読み込み、保存・再読込・再実行した。optional入力の値と接続を維持できた。
- [ ] 同条件のnone/all/対象maskを保存し、元promptのseed42/444444、正面スタジオ、腰手・腕組み、公園を強度2/4で評価した。
- [ ] 対象の顔・頭身・胴脚、保護側の年齢感・体格、衣服・手足、背景中央の段差、mask越境を別々に評価した。
- [ ] 中断後の再実行、Phase 1失敗後のmanual、base→Slider→baseの連続実行が動き、Sliderが残留しない。

| 条件 / seed / strength / run_id | 対象の顔・体格 | 保護側の変化 | 顔・全身のmask被覆 | 背景・手足・境界 | 採用 / 要調整 / 不適合 |
|---|---|---|---|---|---|
| 元prompt | | | | | |
| 正面スタジオ | | | | | |
| 腰手・腕組み | | | | | |
| 公園 | | | | | |

PNG、`_effective_mask.png`、safetensors、JSONの4点を**元の名前のまま同じフォルダー**に保持してください。JSON内の固定falseの検証フラグだけで実行失敗と判断せず、NFE・監査値・画像を評価します。

## 旧hook方式のチェック項目

確認対象: Krea2、attention-target Slider 1本、男女各1人、Euler/simple、8 steps、CFG1、1024×1024。

## 接続・互換性

- [ ] 通常生成が動くモデル・CLIP(type=krea2)・VAEを選んだ。
- [ ] 旧hookノードも検索で見つかり、workflowを保存して再読込できる。
- [ ] 現行`tools/validate_comfy.py`のnative12件がskipなしで成功した。未実施/失敗を成功と混同しない。
- [ ] `tools/probe_krea2_slider.py`で実Sliderの全キーが対応し、代表層のoutside誤差とpacked weight保持が合格した。
- [ ] Samplerに選ぶ女性Sliderを通常LoRA Loaderでも全体適用していない。
- [ ] 最初は手動workflowでMASKと実人物位置が対応することを確認した。

## 比較表

prompt/model/CLIP/VAE/darkbrush0.8/解像度/stepsを同じにし、次の条件を比較する。Aは通常KSamplerでSliderなし、Bは通常LoRA Loaderで同じSliderを全体適用、Cは本Samplerで女性限定。まず+1を使い、必要なら単体で確認済みの強度へB/C双方を揃える。

| seed | A: 人物数・構図 | B: 女性/男性への効果 | C: 女性への効果 | C: 男性への同属性漏れ | C: 欠損/継ぎ目/縮尺/照明 | C: mask位置 |
|---|---|---|---|---|---|---|
| 42 | | | | | | |
| 4444 | | | | | | |
| 4444444444 | | | | | | |
| 123 | | | | | | |
| 777 | | | | | | |

Bでも男性が変化しない属性だけでは、男性保護の有効性を評価できない。Aで2人が成立しないseedはそのまま記録し、基礎生成と局所適用の失敗を分ける。初期採用目安はCの4/5seed以上で「男女各1人・女性の効果・男性への明瞭な同属性漏れなし・新たなタイル状分断なし」。5seedから一般的な成功率は推定しない。

## 動作継続・復帰

- [ ] 同じseedのstrength0と通常生成を比較した。
- [ ] 本Sampler→通常KSampler→本Samplerの順に実行し、通常生成へSliderが残留しない。
- [ ] 中断後の通常生成と再実行が動く。
- [ ] debugログでmatched_modulesとreached_modulesが一致する。
- [ ] adapter_groupsがtarget1/protected0/background0で、owned_hooks_removedがtrue。
- [ ] 自動版で両人物と背景のMASKを確認した。女性のeffectと同時に人物がmask外へ動いていない。
- [ ] 初回ロードとwarm runの時間、Phase1/2評価回数、VRAMを別記した。

PNGの生成metadataと、対応するrun_idのログを一緒に保存する。失敗時はComfyUI/extensionのcommit、モデルのquant_format、Slider形式、manual/auto、収集step/blockを添える。

## 0.1.2のmask後処理

- [ ] [処理なし](../workflows/krea2_female_slider_postprocess_off.json)と[穴埋め8](../workflows/krea2_female_slider_postprocess_fill8.json)を同条件で比較した。
- [ ] original_target_mask、処理後target_mask、added_target_maskを確認した。
- [ ] protectedが処理前後で同じで、mask_postprocess.protected_changed_token_count=0/partition_valid=trueを確認した。
- [ ] 穴埋め後も女性への効果があり、男性の同属性変化・背景破綻が増えていない。
- [ ] 膨張1を試す場合は最大成分だけが広がり、小断片が拡大しないことを確認した。
- [ ] 旧manual/auto workflowは設定省略または両方0で動き、UI保存・再読込後も設定が維持される。

## 0.1.3の対象文章行への適用

- [ ] [比較版](target-text-routing.md)の`target_text_scale=0 / 0.5 / 1`を同じ画像strength・seed・promptで生成した。
- [ ] target/protected mask、initial_noise/full_sigmas hash、収集条件が3版で同じことを確認した。
- [ ] 女性への効果と男性の同属性変化、構図・照明・人物欠損を比較した。
- [ ] 通常全体適用の参照版で同じSlider・強度を一度だけ使用した。
- [ ] 診断のtarget/protected token位置とtext policy、実行回数を確認した。
- [ ] 旧workflowの省略時は0、UI保存・再読込後は指定倍率が維持される。
