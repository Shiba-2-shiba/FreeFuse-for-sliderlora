# 保護付きmask後処理（0.1.2）

自動maskの小さな穴を埋め、必要なら最大の人物成分だけを膨張させます。後処理はPhase1のmask生成後、Phase2のSlider適用前に実行します。生成後のPreviewだけを変更する処理ではありません。

## 設定

Samplerの末尾に次の2項目を追加しました。両方optionalで、旧API/JSONで省略した場合は0です。

| 設定 | 範囲・単位 | 既定値 |
|---|---|---:|
| `fill_holes_max_area` | 0..64、穴の面積token数。0で無効 | 0 |
| `mask_dilate_radius` | 0..1、膨張半径token数。0で無効 | 0 |

最初の比較候補は **fill_holes_max_area=8、mask_dilate_radius=0**。今回の64×64 maskでは1tokenが約16画像pixelで、膨張1でも追加領域は大きくなります。manual modeでは両方0を使用してください。manualで非zeroを指定すると説明付きで停止します。

## 保護ルール

- 小穴はtargetの補集合を4連結で調べます。canvas外周につながる部分や上限より大きい部分は埋めません。外周へ斜めにつながるだけなら4連結では閉じた穴です。
- 穴全体が元backgroundに属し、protectedを一つも含まない場合だけ、targetに移します。保護を含む穴は全体を保留します。
- 膨張は穴埋め後targetの8連結最大成分だけが対象です。小さな断片は保持しますが広げません。同面積ならrow-majorで最初の成分を選びます。
- 追加できるのは元backgroundだけです。protectedに入る候補は除外し、protected自体は変更しません。
- backgroundを補集合で再計算し、target/protected/backgroundの和1・重なり0を維持します。
- 生類似度mapとモデルの全体attentionは後処理の対象ではありません。

protectedは推定maskであり、男性全体の正確なsegmentとは限りません。数値上の非重複を守っても、男性の未捕捉部分や共有attentionによる変化は実画像で確認する必要があります。

## 比較ワークフロー

LoRA、strength4、seed42、woman/man、top_k0.2、temperature10000、collect_step2/block18、8steps/CFG1を固定した4条件です。

| 条件 | ファイル |
|---|---|
| 後処理なし | [postprocess_off](../workflows/krea2_female_slider_postprocess_off.json) |
| 小穴4以下 | [postprocess_fill4](../workflows/krea2_female_slider_postprocess_fill4.json) |
| 小穴8以下 | [postprocess_fill8](../workflows/krea2_female_slider_postprocess_fill8.json) |
| 小穴8＋最大成分膨張1 | [postprocess_fill8_dilate1](../workflows/krea2_female_slider_postprocess_fill8_dilate1.json) |

API版は同名の`_api.json`です。更新後にComfyUIを再起動し、まずoffとfill8を比較してください。その後fill4・fill8_dilate1や別seedで確認します。

## Previewと診断

既存Previewの先頭5出力の順序を維持し、末尾へ次を追加しました。

- `original_target_mask`: 後処理前target。
- `added_target_mask`: 穴埋め・膨張で新しく追加された領域。

先頭の`target_mask`は生成に使った**処理後**maskです。新ワークフローには全7出力のPreviewを配線しています。旧bankではoriginal=current、added=黒にフォールバックします。manual/後処理0でもoriginal/addedを確認できます。

診断文字列/`[Krea2SliderFuse]`ログの`mask_postprocess`に、設定、before/after白数・成分数、充填/保留した穴数、保護による除外数、追加token数、partition_validを記録します。protected_changed_token_countは0であることを確認します。

filled_token_countは穴埋めで増えた数、dilated_token_countは穴埋め後から膨張で増えた数です。合計はadded_target_maskの白数と一致し、重複計上しません。無効な処理の未計算候補数はnullです。

## ローカル検証と実機の区別

CPUでは閉じた穴・外周への通路・斜めの接続・protected混在・最大成分tie・四辺の膨張・不正入力・旧API・Phase2への実接続を検証しています。

提供3maskを`fxezv=protected / uuhis=background`と対応させたローカル計算では、target583→fill4:612→fill8:627、fill8+dilate1:813を再現しました。protectedは446で固定です。画像名からの対応は未確認のため、これはその対応での数学的検証です。逆の対応でもprotectedを上書きせず、内部キーに従って処理することを確認しています。提供画像はリポジトリに含めていません。

新後処理を使った実GPU生成・画質・UI保存/再読込は未確認です。女性への効果と、男性・背景・画像全体への影響を同じseedで比較してください。実機環境の依存を更新する処理はありません。
