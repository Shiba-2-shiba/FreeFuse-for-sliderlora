# 第一段階：膨張後の男性core除外

## 目的

全身の編集余地を残しながら、最終的なSlider予測の選択範囲から男性の顔・主な胴体を除く。既存のD余白0/4との比較で、余白縮小と局所的な除外の効果を分ける。最初の対象は公園seed444444と男性手前seed42。

これはマスク準備と既存manual再生の拡張であり、Sampler・attention・LoRA学習は変更しない。体の相対的な大きさを制御する機能でもない。

## 実行例

元のD余白4の実行graphと、そのrunの保存maskを使う。以下のパスは実際の保存名へ置き換える。`core-human-verified` は確認状況を正直に指定する。

```bash
python tools/prepare_core_exclusion.py \
  --effective D_effective_prediction_mask_00001_.png \
  --protected D_reference_protected_00001_.png \
  --core male_face_torso_exclusion.png \
  --expected-grid 64 64 \
  --effective-provenance actual_saved \
  --core-human-verified false \
  --workflow-api workflow_api.json \
  --workflow-ui workflow_ui.json \
  --output-prefix E_core_clipped/man_front_seed42 \
  --output-dir prepared/man_front_seed42
```

Windowsでは1行へまとめるか、使っているshellの改行継続記法へ直す。既存出力フォルダは上書きしない。Pillowを使用するが、モデル・ComfyUI・torchを読み込んで推論するツールではない。

出力は `target_final.png`、`protected_final.png`、`core_removed.png`、`manifest.json`。実行graphを両方指定すると `replay.json` と `replay_api.json` も作る。2つの完成maskをComfyUIのinputへ置き、replayを読み込み、LoadImageMaskがそのファイルを指すことを確認する。複数ケースの同名maskを取り違えない。サブフォルダを利用する場合は `--input-subdir` にinput内の相対フォルダを指定し、実機で選択可能か確認する。

このreplay補助は配布manual-D graphの構造を検証するもので、任意のComfyUI workflowの変換器ではない。元のmodel・prompt・seed・trial_id等を保持し、mask入力、出力prefix、選択余白0だけを変更する。標準Krea2の1grid cell=16画像pixelという前提でlatentの画像サイズと期待gridを照合する。異なるpatch構造のモデルへはそのまま流用しない。

再構成maskを診断目的で使う場合は `--effective-provenance reconstructed` と明示する。manifestはその区別を保存するが、実機で使われたmaskの真正性や人物同一性を自動証明しない。

## 入力と処理順序

- M：元のD余白4で実際に使用した `effective_prediction_mask`。
- P：同じrunの `reference_protected`。
- E：独立して確認した男性の顔・主な胴体の除外mask。対象の許容する手・腕の調整範囲とは区別する。

3つとも同じ**実サンプリングgrid**の二値PNGが必要。今回の1024×1024のKrea2生成では64×64。元画像サイズのmaskや半透明overlayをそのまま使わない。画像サイズからの自動リサイズは行わない。

MとPは元から排他的でなければ入力エラー。完成targetは `M AND NOT E`、完成protectedは `P OR E` とする。完成targetが空の場合は拒否する。除去した場所と各入力のhashを保存し、削られた部位を目視できるようにする。

完成target/protectedを同じ条件のmanual Samplerへ渡し、**selection_dilate_radius=0**で再生する。穴埋め・参照膨張も0を維持する。後から余白を加えると除外部へ再侵入し得るため、禁止する。

## 実験の比較方法

新条件は `E_core_clipped` と呼び、元のDの結果・保存名を上書きしない。

| 条件 | 選択範囲 | 意味 |
|---|---|---|
| 既存D r0 | Dの元target | 余白なしの比較基準 |
| 既存D r4 | Dの膨張後M | 旧身体周辺の編集余地を持つ比較基準 |
| E_core_clipped | Mから男性coreを最終除外 | 広い編集余地と局所的保護の両立を調べる |

ケース内のprompt、seed、モデル、style、Slider強度、steps、初期latent/noise、conditioningを一致させる。変更は最終選択範囲とその由来であり、maskが異なる結果を既存比較CLIの同一partition試験へ無理に通さない。予測選択の数値監査は各runで行う。

後からA/C/Dのtarget生成方式自体を比較する場合は、全条件へ同じEを同じ順序で適用する。Dだけ保護を強化した結果を、自動target抽出の優位性とは解釈しない。

## 合否

数値の必要条件：完成targetとP/Eの重複0、空targetでない、grid一致、再生側の余白0。これらの成立だけでは男性の外見保持を保証しない。

目視では、男性頬の余分な顔片、首・肩・腕の融合、旧服・旧身体の残り、対象の全身体型、男性の顔・眼鏡・髪・主要体格を別々に確認する。手や腕の小移動そのものは許容するが、別の顔の発生や融合は不合格とする。

除外mask Eは人による確認が必要。ツールの「確認済み」記録は入力者の申告であり、正しい人物輪郭を自動認定した証拠ではない。agent作成候補のまま試す場合は、その状態を記録し、人が確認した正解maskと呼ばない。

## 証拠の限界と第二段階

最新の受け渡しZIPには実際の実効mask PNGと診断safetensorsが含まれていない。入力候補からの再構成と報告hashの照合は可能だが、実際の保存ファイルを受け取ったこととは区別する。初回の実機比較にはローカルに保存されている元のmask PNGを使う。

外側の予測は、編集された途中latentに対するbase予測である。編集をしなかった独立したOFF画像の外側を固定する処理ではない。直接侵入がなくても外見が変わる場合は、別の評価・対策が必要になる。

最終目標は、完成したOFF参照画像の生成に依存せず、対象の全身編集範囲と保護領域を自動で得ること。このツールは、与えられたmaskを処理する検証基盤であり、男性coreや全身領域を自動推定する実装ではない。

OFF参照画像から人物輪郭を抽出する案は、良い編集範囲を用いた比較対照として位置づける。内部でOFF画像を自動生成しても、手動準備が不要になるだけで、OFF画像生成の処理・費用は残る。その方式がそのままOFF画像不要の自動化へ移行できるとは主張しない。

OFF完成画像を使わない第二段階では、生成途中の内部特徴から人物への割当、全身の欠落、保護領域をどう推定するかの設計と検証が必要。既存の短いOFF予測・収集passまで不要になることも、これだけでは保証しない。単純な最大成分選択や全成分の結合を、全身の自動抽出と同一視しない。第二段階の方式は未確定で、現時点の成果物に抽出backendは含まれない。
