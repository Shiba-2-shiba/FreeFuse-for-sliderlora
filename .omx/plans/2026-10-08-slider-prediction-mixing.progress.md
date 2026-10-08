# 局所Slider予測混合の実装記録

Plan: `2026-10-08-slider-prediction-mixing.md`

2026-10-08、作業ブランチ`feat/slider-prediction-mix`で実装。ユーザーの指示により実ComfyUI/GPU/UIの検証はユーザーが担当する。

| Task | ローカル成果物 | 検証 |
|---|---|---|
| 1 | 全層・全stepの直接差分audit、step trace | RED4→GREEN。2block×5投影×2stepの20記録、非選択行故障検出、RNG/出力不変 |
| 2 | 二値maskのpatch展開/cropと厳密予測選択 | RED7→GREEN。BF16/FP32、端点、奇数4D/5D、入力不変 |
| 3 | shared-core NativePairRunnerとlazy CFGGuider、単一Eulerループ | RED4とboundary-order1→GREEN。順序/identity、例外、保存バッファ、8/16NFE。native3件追加（合計10） |
| 4 | 7ノード、v2保存/manifest/provenance、0.1.4 | RED3→GREEN。旧契約、新ノード、trace破損/途中保存拒否 |
| 5 | 同入力/別軌道の比較、直接違反、端点参照 | RED5→GREEN。旧report互換、実装identity、trace hash、終了コード |
| 6 | mix4条件のUI/API8ファイル、実機ガイド | RED4→GREEN。設定/リンク/型/seed固定、Slider二重適用なし |

## 設計上の判断

- 指定checkout内の専用ブランチで実装。外部へのpush/merge・依存の自動導入は行っていない。
- ComfyUI依存のCFGGuider subclassはlazy factoryで生成し、extension importがComfyUIを要求しない契約を維持。
- 全面の実効maskだけでは対象/保護領域を評価できないため、参照partitionを3つの小さなtensorとして保存した。
- mix_noneの指定strength4とnative_zeroのstrength0は、`--endpoint-reference`でeffective strength0とslider NFE0を実証できる場合だけ比較できる。その他の条件は一致を要求。
- 巨大な全層audit/step traceはJSON/safetensorsへ保存し、通常ログへ全レコードを展開しない。
- 計画上の近接した合成テストを観測可能な試験へまとめた。実装内部の文字列を検査するテストを追加していない。

## 独立レビュー

2件の高優先指摘を修正した。

1. `sampler_calc_cond_batch_function`やnested transformer wrappers/callbacksが入れる不足を、fail-closed試験で再現して拒否した。
2. 一部のauditだけでmeasured_zeroとなる不足を、欠落/端点の捏造行を含む試験で再現した。hookは全step×moduleの一意性と呼出数、部分mixは全step、端点は正しいbranch数とaudit空を要求する。保存/読込/分類で検査する。

6件の失敗を確認して修正し、端点の追加3件もRED→GREEN。最終レビューは**APPROVE（ローカル/静的範囲）**。

## 最新の検証

- `python -B -m pytest -q -p no:cacheprovider tests` → **282 passed in 8.50s**、skip0/失敗0。
- 自作Python42ファイルのAST解析 → 成功。
- workflow JSON62ファイルのparse → 成功。
- `git diff --check` → 成功（WindowsのLF正規化案内を除く）。
- native validatorは10件を用意。ローカルsnapshotは`comfy_aimdo.storage`不足でimport停止、**0/10件実行**。doubleをnative実行の代用にしない。

## ユーザー実機で確認する項目

1. 7ノード登録とUI保存/再読込、必要ならnative validator10件。
2. 同じ新版のnative_global/hook_all_all/hook_half_noneをaudit・trial0/1で反復。
3. mix_zero → mix_none → mix_allでゼロ/全黒/全白の端点を確認。
4. mix_halfと既存画像側hookを比較し、対象の顔/頭身/体格、保護人物の同属性、境界/構図を評価。
5. 有望な方式だけstrength2・seed123/777へ展開。

受入条件A1〜A8のローカル部分は完了。A4/A5の実native部分とA9/A10の実INT8・画像評価は未確認。手順は`docs/prediction-mixing.md`。本機能は診断実験版で、通常Samplerの既定方式へ昇格していない。
