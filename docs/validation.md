# 検証記録

2026-10-08 診断機能: 通常Loader相当/nativeと独自hookを比較する診断Sampler、実効mask/初回prediction/最終latent/JSONの保存、比較ツール、11条件22 workflowを追加。独立レビューの3指摘を回帰試験で再現・修正した後、最終全CPU suiteは**237件成功、skip0・失敗0**。Python構文/compile33ファイル、workflow JSON54ファイル、git diff --checkも成功。実INT8/GPU・UI保存/再読込・画像品質はユーザー側で評価するため、この実装作業では未実施です。native validatorは新しい全系列明示式チェックを含む7件を対象に変更しました。実行手順と検証限界は[診断ガイド](diagnostic-parity.md)、作業記録は`.omx/plans/2026-10-08-slider-diagnostic-parity.progress.md`に記載しています。

初期実装: 2026-10-07。実機確認はユーザーがGitHubから取得後に行う。

| 項目 | 状態 |
|---|---|
| ローカルsuite | 0.1.0は70件、0.1.1は78件、0.1.2は120件、0.1.3は165件成功、skip0・失敗0 |
| CPUのマスク・LoRA数値・alpha/rank・正負強度・対象文章行の選択 | ローカルsuiteで確認。文章倍率0では従来のtext差分0を維持 |
| 二段階noise/latent/sigma再利用、例外復帰、2人物の独立した収集 | ローカルsuiteで確認 |
| V3 schemaの契約、LoRAファイル変更、JSONリンク/型 | test double・静的検査で確認 |
| ネイティブComfyUI小型CPU6件 | import前提不足で0件実行。合格ではない |
| ConvRot INT8実モデル・実Sliderのprobe | 未実施 |
| 実ComfyUI UIの保存/再読込・生成 | ユーザーのmanual/auto生成画像を受領。保存/再読込の検証結果は未受領 |
| 自動maskと男女の画質・局所性 | manualの効果とautoの顔mask欠落を確認。LoRA/強度不一致のため統制比較は未了 |
| 速度・VRAMの実測 | 未実施 |

Python3.10.11 / torch2.10.0+cpu / CUDAなし。検証用ComfyUI checkoutは`b26625f23a888367b92153b28d93e159e83e677b`。native validatorは`ModuleNotFoundError: comfy_aimdo.storage`でexit1。既存comfy_kitchenは`TensorCoreConvRotW4A4Layout`も不足している。環境の依存は変更していない。

実装時のAPI照合はユーザー生成ログのComfyUI commit `3d9b2d551788d4fe80ede5743417077d1795cbd2`の公式ソースを対象にした。検証用checkoutの新しいSHAと実機照合先を区別する。

ローカルsuiteの最終件数とレビュー対応は`.omx/plans/2026-10-07-krea2-female-only-slider-freefuse.progress.md`に記録する。native validatorのimport失敗やtest doubleを、実機動作の証明には扱わない。

0.1.1の追試条件と追加診断は[auto-mask-investigation.md](auto-mask-investigation.md)、作業証拠は`.omx/plans/2026-10-07-auto-mask-investigation.progress.md`に記録する。mask算法の修正は未実施で、診断更新を画質修正済みとは扱わない。

追加実機観測: seed42の候補設定で、target maskの最大断片比率が91.42%となり、顔への適用範囲とSliderの効果が改善したとの報告を受領。[候補記録](auto-mask-candidate.md)の強度0/4比較と他seedでの検証は未了。全体の品質合格条件を満たしたとは扱わない。

0.1.2: [mask後処理](mask-postprocessing.md)のCPU数値/旧API/Phase2接続を追加検証。提供maskの条件付き計算で583/612/627/813を再現し、protected不変とpartitionを確認した。新後処理の実GPU生成・UI・画質は未確認。最終suiteとレビュー結果は`.omx/plans/2026-10-07-protected-mask-postprocess.progress.md`へ記録する。

0.1.3: [target文章行の任意適用](target-text-routing.md)を追加。倍率0/0.5/1と正負・ゼロ強度の明示式、protected/その他文章への直接差分0、auto収集の同一性、manual接続、例外後のhook・cache復元をCPUで確認。3比較版と通常全体参照版のUI/API接続・設定を静的検証。実機の効果回復・男性への漏れ・INT8・UI互換性は未確認。証拠は`.omx/plans/2026-10-07-target-text-lora.progress.md`へ記録する。
