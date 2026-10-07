# 検証記録

初期実装: 2026-10-07。実機確認はユーザーがGitHubから取得後に行う。

| 項目 | 状態 |
|---|---|
| ローカルsuite | 0.1.0は70件成功。0.1.1の診断・比較追加後は78件成功、skip0・失敗0 |
| CPUのマスク・LoRA数値・alpha/rank・正負強度・text差分0 | ローカルsuiteで確認 |
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
