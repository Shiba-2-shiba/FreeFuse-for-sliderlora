# 検証資料の共有と再検証

2026-10-12までの実装・検証結果を、既存の `docs/slider-mask-collection-comparison` ブランチへまとめる。現在の成果と限界は [README](../README.md) と [step/block sweepからの検討経緯](attention-step-block-sweep.md) に記載した。公開の通常Samplerと既定auto設定は維持し、追加の観測・読込ノードはExperimentalとして提供する。

## Gitに含めるもの

- 実装、回帰テスト、実行・評価・監査・レポート作成スクリプト。
- `attention-ablation-results`、`mask-source-comparison-results`、`attention-step-block-sweep-results` の実行計画、投入graph、history、manifest、数値評価・hash・監査記録とHTMLレポート。
- HTMLと検討文書から参照する完成画像、mask、比較図。画像は元ファイルをそのまま保持する。
- 検討文書からリンクする2026-10-12の計画・progress資料。`.omx/state`などの実行状態は含めない。

[共有成果物一覧](validation-artifacts.json)には結果フォルダー内の収録ファイル、サイズ、SHA-256を記録する。複数試行・旧版レポートも検討経緯として保持し、過去の不合格・未検証を最新結果で上書きしない。

## ローカルに保持するもの

生のattention・予測trace・mask tensor（`.safetensors`）、レポートから直接参照しない大量のpreview PNG、pytest一時ディレクトリ、一般の実行logはローカルに残す。元データは削除しない。実験フォルダー全体は約20GBで、Gitは比較資料と数値記録を共有する範囲とする。レポートからリンクする少量の検証logは収録する。

manifestや監査JSONには元実機の絶対パスと生tensorのhashが残る。これは当時の実行証拠であり、clone直後に生tensorまで揃うという意味ではない。GitだけでHTMLの画像比較・数値結果を閲覧できるが、tensor再計算を含む監査の再実行には同じローカル成果物が必要。PNGと疎な補助labelを全身の正解セグメンテーションとして扱わない。

## コミット前の再検証

```powershell
python -B -m pytest -q -p no:cacheprovider tests --basetemp=test-results/commit-verification-pytest
python -B docs/audit_seed_bootstrap_validation.py --experiment-plan docs/attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/experiment.json --output test-results/commit-seed-bootstrap-audit.json
git diff --check
```

CPU回帰は **780 passed、2 skipped、85 subtests passed**。skipはWindowsの不正filename条件。最新試行の保存済み監査は [verification-r2.json](attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/verification-r2.json) にあり、284 diffusion NFE・17完成画像・80 raw bank・17 trace bundleを記録している。コミット前の再監査も6ケース全て成功し、同じNFEと成果物数を確認した。新しい監査結果は `test-results/commit-seed-bootstrap-audit.json` へ保存し、当時の監査JSONを変更していない。今回、新たなGPU生成やbackend再起動は行っていない。

Python構文、共有成果物のhash、HTMLと更新したMarkdownの参照先、Git差分を確認する。lint/typecheckはリポジトリに設定がない。画像品質の結論は保存済み実機結果の範囲で、補助目視は人の承認ではない。
