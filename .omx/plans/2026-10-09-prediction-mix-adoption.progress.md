# Prediction Mix採用の実装記録

基準: fb3cd59 / 実装: 0.2.0

- T1: 旧manualの省略入力・位置引数・端点・旧8workflowを固定。
- T2: base-onlyマスク収集helper、prefix sigmaと初期入力、例外後解除、親設定保持の検査を追加。
- T3: auto/manualをPrediction Mixへ接続。同じmaskのPhase 2一致、端点NFE、auto収集後のbase再activateを検証。
- T4: schema3、保存raw mapからのmask再生成、summaryの収集sigma/観測情報/統計照合、same-mask比較を実装。schema1/2は読込維持。
- T5: 旧入力順を保ち7optionalを末尾追加。推奨manual/autoのUI/API計4workflow、version0.2.0、ガイドを更新。
- T6: native12件用ソース、同一mask書き出しツール、手動実機チェックリストを整備。

独立レビュー: Critical0、Important2。summary情報の改変受理と親base復帰の検証不足を修正。前者は9種の改変回帰、後者はbase→slider端点の起動順と実nativeの成功/強制失敗後のstyle付きbase確認を追加。実nativeソースはユーザーが実行する。

検証: 全CPU327件成功。旧mix8workflow byte-identical。最終構文・リンク検査は最終報告に記録。

判断: auto/manualの実機比較を再現できるよう、検証済み参照maskの書き出しツールを追加。runtimeの親base復帰はconditioning準備後のguiderで行う。例外時は解除して失敗を返し、次の実行で親baseを再activateすることを検査する。

実機native/INT8/GPU/UI/auto画質: ユーザーの手動確認待ち。CPU境界doubleの成功をnative成功としていない。

追加レビュー対応: 選択専用の背景拡張0〜16 grid（省略0、auto候補4）、保護境界非越境、参照partition保持、旧auto/new helperのmap・mask一致試験、runtime version定数を追加。追加範囲の独立レビュー指摘0。全CPU339件成功。Phase 1早期打ち切りは旧auto回帰を優先して延期。診断詳細は実装済みのため保持。

元のリポジトリでの最終検証: 全CPU339件成功、Python52ファイルAST、workflow66 JSON/link、文書/計画81リンク、git diff --check成功。旧mix8ファイルはfb3cd59とbyte-identical。実機native12件・GPU・UI・選択余白の画質評価はユーザー手動確認待ち。
