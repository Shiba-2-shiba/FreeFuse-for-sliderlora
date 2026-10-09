# Prediction Mix採用の実装記録

## 最新進捗（2026-10-09、ユーザー追加指示反映）

| 項目 | 状態 | 証拠・残作業 |
|---|---|---|
| T1〜T7 | 実装完了 | CPU339件、native12件skip0、実INT8全140対象照合・代表5投影成功 |
| U1 | 完了 | ComfyUI0.37.0 / 15ef24d1、AMD R9700、Python3.12.11、torch2.12.0+rocm7.14.0 |
| U2 | 対象外 | ユーザーが不要と指定。旧commit回帰を合格と偽らない |
| U3 | 完了 | 同一mask auto/manualの全step・latent・RGBが完全一致。強度0/4の収集一致、余白の参照不変 |
| U4 | 完了 | 3graphをComfyUIユーザーフォルダーへ保存。専用UI再起動後のJSON構造・widget・link一致、UIからの再実行3件成功。backendは再起動していない |
| U5 | 評価完了・auto画質未達 | 4prompt×2seed×強度2/4×none/all/targetの48条件を生成・監査・目視判定。姿勢の頭部crop/増殖、公園の顔/服融合、床の矩形artifactを残した |
| U6 | 完了 | 既存Phase1/2中断後復帰に加え、標準KSampler→Slider→標準KSampler、不正Subjects後の標準KSamplerを実行。cacheを使わない前後latent/RGB完全一致 |

既存実機成果物: `test-results/live-20261009/summary.json`、`test-results/overlap-20261009/summary.json`。新しい残作業は`test-results/u456-20261009/`へ保存する。画質は条件付きで、男性が手前の重なりではauto余白1〜4すべてに顔断片が残る。manual半分で断片は消えるが、mask形状と収集モードを同時に変えた比較なので原因の完全な切り分けではない。詳細は`docs/validation.md`冒頭。

U4/U5/U6の証拠は`test-results/u456-20261009/u4-summary.json`、`u5-summary.json`、`u6-summary.json`。U5部分混合16run全128stepで予測選択が完全一致し、全8prompt/seed群で参照partitionとnone強度2/4結果が完全一致。公園seed42のall強度4のみ、生protected map120要素に最大4.0745362639427185e-10の差があり、完全一致ではない。二値maskは同一。許容値を追加して合格へ変えず、差を測定記録へ保持した。

Ruling: U5は比較表を実施し失敗を含む採用判定を残した時点で「評価完了」。autoの汎用画質合格とは扱わない。姿勢の基準頭部欠落は基礎生成不成立として記録し、seedを差し替えない。

ユーザーがチャット終了・引き継ぎを希望したため、新規評価と生成実装変更を追加しない。次は`.omx/notepad.md`と`docs/validation.md`を読み、自動maskの誤割当・穴・人物形状変化後の境界を切り分ける。U2は再導入しない。現在の未コミット変更・保存済み評価を維持する。

以下は実装時点の履歴。「実機確認待ち」は当時の状態であり最新状態ではない。

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
