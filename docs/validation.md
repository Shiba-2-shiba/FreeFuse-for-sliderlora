# 検証記録

2026-10-09 / **U4・U5・U6評価完了とチャット引き継ぎ**：ユーザー指定でU2の旧commit回帰を完了条件から除外した。T1〜T7とU1/U3/U4/U6は完了、U5は指定比較表の評価・採用判定を完了した。autoの汎用画質は未達のまま。ユーザーがチャット終了を希望したため新しい生成・実装修正を追加せず、[引き継ぎノート](../.omx/notepad.md)と最新plan/progressを残した。

- **U4完了**：new manual、new auto、旧mix halfをComfyUIユーザーフォルダーの`workflows/codex_u4_20261009/{manual,auto,legacy}.json`へ別名保存。専用Edgeを終了・再起動し、保存JSON構造・widget値・linkが一致することを確認。UIの`app.queuePrompt`から3条件を再実行し、キャッシュではなくSamplerを評価してすべて成功した。バックエンドの再起動は行っていない。証拠`test-results/u456-20261009/u4-summary.json`。
- **U5評価完了**：元prompt、正面スタジオ、腰手/腕組み、公園、それぞれseed42/444444、強度2/4、none/all/targetで48生成成功。16部分run全128stepの予測選択は完全一致。全8prompt/seed群で参照partition一致、none強度2/4の全step入力・予測・最終latent・RGBも完全一致。生protected mapは公園seed42のall強度4に限り120要素・最大4.0745362639427185e-10の差を確認した。二値maskは同一だが、生mapを完全一致とは報告しない。事後の許容値で合格へ変更せず測定を保持した。証拠`u5-summary.json`、`u5-raw-map-difference-investigation.json`、比較図8枚、`u5-visual-observations.json`。
- **U6完了**：既存のPhase1/2中断後manual復帰に加え、標準KSampler→Slider→標準KSampler、不正Subjects拒否後の標準KSamplerを実行。前後のlatent・RGBは完全一致、latent最大誤差0。1倍の空latent補間で入力値を維持してcacheを無効化し、各標準Samplerが実際に実行されたことをhistoryで確認した。追加の不正SubjectsはSampler前に拒否される検査で、GPUのPhase1内部強制例外とは区別する。証拠`u6-summary.json`。

| U5条件 | 評価結果 |
|---|---|
| 元prompt / 両seed / 強度2・4 | 対象に強度差、保護側は成人らしい外見。今回の条件では比較的良好 |
| 正面スタジオ / 両seed / 強度2・4 | 顔・体格の効果と保護側の外見を確認。seed444444・強度4の床に白い矩形artifactがあり要調整 |
| 腰手/腕組み / 両seed / 強度2・4 | 基準画像から頭部が画面外で顔比較不可。強度4で新しい顔と古い身体が重なり、増殖/融合する不適合例 |
| 公園 / seed42 / 強度2・4 | 対象効果と保護側の成人らしい外見、道・樹木・影の連続性を確認 |
| 公園 / seed444444 / 強度2 | 対象の顔に効果、体格の変化は限定的。保護側の成人らしい外見を維持 |
| 公園 / seed444444 / 強度4 | 元の服が新しい顔へ重なり、首・胴の接続が崩れる不適合例 |

既存の重なり構図の顔断片も未解決。現在の判定はmanual推奨・auto実験扱いの継続であり、「評価完了」を「画質合格」と読み替えない。次の課題は失敗runのraw/reference/effective maskを基準・生成画像と照合し、誤割当・穴・孤立成分・形状変化後の境界を切り分けること。手動halfで改善した比較はmask形状と収集モードが同時に異なり、原因を単独で証明していない。算法・追跡・境界処理の変更は次の指示に基づく別計画とする。

保存済み評価は繰り返さず、次チャットは引き継ぎノートと各summaryから再開する。生成実装・配布workflow・依存は変更していない。未コミット変更は最新plan/progress、README、本記録、実機チェックリスト、nativeテストの推論モード修正で、commit/pushは行っていない。専用ブラウザーは終了済み。ComfyUIバックエンドは維持し、終了時のqueueは空。

2026-10-09 / **男女が重なる構図とauto選択余白1〜4の追加評価**：同じ実機・モデル・Slider、seed42、強度4で、「女性が手前、男性が後ろから肩に手を置く」「男性が手前、女性が後ろから肩と腕に触れる」の2promptを評価した。基準画像で肩・腕・胴体の重なりを確認してから、それぞれbase、全体適用、auto余白1/2/3/4を生成した。男性が手前の構図にmanual左半分を1条件追加し、合計13生成が成功した。

| 余白 | 女性が手前 | 男性が手前 |
|---|---|---|
| 1 | 対象の顔の効果、保護人物の成人らしい外見。比較的まとまるが全体適用より体格変化は限定的 | 保護人物の顔の横に別の顔の断片、下肢にも残像 |
| 2 | 対象に効果。保護人物の脚・足が対象の左側へ回り込む不自然な配置 | 顔と手の断片が残る |
| 3 | 対象に効果、保護人物の成人らしい外見。今回の女性手前条件では比較的まとまる | 顔の断片が保護人物の頬に重なり、服の境界も崩れる |
| 4 | 対象に効果。保護人物の首・襟に切り欠き状の変化 | 顔の断片・頬・服の境界の崩れが残る |

男性が手前の同条件manual左半分では顔の断片が消え、2人の像が成立した。manualは参照mask形状と収集モードの両方がautoと異なるため、原因を一つに確定する比較ではない。自動maskの誤領域や、体格変化に伴う選択境界のずれが有力な課題で、**余白1〜4を重なり構図全般の安全な範囲とは判定しない**。固定半分maskも人物追従の解決策ではない。

数値監査では、auto部分選択8runとmanual1runの全stepでinside=native / outside=baseが完全一致した。autoのraw map・参照partitionは各prompt内の強度0/4・余白1〜4で完全一致し、初期noise/latent/full sigmas/conditioningも同一。余白拡張は参照partitionを保ち、推定protected maskへの追加は0だった。これは実人物の全画素を正しく保護できたことを意味しない。画像と参照maskの重ね図には背景の誤割当や人物内の穴も見える。演算の合格と画質の不合格を分けて記録する。

保存先はローカルの`test-results/overlap-20261009/`。`summary.json`、`woman_front-comparison.png`、`man_front-comparison.png`、`manual-control-comparison.png`、`reference-mask-overlays.png`と各runの元4点セットを保持する。1seed・2prompt・1Sliderの探索結果であり、年齢や同一人物性の自動スコアにはしていない。生成実装と配布workflowの変更はない。

2026-10-09 / **0.2.0の起動中ComfyUIでの実機追試**：ユーザーの依頼により、`http://127.0.0.1:8000`のComfyUI 0.37.0 / commit `15ef24d1c0333a3eba56c5cd153d8db65363ff8f`、Python 3.12.11、torch 2.12.0+rocm7.14.0、AMD Radeon AI PRO R9700で検証した。モデルは`intorealismAsian_k2JAVFLASHV1.safetensors`（対象投影はint8_tensorwise）、CLIPは`qwen3vl_4b_bf16.safetensors`、styleはdarkbrush 0.8、Sliderは`archive/krea2_deaging_20261001T075826Z_c06cb035.safetensors`。seed42、1024×1024、8steps、Euler/simple、CFG1を使用した。

- native12件はskip0で成功。初回は直接forward比較をautograd有効のまま呼ぶテスト側の不備で1件失敗した。`tests/native_checks.py`の当該比較を実Samplerと同じ`torch.inference_mode()`に揃えて再実行した。生成実装の変更はない。変更後のCPU339件も成功（6.41秒）。
- 実GPUで14条件の生成・診断保存が成功し、2件の意図的中断を確認した。保存物のhash・manifest・mask再生成を既存readerで検証した。zero/none、all/native全体適用、同じmaskのauto/manualは初回予測・最終latent・RGBが完全一致した。保存traceが双方にある比較では全stepの入力・予測も完全一致し、許容誤差は0とした。
- 部分選択は全stepでinside=native / outside=baseが完全一致。autoの強度0/4、余白0/4/8でraw mapと参照partitionが一致した。余白だけが拡大し、protected参照領域への追加は0。auto部分選択のNFEは収集2＋base8＋slider8＝18だった。
- Phase 1収集中とPhase 2生成中に、対象prompt_idだけをキャンセルした。それぞれ直後のmanual strength0が成功し、中断前baseの全step入力・予測・最終latent・RGBと完全一致した。成功runの`owned_hooks_removed`はすべてtrue。
- 実INT8の全140対象moduleを照合し、gate/wq/wk/wv/woの代表5投影probeが成功した。非対象位置・明示式との誤差は0、packed weightは不変。probe終了後にComfyUIの`ModelPatcher.__del__`がPython終了時の`ON_DETACH`参照で警告を出したが、probe結果と終了コードは成功。実サーバーの停止や依存変更は行っていない。
- 実frontend 1.53.6を専用の非表示Edgeで開き、新manual、新auto、旧halfの3graphを読込・serialize・再読込した。専用ノードのwidget値と入出力linkを保持し、再読込後のAPI exportもlive preflightに合格した。ファイル名は実環境へ合わせたコピーを使い、配布workflowは変更していない。ブラウザーは終了済み。

**画質の判定は条件付き。** このseedではmanual halfとauto余白4/8に対象の顔・体格変化が見え、保護人物は成人の外見を維持した。一方、auto余白0では対象の顔の二重化・崩れを確認した。予測選択の数値監査には違反がなく、狭い固定maskと大きな形状変化の組合せは引き続き画質評価が必要。他seed・他prompt・他Slider、同一環境での旧commit数値回帰、アプリ再起動を跨ぐユーザー保存workflowの維持は今回未検証。

元のPNG・mask・safetensors・JSON、実行prompt_id、比較結果、UI export、ログはローカルの`test-results/live-20261009/`へ保存した。集約は`summary.json`、画像比較は`visual-comparison.png`、実INT8 probeは`real-int8-probe.json`。これらはGit追跡対象外で、モデル・画像を公開しない。低temperatureの探索run（保存名`auto_invalid_map`）は実際には成功したため、強制失敗の証拠には数えていない。生成JSONの固定falseフラグは書き換えていない。

以下は実装完了時点の履歴。「実機未実施」は各記録の作成時点の状態であり、上記の追試と区別する。

2026-10-09 / **0.2.0 Prediction Mix採用**：既存Prediction Mixへautoのbase-onlyマスク収集を接続。初期noise/latent/full sigmasを再利用し、Phase別NFEを分離。既存ノードID・位置引数・manual既定と旧workflowを維持し、新manual/auto workflowを追加した。新診断はschema 3で、raw mapからのmask再生成、予測選択の再計算、収集provenance・summaryのNFE検証を行う。旧schema 1/2の読込を維持する。

独立レビューで2点（summaryの収集由来検査不足、親base復帰の検査不足）を確認し、改変受理と親base選択の回帰試験をRED→GREENで修正した。追加レビュー対応後の最終CPU suiteは339件成功（失敗・skip0）。テストはComfyUI境界doubleを含み、実native検証の代用ではない。

native validatorの対象は12件へ更新した。**0.2.0の実native・INT8/GPU生成・UI保存再読込・auto画質は未実施で、ユーザーが手動確認する。** 実装担当の完了条件はコード・CPU/static検証・検証用workflowと手順の整備まで。[実機確認手順](real-machine-checklist.md)を参照。0.1.4のmanual実機結果と、0.2.0のauto品質を区別する。

追加レビューでは、参照partitionを保った選択専用背景拡張（0〜16 grid、既定0、auto候補4）、旧autoと新helperのraw map/partition一致試験、lowvram注意、runtime version共通化を追加した。追加範囲の独立レビューで指摘0。選択拡張後の実機画質はユーザーの手動確認待ち。

以下は各版の作成時点の履歴。

2026-10-08 / **0.1.4 診断実験版**: 全層・全stepの直接差分audit、単一Eulerループのbase/native予測混合Sampler、v2 trace/参照mask保存、軌道を区別する比較CLI、4条件8 workflowを実装。最新CPU suiteは**282件成功、skip0・失敗0（8.50s）**。自作Python42ファイルのAST解析、workflow JSON62ファイル、git diff --checkも成功。独立レビューの2件の高優先指摘（未知実行フックの拒否不足、部分auditを合格にできる不足）を6件の失敗試験と端点の追加3件で再現して修正し、全suiteを再実行した。

native validatorは現在10件。ローカルComfyUI sourceでは`comfy_aimdo.storage`不足によりimportが停止し、0/10件実行。CPU境界doubleの成功をnative/INT8成功としない。ユーザーが実ComfyUI・実GPU生成・UI保存/再読込・画質評価を担当する。[実行順と比較コマンド](prediction-mixing.md)を参照。実機での端点一致・切替費用・人物保護の採用判定は未完了。

2026-10-08 診断機能: 通常Loader相当/nativeと独自hookを比較する診断Sampler、実効mask/初回prediction/最終latent/JSONの保存、比較ツール、11条件22 workflowを追加。独立レビューの3指摘を回帰試験で再現・修正した後、最終全CPU suiteは**237件成功、skip0・失敗0**。Python構文/compile33ファイル、workflow JSON54ファイル、git diff --checkも成功。実INT8/GPU・UI保存/再読込・画像品質はユーザー側で評価するため、この実装作業では未実施です。native validatorは新しい全系列明示式チェックを含む7件を対象に変更しました。実行手順と検証限界は[診断ガイド](diagnostic-parity.md)、作業記録は`.omx/plans/2026-10-08-slider-diagnostic-parity.progress.md`に記載しています。

初期実装: 2026-10-07。実機確認はユーザーがGitHubから取得後に行う。

| 項目 | 状態 |
|---|---|
| ローカルsuite | 0.1.0は70件、0.1.1は78件、0.1.2は120件、0.1.3は165件成功、skip0・失敗0 |
| CPUのマスク・LoRA数値・alpha/rank・正負強度・対象文章行の選択 | ローカルsuiteで確認。文章倍率0では従来のtext差分0を維持 |
| 二段階noise/latent/sigma再利用、例外復帰、2人物の独立した収集 | ローカルsuiteで確認 |
| V3 schemaの契約、LoRAファイル変更、JSONリンク/型 | test double・静的検査で確認 |
| 現行ネイティブComfyUI小型CPU10件 | import前提不足で0/10件実行。合格ではない |
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
