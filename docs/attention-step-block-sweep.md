# Attention step / block sweep の続行結果

**現在の到達点（2026-10-12）:** 顔mask欠如と三人化は、同一Mix入力でのbranch観測・再分類とbase-only初期化で、検証条件内ではある程度改善した。最新の診断2構図とholdout4条件で完成顔を維持し、独立した第三の頭部・人物は見当たらない。一方、重なりseed123の旧身体状残片、背景漏れ・四肢欠落が残る。R1〜R4からの新候補選定と既定auto採用は未実施。末尾の「Seed固定・base-only初期化・片側支持の追加検証」が最新結果で、以下は検討順の記録。[画像・mask一覧](attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/visual-summary-v2/index.html) / [保存範囲と再検証](validation-artifacts.md)。

2026-10-12（日本時間）、中断した `20261011T111323Z` の凍結計画を、起動中のComfyUI `http://127.0.0.1:8000` で完了した。再起動は行っていない。

[画像・mask・attentionの比較レポート](attention-step-block-sweep-results/20261011T170454Z/report.html) / [機械検証](attention-step-block-sweep-results/20261011T170454Z/verification.json) / [補助目視](attention-step-block-sweep-results/20261011T170454Z/assistant-visual-observations.json)

## 結果

step 1 / block 18は両構図で有望だが、顔中心の問題の完全な解決には至っていない。公園では大きな顔欠損が減り、頭身・四肢にも変化が見える一方、口・顎に白い継ぎ目が残った。重なりでは顔と頭身が比較的まとまった。step 3や遅いblockへの単純な移動は、顔・腕の欠損や旧人物の残像を安定して解消しなかった。

168条件の観測を完了し、148条件でmaskが成立、20条件は高信頼seed不足として記録した。予定18画像中11画像を生成した。7条件は元maskが不成立のため生成せず、代替maskや別条件へ置換していない。

| 構図 | OFF成立 / 42 | ON成立 / 42 | 編集画像 | 生成対象外 | 実測NFE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 公園 / seed 444444 | 39 | 39 | 6 | 3 | 102 |
| 女性手前の重なり / seed 42 | 34 | 36 | 5 | 4 | 86 |
| 合計 | 73 | 75 | 11 | 7 | 188 |

block 27は全step・全構図・OFF/ONで対象seed不足。重なりON step3/block22も保護seed不足で不成立だった。mask成立は人体領域の正確さや生成画質の合格を意味しない。

## 顔中心付近の固定70セル

前回公園ON step2/block18の閉じた穴の座標を固定して比較した。独立した顔labelではなく、生成後の顔位置も条件ごとに変わる。

| ON条件 | 対象 / 70 | 保護 / 70 | 背景 / 70 | 固定OFF保護を引いた実効対象 / 70 | 画像 |
| --- | ---: | ---: | ---: | ---: | --- |
| step1 / block14 | 57 | 0 | 13 | 57 | 事前候補外・未生成 |
| step1 / block18 | 49 | 20 | 1 | 49 | 改善するが口・顎の継ぎ目が残る |
| step2 / block18 | 0 | 62 | 8 | 0 | 前回の大きな欠損を再現 |
| step3 / block18 | 12 | 58 | 0 | 12 | 顔・目・頬が崩れる |
| step1 / block22 | 22 | 39 | 9 | 22 | 口・顎と襟の融合 |
| step2 / block22 | 2 | 64 | 4 | 2 | 大きな欠損は少ないが顔位置・形状が変わる |
| step3 / block22 | 11 | 53 | 6 | 11 | 頭部の残存像・混合 |

同じblock18で、穴領域のQ/K対象役割スコア平均はstep1で0.712、step2で0.372。保護側は0.280から0.620へ変わる。ROI平均で対象が最大のheadは38/48から12/48へ減る。step2の最終prototypeスコアでは対象0.1538、保護0.8455となり、保護側への競合がさらに強い。

この結果は、顔情報がblockに存在しないという説明より、収集時期・head集約・prototype競合の影響と整合する。step1でも21セルは対象に戻らない。Q/K役割スコアは3役割内のheuristicで、本来の全token attention確率や校正済みconfidenceではない。

重なりblock22/step1では頭上と脚に旧領域が残り、step2では背後に別の大きな顔・身体が発生した。mask量や公園のROI被覆だけで採用判断はできない。

## 統制と検証

- 2構図のprompt・seed、Slider +4、1024×1024、Euler/simple・8 steps・CFG 1、k3方式と閾値を固定。収集はstep1/2/3 × block14〜27の共有3 forward。穴埋め・mask膨張・selection余白は全て0。
- 編集対象は事前指定block18/22/27のON maskから、固定OFF step2/block18保護を引く。保護maskは全候補で同じ。本生成のMODELには収集用Sliderを事前適用しない。
- 全tapのfile/tensor hash・PNG、候補status、tap網羅性、hook除去、forward入力・出力の非改変を検査。保存スコアから二値maskを再構成した。
- 11編集bundleのhash・固定target/protected/background、全stepのinside=Slider/outside=base選択を検査。各構図の全候補で初回base/Slider予測が一致した。
- 両構図のON step2/block18は、前回Cとpartition・最終latent・Slider hashまで完全一致した。コードのsource hashは前回と異なり、数値の一致とは別に記録している。
- 編集段階の追加収集はcache hitにより0 NFE。収集12 + 編集176 = 188 NFE。中断試行の既知3 NFEと不明な追加分は含めない。
- 対象CPUテストは152 passed・2 skipped。スキップはWindowsで作れない不正filenameのケース。既存pytest一時領域の権限制限は、新しい試行内の一時領域を指定して回避した。

補助目視はCodexの観察で、人の承認ではない。男性の外見には候補間の差が残り、未編集OFF/allへの完全保持・独立labelによる精度・別seedへの汎化は未検証。

## 保存と再検証

旧ジョブは現サーバーの履歴・キューに存在しなかったため、旧試行を保持し、[続行フォルダー](attention-step-block-sweep-results/20261011T170454Z/continuation.json)でOFF/ONを再収集した。旧フォルダーの [CONTINUATION.md](attention-step-block-sweep-results/20261011T111323Z/CONTINUATION.md) からも結果へ辿れる。

`docs/audit_attention_step_block_sweep.py` は保存済み成果物だけを検査し、新しい `verification.json` を排他的に作る。`docs/build_attention_step_block_sweep_report.py` は検証JSON・補助目視からHTMLを再生成する。`--figures-only` で全mask一覧・編集contact sheet・固定ROIの図を再生成できる。

既定step/blockやmask算法は変更していない。次の検証軸はstep1/block18を対照にしたhead集約とprototypeの対象/保護競合。block14/step1はmask上の候補だが画像は未生成であり、全身・保護側・別seedの確認を経て判断する。

## block 1〜13の追加検証

前回は凍結計画がblock14〜27のみだったため、ユーザーの指摘を受けて起動中の同じComfyUIでblock1〜13も追加した。[追加mask・attentionとblock1〜27の統合比較](attention-step-block-sweep-results/20261011T173209Z-blocks1-13/report.html) / [追加検証JSON](attention-step-block-sweep-results/20261011T173209Z-blocks1-13/verification.json)。上の結論は前回の画像生成候補に対するものであり、早いblockを除いたまま全blockの最適条件とは扱わない。

追加156条件中103条件でmaskが成立し、53条件は高信頼seed不足として保存した。block18も12対照tapを再収集し、全保存tensor hash・候補statusが前回と完全一致した。追加12 NFE、前回と合わせ200 NFE。新しいblockの編集画像はまだ生成していない。

| 構図 | OFF成立 / 39 | ON成立 / 39 |
| --- | ---: | ---: |
| 公園 | 27 | 27 |
| 女性手前の重なり | 27 | 22 |

両構図のblock1〜4は全step・両branchで不成立だったが、観測自体と失敗理由は保存している。結果には空欄ではなく失敗条件として表示する。

公園ではstep3/block7・8・9・10、step2/block10・13で旧70セルのうち69セルが実効対象へ戻った。前回のstep1/block18の49セルより多く、早いblockにも顔中心被覆を改善する候補がある。maskの周辺には散在成分も見えるため、これを画像品質の合格とはしない。次の画像比較では早いblockも候補へ含め、頭部・全身・保護側と残像を評価する。

今回の表示は前回と同じ実装上のblock indexで、block1〜27の324条件を網羅した。実装は0〜27の添字だが、block0は今回指定された1〜13の補完対象には含めていない。対象sweepのCPUテストは13 passed。

## 人物構図・輪郭・役割割当の再評価

ユーザーから、女性手前の重なりON step3/block26のmaskが三人に見え、公園は前半blockが粗く、後半step3が人物・手足へ沿う一方で顔が保護側へ入るという所見を受けた。[確認所見](attention-step-block-sweep-results/20261011T173209Z-blocks1-13/mask-shape-review/observations.json)と[検査済みmaskの形状統計](attention-step-block-sweep-results/20261011T173209Z-blocks1-13/mask-shape-review/selected-mask-evidence.json)に記録した。

重なりON step3/block26は中央に対象の頭部状領域、その左右に保護の頭部状領域があり、三頭のような配置になっている。ON step3/block18・23にも同様の傾向が見える。同じstep3/block18のOFFは左対象・右保護の二人状配置で、ON収集による配置または役割分離の崩れと整合する。[OFF/ONとblockの拡大比較](attention-step-block-sweep-results/20261011T173209Z-blocks1-13/mask-shape-review/woman_front_strong_overlap-shape-comparison.png)。二人の人物領域・役割分離として、このON maskを適合とは扱わない。block26の対象最大成分比は97.21%でも、この問題は検出できない。

公園ON step3/block7・10は広い人物領域と輪郭周囲の散在が目立つ。後半step3/block18・23・26では手足・輪郭の形がより明瞭になる一方、対象の顔中心が保護側へ分類される。block26は対象の胴・脚の欠落も大きい。[前半・後半とstepの拡大比較](attention-step-block-sweep-results/20261011T173209Z-blocks1-13/mask-shape-review/park-shape-comparison.png)。早いblockの69/70という顔ROI被覆だけで最良と判断せず、輪郭・全身の被覆・背景への散在・人物対応を併せて見る。

保存された `status=ok` は計算成立を表し、人物数や対象/保護の正しさを保証しない。既存編集は固定OFF保護を使うが、ONの対象maskから既に落ちた顔は、保護maskの差し替えだけでは復元されない。

三頭状の配置が収集途中の像に既に存在するのか、同一人物の特徴が分類で二役へ分割されるのかは、maskだけでは未確定。完成参照RGBは未保存なので、同一seed・同一sigmaのSliderなし/ありの収集像と役割mapの照合が次の切り分けとなる。公園では後半step3の輪郭を評価対象に残し、顔の誤割当を別問題として調べる。

## Prediction Mixに合わせた方式の再検討

保存済みhashを再確認すると、両構図ともOFF/ON収集の入力はstep1のみ一致し、step2・3は異なっていた。従来のON収集は全体LoRAありの独立軌道であり、対象領域だけを混合する実際のMix軌道上の観測ではない。次の比較は同じMix入力・sigma上でbase/LoRAの両branchを観測し、重みの影響と途中latentの違いを分ける。

公園の旧穴70セル中、ON step2/block18では32セルが保護seedに選ばれていた。現行候補はそのseedをone-hotで固定し、伝播でも固定を維持するため、後段のmask閾値や穴埋めだけでは再分類できない。前半〜中盤の人物対応を手掛かりに、後半step3の輪郭を再分類する方式を第一候補として検討する。顔被覆の高い前半mask全体や、輪郭の詳しい後半protected mask全体をそのまま正解にはしない。

[再検討と追加検証の計画](../.omx/plans/2026-10-12-prediction-mix-mask-validation.md)に、既存traceの正しい復号、OFF/ON/Mixの3軌道×base/LoRA重みの交差実験、三人状配置の原因別対策、固定maskの再生比較、動的更新へ進む条件を記録した。[今回の読み取り証拠](../.omx/plans/2026-10-12-prediction-mix-mask-evidence.json)を併記する。提案方式の実装・追加生成はまだ行っていない。

## 計画に沿った実装と追加生成の結果

上記計画の観測・復号・seed再分類・輪郭再分類・固定mask再生を実装し、同じComfyUI backendを同じ起動引数で再読み込みして実機検証した。[追加生成10画像と交差比較のレポート](attention-step-block-sweep-results/20261011T185947Z-paired-mix/report.html)、[数値監査](attention-step-block-sweep-results/20261011T185947Z-paired-mix/verification.json)、[実行記録と計画差分](../.omx/plans/2026-10-12-prediction-mix-mask-validation.progress.md)。上の「未実施」は計画作成時点の記録。

三人状配置は重なり構図の全体ON軌道でstep3時点に既に発生していた。**同じON入力をbase重みで再評価しても三人**で、全体ONの完成画像も三人。一方、OFF軌道と実際のMix軌道の同step入力を両重みで評価した予測は二人だった。独立した全体ON収集を、そのままMixのmask収集元にする方法の問題を確認した。

両構図の観測付きDは、保存済みDと全stepの入力・base/LoRA/混合予測・最終latentが完全一致。交差再評価では元重みの予測と全tap tensor hashが完全一致した。200 tapを保存・検査し、3軌道6完成画像＋2修正候補×2構図の4画像を生成した。実測144 NFE（交差観測80、候補再生64）。保存済み6条件のtrace復号は0 diffusion NFEで、最終RGBを元画像と一致させた。

候補は同じMix入力の前半block8/10・両branchから一致したsemantic seedを作り、後半block18/23のfeaturesと全48headの画像Q/K親和性で再分類する。seedのみ／輪郭のみ／両方のablationを保存した。初回の生成比較では保護をDに固定し、旧D対象と対応付きbase/LoRA対象の和集合から保護を引いた。mask衝突と追加領域を記録している。

修正4画像は二人構図を維持し、公園の足元の切り欠きを減らした。block18は継続候補だが、両構図で全身体型の一貫した追加改善は未確認。重なりblock23は前腕に明瞭な切り欠きがある。計画の採用gateを通過とせず、holdoutと動的maskへの展開は行っていない。実装・追加検証の完了と、本番の既定maskとしての採用を区別する。

CPU回帰は713 passed・2 skipped・85 subtests passed。Windowsのsymlink表記テストを正規化し、保存maskの読み込みには未完了拒否・改変検査・cache fingerprint・所有tensorコピーを追加した。最終版を読み込んだbackendでmask読込・保存の実機smokeを行い、PNGの画素一致を確認。最終レビューは設定native reviewerが利用できなかったため別パスの自己レビュー。

## Dとの和集合・部位・固定保護の分離検証

2026-10-12、[実行計画](../.omx/plans/2026-10-12-paired-mask-attribution-validation.md)に沿って実装し、現在起動中のComfyUI `127.0.0.1:8000` で検証した。サーバーの再起動はしていない。[10画像・部位・衝突・途中予測の比較レポート](attention-step-block-sweep-results/20261011T235655Z-mask-attribution/report.html)、[数値監査](attention-step-block-sweep-results/20261011T235655Z-mask-attribution/verification.json)、[部位指標](attention-step-block-sweep-results/20261011T235655Z-mask-attribution/attribution-r2.json)、[実行記録](../.omx/plans/2026-10-12-paired-mask-attribution-validation.progress.md)。

seed修正のS0/S1と、seed修正＋輪郭伝播のC0/C1を比較した。0は旧D対象との和集合なし、1はありで、全条件の保護はDに固定した。0でも同じMix入力のbase/LoRA対象の和集合は使用する。新鮮なD対照は両構図とも全stepの入力・base/LoRA/混合予測・最終latentが旧Dと完全一致した。

| 構図・条件 | 対象セル | D比追加 / 削除 | 両branchで背景と確認した領域への選択セル |
| --- | ---: | ---: | ---: |
| 公園 D | 1211 | 0 / 0 | 210 |
| 公園 S0 / C0 | 945 / 944 | 32 / 298、34 / 301 | 110 / 109 |
| 公園 S1 / C1 | 1243 / 1245 | 32 / 0、34 / 0 | 215 / 215 |
| 重なり D | 1236 | 0 / 0 | 155 |
| 重なり S0 / C0 | 1112 / 1116 | 130 / 254、130 / 250 | 178 / 177 |
| 重なり S1 / C1 | 1366 / 1366 | 130 / 0、130 / 0 | 219 / 218 |

和集合なしでも、顔と二人構図を維持した。今回の顔維持に旧D支持が不可欠とはいえない。公園では足首・靴の切り欠きが減ったが、両構図で頭身・胴脚比が一貫して追加改善したとは判定できない。重なりのblock18候補には前回block23のような大きな前腕の穴は見当たらなかった。

最終候補のDへの追加セルのうち、RGBから確認した顔内部labelに入るセルは両構図とも0だった。公園の追加32/34セルには足部4・腕1・共通背景5セルが含まれ、重なりの追加130セルには足部1・腕1・共通背景64/63セルが含まれる。顔維持・足元の変化・背景への漏れは分けて評価する。

部位labelは同じpilot入力のbase/LoRA各RGBから確認した疎な内部セルで、`human_verified=false`。全身正解labelは供給せず、重なりLoRA側の左腕は接触・ぼけにより判定不能とした。全体precision/IoUや生成後の人物精度をこれらから推定しない。部位の一部には欠落が残り、重なりの共通背景選択はDより増えたため採用gateは不合格、既定値・holdout・動的更新への展開は保留した。

S0 rawと固定D保護の衝突は公園16セル、重なり53セル。両branchで対象と確認できたセルは0だった。公園では共通背景1・未確定15、重なりでは保護内部9・共通背景19・未確定25セルを記録した。解除対象を人工的に作らず、S0-releaseは両構図とも`not_applicable`として生成しなかった。これにより、固定保護が切り欠きの原因かどうかは未確定のまま残る。

実測160 NFE（D対照32＋4条件×2構図128）。保存Mixの60 tapを再検査し、32 raw variant bankを再計算・監査した。10画像のhash・partition・全step選択・初回branch一致・hook除去・実行graph・cache/NFEを監査し、10 trace bundleの復号は0 diffusion NFE、最終RGBを生成画像と一致させた。CPU回帰は745 passed・2 skipped・85 subtests passed。設定されたnative reviewerのモデルが非対応だったため、別パスの著者自己レビューで入力条件・参照hash・評価の分母・未確定領域・解除境界を確認した。

## Seed固定・base-only初期化・片側支持の追加検証

画像を先に見たい場合は、[完成画像とmaskの見比べ一覧](attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/visual-summary-v2/index.html)を参照。6条件の完成像・実効maskを同じ順序で並べ、D/A比較、raw対象→保護減算の6段階、R0〜R4の顔分類、1セル解除と残片の拡大をまとめた。[完成画像一覧PNG](attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/visual-summary-v2/01-results-overview.png)と[実効mask一覧PNG](attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/visual-summary-v2/02-effective-mask-overview.png)も保存できる。

2026-10-12、[計画](../.omx/plans/2026-10-12-seed-lock-bootstrap-validation.md)のTasks 1〜5を実装し、起動中の同じComfyUI `127.0.0.1:8000` を再起動せず検証した。[17画像の比較レポート](attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/report-r2.html)、[数値監査](attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/verification-r2.json)、[実行記録](../.omx/plans/2026-10-12-seed-lock-bootstrap-validation.progress.md)。

prototype用のpoolとone-hot固定セルを分離し、P_full＋全固定のR0、固定なしR1、最大9固定R2、base支持でprototype poolを絞るR3、R2と同じ固定でpoolを絞るR4を比較した。R0は保存された2構図×2branchと全score・seed・prototype index・partitionが完全一致した。背景target seedは公園14セル・重なり126セル、固定保護による公園顔9/9→8/9と(x=23,y=15)を再現した。R1〜R4は顔・手足・保護の非悪化条件を満たさず、R*は未選定。固定の緩和／prototype poolの絞り込みだけでは解決しなかった。[候補選定](attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/selection.json)、[prototype poolと固定セルの別集計](attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/seed-pool-attribution.json)。

baseだけの2 forwardからB_T/B_Pを新規収集し、D由来の初期化I_Dとbase-only初期化I_Aを比較した。両構図のBは旧Aと完全一致し、B_Pは旧D保護とも一致。D/Aの観測付きpilotは、それぞれ旧D/Aの全step入力・base/Slider/Mix予測・最終latentと完全一致した。Aのgraphとmask計算に全体ON収集やD mask入力はない。方式選定にはD診断データを使っており、開発過程までD非依存とはしない。

新しい7分類では、公園の顔の1セルをbase-only targetとして識別した。反対Slider RGBの同セルが対象女性の頭髪内部にあることを個別確認し、この1セルだけを診断解除した。target増分／protected減分が同じLだけに一致し、base顔labelは8/9→9/9になった。一方、完成顔は対照でも既に保たれており、解除による明瞭な追加画質改善は確認できなかった。重なりは解除対象なし。自動解除方式は導入していない。

診断A-R0は顔・二人配置・保護の必須gateを通過したため、凍結した2prompt×seed123/777をR0の経路確認として一度だけ実行した。4完成像の顔は保たれ、独立した第三の頭部・人物は見当たらない。ただし重なりseed123には左肩後方の白い衣服／旧身体状残片があり、G2は不合格、接触境界の保護G3は判定不能。その他3条件はこの限定gateを通過した。背景・部位の欠落と全身改善の未確定が残り、R*の成功・汎化や既定採用は主張しない。[最終holdout評価](attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/holdout-assessment.json)。

実測284 diffusion NFE、17完成画像（診断9＋holdout8）、80 raw bankを監査。17 trace bundleは0追加diffusion NFEで最終RGBが完全一致した。checkpoint/style/CLIP/VAE/Sliderの現在のファイルhashは前記runtimeパスでの記録と一致。CPU回帰は780 passed・2 Windows platform skips・85 subtests passed。native roleが非対応modelへ解決され起動できなかったため、最終レビューは別パスの著者自己レビューで行った。公開node schema／既定autoは変更せず、新しい検証スクリプトとoffline scoringとして提供する。
