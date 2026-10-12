# Seed Lock / Base Bootstrap Implementation Plan

> **For agentic workers:** 実行時は `superpowers:executing-plans` に従い、以下のcheckboxをtask単位で実装・検証する。独立した作業はnative subagentへ委譲できるが、非対応role/modelの再試行を完了条件にしない。AGENTS.mdの権限、既存変更保全、実機検証の規則を優先する。

**Goal:** 顔の誤割当を改善するseed修正を維持しつつ背景の誤固定を減らし、全体ON・Dに依存しない初期化と、移動する顔・手足を考慮した保護衝突の扱いを検証する。

**Architecture:** 保存された同一Mix入力でprototypeと固定seedを分離したoffline比較を行う。次にbaseだけで作った初期maskからMixを観測し、D pilotを対照に同じ分類方式を適用して固定再生する。片側だけの対象支持を含む衝突診断は独立した局所解除実験とし、顔・人数・保護・背景・体型を別gateで評価する。

**Tech Stack:** 現在のPython 3.12、PyTorch、safetensors、Pillow、pytest、既存ComfyUI REST／実験ノード。新しい依存・外部segmenterは追加しない。

**Spec:** 下の「要求・判断する問い」に固定した追加提案4点、[これまでの結果](../../docs/attention-step-block-sweep.md)、[直前の実行計画](2026-10-12-paired-mask-attribution-validation.md)と[実行記録](2026-10-12-paired-mask-attribution-validation.progress.md)。2026-10-12作成・再開改訂。会話の参照だけに依存せず、この文書と保存済み根拠から実装できるようにする。

**Status:** Tasks 1〜5の実装・条件付き検証完了。R*未選定、base-only＋R0のholdoutは4条件中1条件で残片gate不合格。既定採用なし。[17画像・284 NFEの結果](../../docs/attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/report-r2.html)と[実行記録](2026-10-12-seed-lock-bootstrap-validation.progress.md)を参照。

## 再開時点の確認

- 直前計画のTasks 1〜5は完了。progress冒頭の古い `executing` 表記より末尾の完了記録を採る。保存された[数値監査](../../docs/attention-step-block-sweep-results/20261011T235655Z-mask-attribution/verification.json)はsource integrity、再集計一致、32 raw bank、160 NFEを記録している。10画像とCPU `745 passed / 2 skipped / 85 subtests passed` は前回の検証結果であり、今回の再実行結果ではない。
- 顔と二人配置は診断条件で維持された一方、重なりの共通背景選択はDの155セルからS0の178セルへ増加した。全身改善と既定採用は未成立。[結果本文](../../docs/attention-step-block-sweep.md):114〜135。
- 最新のseed/bootstrap計画には実行progressも新runnerもなかった。今回、実装開始前の契約・失敗分岐・受入条件を補完する。過去sessionの `.omx/state`、hook状態、既存実験成果物は編集しない。

## 要求・判断する問い

| 要求 | 実装・比較 | 完了時に答える問い |
| --- | --- | --- |
| seed固定とprototypeを分ける | prototypeを固定したlock比較、lockを固定したprototype比較 | 顔を回復した要因と背景を固定した要因は何か |
| 全体ON・D依存を外す | base由来初期mask → 観測付きMix → 固定mask再生 | Dの最終maskだけでなく、D pilotと全体ON収集も不要にできるか |
| 片側の対象支持を扱う | base-only／LoRA-only／役割競合の区別、確認済み局所解除 | 固定保護が、移動前または移動後の対象を削っているか |
| 未使用seedで確認する | 診断条件で方式を凍結後、seed123/777で対照比較 | 顔欠損と三人化への対策が診断seed以外でも成立するか |

実装と検証の完了は、各問いの結論・否定的結果・判定不能理由を保存することで判定する。すべての品質gate通過や既定採用を、作業完了の条件にはしない。

## 現在の根拠

| 事実／再集計 | 解釈と今回の扱い |
| --- | --- |
| 公園・Mix step3/block18の顔内部はbaseで現行3/9→seed修正9/9、LoRAで0/20→20/20。輪郭のみは4/9、0/20 | 顔の分類にはseed修正が有望。疎なagentラベル基準であり、顔全体の精度や汎化ではない。[部位指標](../../docs/attention-step-block-sweep-results/20261011T235655Z-mask-attribution/attribution-r2.json) |
| 修正seedのうち、公園14セル、重なり126セルが両branchの共通背景ラベル内。重なりの対象seed総数は819 | 4観測の一致でも誤seedが残る。Task 1で保存seed tensorとannotationから再集計をJSON化する。[保存seed統計](../../docs/attention-step-block-sweep-results/20261011T235655Z-mask-attribution/woman_front_strong_overlap/ablations/base_s3_b18_seed_only/statistics.json)、[annotation](../../docs/attention-step-block-sweep-results/20261011T235655Z-mask-attribution/annotations/manifest.json) |
| 全seedをone-hotにし、伝播でもseedを維持する | prototype用のpoolと固定するセルを分離する。[anchored_masks.py](../../slider_fuse/anchored_masks.py):18〜29、[experimental_masks.py](../../slider_fuse/experimental_masks.py):203〜255 |
| S0/C0もD pilotの観測とD保護を使用する | D unionを外しただけでは、全体ONからの由来がなくなったとはいえない。[候補構築](../../docs/build_paired_mask_candidates.py):132〜176 |
| 全体ONの同じstep3入力はbase/LoRA両重みで三人、OFF/Mix入力は二人 | 全体ON収集を実運用のmask源から外す方向を検証する。診断2条件以上への一般化は未確認。[交差比較](../../docs/attention-step-block-sweep-results/20261011T185947Z-paired-mix/woman_front_strong_overlap/causal-step3-b26.png) |
| 公園S0のbase顔内部はraw9/9から保護減算後8/9へ落ち、削除セルはnative gridの(x=23,y=15) | 解除対象0は「両branchで同位置が対象」という条件の結果。固定保護の影響がない証拠ではない。Task 1でbefore/afterを再集計する。[衝突分類](../../tools/evaluate_paired_mask_attribution.py):108〜125 |
| 新方式では公園の共通背景選択が減るが、重なりではDより増える | 顔、人数、保護、背景、全身変化を独立に判定する。[直前レポート](../../docs/attention-step-block-sweep-results/20261011T235655Z-mask-attribution/report.html) |

## Global Constraints

- 診断条件は公園seed `444444`、女性手前の重なりseed `42`。prompt、モデル・style・Sliderのhash、Slider `+4`、`1024×1024`、`Euler/simple`、`8 steps`、`CFG 1`、穴埋め・膨張・選択余白 `0` を維持する。[直前experiment](../../docs/attention-step-block-sweep-results/20261011T235655Z-mask-attribution/experiment.json)
- 主分類は `step3/block18`、anchorは同stepの `block8/10 × base/LoRA`。prototype最大3、既存閾値・温度を固定。今回の主比較に輪郭伝播は入れない。既存C0は保存済みの副次対照とする。
- 凍結値は `top_k_ratio=0.2`、`seed_confidence=0.65`、`seed_margin=0.15`、`mask_confidence=0.55`、`mask_margin=0.10`、`context_temperature=1.0`、`feature_temperature=0.15`、`temperature=10000.0`、`prototypes=3`。保存configと照合し、不一致をdefaultsで埋めない。[保存config](../../docs/attention-step-block-sweep-results/20261011T235655Z-mask-attribution/park/ablations/base_s3_b18_seed_only/statistics.json)、[実装既定値](../../slider_fuse/experimental_masks.py):28〜32。
- 人物意味を決めるQ/K head集約・prompt token・LoRA強度・step数を同時に変更しない。新たなblock sweepは行わない。
- 実行時は現在起動中のComfyUIを確認して使用する。既存実験ノードで実行可能なら再起動しない。必要なbackend変更が生じた場合はCPU検証後に起動条件を保存して再読込する。
- 既存データ、実験、annotation、検証JSONを上書きしない。新しい `<UTC>-seed-bootstrap` rootとrevisionを使う。前回コードはdirty/untrackedを含むため、既存変更を巻き戻さない。
- 公開node schemaと既定autoの挙動は変更しない。まず実験スクリプトとoffline scoringとして実装する。
- agent annotationは `human_verified=false`。候補maskから正解ラベルを作らず、unknownを背景や非保護に読み替えない。annotationによる解除は診断対照に限定する。
- 初期化が違えばstep2以降のlatent・顔位置が違う。D pilotのラベルをA pilotへ移植しない。state/branch/input/sigma/RGB hashが一致する場合だけ再利用する。
- 全step固定maskで元noiseから再生成する。毎step更新、無制限pilot反復、追加の全体ON生成、完成OFF画像を必須入力にする方式は今回導入しない。

## 1. seed pool・prototype・固定セルの分離

記号はroleごとに以下とする。

- `E`：現行の4観測anchorから `_seed_pool` で選ぶ候補pool。現行confidence/margin、top-k上限、stable row-major tie breakを維持する。
- `B_T/B_P`：独立したbase-only step2/block18 k3収集のtarget/protected。既存診断条件のoffline段階は検査済み旧Aのmaskを使い、実機段階では新規base収集で再取得する。
- `E_supported`：targetは `E_target ∩ B_T`、protectedは `E_protected ∩ B_P`。backgroundは `E_background` を維持する。Bの残余領域を正しい背景seedとして追加しない。
- `P_full/P_supported`：それぞれE／E_supportedで推定したprototype bank。branchごとに同じlate featureを用いる。
- `K`：E_supportedの既存margin順上位3セル／role（3未満なら存在する全セル）。anchorとBだけで決まり、P_full/P_supportedのどちらを採るかに依存しない。最大9セル。

E_supportedは信頼できる人物正解ではなく、古い位置の支持を使う検証仮説である。正しい変形先の顔・腕を失う可能性を評価する。Eのrole poolが空ならR0を含む全方式が不成立。E_supportedのrole poolだけが空なら、それを必要とするR2/R3/R4をfailedにし、Eを使うR0/R1は継続する。部分的なKやEへのfallbackで不成立を隠さない。Kの数や支持条件は結果を見て変更しない。

| ID | prototype | one-hot固定 | 切り分け |
| --- | --- | --- | --- |
| R0 | P_full | E全セル | 現行seed_only／S0相当の対照 |
| R1 | P_full | なし | R0対R1で全seed固定の影響 |
| R2 | P_full | Kのみ | R1対R2で少数固定の影響 |
| R3 | P_supported | なし | R1対R3でprototype pool変更の影響 |
| R4 | P_supported | R2と完全同じK | R2対R4で固定セルをそろえたprototype比較 |

固定しないセルは、同じ `softmax(feature_score/T_feature + log(late_context))` で分類する。lock比較ではclamp前scoreとprototype tensorのhashが一致し、prototype比較ではlock maskのhashが一致することを監査する。backgroundを含む3役割に同じ規則を適用し、変更箇所をmanifestに保存する。

anchorは4観測のrole softmax平均とし、argmaxが4観測で一致しないセルを3役割とも `1/3` に置く現行順序を保持する。E_supportedはEの順序を保持したintersection、Kはその先頭3件とする。E、E_supported、Kは同じpilotの両branchで共有し、prototype bankは各branchのlate featureから独立に作る。[anchor構築](../../docs/build_paired_mask_candidates.py):137〜141、[poolの順位とprototype計算](../../slider_fuse/experimental_masks.py):109〜144。

新scorerは `prototype_pools` と `lock_masks` を分け、旧 `_candidate_bank` へ渡す互換field `seeds` は **lock_masksのみ** とする。`seed_counts` も固定数とし、`prototype_pool_counts` を別保存する。R1/R3では3役割のseedsを全Falseにする。pool全体を互換fieldへ渡して非固定セルを固定扱いしない。R0だけは旧fieldも完全一致する。[bankのseed参照](../../slider_fuse/experimental_masks.py):182〜200。

R0は既存 `anchored_candidate` のscore、seed mask、prototype index、最終partitionを完全再現する。既存署名を維持し、共通計算を抽出する場合も計算順を変えない。R0の再現に失敗したら他方式の品質比較へ進まない。

offlineで各branchの顔・胴・左右手足・保護・背景、固定seed内の誤ラベル、prototype用pool内の誤ラベル、uncertain領域を集計する。旧70セルは副次指標。硬い固定だけで顔が維持され、非固定では失われるなら、その限界も結果として残す。

### R*の選定

診断2条件のD pilotの保存tensorだけでR1〜R4を比較し、同じ方式IDを両構図に使うR*を最大1つ選ぶ。構図ごとの別方式選択はしない。

候補資格は、両構図・両branchで確認済み顔内部と各可視手足の欠落がR0より増えず、保護内部侵入と両branch共通背景侵入も増えないこと。いずれかの構図で共通背景侵入または固定seedの背景誤分類が厳密に減る候補だけを選ぶ。欠けているlabelはunknownとして保持し、「全手足精度を確認済み」とは記載しない。

比較単位を固定する。顔・手足・保護侵入は各branchのraw targetと、同じpilotのbranch固有labelでR0との差を評価する。さらにpaired unionからB_Pを引いた最終targetでも部位欠落・保護侵入の非悪化を確認し、共通背景侵入はその最終targetと両branch背景labelの積で数える。固定seedの誤分類はtarget lockと共通背景の積、固定セル数は3役割のlockの和を構図ごとに一度だけ数える。branchで重複加算しない。顔labelの欠如・必須比較のnullは選定不可、既知の遮蔽部位のnullは制約として別記する。[部位指標とunknown](../../tools/evaluate_paired_mask_attribution.py):81〜105。

複数候補なら、両構図の共通背景侵入セル合計→固定seedの背景誤分類合計→固定セル数→R1,R2,R3,R4の順で決定する。選定は暫定agentラベル基準と記す。適格候補がなければ `R*=not_selected` とし、seed改善の検証は否定的結果で完了する。独立したbase初期化の検証は、予定された対照R0で続行できる。

## 2. D pilotとbase-only pilotを比較する

初期化は以下の2種類。

- `I_D`：保存済みD target/protectedで観測付きMixを実行する診断対照。Dの由来に全体ONが含まれることを記録する。新たな全体ON収集は実行しない。
- `I_A`：style付きbaseだけから `Krea2SliderFuseExperimentalMaskCollect` で2 forward収集し、`multi_proto_bg` のstep2/block18を選ぶ。得たB_T/B_Pで観測付きMixを実行する。こちらのgraphとmask計算はD／ON収集結果を参照しない。

`AttentionSweepCollect` は3 forward専用なので、2 forwardの初期化へ転用しない。[既存collector構成](../../docs/mask-source-comparison-results/20261011T104616Z/park/queued_api.json) のnode8、[sweepの契約](../../slider_fuse/attention_sweep.py):19〜38

両pilotで実際の同じx・sigmaをbase/LoRAで観測し、step3のblock8/10の文脈とblock18のraw feature/QKを保存する。[観測ノード](../../probe_nodes.py):82〜110、[raw保存](../../slider_fuse/paired_observation.py):64〜102

| 完成画像条件 | 観測元 | 分類 | 保護 | 対象の構成 |
| --- | --- | --- | --- | --- |
| D-R0 | I_D | R0 | 同じB_P | paired target union − B_P |
| D-R* | I_D | 選定済みR* | 同じB_P | 同上 |
| A-R0 | I_A | R0 | 同じB_P | 同上 |
| A-R* | I_A | 同じR* | 同じB_P | 同上 |

B_Pは新規base収集で得たものを全4条件に共有する。診断2条件ではB_Pが旧D保護と一致すること、B_T/B_Pが旧Aと一致することを最初に検証する。不一致を許して初期化差と保護差を混ぜず、原因を確認してから比較する。

bootstrap graphは旧graphのstyle付きMODEL、conditioning、subjectsのテキスト指定、latent、VAEとOFF collectorだけから組む。collector configの `collect_step=2 / collect_block=18 / selected_steps=[1,2] / selected_blocks=[16,18]` を保持し、選択は `multi_proto_bg` とする。subjectsの対象／保護mask入力は未接続にする。保存用outputを含むgraph全体にON collector、Sliderを適用する上流MODEL、D mask loaderを残さない。[旧OFF収集](../../docs/run_mask_source_comparison_live.py):46〜66。

`docs/run_paired_mix_validation.py:35〜41` の `common()` はD loaderを自動挿入するためA graphには呼ばない。A pilotへは新Bのretained-mask manifestだけを渡し、通常のMix sampler内部でSlider branchを作る。`independent_runtime_inputs=true` はAの入力graph／mask計算についての主張に限定する。R*の方式選定自体はD診断データを利用しており、「開発・選定もD非依存」とは記載しない。

どの最終候補にも旧D対象を足さない。同入力base/LoRAの対応する対象支持の和集合は維持する。R*のルールはD側offlineで凍結したものを使い、A側の結果に合わせて再調整しない。

I_Dの全trace／tapを旧D観測と照合する。I_Aの全trace／最終latentは同条件の旧A_offと照合し、観測追加と初期mask再取得が生成を変えていないことを確認する。seed123/777では旧Aがないので、passive observerの非改変監査を使い、存在しない旧対照との一致を主張しない。

## 3. 片側支持の衝突診断

従来の5分類は互換性のため維持し、新しい関数／schemaで次の7分類を出す。対象は `C = raw_target ∩ protected`。

1. `role_conflict`：base対象とLoRA保護、またはLoRA対象とbase保護が競合。
2. `both_target`：両branchで対象内部。
3. `base_only_target`：base対象、LoRA側は対象として未確認。role_conflictは除く。
4. `slider_only_target`：LoRA対象、base側は対象として未確認。role_conflictは除く。
5. `protected`：保護内部として確認でき、対象との競合がない。
6. `common_background`：両branchで背景。
7. `unresolved`：残り。

この順で分類し、7分類の排他性と総和Cを検査する。片側支持は自動解除条件ではない。別の `release-review.json` に、各セルのbranch別画像hash・確認した部位・反対branchの可視状態・解除理由・作成者種別を保存する。反対branchがunknown／遮蔽／不鮮明のままなら解除不可。保護人物・role_conflictも解除不可。単に保護ラベルが付いていないだけでは「非保護と確認済み」にしない。

release-reviewの各rowは `cell_index=y*64+x`、`support_branch`、`target_part`、`opposite_status`（`target/known_background/protected/unknown/occluded/unclear`）、`reason`、`author_kind`、`human_verified` を必須にする。branch別referenceは既存6項目（input hash、sigma、branch、step、RGB hash、prediction hash）に対応付ける。解除許可は `both_target/base_only_target/slider_only_target` かつ反対側 `target/known_background` の明示確認のみ。重複・範囲外・C外・参照不一致は拒否する。既存 `build_diagnostic_releases()` はschema 1の `confirmed_target` 専用なので流用せず、新runnerから検証済みLを `apply_diagnostic_release()` へ渡す。[既存reference検査](../../tools/evaluate_paired_mask_attribution.py):75〜78、[旧解除入口](../../docs/build_paired_mask_candidates.py):190〜224。

局所解除の生成対照は事前固定した **D-R0** とし、最大1条件／構図。前回の公園(x=23,y=15)を含む衝突を両RGBで再確認する。LはC内で対象支持があり、反対branchでも対象または明瞭な非保護領域と個別確認できたセルに限る。

`P_release = B_P − L`、`T_release = raw_target − P_release` とし、`T_release − T_D-R0 = L`、`B_P − P_release = L`、他条件同一を証明する。Lが空ならnot_applicable。解除画像は原因診断のみで、R*選定とholdoutの自動方式には混入させない。A条件の衝突も数値・overlayを保存するが、解除画像を追加で増やさない。

## 4. 問題別のgateとholdout

| Gate | 判定内容 |
| --- | --- |
| G0 数値 | source／asset／条件の一致、R0完全再現、pilot対照一致、同入力両branch、hook除去、partition、全step routing、cache/NFE。失敗した比較群は停止 |
| G1 顔 | branchごとの顔内部欠落が基準R0より増えず、完成像に顔・口・顎の明瞭な穴や混合がない。顔全体は目視も併記 |
| G2 人数 | pilotと完成像が二人配置を維持し、追加の頭部・人物・旧身体残像がない。成分数や面積で人数を代用しない |
| G3 保護・境界 | 確認済み男性の顔・胴への侵入が基準より増えず、完成像に男性の顔増殖・主要体格の意図しない変化・腕の融合がない。軽微な接触変化を別記 |
| G4 背景・部位 | 共通背景侵入、各可視手足の欠落、unknown率を個別表示。改善／不変／悪化／判定不能を残す |
| G5 体型 | 頭身・胴脚比・編集強度の追加変化を記述する。G1/G2の成功を、G5の変化の小ささだけで不合格にしない |

顔・人数の限定的なholdoutへ進む条件は、診断2条件のA経路でG0〜G3を満たし、明瞭な人体の欠損がないこと。G4が悪化または未評価なら、その制約を明示して限定検証とし、総合採用には進めない。G5の追加改善は限定holdoutの必須条件にしない。

gateの基準R0は **同じcase・同じpilot初期化** のR0に固定する。DラベルによるAのセル精度比較はしない。R0自身の数値差は0だが、顔・人数・保護・欠損の目視条件は独立に判定する。R*選定時は両診断caseの **A-R*** のG0〜G3が全てpassでholdout可とし、A-R0は失敗も保存する対照。R*未選定時だけA-R0を経路確認の資格対象とする。R*選定後にそのgateが失敗した場合、A-R0へ切り替えてholdoutを続けない。必須gateのunknown/not_applicableはpassとして扱わない。両caseの目視field `major_anatomy_defect` は明示的なfalseを必須にし、true/null/未記入ならholdout不可とする。G4の軽微な悪化を許容することと、明瞭な人体欠損を許容することを混同しない。

holdoutは同じ2prompt × seed `123/777` の4条件。方式・閾値・K・prototype数・block・LoRA強度を凍結した後に一度だけ実行する。各条件でbase収集→A pilot→A-R0とA-R*を再生する。R*未選定の場合は、G0〜G3を満たすA-R0の経路確認だけを行い、新方式の検証成功とは呼ばない。

新しいstateのRGBに新しいannotationを付け、診断seedの座標を転用しない。方式選定にholdout結果を戻さず、失敗したseedの差し替え・再調整をしない。ラベルが不鮮明で判定できないgateはunknownとし、強制的にpassにしない。今回の上限内では強度2などの追加軸に広げない。

holdoutの「未使用」は当該prompt・asset・強度・生成条件での診断／選定への未使用を指す。freeze時に保存実験のcase/seed一覧を記録し、123/777の既使用が見つかればその条件を `not_applicable: previously_used` とする。代替seedを自動で補充しない。

## 実装範囲とタスク

### Task 1: source固定と分離可能なseed scoring

**Files:** Modify `slider_fuse/anchored_masks.py`、`docs/build_paired_mask_candidates.py`。Create `docs/run_seed_bootstrap_validation.py`、`tests/test_seed_controls.py`。

**Interfaces:** anchored_masks.pyに `build_seed_pools(anchor_context, bootstrap_masks, config) -> dict`、`fit_role_prototypes(features, pools, config) -> dict`、`score_with_seed_controls(features, late_logits, anchor_context, prototype_bank, lock_masks, config) -> dict` を追加する。role順はtarget/protected/background。anchor/lateは `[N,3]`、featuresは `[1,N,C]`、poolはrole別long index、lockはrole別 `[N]` bool。実験のNは4096、bootstrap maskは `[1,64,64]`。小さい合成gridのunit testは許す。

`build_seed_pools` は `full/supported/locks/supported_failures` を返す。full/supportedはrole別pool、locksはK、supported_failuresは空poolのroleと理由を持ち、R0/R1の計算を妨げない。`fit_role_prototypes` はrole別 `centers/prototype_indices/prototype_pool_counts` を返す。`score_with_seed_controls` は旧candidate互換fieldと `prelock_scores/lock_masks/lock_counts/prototype_pool_counts` を返す。prototype比較でbank以外、lock比較でlock以外を変更しない。未clamp scoreはcloneして保持し、後段のin-place clampで証拠を変更しない。

- [x] 保存sourceのhash、raw tensor、旧A/D、annotationを検査し、新しいexperimentと `baseline-recheck.json` を排他的に作る。14/126背景seed、顔9→8と座標(23,15)を再集計する。
- [x] prototypeを同じにして固定だけ外すfixture、同じKでprototypeだけ変えるfixture、入力非変更、空pool、非有限、grid不一致、role衝突、stable tie breakのテストを先に書いてREDを確認する。
- [x] R0〜R4を実装し、prototype hash／clamp前score hash／lock mask hash／poolと固定数／clamp前後の役割を保存する。固定されていないセルを暗黙にseed扱いしない。
- [x] R0を既存実装と全tensor比較する。テスト用例に加え保存2構図×2branchで完全一致させる。R1〜R4は同じ保存入力でoffline評価し、selection.jsonへR*またはnot_selectedと理由を固定する。
- [x] 全tensor一致にはscores・seeds・prototype index・growth_allowedと、bankのtarget/protected/background/real_background/uncertainを含める。旧 `anchored_candidate` を独立oracleとして残し、新実装を呼ぶwrapperに置換して自己比較にしない。空supported poolではR0/R1を継続しR2〜R4だけ不成立となることも検査する。
- [x] `python -m pytest -q -p no:cacheprovider tests/test_seed_controls.py tests/test_paired_mask_attribution.py tests/test_paired_observation.py tests/test_experimental_masks.py --basetemp=test-results/seed-controls-01` を実行し、失敗0を確認する。

### Task 2: base-only初期化と観測付きpilot

**Files:** Modify `docs/run_seed_bootstrap_validation.py`。Create `tests/test_seed_bootstrap_validation.py`。再利用：`docs/run_paired_mix_validation.py` のbundle保持、`docs/run_attention_step_block_sweep.py` のqueue処理、既存collect/select/observed/saveノード。通常autoとsweepの3 forward契約は変更しない。

**Interface:** `build_bootstrap_graph(case_config: dict, *, trial_id: int, prefix: str) -> dict`、`build_pilot_graph(case_config: dict, initial_mask_manifest: Path, *, trial_id: int, prefix: str) -> dict`。case_configはpromptグラフ・seed・assetとrole token・B/D由来を明示し、既存2case固定のhelperにholdoutを無理に渡さない。

case_configは `case_id/prompt_case/seed/base_graph/collector_config/slider_path/asset_sha256/role_positions/background_phrase` を必須にする。diagnosticだけに `reference_A_report/reference_D_report/reference_D_probe` を持たせる。`case_id` は `{prompt_case}__seed_{seed}` とし、4holdoutが2prompt用フォルダーを上書きしない。trial_idはseedと分けた0〜2147483647の整数で、実行root内のcold runごとに予約しmanifestへ保存する。[旧trial検証](../../docs/run_mask_attribution_validation.py):44〜49。

- [x] A graphがstyle付きbaseを収集し、Slider上流適用・ON収集・D mask読込を含まないことをテストする。style LoRAと対象Sliderを区別し、loader名だけで一律拒否しない。
- [x] seed変更がcollector/sampler/ノイズを決める全入力へ伝わり、conditioning・role positions・他条件を変えないことをテストする。old graphやcase_configをin-place変更しない。
- [x] 新規trialでbaseを2 forward収集し、multi_proto_bgのB_T/B_Pを保存する。診断2条件では旧A maskと完全一致、B_Pは旧D保護とも完全一致を確認する。
- [x] I_D/I_Aを別graphで観測付き生成する。初期noise・sigmas・conditioning・assetを共通にし、変えるのは初期maskだけとする。両方のstep3 raw情報を保持する。
- [x] I_Dは旧D trace/tap、I_Aは旧A_off trace/最終latentと照合する。元の観測tapがないAについて、存在しないtap一致を主張せず、observerの前後非改変とtrace一致を証拠にする。
- [x] `python -m pytest -q -p no:cacheprovider tests/test_seed_bootstrap_validation.py tests/test_attention_sweep.py tests/test_paired_observation.py --basetemp=test-results/seed-bootstrap-01` を実行する。

### Task 3: 片側支持を失わない衝突診断

**Files:** Modify `tools/evaluate_paired_mask_attribution.py`、`docs/build_paired_mask_candidates.py`。Create `tests/test_one_sided_collision_review.py`。既存`classify_collisions`とschema 1の意味を変更しない。

**Interfaces:** `classify_collisions_v2(raw_target, protected, base_regions, slider_regions) -> dict[str, list[bool]]`、`validate_release_review(collision_classes, review_manifest, reference_identity) -> list[bool]`。引数maskは長さNのbool列、前者のkeyは上記7分類、後者は長さNの解除L。新規 `collision_schema=2` と `release_review_schema=1` を使う。解除tensor適用は既存 `apply_diagnostic_release` を再利用する。schemaはJSON envelopeに置き、分類bool列のdictへ混在させない。

- [x] 7分類の排他・総和、base-onlyの顔セル、LoRA-onlyの移動先、role_conflict優先、反対側unknownの解除拒否、参照RGB改変、C外の解除拒否をRED→GREENで検査する。
- [x] D/Aの各候補でraw前→保護減算後の欠落と、分類前後のroleを部位別に保存する。公園(x=23,y=15)の分類変化を保存データで再現する。
- [x] 各branchのRGBを別々に確認し、片側支持の反対側が可視の非保護かを記録する。人が確認していないものはhuman_verified=falseを維持する。
- [x] D-R0の確認済みLだけを局所解除候補にする。L=0はnot_applicable。画像生成に進む場合は両差分がLのみであることを監査する。
- [x] `python -m pytest -q -p no:cacheprovider tests/test_one_sided_collision_review.py tests/test_evaluate_paired_mask_attribution.py tests/test_paired_mask_attribution.py --basetemp=test-results/seed-collision-01` を実行する。

### Task 4: 初期化×方式の固定再生と問題別gate

**Files:** Modify `docs/run_seed_bootstrap_validation.py`、`docs/build_paired_mask_candidates.py`、`tools/evaluate_paired_mask_attribution.py`。Tests：`tests/test_seed_bootstrap_validation.py`。

**Interfaces:** runner CLIは `--stage freeze --source-root PATH --output-root NEW_PATH --trial-id INT --server URL` でexperimentを作り、以後 `--experiment-plan PATH --stage offline|bootstrap|pilots|replay|gates|release|holdout|traces` を使う。gatesは `--visual-review PATH` を必須にする。入力case listとselected variantをmanifestで受け、holdoutに固定2case前提や旧D必須参照が混入しないようにする。新rootのscoring結果は既存の `anchored_mask_schema=1` でSamplerへ渡す。[既存保存形式](../../docs/build_paired_mask_candidates.py):99〜118。

- [x] D/A×R0/R*を構築し、B_Pの同一性、旧D targetを含めないこと、state対応、変換前後のtarget unionと未知領域を検査する。annotationを通常候補の入力にしない。
- [x] Aの新RGBでラベルを凍結する。Dの参照hashが一致するときだけ旧labelを再利用する。候補自身の再生trajectoryの途中像へpilot labelを正解として転用しない。
- [x] 元noiseから固定再生し、各条件の全step routing、初回branch予測、mask由来、queue/cache、実測NFEを監査する。raw maskの成功と画像の成功を分けて保存する。
- [x] G0〜G5を独立fieldとしてpass/fail/unknown/not_applicableで記録し、限定holdoutの資格を機械的な必須条件＋記録付き目視から決定する。invalidなG0や明瞭な増殖を見逃して先へ進めない。
- [x] `evaluate_holdout_eligibility(selection: dict, gates: dict) -> dict` をrunnerへ追加する。出力は `eligible/mode/required_variant/reasons`、modeは `selected_method/path_only/not_eligible`。必要な2case×G0〜G3の欠落・unknownとmajor_anatomy_defectの非falseを拒否し、R*選定後の失敗をR0成功で置換しない。
- [x] R*不適格時も失敗条件を残す。R0は予定された対照であると明示し、R*の成功例として置換しない。

### Task 5: 条件付きholdoutと最終監査・報告

**Files:** Create `docs/audit_seed_bootstrap_validation.py`、`docs/build_seed_bootstrap_report.py`、`tests/test_audit_seed_bootstrap_validation.py`。Modify `docs/run_seed_bootstrap_validation.py`、`docs/attention-step-block-sweep.md`。Tests：新auditorテストと `tests/test_seed_bootstrap_validation.py`。

**Interfaces:** auditor `--experiment-plan PATH --output NEW_JSON`、report builder `--experiment-plan PATH --verification PATH --output NEW_HTML`。旧auditorの固定した2case/Dパリティを緩めず、新しい監査入口から既存`verify_edit`／readerを利用する。

- [x] selectionとgateのhashを固定し、未使用seed123/777を両構図に適用する。1つが失敗しても原因・品質結果を残し、比較が成立する他の予定条件は続ける。
- [x] holdoutで方式選択をやり直さないこと、初期mask失敗時にDへ戻らないこと、unknownをpassにしないこと、同一maskの再利用を独立生成と数えないことをテストする。
- [x] traceを既存decoderで復号し、最終RGB一致、全phaseの実測NFE、失敗試行とVAE費用の別計上を確認する。新しいレポートに初期化・方式・顔・人数・保護・背景・体型・未確定衝突を掲載する。
- [x] `python -m pytest -q -p no:cacheprovider --basetemp=test-results/seed-bootstrap-final`、変更Pythonの構文検査、`git diff --check`、全reportリンク、実行後queueを確認する。設定されていないlint/typecheckを成功扱いしない。
- [x] native reviewerが利用できれば独立レビューし、利用不能なら事実を残して別パスの自己レビューを行う。既存成果物は保持し、新しいprogressと本計画から結果へリンクする。

## 実行順・成果物契約

1. **freeze → offline（Task 1）:** sourceは直前attribution root、そのexperimentからpaired Mixと旧A/Dの参照を解決する。新rootを排他的に作成し `experiment.json`（`seed_bootstrap_schema=1`）、`baseline-recheck.json`、各caseのR0〜R4 evidence、`selection.json` を保存。selectionは両case共通の `selected_variant=R1|R2|R3|R4|null`、不適格理由、比較指標、config/annotation/source hashを必須にする。raw・最終maskの失敗をcase/branch/variant単位で残す。
2. **bootstrap → pilots（Task 2）:** 新Bを収集・旧A/Dとの一致確認後、I_D/I_Aのobserved Mixを生成する。`Krea2SliderFusePairedProbeSave` のbranch別RGB、raw tap、input/sigma/prediction hashを保存する。I_DのG0が成立しない間は後続の方式比較を止める。I_Aの参照不一致はA比較を止め、既に成立したDのoffline結果を取り消さない。
3. **pilot annotation → replay（Tasks 3・4）:** Aの新RGBだけを見てannotation revisionを固定してから候補overlayを照合する。候補maskを見てlabelを決めない。Dはreference完全一致時だけ旧labelを再利用する。R*の選び直しは行わず、同じclassifierを新pilotに適用してD/A×R0/R*を再生する。R*がAで成立しない場合は失敗を保存し、予定R0対照は継続する。
4. **traces → gates → release（Tasks 3・4）:** 途中像を含む目視前に利用可能な生成結果をdecodeする。trace復号を最終段階だけに置いてgateの入力を欠かさない。gate JSONはmask定量値と、参照画像hashを持つ目視reviewを合成する。局所解除はD-R0の独立診断で、AやR*の成否だけで中止しない。対象Lがなければ生成なしの理由を保存する。
5. **条件付きholdout → traces → audit/report（Task 5）:** selectionと診断gateのhashを凍結して4caseを一度だけ実行する。各caseはbase収集→A pilot→固定済み方式のscoring→R0/R*再生。Aの新annotationと目視reviewで評価する。証拠待ちは `awaiting_annotation/awaiting_visual_review` として保存し、再開時に生成を再実行しない。holdout不可の場合もauditor/reportを実行し、対象外理由を残して完了する。

全phaseは `case_id/phase/variant/attempt_id` で識別し、queued graph、prompt_id、history、artifact hash、cached nodes、実測NFEを保存する。再開は保存済みidentityとhashが一致する完了runの再利用だけを許す。history不明のjobは照会で解決するまで再投入しない。未完了artifactを完成品と扱わず、失敗試行は新attemptに分ける。trial_idとprefixだけ変えて同一条件を独立成功として水増ししない。[既存queue処理](../../docs/run_attention_step_block_sweep.py):120、[cache/NFE監査](../../docs/audit_mask_attribution_validation.py):102〜127。

Aのgraph検査は実際の推論入力とmask計算の依存を対象にする。experiment内の **比較用** D参照は許すが、A builder・scorerへ渡さない。bootstrapはSlider適用なし、A pilotは通常MixのSlider branchあり、全体ON collectorなしを別々に検査する。実験スクリプトのHTTPクライアントは既存system Pythonを使い、backendは実行時のlive capabilityを確認する。サーバーの再起動や公開node変更は現計画の必須作業ではない。

## 受入テストの固定

以下はこれから追加するテスト名と主要assertion。fixtureは既存テストのCPU tensor・偽graphを使い、実機source比較は別の保存evidenceで行う。REDは期待する未実装／契約違反での失敗を確認し、import環境の失敗を代用しない。

| 所有Task／test名 | 必須assertion・期待結果 |
| --- | --- |
| 1 `test_r0_matches_legacy_candidate_and_bank` | `torch.equal(new['scores'], old['scores'])` と全互換field・bank一致。2case×2branchの保存データでも差分0 |
| 1 `test_lock_only_change_preserves_prelock_scores` | R0/R1/R2でprototype/prelock hash一致、R1のlock総数0、R2のlockはKのみ |
| 1 `test_supported_prototype_change_preserves_locks` | R2/R4のlock hash一致、P_supportedのpoolはEとBの積に等しい |
| 1 `test_empty_supported_pool_does_not_block_full_pool_controls` | `status['R0']==status['R1']=='ok'`、R2/R3/R4は理由付きfailed |
| 2 `test_bootstrap_and_pilot_have_no_d_mask_dependency` | A graphのD loader/ON collector数0、bootstrapのSlider適用数0、pilotのmix_scopeはtarget_mask |
| 2 `test_holdout_seed_is_applied_without_mutating_prompt_template` | collector/sampler seedがcase.seed、元graph不変、4つのcase_id/prefixが一意 |
| 3 `test_one_sided_collision_requires_explicit_opposite_review` | 片側対象＋unknownでは `not any(L)`、明示background確認セルだけTrue、role_conflictは常にFalse |
| 3 `test_v2_collision_partition_and_v1_compatibility` | 7分類のセルごとの和がC、schema 1の既存テストは出力不変 |
| 4 `test_unknown_gate_and_failed_selected_variant_block_holdout` | `eligible is False`。R*失敗＋R0成功でもFalse。major_anatomy_defectがtrue/nullでもFalse。R*未選定かつR0全pass・欠損falseのみmode=path_only |
| 5 `test_resume_reuses_verified_run_and_rejects_changed_identity` | 完了run再開でsubmit回数0、hash不一致は例外、失敗試行NFEも別fieldに保持 |
| 5 `test_audit_accepts_skipped_holdout_but_rejects_missing_evidence` | 正当な対象外理由付きreportは作成可、必須source/parity証拠欠如は監査失敗 |

Task 5の追加対象テストは `python -m pytest -q -p no:cacheprovider tests/test_seed_bootstrap_validation.py tests/test_audit_seed_bootstrap_validation.py --basetemp=test-results/seed-bootstrap-audit-01`。上記の全受入条件と最終CPU回帰・artifact監査を満たせば実装検証作業は完了。画質gate不合格やholdout対象外を成功へ変換することは完了条件に含めない。

## Review Focus

- prototypeを変えたときにKも変わり、因果比較が崩れる：Task 1でprototype／score／lockのhash同一性を検査。
- R0のrefactorで既存seed・浮動小数点計算順が変わる：Task 1で保存2構図の全tensor完全一致を検査。
- A経路のgraphやmask計算にD／ON由来が残る：Task 2・4で独立した入力由来とdependency graphを検査。
- 対象が動いたセルをunknown＝非保護として解除する：Task 3で明示的な反対branch確認とconflict拒否を検査。
- holdoutの失敗を見て方式・seed・閾値を差し替える：Task 5で事前selectionとcase listのhashを監査。

## 生成量・NFE上限

| 段階 | 最大完成画像 | 追加diffusion NFE |
| --- | ---: | ---: |
| 保存tensorでR0〜R4比較・衝突再集計 | 0 | 0 |
| 診断2構図のbase初期mask収集 | 0 | 2×2 = 4 |
| I_D/I_Aの観測付きpilot、各2構図 | 4 | 4×16 = 64 |
| D/A×R0/R*の固定再生、各2構図 | 8 | 8×16 = 128 |
| 基本比較小計 | 12 | 196 |
| 条件付きD-R0局所解除、最大2構図 | 2 | 32 |
| 条件付きholdout：2構図×2seed、各base収集＋pilot＋R0/R* | 12 | 4×(2+16+16+16) = 200 |
| 全段階の上限 | 26 | 428 |

R*未選定なら該当する再生を実行せず、画像数・NFEを減らす。重複maskを流用する場合も独立生成数を増やさない。decodeは0 diffusion NFEだがVAEの時間は別記する。再試行・再収集が必要になれば、成功分上限へ紛れ込ませず、原因と追加費用を進捗記録に記載する。

## 停止・採用・残るリスク

- source改変、R0再現失敗、asset差、pilot対照不一致、入力非改変違反は該当比較を止めて原因を解決する。部分的な成功でG0通過としない。
- R1〜R4が全て顔を失う場合は「seed固定の緩和だけでは解けない」と結論し、閾値変更や別head探索をその場で追加しない。base初期化の独立した問いはR0対照で確認できる。
- E_supportedは過去の形状に依存し、変形先の顔・腕を落とす可能性がある。各branchの部位欠落とpool除外の座標を保存する。失敗時に最大成分や強制穴埋めを黙って追加しない。
- 局所解除で欠落が減っても保護人物の外見が変われば、対策としては不合格。解除不要を証明したことと、確認可能なLがなかったことを区別する。
- 顔／人数のholdout成功だけでは本番採用としない。G3/G4の悪化・unknown、agentラベル、4holdoutという範囲を残す。既定値変更は別判断とする。
- 拡大した境界labelが必要な場合は新revisionを作り、全候補を同じrevisionで再評価する。追加ラベルを使って既に見たholdoutを調整用へ転用しない。

## 成果物・自己点検

- [x] experiment、baseline-recheck、R0〜R4のpool/prototype/lock/score、selectionとhash。
- [x] 新規B mask、I_D/I_A観測、D/A×R0/R*のmask・画像・全trace・実行履歴。
- [x] collision schema 2、セル別release review、条件付き局所解除またはnot_applicableの理由。
- [x] gate別判定、凍結holdout条件、実行／対象外／失敗／unknownの記録。
- [x] CPUログ・数値監査・目視所見・HTMLレポート・実行progress・結果へのリンク。

計画作成時の自己点検：4提案をTasks 1〜5へ対応付け、prototypeとlockの比較を独立させた。2 forwardには既存ExperimentalMaskCollectを使い、3 forward専用sweepの契約を変更しない。A経路のD依存、片側支持の不確実性、限定holdoutと総合採用の差、最大26画像・428 NFEを明示した。
