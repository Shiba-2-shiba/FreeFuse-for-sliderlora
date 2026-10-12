# Paired Mask Attribution Implementation Plan

> **For agentic workers:** 実行時は `superpowers:executing-plans` を使用して以下のtaskを順に実装・検証する。独立した作業を委譲できる場合は `superpowers:subagent-driven-development` を利用できるが、利用不能なrole/modelの再試行を前提にしない。AGENTS.mdの権限・既存変更保全・検証契約を優先する。

**Goal:** 新方式単独の効果、Dへの領域追加の効果、固定保護による切り欠きを分離し、顔・全身・保護人物・境界について説明可能な実装と検証結果を得る。

**Architecture:** 保存済みの同一Mix入力の観測から候補を再構成し、旧Dとの和集合を独立因子として固定mask再生する。RGBから独立に記録した部位ラベルでraw分類・編集支持領域・保護衝突を別々に測り、保護の局所解除は診断対照として追加する。既存Sampler、retained-mask reader、数値監査を再利用する。

**Tech Stack:** Python 3.10以上、既存のPyTorch・safetensors・Pillow・pytest、起動中のComfyUI REST。新しい依存・外部segmenterは追加しない。

**Spec:** この会話で指定された提案1〜3、および[現行結果](../../docs/attention-step-block-sweep.md)、[前段計画](2026-10-12-prediction-mix-mask-validation.md)。本書は2026-10-12に作成した事前計画。実施状況は下記の実行結果を参照。

**実行結果（2026-10-12）:** Tasks 1〜5を実装・検証済み。[実行記録](2026-10-12-paired-mask-attribution-validation.progress.md)と[比較レポート](../../docs/attention-step-block-sweep-results/20261011T235655Z-mask-attribution/report.html)を参照。実測10画像・160 NFE。条件付きS0-releaseは確認対象衝突セル0のため対象外。品質の採用gateは不合格で、下記は事前に固定した計画を保存したもの。

## 要求と今回の到達点

| 要求 | 実装するもの | 検証で答える問い |
| --- | --- | --- |
| 1. 新方式単独とDへの追加を分ける | seed修正／seed修正＋伝播、D unionなし／ありの2×2候補 | 顔の回復・全身編集は新候補自体で成立するか。Dの領域が必要か |
| 2. 顔回復と領域追加を別々に測る | 部位ラベル、raw／実効maskの指標、D差分の部位内訳 | 増えたセルは顔・手足・旧身体の消去域・保護人物・背景のどこか |
| 3. 固定保護との衝突を確認する | 衝突の分類・可視化と、確認領域だけを解除した固定再生 | 切り欠きは保護優先で説明できるか。解除すると男性へ影響するか |

完了条件は、上の3問それぞれに証拠付きの結論または判定不能理由があり、実装の検査と全予定条件の状態が記録されること。画質改善が得られなかった場合も検証結果として完了できる。既定方式への採用・自動保護推定・動的mask・holdoutへの展開は今回の完了条件に含めない。

## Global Constraints

- 公園seed `444444` と女性手前の重なりseed `42` を使用する。prompt、モデル・style・Sliderとそのhash、Slider `+4`、`1024×1024`、`Euler/simple`、`8 steps`、`CFG 1`、全radius `0` を固定する。[現行統制](../../docs/attention-step-block-sweep.md)
- 候補の主比較は同じD pilotの `step3/block18`。anchorは同stepの `block8/10 × base/slider` の4観測。prototype数・閾値・温度・伝播1回・全48head・strength `0.5` は保存configと照合し、今回最適化しない。[候補生成コード](../../docs/build_paired_mask_candidates.py) の34〜59行
- `block23` は保存データによる副次解析・既知の失敗対照として残す。主比較の結果を見てblockを差し替えない。[既存品質判定](../../docs/attention-step-block-sweep-results/20261011T185947Z-paired-mix/assistant-visual-observations.json)
- 新しいUTC run IDの `docs/attention-step-block-sweep-results/<UTC>-mask-attribution/` を出力先とする。既存の結果・検証JSON・pytest一時領域を上書きしない。
- `seed` は生成乱数、文脈seedは分類の確信セルとして区別する。生成時は新しい `trial_id` を全比較条件で共有し、seedを変えずcache再利用を識別する。[Sampler契約](../../nodes.py) の182〜217行
- ラベルは候補maskから作らない。agentがRGBから作成したラベルは `human_verified=false` として保存する。人の承認がなくても記述的な検証は進められるが、人が確認した正解とは記載しない。[既存評価契約](../../tools/evaluate_attention_masks.py) の1〜44行
- 保護maskの一致・予測選択の一致・CPUテスト成功を、完成画像の人物保持や人体精度の合格に読み替えない。
- 編集用MODELにはSliderを事前適用しない。元noiseから全8stepを再生し、途中latentを開始点にしない。既存node ID・公開入力・通常autoの既定値は維持する。
- 既存dirty変更を保全する。2026-10-12確認時点で `__init__.py`、`slider_fuse/{diagnostics,experimental_collection,experimental_masks,native_pair}.py`、一部docs・testsに変更がある。作業開始時に再確認し、他の変更を巻き戻さない。

## 現状の根拠と残る交絡

| 証拠 | 含意・本計画での扱い |
| --- | --- |
| [候補生成](../../docs/build_paired_mask_candidates.py):74〜77 は `old_target ∪ base_combined ∪ slider_combined` からD保護を引く | 現行4画像は新方式単独の検証ではない。D unionを因子にする |
| [seed修正](../../slider_fuse/anchored_masks.py):18〜29 はseed poolとprototypeを作り直す | `seed_only` はseed取得元＋prototype再推定の処理全体。seed固定だけの単独効果と呼ばない |
| [現行分類・伝播](../../slider_fuse/experimental_masks.py):147〜172、203〜255 はseedを固定する | 輪郭伝播のみで誤seedを解消できると仮定しない |
| 前ターンの保存PNG再集計では、公園LoRA側block18の旧70セルが baseline 17、seed_only 69、shape_only 15、combined 69。Dと生成候補はすべて70 | 再集計をTask 1でJSON化する。旧70セルは診断値であり、独立した顔精度ではない。[保存比較](../../docs/attention-step-block-sweep-results/20261011T185947Z-paired-mix/park/ablation-b18.png) |
| [同入力交差比較](../../docs/attention-step-block-sweep-results/20261011T185947Z-paired-mix/woman_front_strong_overlap/causal-step3-b26.png) と完成像では全体ONだけ三人化 | 全体ONの新規収集を今回のmask源にしない。旧Dに含まれる由来は対照として明記する |
| [前段監査](../../docs/audit_paired_mix_validation.py):65〜93 はDの全trace一致と各候補の選択を確認 | 新しい対照の再現性とmask差による結果差を分けて監査できる |
| [既存評価器](../../tools/evaluate_attention_masks.py):70〜121 はfull/torso/legs/protected/backgroundを別評価する | 厳密なPNG読込とnull規約を再利用し、顔・左右手足・状態対応の専用評価を追加する |

今回の候補はD軌道で観測した情報による「pilot→固定再生」である。候補を変えると再生中のlatentも変わるため、候補自身の全stepで観測したmaskと同一だとは主張しない。

## 比較条件と因子

`D_T/D_P` は検証済みDの対象／保護、`T[v,b]` はvariant `v`・branch `b` の対象maskとする。

`R_v = T[v,base] ∪ T[v,slider]`

| ID | 分類variant | D union | 最終target | protected |
| --- | --- | --- | --- | --- |
| D | 保存D | 対照 | `D_T` | `D_P` |
| S0 | seed_only | なし | `R_seed − D_P` | `D_P` |
| S1 | seed_only | あり | `(D_T ∪ R_seed) − D_P` | `D_P` |
| C0 | combined | なし | `R_combined − D_P` | `D_P` |
| C1 | combined | あり | `(D_T ∪ R_combined) − D_P` | `D_P` |

- S0対S1、C0対C1で旧Dの寄与、S0対C0、S1対C1で伝播追加の寄与を比較する。今回の設計だけで全head平均と個別headの優劣は判断しない。
- rawのbaseline／seed_only／shape_only／combinedはbase・sliderそれぞれ保存する。画像再生は上表に限定し、shape_onlyはofflineの機序対照とする。
- 「D unionなし」でも同入力のbase/slider対象の和集合は使用する。branch間の和集合まで除いた方式ではないことを全manifestと表に記す。
- すべて二値64×64。`background = 1 − target − protected`。空target/protected、非有限、grid不一致は理由付きfailedとし、別maskを代入しない。
- D比のadded／removed、rawのbranch一致／不一致、保護を引く前のcollision、保護により削られたセルを別々に保存する。
- C1は旧block18候補とのtensor完全一致をoffline回帰とする。不一致なら、先に原因を記録して比較条件を修正し、生成へ進まない。

## 部位ラベルと評価契約

1. 各構図のD pilot `step3` について、base予測RGBとslider予測RGBを別々にラベル化する。RGBとinput/sigma/conditioningのhash、branch、step、grid、作成者種別を `annotations/manifest.json` に保持する。同じxでも両予測の人物位置は同一とは限らない。
2. 各RGBに `target_face`、`target_torso`、`target_arm_left/right`、`target_leg_left/right`、`target_foot_left/right`、`protected_face`、`protected_body`、`known_background` を付ける。左右は人物本人の左右。判別不能・隠れた部位はnull理由を残し、推定して埋めない。明瞭な内部セルだけを数値対象にする。
3. 部位間は非重複。任意の `target_full` は全可視対象を独立に描ける場合だけ供給し、各部分の包含を検査する。髪・輪郭など部位外の対象もあるため、fullを部分ラベルの和から推定しない。unknownは補集合から背景へ変換しない。
4. 1024画像を64グリッドに重ね、各16×16画素区画が明瞭に同じ領域の内部にある場合だけラベルを付ける。境界をまたぐセル、ぼけで判別不能なセル、両branchで人物役割が変わるセルは別表示する。保存する評価maskはnative gridで、実行時に自動resizeしない。
5. ラベルを候補名・成績を見る前に凍結する。修正する場合は新revisionを作り、変更理由と全候補の再評価を記録する。既存の70セル診断は別欄に残す。

raw分類は同じbranchのRGBラベルに対して部位recall／欠落セル、保護への侵入、背景への漏れを測る。partialラベルから全体precision・IoUを算出しない。fullラベルがある場合のみ既存定義のprecision・IoUも表示する。[既存の分母とnull規約](../../tools/evaluate_attention_masks.py):79〜101

編集maskは変更前後の占有範囲を含むため、baseとsliderの対象支持を別々に測る。片方だけに存在する旧身体・変形先の領域を、もう片方の背景という理由だけで漏れ扱いしない。両方でbackgroundと確認できる共通背景への侵入を別指標にする。双方で対象と保護が競合するセルは `role_disagreement` として独立計上する。

各候補で `added_to_D`／`removed_from_D` を上記部位・保護・共通背景・role_disagreement・unlabelledへ配分し、セル数と分母を併記する。unknown率を必ず表示し、未評価領域の多い候補を高精度と判定しない。完成画像の顔・全身・男性・境界は、固定ラベルの数字とは別の目視判定にする。

## 保護衝突の診断

主比較後、事前固定したS0のraw `R_seed` と `D_P` の衝突 `C = R_seed ∩ D_P` を解析する。成績を見て別variantを選ばない。

- `confirmed_target`：baseとsliderの両方で対象内部と確認できるセル。
- `confirmed_protected`：いずれかで保護人物内部と確認でき、反対側で対象との役割競合がないセル。
- `role_disagreement`：片方の対象と他方の保護が衝突するセル。
- `common_background`：両方で背景と確認できるセル。
- `unresolved`：残り。片方だけの対象支持や不鮮明な接触部もここに残す。

分類順はrole_disagreement→confirmed_target→confirmed_protected→common_background→unresolvedとし、5分類は互いに素で和がCに等しいことを検査する。各成分を両RGB・既存D完成像・S0完成像に重ね、顔・腕・胴・脚・靴のどこが削られたか記録する。

診断解除領域は `L = C ∩ confirmed_target` に限定する。`P_release = D_P − L`、`T_release = R_seed − P_release` をS0-releaseとして同じseedから再生する。S0との違いは保護方針だけに限定し、`T_release − T_S0 = L`、`D_P − P_release = L` を監査する。Lが空ならnot_applicableとし、空実験のために別のセルを解除しない。

これはRGB確認ラベルを使用した診断対照であり、自動方式の採用候補ではない。解除して改善した場合は「固定保護が原因の一部」と判断する。改善しない場合は候補自身の軌道・旧領域・混合境界に原因が残る。role_disagreementを解除しないため、未解決の遮蔽問題が残る可能性も結論に含める。

## Review Focus

- 同じ64×64でも別seed・別step・別branchのラベルを誤用しない：Task 3で参照hash／状態の不一致を拒否する。
- unionなしなのに旧Dセルが紛れ込む：Task 2で候補が旧Dの偽陽性を削れるfixtureを置く。
- 非ラベル領域や空の顔ラベルが完璧な精度になる：Task 3でnull理由とunknown数を検査する。
- 接触部の役割不一致を女性と決め打ちして保護解除する：Task 4で不一致・保護・unknownセルを解除しない検査を置く。
- cache出力や別maskのhashを新規生成・同一対照と扱う：Task 5でqueued graph、trial、partition、cache履歴、NFEを照合する。

## 実装タスク

### Task 1: 証拠の固定と再集計

**Files:** Create `docs/run_mask_attribution_validation.py`、`tests/test_mask_attribution_validation.py`。参照：`docs/audit_attention_ablation_live.py:48/68`、`slider_fuse/paired_observation.py:27/45`。

**Interface:** `freeze_experiment(source_root: Path, output_root: Path, trial_id: int) -> dict`。既存paired-mix rootを入力し、`experiment.json` と `baseline-evidence.json` を排他的に作成する。後続taskはこれらのhashで入力を特定する。

- [ ] 既存rootへの出力、改変済みtensor、missing tap、非対応config、重複run IDを拒否するテストを先に作る。
- [ ] source root `20261011T185947Z-paired-mix` のtap／D／旧C1／RGBのmanifestを検証する。前ターンの70セル再集計、D比added/removed、候補hashを再計算してJSON化する。
- [ ] 2構図、比較ID、annotation契約、コードとassetのhash、全固定値、planned NFE、trial IDをexperiment.jsonへ固定する。sourceへのリンクとhashを保持し、巨大tensorを不要に複製しない。
- [ ] `python -m pytest -q tests/test_mask_attribution_validation.py` が成功すること。offline処理のdiffusion NFEは0。

### Task 2: 2×2候補と由来の保存

**Files:** Modify `slider_fuse/anchored_masks.py`、`docs/build_paired_mask_candidates.py`。Create `tests/test_paired_mask_attribution.py`。既存`tests/test_paired_observation.py:56/83`も回帰対象。

**Interfaces:** `compose_paired_masks(base_target, slider_target, pilot_target, protected, *, include_pilot: bool) -> dict[str, torch.Tensor]` をanchored_masks.pyに追加する。入力は同gridの二値 `[1,H,W]` tensor。戻り値は `target/protected/background/raw_target/collision/added_to_pilot/removed_from_pilot`。

- [ ] 旧Dの孤立した偽陽性セルがS0/C0では消え、S1/C1では残ること、保護重複0、partition全域1、raw collision保存、入力tensor非変更、空対象拒否をテストする。
- [ ] builderに `--experiment-plan PATH --output-root PATH` を追加する。既存rootのみの呼出しを維持し、新モードはfrozen sourceを読み、同じseed_only/combinedからS0/S1/C0/C1を生成する。
- [ ] raw4variant×2branch×2block×2構図を保存する。各候補は既存reader互換の`anchored_mask_schema=1`、最終3mask tensor、PNG、source/variant/union/protection/全中間maskのhashを持つ。追加根拠tensorは別ファイルで保存する。
- [ ] C1の最終maskが旧block18候補に完全一致することと、全候補が既存`read_retained_masks`で読めることを検証する。
- [ ] `python -m pytest -q tests/test_paired_mask_attribution.py tests/test_paired_observation.py tests/test_experimental_masks.py` が成功すること。

### Task 3: 状態に対応する部位評価

**Files:** Create `tools/evaluate_paired_mask_attribution.py`、`tests/test_evaluate_paired_mask_attribution.py`。既存`tools/audit_mask_support.py:36/58`と`tools/evaluate_attention_masks.py:79/103`を再利用し、既存CLI/schema 1は変更しない。

**Interfaces:** `evaluate_attribution(experiment_path: Path, annotation_manifest: Path, candidate_index: Path) -> dict`。CLIは `--experiment-plan --annotations --candidates --output`。出力は `mask_attribution_schema=1`、reference/annotation/candidateのhash、部位指標、added/removed内訳、unknown率、注意点。

- [ ] 参照hash不一致、grid不一致、非二値PNG、部位重複、full外のpart、empty/missingラベル、片方のbranchだけの対象、role_disagreementのfixtureを作り、拒否／null／分母を検査する。Torch・ComfyUIをimportしないことも検査する。
- [ ] 本書のラベル契約に従いRGBへグリッドを表示する確認用画像を出力し、agentまたは人が記録したセル一覧からPNGを作る。作成者種別とhuman_verifiedを正直に保持する。
- [ ] raw分類と編集支持を別reportにし、顔recallだけによるランキングを作らない。対象部位ごとの欠落、保護侵入、共通背景漏れ、評価不能数を全候補に同じ定義で出す。
- [ ] S0/S1とC0/C1、S0/C0とS1/C1の差分を同じannotation revisionで比較する。pixel総数の増加を顔改善として計上しない。
- [ ] `python -m pytest -q tests/test_evaluate_paired_mask_attribution.py tests/test_evaluate_attention_masks.py tests/test_mask_support_audit.py` が成功すること。

### Task 4: 衝突分類と診断解除

**Files:** Modify `tools/evaluate_paired_mask_attribution.py`、`slider_fuse/anchored_masks.py`、`docs/build_paired_mask_candidates.py`。Tests：Task 2・3の新規testファイル。

**Interfaces:** 評価器に `classify_collisions(raw_target: list[bool], protected: list[bool], base_regions: dict, slider_regions: dict) -> dict[str, list[bool]]`、anchored_masks.pyに `apply_diagnostic_release(raw_target, protected, release_mask) -> dict[str, torch.Tensor]` を追加する。分類器は5分類＋release PNGを生成し、builderはそのhash付きmanifestを読む。

- [ ] 5分類の排他性と総和C、解除LがCの部分集合、保護・role_disagreement・unresolvedを解除しないこと、非衝突セルを含むLの拒否をテストする。
- [ ] S0に対する衝突一覧と、両branchのRGB・D/S0完成像へのoverlayを出す。後者の完成像が未生成ならpendingとして残し、Task 5で完成させる。
- [ ] Lが非空の場合だけS0-release候補を作成する。`T_release − T_S0 = L`、`D_P − P_release = L`、他セル不変を保存tensorから再構成できるようにする。
- [ ] agentラベルだけの場合も診断生成は可能だが、結果に`annotation_dependent_diagnostic`と`human_verified=false`を保持する。自動的な保護精度の証拠にしない。
- [ ] Task 2・3のテストを再実行して成功すること。

### Task 5: 固定再生・監査・比較レポート

**Files:** Modify `docs/run_mask_attribution_validation.py`。Create `docs/audit_mask_attribution_validation.py`、`docs/build_mask_attribution_report.py`。Tests：`tests/test_mask_attribution_validation.py`。既存の`run_stage/retain_edits`、`verify_edit`、`checked_diagnostic`を利用し、旧runner/auditorを弱めない。[再生の既存実装](../../docs/run_paired_mix_validation.py):35〜99

**Interfaces:** runnerに `--experiment-plan PATH --stage control|factorial|release`、監査器に `--experiment-plan PATH --output PATH`、レポートに `--experiment-plan PATH --verification PATH --output PATH`。各出力は既存ファイルを拒否し、新しいrevisionまたはattemptへ保存する。

- [ ] dry-run graphのテストでモデル・positive・seed・8steps・CFG1・全radius0・manual mask・trial ID・候補manifestの対応を検査する。failed候補やcandidate IDの重複を拒否する。
- [ ] 現サーバーのobject_info／queue／起動条件を確認する。CPU側変更だけで既存nodeを使える場合は再起動しない。読込中の実装と必要機能が異なる場合に限り、変更を完成・CPU検証してから同じ起動条件で再読込する。
- [ ] 新鮮なDを2構図で再生する。全8stepの入力・base/slider/混合予測・最終latentが旧Dと一致することを確認する。trial metadataの差は許し、数値差は許さない。不一致なら候補生成画像の比較を止め、環境／asset／実装差を解決する。
- [ ] S0/S1/C0/C1を各構図で生成する。候補の自身の軌道の違いは結果として保持する。各runのinside=Slider/outside=base、partition、初回base/slider予測一致、hook除去、graph provenanceを監査する。
- [ ] 条件を満たすS0-releaseを生成する。S0とのraw target一致、解除前後の保護差L、初期noise・条件一致を監査する。
- [ ] 保存済みtraceを既存decoderで正しく復号し、step3の生成途中像も併記する。必要ならRGB確認用に使用するが、D pilotの固定ラベルを別軌道の正解に流用しない。VAE費用はdiffusion NFEとは別に記録する。
- [ ] レポートはraw4variant、実効mask、D差分、部位表、衝突overlay、完成画像全身／顔／腕／足元／男性cropを並べる。3問の結論、failed/skipped/not_applicable、数値監査、目視、ラベル検証状態を別欄にする。
- [ ] `python -m pytest -q` を新しい専用basetempで実行する。プロジェクトに設定されたlint/typecheckがあれば実行し、未設定のものを成功扱いしない。変更Pythonには構文検査も行う。画像品質はこのCPU回帰とは別判定にする。

## NFE・実行量

| 段階 | 画像数上限 | 追加diffusion NFE上限 |
| --- | ---: | ---: |
| 保存観測からの再計算・ラベル・評価 | 0 | 0 |
| Dの新鮮な対照：2構図 | 2 | 32 |
| 2×2候補：4条件×2構図 | 8 | 128 |
| S0-release：最大1条件×2構図 | 2 | 32 |
| 合計 | 12 | 192 |

通常の部分混合は1画像8step×2branch=16 NFE。[前段実測](../../docs/attention-step-block-sweep-results/20261011T185947Z-paired-mix/verification.json)。予算は成功時の上限で、失敗試行・復旧・追加再収集は別計上する。maskが完全同一なら同じtrialの検証済み画像へ参照を張り、重複生成を省けるが、独立した生成成功数には数えない。cache hitは履歴から識別し実測NFEを記録する。保存tapを再利用できない場合はここへ混ぜず、再収集の条件・費用を先に計画へ追記する。

## 合否基準と停止条件

| 判定 | 必須条件 |
| --- | --- |
| 実装・監査合格 | 上記CPU検査成功、出力の由来とhash照合、入力非変更、全partition排他、全step選択一致、D数値再現、failed条件の理由保存 |
| 新方式単独の成立 | S0またはC0が両構図で二人配置・対象の顔・全身を維持し、余分な顔／旧身体残像／明瞭な切り欠きを生じない。部位ごとの欠落と侵入をDおよびunionあり対照と比較して提示する |
| 明確な改善候補 | 事前確認した欠損の少なくとも1つが画像と対応部位の数値の両方で減り、両構図で顔・各可視手足の欠落セル、確認済み保護内部侵入、共通背景侵入が対照より増えない。欠けた評価部位は判定不能として残し、総合合格にしない |
| 保護衝突の寄与 | S0-releaseの変更がLだけで、対応する切り欠きが改善するかを記録。改善しても男性の顔・体格・境界が悪化すれば解決策としては不合格 |
| 人物保持 | 男性の顔・髪・眼鏡等の外見、身長・主要体格、衣服境界をDとOFF参考画像に対して確認。接触上の軽微な手・前腕移動と、融合・別顔・主要体格変化を分けて記録 |
| 判定不能 | 小さすぎる／ぼけた部位、役割不一致、未確認label、対照の数値不一致は理由付きunknown。閾値を後から緩めて合格にしない |

明瞭な人数増加・頭部増殖・人体の穴は当該候補を不合格として保存する。比較が成立する他条件は続行する。D再現失敗・成果物改変・参照状態不一致はその比較群を停止し、原因を解決する。安全に計算できる失敗候補の否定的結果も省略しない。

agentラベルによる数値は暫定の記述的根拠であり、画質の人間承認や汎化の合格とは区別する。今回は2構図×各1seedの機序検証なので、結果が良くても既定値は変更しない。

## リスクと対策

- **Dの継承が改善を隠す／作る:** 2×2比較とadded/removedの部位内訳で識別する。C1は旧候補に完全一致させる。
- **branch間で形状が変わる:** raw分類は各branchのRGB、編集maskは両方の支持で評価し、役割競合は独立表示する。
- **前半anchorの共通誤り:** 4観測の一致を正解と扱わず、RGBラベル上の背景侵入・保護侵入を測る。
- **pilotから候補軌道がずれる:** 固定再生の途中像と完成像を確認し、pilot上の良いmaskが再生でも良いという推論を検査する。動的化へ自動移行しない。
- **診断用ラベルが方式へ混入する:** S0-releaseは明示的な診断IDに限定する。通常候補の生成にannotationを入力しないことをテストする。
- **局所解除でも外側が変わる:** 数値の直接選択監査に加えて男性・全身・接触境界を画像評価する。

## 成果物と実行時チェックリスト

- [ ] `experiment.json`、source hash一覧、`baseline-evidence.json`、annotation manifestとgrid PNG。
- [ ] raw variant群、D/S0/S1/C0/C1候補、collision分類、条件付きS0-release、各tensor/PNG/manifest。
- [ ] 部位指標JSON、union／伝播の差分表、D差分の部位内訳、unknown率。
- [ ] queued API・history・cache情報・実測NFE・各画像診断・CPU検証ログ。
- [ ] 数値検証JSON、目視所見JSON、HTMLレポート、提案1〜3それぞれの結論。
- [ ] 実装時に進捗ファイルを作り、本計画から結果へリンクする。完了・不合格・判定不能・条件付き未実施を区別する。

本書作成時の自己点検：提案1〜3を比較条件／評価契約／衝突診断／Tasks 1〜5へ対応付けた。主比較、解除条件、最大12画像・192 NFE、source再現失敗時の停止、ラベルの限界を事前に定義した。実装と実機検証の実施結果はまだ含まない。
