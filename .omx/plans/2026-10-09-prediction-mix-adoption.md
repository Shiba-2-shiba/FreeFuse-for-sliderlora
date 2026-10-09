# Prediction MixをFreeFuseの推奨LoRA適用方式にする実装計画

> **For agentic workers:** 本計画は実装前の設計・作業計画です。実装開始時は `superpowers:executing-plans`、または選択された場合に `superpowers:subagent-driven-development` を読み、以下のチェック項目を順に実施してください。AGENTS.mdの役割・権限・検証ルールを優先します。

**Goal:** 検証済みのPrediction Mixを通常利用の推奨経路にし、既存の自動マスク収集と接続して、手動・自動の両方で利用できるようにする。

**Architecture:** 既存の`Krea2SliderFusePredictionMixSampler`を拡張する。自動モードではSliderなしのPhase 1でマスクを決め、同じ初期noise・latent・全sigma列からPrediction MixのPhase 2を開始する。旧hookノードと既存の評価workflowは保持し、新しい推奨workflowから新経路を使う。

**Tech Stack:** Python、PyTorch、safetensors、ComfyUI V3、既存のKrea2実装。新しい依存ライブラリは追加しない。

**Spec:** 本書「2. 採用設計」を設計仕様とする。採用の根拠は[これまでの検討まとめ](../../docs/slider-attenuation-investigation.md)。

**作成日:** 2026-10-09

**最新の実行指示（同日）:** ユーザーが起動中ComfyUIでのエージェントによる実機検証を依頼し、U2の旧commit回帰を不要と指定した。U2は完了条件から除外する。T1〜T7の実装、U1/U3の実機検証は完了。U4の永続保存・再実行、U5の指定比較表、U6の通常KSamplerを含む復帰確認をエージェントが実行する。以下の「ユーザーが手動実施」は当初の担当割当の履歴で、この追加指示が優先する。U5の完了は評価の実施・採用判定・失敗記録までであり、未達のauto画質を合格へ書き換えない。

**調査時点のコード:** `dev` / `fb3cd592e7d115809ab125ef524612b7bfc62e19`

**現在の状態:** T1〜T7およびU1/U3/U4/U6完了、U2対象外。U5は48条件の評価・判定を完了したが、auto画質は未達例が残る。ユーザーのチャット終了・引き継ぎ希望により評価を区切った。最新証拠と次の課題は対応するprogress、`docs/validation.md`、`.omx/notepad.md`を参照。画質の原因切り分け・修正は次の指示を受けて別作業とする。

**この依頼の成果物:** 実装計画書。生成処理の変更・commit・push・公開は、この計画作成には含めない。

## 1. 要件と現在地

### 1.1 ユーザーの決定と、計画上の提案

- **決定済み:** `krea2_slider_mix_half.json`で有効だったPrediction Mixを、FreeFuseのLoRA適用に採用する。
- **決定済み:** 実装に先立って計画を作成する。
- **決定済み:** 実機確認はユーザーが手動で行う。GPU生成・実ComfyUIでのUI操作・native検証の実行・画質評価を、エージェント側の実行作業には含めない。
- **計画上の提案:** 既存の自動マスク機能も新方式へ接続する。手動版を先に互換検証し、自動版の実機評価を後段の独立した判定とする。
- **計画上の提案:** 「採用」は、既存Prediction Mixノードの機能拡張と、新しい推奨workflow・READMEの導線変更で行う。旧Samplerの保存済みworkflowを読み込んだだけで生成方式が変わる移行は行わない。

実装担当はコード、CPUテスト、静的検証、実機確認用workflow・ツール・手順を整備する。ユーザーの実機結果を待たずに、これらを満たした時点で「実装完了」として報告できる。「実機評価完了／運用上の採用判定」は別の状態として管理する。

### 1.2 根拠となる現行コード

以下の行番号は調査時点のもの。実装中に移動した場合は関数名も使って追跡する。

| ID | 現状の事実・制約 | 根拠 |
|---|---|---|
| E1 | 旧Samplerはauto/manualと対象文章倍率を持ち、診断出力は文字列 | [nodes.py:66](../../nodes.py#L66) |
| E2 | Prediction Mixは別ノード。出力はlatent/mask_bank/専用診断型、診断レベルの既定はaudit | [nodes.py:170](../../nodes.py#L170)、[schemaテスト](../../tests/test_prediction_mix_nodes.py) |
| E3 | Prediction Mix本体は現在manual固定。modeによってbase/slider/bothを選ぶ | [native_pair.py:205](../../slider_fuse/native_pair.py#L205) |
| E4 | 同じcoreのモデルをcleanup→load→pre_runで切り替え、current_patcherを検査する | [native_pair.py:39](../../slider_fuse/native_pair.py#L39) |
| E5 | 部分混合は同じ入力・sigmaから二経路を評価し、二値maskで予測を選ぶ | [native_pair.py:145](../../slider_fuse/native_pair.py#L145)、[prediction_mixing.py:15](../../slider_fuse/prediction_mixing.py#L15) |
| E6 | 旧autoは短いPhase 1の後、元のnoise/latent/full sigmasからPhase 2を開始する | [sampling.py:96](../../slider_fuse/sampling.py#L96)、[test_sampling.py:14](../../tests/test_sampling.py#L14) |
| E7 | Collectorは一時的なmodule hookを使い、対象sigma・blockの情報だけを観測する | [attention.py:50](../../slider_fuse/attention.py#L50) |
| E8 | 自動mask生成・手動mask変換・保護付き後処理は既存関数で実装済み | [masks.py:93](../../slider_fuse/masks.py#L93)、[masks.py:196](../../slider_fuse/masks.py#L196)、[masks.py:228](../../slider_fuse/masks.py#L228) |
| E9 | 入力条件、未知hook、空latent、auto/manual接続を検査する | [sampling.py:59](../../slider_fuse/sampling.py#L59)、[sampling.py:114](../../slider_fuse/sampling.py#L114)、[native_pair.py:97](../../slider_fuse/native_pair.py#L97) |
| E10 | Saveのprovenanceは現在auto収集設定を照合しない。auditのbranch NFEはPhase 2を前提とする | [diagnostics.py:335](../../slider_fuse/diagnostics.py#L335)、[diagnostics.py:374](../../slider_fuse/diagnostics.py#L374) |
| E11 | 比較CLIはschema 1/2を読み、生成条件・環境・入力・mask partition・実装を照合する | [compare_slider_diagnostics.py:13](../../tools/compare_slider_diagnostics.py#L13)、[同:139](../../tools/compare_slider_diagnostics.py#L139) |
| E12 | mixの既存4条件はwidgetsの順序・値・接続をテストしている | [test_prediction_mix_workflows.py](../../tests/test_prediction_mix_workflows.py) |
| E13 | native validatorは現在10件を期待し、import不足やskipを成功扱いしない | [validate_comfy.py](../../tools/validate_comfy.py)、[native_checks.py](../../tests/native_checks.py) |

### 1.3 実機結果を設計へ反映する

[検討まとめ](../../docs/slider-attenuation-investigation.md)で確認した事項を、次の設計制約にする。

- 予測選択を二値のまま維持する。全ステップで「内側＝同入力のnative、外側＝同入力のbase」の一致を検査できることが、この方式の根拠である。
- 自動マスクの品質改善は、Prediction Mix採用とは別問題として扱う。今回確認済みなのは主に手動の左半分maskである。
- 背景の縦段差、人物のmask越境、保護人物の細部変化は、既知の実機評価項目として残す。
- 強度2/4で効果の調整を確認できたが、強度と効果量・保護側の変化に線形性は要求しない。

## 2. 採用設計

### 2.1 方針の比較

| 案 | 利点 | 問題 | 判断 |
|---|---|---|---|
| A. 既存Prediction Mixノードをauto対応へ拡張し、推奨workflowを追加 | 検証した経路を再利用できる。既存graphの意味を保てる | 移行時に利用者が新workflowを選ぶ必要がある | **採用案** |
| B. 旧Samplerの内部を一括でPrediction Mixへ置換 | 旧workflowも新方式になる | 過去画像の再現性、対象文章倍率、出力診断型の意味が変わる | 採用しない |
| C. 旧Samplerにbackend選択を追加 | 一つのノードで比較できる | 方式ごとに無効な設定が増え、旧文字列診断と新専用型の扱いが複雑になる | 今回は採用しない |

E1/E2の入出力契約が異なるため、Aが最小の互換移行になる。

### 2.2 通常利用の入口

登録ノード数と既存node ID・display_nameは維持する。`Krea2SliderFusePredictionMixSampler`のcategoryを現行の`CATEGORY + "/Diagnostics"`から`CATEGORY`へ移し、ノード検索でも通常利用の入口にする。descriptionはmanual対応、autoの画質は実験段階、対象外の完成画素は固定されないことを説明する。診断SamplerとDiagnosticSaveは既存categoryのままにする。次のUI/API workflowを新規追加する。

- `workflows/krea2_female_slider_prediction_mix_manual.json`
- `workflows/krea2_female_slider_prediction_mix_manual_api.json`
- `workflows/krea2_female_slider_prediction_mix_auto.json`
- `workflows/krea2_female_slider_prediction_mix_auto_api.json`

既存の`krea2_slider_mix_zero/none/all/half`計8ファイルは、audit付きの再現・比較用として内容を維持する。旧hook系workflowも削除しない。[E2、E12]

新推奨workflowは`diagnostic_level=summary`を明示し、日常生成で全ステップの大きなtraceを保存しない。診断保存ノードとの接続は残し、auditへ切り替えるだけで詳細な検証を行えるようにする。schemaの既定`audit`は互換性のため維持する。[E2、E10]

### 2.3 入力・出力の契約

既存Prediction Mixの入力順と`execute`の既存位置引数を変えず、**現在の`diagnostic_level`の後ろ**に以下のoptional入力を追加する。

| 新入力 | 省略時の値 | 契約 |
|---|---:|---|
| `mask_mode` | `manual` | `manual / auto`。古いworkflowはmanualとして動く |
| `collect_step` | 2 | autoの収集step。1始まり、1〜steps |
| `collect_block` | 18 | autoの収集block。0始まり、モデル内の範囲を検査 |
| `top_k_ratio` | 0.3 | 既存auto検査を再利用 |
| `temperature` | 4000 | 既存auto検査を再利用 |
| `fill_holes_max_area` | 0 | autoだけで使用。既存の0〜64の制約を再利用 |
| `mask_dilate_radius` | 0 | autoだけで使用。既存の0〜1の制約を再利用 |
| `selection_dilate_radius` | 0 | 予測選択だけの拡張半径。0〜16 token grid。参照partitionは変更しない |

旧Prediction Mixの`strength/seed/steps/cfg/mix_scope/trial_id/diagnostic_level`の順序・既定値と、3出力の順序・型は維持する。[E2、E9]

`target_text_scale`は追加しない。Prediction Mixのnative経路は通常Loaderによる全文適用であり、旧hookの対象語倍率は移行対象の設定ではない。[E1、E3、E5]

### 2.4 autoからPrediction Mixへの接続

```text
外部入力を検証 → 同じmodel coreの実行権を1回だけ取得
  → initial latent / noise / full sigmasをそれぞれ1回作成
  → manual: 既存のmanual_masksでmask_bankを作成
     auto: Sliderなしの専用cloneでPhase 1だけ実行
           → raw map → 二値mask → 保護付き後処理
           → 観測用hook / wrapperを解除してcloneを終了
  → 参照targetを維持し、選択専用の背景拡張を行って実効maskを決定
  → Phase 2を元のinitial latent / noise / full sigmasから開始
  → base/nativeを同入力で評価し、二値予測選択
  → 診断を確定 → 所有する処理を解除して返却
```

Phase 1専用の小さなモジュール`slider_fuse/mask_collection.py`を追加する。`AttentionCollector`、`generate_masks`、`postprocess_masks`、既存のvalidationを再利用し、旧`sample_krea2`を丸ごと呼ばない。旧関数はcore_guardとhookによるPhase 2まで所有しているため、その呼び出しでは不要な生成と二重所有が発生する。[E6〜E9]

新helperの契約は以下とする。`dict`のmask_bankは既存の構造を使い、新しい汎用frameworkやモデル階層は導入しない。

```python
collect_auto_mask(
    base, positive, negative, prompt_info, subjects,
    image, noise, sigmas, *, grid, seed, steps, cfg,
    collect_step, collect_block, top_k_ratio, temperature,
    fill_holes_max_area, mask_dilate_radius,
) -> tuple[dict, dict]  # mask_bank, collection_report
```

所有権と状態遷移を固定する。

- `core_guard`は`sample_krea2_prediction_mix`だけが取得し、helperは取得しない。
- helperは渡されたbaseから観測用cloneを作り、そのcloneのinjection・wrapper・collectorを所有する。元のbaseのmodel_optionsを直接変更しない。
- Phase 1では局所Sliderをロードせず、`SliderHook`も作らない。全体用styleはbaseから引き継ぐ。
- 観測用forwardは、既存と同じruntime text/grid/positive branch検査を行い、`collector.begin_forward`を呼ぶ。観測処理はモデルの返却値を置換しない。
- Phase 1は`noise.clone()`、`image.clone()`、`sigmas[:collect_step + 1]`を使う。途中latentをPhase 2へ渡さない。[E6]
- collectorをresetする前に、raw map・observation・mask情報を所有するCPUコピーとして確定する。
- 成功、例外、中断のすべてでcollectorの解除とcloneの終了を試みる。途中のcleanup失敗で元の例外を隠さない。Phase 2前に観測hookが残っていないことを確認する。[E4、E7]
- 観測cloneのunloadで共有coreのロード状態が変わることは許容するが、親baseのpatch定義・model_options・style設定をclearしない。終了直後に親baseを再activateし、同じ予測を返せることを成功時・失敗時の両方で検査する。
- 既存の旧hook経路はこのhelperへ無理に移行しない。今回共有するのは既存Collector・mask生成関数であり、旧Sampler全体のリファクタリングは行わない。

### 2.5 mask・端点・実行回数

- `manual`では両MASKを必須にし、後処理パラメーターは0を要求する。`auto`では両MASK接続を拒否する。旧validationと同じ境界を維持する。[E8、E9]
- 空・非有限・拡散したauto mapや、空のtarget/protected maskは停止する。全面maskや旧hookへ自動的に切り替えない。[E8]
- `mix_scope=none/all`やstrength0でも、autoを指定した場合はPhase 1を実行し、参照maskとPreviewを確定する。scope/strengthによってマスク収集条件を変えない。
- 後処理は既存ルールをそのまま使い、protectedを増減させず、target/protected/backgroundの排他partitionを維持する。
- 全黒・全白の実効maskは`mix_scope=none/all`で表す。両人物の参照maskに空maskを許可する変更はしない。[E3、E8]

`phase2_nfe`と`sampler_nfe`は**生成ループのstep数**、`branch_nfe`は**Phase 2各経路の評価回数**のまま維持する。Phase 1を`branch_nfe.base`へ混ぜない。[E5、E10]

| 8 steps、collect_step=2の例 | phase1_nfe | phase2_nfe / sampler_nfe | branch_nfe（base / slider） | total_model_nfe |
|---|---:|---:|---:|---:|
| manual・部分mask・強度4 | 0 | 8 | 8 / 8 | 16 |
| auto・部分mask・強度4 | 2 | 8 | 8 / 8 | 18 |
| auto・scope noneまたは強度0 | 2 | 8 | 8 / 0 | 10 |
| auto・scope all・強度4 | 2 | 8 | 0 / 8 | 10 |

### 2.6 診断・保存・比較

新Prediction Mixのreportを`diagnostic_schema_version=3`とする。旧hook/nativeのschema 2は変更せず、読込側は1/2/3を扱う。新versionを設ける理由は、auto生成条件やPhase 1情報の欠落を、完全な診断として受理しないためである。[E10、E11]

既存の`generation`はモデル・prompt・seed・Sliderなど生成の共通条件として維持し、独立した`mask_generation`を追加する。

```text
mask_generation:
  mode: manual | auto
  collection_branch: null | base
  collect_step, collect_block, top_k_ratio, temperature
  fill_holes_max_area, mask_dilate_radius
```

manualでは収集専用4項目をnull、後処理2項目を0へ正規化する。autoでは入力値を保存する。Save時はpromptの省略値をschemaの既定値で補い、同じ正規化をしてreportと照合する。使われないmanual収集設定の違いで結果を別物と扱わない。

新reportには`phase1_nfe`、既存のPhase 2回数、`total_model_nfe`、`observation`、`map_diagnostics`、`mask_postprocess`を記録する。収集入力のhashは`collection_initial_noise_sha256`、`collection_initial_latent_sha256`、`collection_full_sigmas_sha256`、`collection_used_sigmas_sha256`の4項目とし、最後は実際に渡したprefixを表す。manualでは4項目をnullとする。autoの先頭3項目は対応するPhase 2の初期hashと一致することを検査する。

schema 3のauto artifactには、既存の参照partitionに加えて`raw_target_similarity`、`raw_protected_similarity`、`original_target_mask`、`added_target_mask`を保存する。raw mapは観測値として`[1, H*W]`、maskは`[1,H,W]`へ揃える。mask_bank自体の既存Preview契約は維持する。[E7、E8]

保存・読込では、manifest/hash/shape/dtype/有限性、二値mask、partition、追加領域の整合を検証する。現在のextra_tensors許可キー・shape分岐・`schema == 2`限定のmanifest生成もschema 3へ対応させる。Phase 2のtraceにPhase 1を混ぜない。summaryでもmaskの由来と回数を検証する共通処理を通し、詳細traceの有無だけをaudit/summaryで分ける。summaryではステップごとの予測traceは保存しない。[E10]

比較ルールは以下とする。

- 同じauto設定同士の通常比較では、mask_generation、参照partition、実装・環境・入力条件を照合する。選択半径だけの違いはselection_policy_comparisonとして比較し、反復試験とは扱わない。
- `mix_scope`が違う比較は引き続き許可する。実効maskが異なることだけで`none/all/half`の比較を拒否しない。[E11]
- autoのmaskを手動で再入力した比較には、CLIの明示オプション`--same-mask-reference`を追加する。同じmix_scope・strength・選択拡張設定・実効mask hash・全参照partition hashを要求し、**マスクの作成経路だけ**の違いを許す。結果名は`same_mask_phase2_comparison`とする。
- 旧schemaの収集情報欠落を「同じauto条件」と推定しない。旧manualとして明示されたrecordはmanualへ正規化し、通常の旧診断比較を維持する。
- 既存の`--endpoint-reference`はnone/zeroの比較に限定し、その他の環境・実装・入力検査は緩めない。
- 変更前後でsource hashが違う回帰確認は通常比較CLIの例外にしない。回帰試験として両版・入力条件を記録し、tensorを別途照合する。

### 2.7 今回の範囲外

- soft mask、feather、境界ぼかし、attention bias、maskを毎step動かす処理。
- 新しい人物segmentationモデル、mask生成アルゴリズムや既定値の最適化。
- target_text_scaleをPrediction Mixへ移植すること。
- CFG≠1、Euler/simple以外、動画、複数画像、ControlNet、reference latent、追加の量子化形式、複数GPU。[E9]
- 全画素の保護、服装・人物同一性の完全固定、計算速度改善の保証。

背景境界の問題を隠すために今回の二値選択を変更しない。原因を切り分けた後の別課題とする。

## 3. 全体の制約

- 同じSliderをnative経路で一度だけ適用し、上流の二重適用を推奨workflowに含めない。[E3、E10]
- 旧node ID、旧入力位置、旧出力型、旧manualの数値経路を保持する。[E1、E2、E12]
- base/nativeは同一入力・sigma・conditioning・deviceで評価し、元のモデルcoreへ変更を残さない。[E4、E5]
- 一度だけ用意した初期noise/latent/full sigmasを両Phaseで再利用する。[E6]
- 実装検証と実機画質評価を区別する。import失敗やskipをnative成功にしない。[E13]
- 先に作成した検討まとめは保持する。実装開始時は、その未commit状態を含め作業ツリーを確認し、他の変更を上書きしない。

## 4. 重点レビュー項目

| リスクの高い条件 | 期待する振る舞い | 担当タスク |
|---|---|---|
| auto収集中の中断、map生成例外、cleanup自体の例外 | hook・wrapper・ロックを解放し、次runが実行できる。元の例外を保持 | T2 / T3 |
| 手動ノードを保存・再読込し、新optional値が欠落 | manual・既存audit既定で動き、旧widget値がずれない | T1 / T5 |
| strength0 / scope none / scope allでautoを選ぶ | マスクを収集した上で必要なPhase 2経路だけ実行。回数は表のとおり | T3 / T4 |
| 同じcoreを使うbase→native→baseの切り替え | style保持、Sliderの残留なし、各forward前のpatcher identity確認 | T1 / T3 |
| auto設定やmask tensorの一部だけが改変・欠落 | 保存・読込・比較で拒否。画像だけ正常でも完全artifactとは扱わない | T4 |

## 5. 実装タスク

各実装タスクは、期待する失敗をテストで確認してから最小限の実装を行う。既存コードの整理を伴う場合は、先に既存動作を固定する。記載した新ファイル・関数は**計画対象**であり、現時点で存在するという主張ではない。

### T1. 手動Prediction Mixの回帰基準を固定する

**Files:** `tests/test_native_pair.py`、`tests/test_prediction_mix_nodes.py`、`tests/test_prediction_mix_workflows.py`、`tests/test_prediction_mix_artifacts.py`。

**Consumes:** 既存の`sample_krea2_prediction_mix`、`NativePairRunner`、4条件workflow。[E2〜E5、E12]

**Produces:** 旧省略入力、既存位置引数、端点、強度2/4/0/負値、cleanupを固定するテスト。

- [x] 実装前に`python -B -m pytest -q -p no:cacheprovider tests`を実行し、基準件数・結果を記録する。直近の記録は282件だが固定件数を合格条件にはしない。
- [x] `test_legacy_mix_omitted_options_remain_manual`を追加し、既存引数でmanual・auditとして動くことを固定する。
- [x] `test_manual_reference_outputs_and_branch_counts`でzero/none/all/halfのtensor、branch回数、解放後の状態を固定する。既存assertを再利用し、同じ内容のテストを重複追加しない。
- [x] 既存4条件8 workflowを基準として記録し、後のタスクで変更しないことを確認する。
- [x] 実機manual基準をcommit `fb3cd592e7d115809ab125ef524612b7bfc62e19`の4条件と定義し、ユーザー向け手順へ基準版・workflow名・条件表を残す。既存の`検討結果`を再利用する場合の確認条件と、新たに旧版を実行する場合の手順をT6へ引き継ぐ。
- [x] `python -B -m pytest -q -p no:cacheprovider tests/test_native_pair.py tests/test_prediction_mix_nodes.py tests/test_prediction_mix_workflows.py tests/test_prediction_mix_artifacts.py`を通す。
- [x] 回帰テストだけの差分としてレビュー可能な単位にまとめる。

### T2. Sliderなしの自動マスク収集helperを追加する

**Create:** `slider_fuse/mask_collection.py`、`tests/test_mask_collection.py`。

**Reuse:** `AttentionCollector`、`generate_masks`、`postprocess_masks`、入力検査。[E6〜E9]

**Produces:** 2.4節の`collect_auto_mask(...) -> (mask_bank, collection_report)`。reportの必須項目は`phase1_nfe`、`observation`、2.6節で定義した4つの`collection_*_sha256`。

- [x] helper未実装で失敗する`test_collect_uses_only_base_and_prefix_sigmas`を作る。Slider適用0回、指定sigmaの観測、noise/latentの同一性、呼出回数`collect_step`をassertする。
- [x] `test_collection_preserves_base_prediction_and_rng`、`test_collection_failures_release_all_owned_state`を追加する。観測を有効にしても同じbase予測・乱数状態になり、hook登録中・sample中・mask後処理中の例外後にも残留しないことを確認する。
- [x] 成功・例外の両方で、親baseのpatch/model_options/style設定が変わらず、再activate後の予測が収集前baseと一致することを固定する。
- [x] 既存mask生成関数だけを使ってhelperを実装する。source tensorを返却前にコピーし、reset後もraw mapとPreview用情報が残るようにする。
- [x] 空・定数・NaN/Inf mapで失敗し、勝手に全面maskへ置き換わらないテストを通す。
- [x] `python -B -m pytest -q -p no:cacheprovider tests/test_mask_collection.py tests/test_attention.py tests/test_masks.py tests/test_mask_postprocess.py tests/test_sampling.py`を通す。
- [x] マスク収集だけの機能としてレビュー可能な単位にまとめる。

### T3. Prediction Mixのauto/manual分岐へ接続する

**Modify:** `slider_fuse/native_pair.py`。

**Test:** `tests/test_native_pair.py`、新規`tests/test_prediction_mix_auto.py`。

**Interface:** `sample_krea2_prediction_mix`の既存keyword引数に、2.3節の8項目を同じ名前・既定値で追加する。返却値は`(latent, mask_bank, DiagnosticPayload)`のまま。

- [x] `test_auto_restarts_full_schedule_with_collected_mask`を追加する。Phase 1の途中latentをPhase 2へ渡さず、初期3入力を再利用し、得たmaskを実効選択へ使うことをassertする。
- [x] `test_auto_matches_manual_with_identical_partition`を追加する。固定mapから得たmaskと、同じmaskをmanual入力した結果で、Phase 2の全入力・予測・final latentが一致することをassertする。
- [x] `test_auto_endpoint_counts`をzero/none/all/partialに対して追加し、2.5節の回数表をassertする。負strengthでもpartial経路と選択条件を維持する。
- [x] 元入力を事前検証してcore_guardを取得し、初期3入力を作成、manual分岐またはT2のhelperを呼ぶ。観測解除後に必要なSlider cloneを用意し、既存guiderへ確定maskを渡す。
- [x] 二重core_guard、Phase 1 hookの残留、styleの消失、Phase 2のbase/native切り替え失敗を回帰テストで拒否する。
- [x] `python -B -m pytest -q -p no:cacheprovider tests/test_prediction_mix_auto.py tests/test_native_pair.py tests/test_mask_collection.py tests/test_prediction_mixing.py`を通す。T1の手動基準も再確認する。
- [x] UIへ公開する前に、内部APIでautoが完結する差分としてまとめる。

### T4. schema 3の診断・保存・比較を実装する

**Modify:** `slider_fuse/native_pair.py`、`slider_fuse/diagnostics.py`、`tools/compare_slider_diagnostics.py`。

**Test:** `tests/test_prediction_mix_artifacts.py`、`tests/test_prediction_mix_comparison.py`、`tests/test_diagnostic_artifacts.py`、`tests/test_diagnostic_tools.py`。

**Interfaces:** 2.6節の`mask_generation`、Phase 1/hash情報、auto mask tensor追加、CLIの`--same-mask-reference`。

- [x] `test_auto_provenance_rejects_changed_collection_settings`、`test_auto_artifact_requires_complete_mask_source`を追加する。省略値の正規化を含め、設定改変、raw map欠落、maskの改変、manifest欠落を拒否する。
- [x] `test_auto_audit_counts_phase1_separately`で、Phase 1をbranch_nfeへ混ぜたreportや途中traceを拒否する。summaryのNFE・provenanceも検証する。
- [x] `test_same_mask_reference_accepts_only_identical_partition`を追加する。同一maskのauto/manualだけを許可し、同面積だが位置が違うmask、scope/strength/conditioningの違いは拒否する。
- [x] schema 1/2の既存reportを読めること、schema 3の必須情報欠落を拒否すること、従来のall/half比較とnone/zero比較が動くことをテストする。
- [x] 2.6節のreport、Save、reader、比較CLIを実装する。保存は既存のJSON-last方式を維持する。
- [x] `python -B -m pytest -q -p no:cacheprovider tests/test_prediction_mix_artifacts.py tests/test_prediction_mix_comparison.py tests/test_diagnostic_artifacts.py tests/test_diagnostic_tools.py tests/test_diagnostic_audit.py`を通す。
- [x] 診断契約の変更として独立レビューできる単位にまとめる。

### T5. ノード入力・推奨workflow・利用者向け説明を追加する

**Modify:** `nodes.py`、`README.md`、`docs/prediction-mixing.md`、`docs/real-machine-checklist.md`、`docs/validation.md`、`pyproject.toml`と既存report内のversion表記。

**Create:** 2.2節の4 workflow、`tests/test_prediction_mix_auto_workflows.py`。

**Test:** `tests/test_prediction_mix_nodes.py`、`tests/test_nodes.py`、`tests/test_workflows.py`。

- [x] schemaで既存入力順・出力型が不変で、新8項目だけが後置optionalになる失敗テストを追加する。旧位置引数による`execute`、Prediction Mixだけが主categoryへ移ること、manualと実験段階autoを区別したdescriptionも検証する。
- [x] 2.3節どおりに入力を追加し、内部APIへkeywordで渡す。2.2節のcategory/descriptionへ更新し、node ID・display_name・登録数を維持する。Mask Previewは先頭7出力を保ち、実効予測選択maskと選択追加領域を末尾へ追加する。
- [x] 新manual workflowは成功済みhalf条件を使い、`mask_mode=manual`、strength4、summaryを明示する。新auto workflowは両MASKを未接続にし、`mask_mode=auto`、step2/block18、top_k0.2、temperature10000、fill0/dilate0、selection_dilate_radius4、strength4、summaryを明示する。autoの値は過去の候補であり、最適値とは表示しない。
- [x] UI/APIでprompt・モデル・LoRA・設定・node linkが一致すること、Previewの先頭7出力と追加2出力とDiagnosticSaveが接続されることを静的検証する。
- [x] READMEの推奨入口をPrediction Mixへ変更する。旧hookは互換・比較用、自動版の画質は実機確認前、と明記する。旧文章倍率は新方式に移植しないこと、生成時間、保護画素非固定、背景境界・mask越境を説明する。
- [x] 候補versionを0.2.0として全version表記を一致させる。実機判定が残る場合は検証状態を明示し、未実施を成功へ書き換えない。
- [x] `python -B -m pytest -q -p no:cacheprovider tests/test_prediction_mix_nodes.py tests/test_prediction_mix_workflows.py tests/test_prediction_mix_auto_workflows.py tests/test_nodes.py tests/test_workflows.py`を通す。

### T6. 自動検証を完了し、ユーザーの手動実機確認へ引き渡す

**Modify:** `tests/native_checks.py`、`tools/validate_comfy.py`、`docs/real-machine-checklist.md`、`docs/validation.md`。

**担当:** 以下の「実装担当」はエージェントが実施し、「実機確認」はユーザーが手動実施する。未実施の実機項目を、実装担当の完了条件へ混ぜない。

**実装担当の作業:**

- [x] 実native小型Krea2で`test_native_auto_collect_then_mix`と`test_native_auto_manual_same_mask`の2件を追加する。前者はcollector→mask生成→Prediction Mixを通し、後者は同じmaskのmanual/auto Phase 2を比較する。validatorの期待件数は10→12へ更新し、skip/import不足を成功扱いしない。[E13]
- [x] 全CPU suite、Python構文、全workflow JSON・リンク、`git diff --check`を実行する。
- [x] 実機確認用の比較条件表、操作順、モデル/LoRA名を合わせる場所、保存する4点セット（PNG・mask・safetensors・JSON）、結果記入欄をチェックリストへ追加する。
- [x] ユーザーのComfyUIのPythonで実行するnative validatorと比較CLIのコマンド例を用意する。エージェント側でGPU生成や実UI操作を開始しない。
- [x] 「実装・CPU検証：完了」「実機native/UI/画質：ユーザー確認待ち」を分けて検証記録へ記載し、実装成果物を引き渡す。

**ユーザーが手動で行う実機確認:**

| 順序 | 確認する内容 | 残す証拠 |
|---|---|---|
| U1 | 実ComfyUI環境でnative validatorを実行し、12件実行・skip0を確認 | 実行ログとComfyUI revision |
| U2 | ユーザー指示で対象外。旧commitの実機回帰は実施せず、未実施を合格とは扱わない | 現行版の端点・旧入力互換性の検査は保持 |
| U3 | autoのstrength0/4で収集maskが同じか、autoと同じmaskをmanualへ渡したPhase 2が一致するか | 両runのmask・JSON・tensorと比較結果 |
| U4 | 新旧workflowをUIへ読み込み、保存、再読込、再実行する | 接続・widget値・実行結果。任意入力の値ずれの有無 |
| U5 | 元promptのseed42/444444、正面スタジオ、腰手・腕組み、公園でnone/all/対象maskと強度2/4を比較 | 顔・頭身・保護側・境界・mask越境を記入した評価表 |
| U6 | 中断後の再実行、Phase 1失敗後の手動生成、base→Slider→baseの連続実行 | 状態が復帰し次の生成が可能であることのログ |

実機結果を受領後、エージェントは保存物の整合性と数値を解析し、必要な修正を切り分ける。手動版の実測だけでauto品質を合格としたり、ユーザー未実施の項目を実施済みにしたりしない。

既存の`検討結果`をU2の基準に再利用する場合は、artifact hash/ID、初期noise/latent/sigma/conditioning hash、seed・prompt・強度・style・解像度、ComfyUI/Python/PyTorch/CUDA/GPUが揃うことを確認する。モデル・TE・VAEは旧recordではファイル名による照合で、重みSHA256の一致までは記録されていないため、実機側で同じファイルを変更せず使っていることも確認する。条件が揃わない場合は異なる環境の参考結果として残し、数値回帰の合格根拠には使わない。

## 6. 受け入れ条件

| ID | 完了を判断する条件 | 検証 |
|---|---|---|
| AC1 | 既存Prediction Mix node ID、入力位置、出力型、4条件8 workflowが維持される | 実装担当：T1/T5 schema・workflow試験。実UIはユーザーU4 |
| AC2 | 旧引数省略時はmanual/auditで動き、現行manualの端点と旧入力互換性が維持される | T1/T3と現行実機の端点比較。旧commit実機回帰U2はユーザー指定で対象外 |
| AC3 | auto Phase 1はbaseだけをcollect_step回評価し、観測hookをPhase 2へ残さない | T2/T3のspy・例外・残留検査 |
| AC4 | Phase 2は元のnoise/latent/full sigmasから開始する | T3の入力tensor完全一致 |
| AC5 | 同じpartitionを使うauto/manualのPhase 2予測・最終latentが一致する | 実装担当：T3 CPU試験。実native/実機はユーザーU1/U3 |
| AC6 | 部分混合の全stepでinside=native、outside=baseが完全一致し、NaN/Infなし | 保存traceからの再計算 |
| AC7 | zero/none/all/partialのNFEが2.5節の式を満たす | 実装担当：T3/T4。実機JSONはユーザーU2/U3の受領後に解析 |
| AC8 | 未対応入力・不正map・空maskで説明付き停止し、fallbackやモデル汚染なし | T2/T3例外試験 |
| AC9 | schema 3の収集条件・maskの由来を保存・検証でき、改変を拒否する | T4 artifact/provenance試験 |
| AC10 | 旧schemaを読め、比較のmask/環境/実装不一致を誤って同条件と扱わない | T4 reader/CLI試験 |
| AC11 | 新推奨4 workflowがUI/APIで一致し、summary/auditの両方で保存できる | 実装担当：T5静的・保存試験。実UIはユーザーU4 |
| AC12 | source構文・全CPU suite・diff検査が成功し、ユーザー向け実機手順と未実施状態が記録される | 実装担当：T6の引き渡し確認 |

実装完了は各ACの「実装担当」部分とT6の引き渡しまでで判定する。画質と実機互換性の採用判定はユーザーU1〜U6の結果で別に記録する。未編集参照との顔・体格の比較、maskの全身被覆、背景の連続性を実画像で評価し、未解決例を消さない。自動版で顔がmaskから落ちる場合は「Prediction Mixの演算失敗」ではなく「自動mask品質未達」として分けて記録する。

## 7. リスクと対策

| リスク | 対策・停止条件 |
|---|---|
| Phase 1のhookがPhase 2に残る | helperが所有物をfinallyで解除。残留を検出したらPhase 2へ進まない |
| 旧graphでwidget値がずれる | optionalを末尾追加、旧位置引数とUI roundtripを検証。既存workflowを再保存して一括変換しない |
| 自動maskが顔を覆わず再び弱く見える | raw/original/processed maskを保存。実効maskを確認してからLoRA方式を評価 |
| 同一coreのSliderがbaseへ残る | 既存のpatcher identity検査とbase→native→base試験を維持 |
| 収集設定の違いを混合方式の差と誤認する | provenanceと比較CLIでmask生成条件・実効mask・partitionを照合 |
| 二経路＋Phase 1の時間・VRAM負担 | NFEを分離表示、通常はsummary、auditは検証時に使用。速度保証はしない |
| 背景段差を解消しようとして既存の効果まで変える | 今回は二値予測選択を固定し、境界処理は別計画で判断 |
| CPU成功だけで自動版を公開品質と扱う | 実装完了とは別に、ユーザーによるnative/UI/実画像評価の状態を記録 |

## 8. 実行順序と完了時の成果物

依存順は **T1 → T2 → T3 → T4 → T5 → T6**。同じ`native_pair.py`と診断契約を連続して変更するため、基本は一つの実装担当が順に進め、独立レビューを節目に入れる。コード変更を並列化する場合も、共有ファイルの所有者を一人に固定する。

成果物は、auto対応Prediction Mix、推奨manual/auto workflow、更新された保存・比較機能、互換性テスト、ユーザーの手動実機確認手順、実施済み／未実施が分かる検証記録とする。コード・CPU検証・引き渡し資料が揃えば実装作業を完了として報告する。実機品質はユーザーの確認結果を受領して更新し、未検証のauto品質を採用済みと説明しない。

rollbackは、旧hookまたは既存manual mix workflowへ戻して行える構成を維持する。既存workflow・ノードを削除しないため、データ移行の逆変換は不要である。


## 9. 追加レビューへの対応（2026-10-09）

[受領したレビュー](<../../../計画としての完成度は高く、方針（既存のPrediction Mixノー.txt>)を反映する。レビューの「コード未変更」は計画評価時点の前提で、受領時にはT1〜T6の実装と327件のCPU検証が進んでいた。

### 9.1 選択専用の拡張を今回の範囲に加える

自動マスクが狭い場合、形状・位置が変わった人物の一部がbase側の予測領域へ出る可能性がある。人物に沿った境界と背景の段差も評価対象にする。これを理由に、**参照partitionを変更せず、Prediction Mixの採用範囲だけを広げる**機能を追加する。

- 新しい任意入力`selection_dilate_radius`を、既存入力の最後に追加する。既定0、上限16 grid。通常Krea2では1 grid約16px。
- 後処理の`mask_dilate_radius`（上限1）と別の設定にする。raw/original/processed target、protected、backgroundの参照partitionは保存したままにする。
- 選択範囲は元targetを全て保持し、8連結の最大target成分から背景へ1セルずつ広げる。保護領域を追加せず、保護領域を飛び越えて背景を選ぶ拡張も行わない。小さな孤立成分は保持するが拡張しない。
- 新auto workflowは半径4（約64px）を比較開始候補として明示する。manualは0。これらは画質確認済みの最適値ではない。
- Mask Previewの先頭7出力を維持し、実効予測mask、選択で追加した領域の2出力を末尾に追加する。
- reportに`prediction_selection={dilate_radius, method: largest_component_background_only}`を保存し、Saveの入力照合と読込時の再生成を行う。
- same-mask比較ではauto/manualの参照partitionだけでなく選択半径も揃える。旧manualの省略時は0で、以前の数値経路を維持する。

Voronoi/watershed、soft mask、毎stepの人物追跡は今回追加しない。今回の拡張だけで顔欠落・人物移動・背景段差が解消すると断定しない。

### 9.2 回帰・手順・作業順の補正

- 旧autoと新helperに同じモデル・prompt・noise・sigma・収集設定を与え、生のmap（保存形状を揃えた数値）と参照partitionが一致する試験を追加する。未学習の小型テストモデルのマスク品質はこの試験の対象にしない。
- 診断機能は、最低限の設定/NFE記録をT4a、詳細CLI・完全manifest・raw map保存をT4bとして考える。今後同様の導入ではT1→T2→T3＋T4a→T5→T4b→T6とし、workflowを早く実機へ渡す。本作業ではT4bまで既に実装済みのため削除・巻き戻しせず保持する。
- Phase 1の観測直後打ち切りは、cleanupと旧autoの回帰に影響するため今回は実装しない。時間・VRAMを実測した後の最適化候補とする。
- lowvram/offloadでは経路ごとの重み再適用・ロードが時間を増やす可能性をREADMEへ追加し、単純な2倍の速度保証をしない。
- runtime versionを`slider_fuse/version.py`に一本化し、package metadataとの一致をテストする。

### T7. 追加レビューの反映

**Files:** `slider_fuse/masks.py`、`slider_fuse/native_pair.py`、`slider_fuse/diagnostics.py`、`nodes.py`、比較CLI、新workflow4件、ガイド、`slider_fuse/version.py`。

- [x] 選択専用拡張の参照不変、背景限定、保護境界、小成分非拡張、半径0互換、不正半径をRED→GREENで検証する。
- [x] 非ゼロの背景を持つauto maskで、参照partitionを変えず実効maskだけが広がることを確認する。
- [x] 保存後の半径・採用範囲の改変を拒否し、同じmaskでも異なる選択方針をsame-mask比較へ混ぜないことを検証する。
- [x] 旧autoと新helperのmap/partition一致を確認する。
- [x] optional入力、Previewの追加出力、新workflowのwidget/link、versionの一致を確認する。
- [x] 実機U3で半径0/4/8を同じ収集mapから比較できる手順を用意し、ユーザーが顔被覆・越境・境界を評価する。

実装担当の受け入れ条件に、参照不変・選択拡張の保存再生成・旧auto一致を追加する。画質の採用判定は引き続きユーザーの手動確認とする。
