# Slider診断機能・評価ワークフロー Implementation Plan

> **For agentic workers:** 実装時は `superpowers:executing-plans`、または明示的に選択した `superpowers:subagent-driven-development` でタスク順に実行する。チェックボックスは実装・検証後に更新する。この文書の作成は実装開始を意味しない。

**Goal:** 通常Loaderと独自LoRA経路の差、画像への適用範囲、文章への適用範囲を独立に評価し、局所Sliderの体格変化が弱い原因を絞れる診断機能と比較ワークフローを提供する。

**Architecture:** 通常用Samplerの公開入力・既定動作を維持し、診断専用Samplerと結果保存ノードを追加する。既存Adapter・hook・サンプリング・復元処理を共用し、同じサンプリング経路に標準LoRA Loader相当のbackendと独自hook backendを接続する。診断は初めにmanual左半分と画像全面を比較し、自動マスク再設計を混ぜない。

**Tech Stack:** Python >=3.10、既存PyTorch/safetensors、ComfyUI V3 API、既存pytest。依存追加なし。

**Spec:** この会話のユーザー要求「診断用変更と評価用のワークフローの作成をするための実装計画」と、本書の要求・比較表・受入条件を仕様とする。成果物は本計画書のみ。コード、既存workflow、学習済み重みは本ターンで変更しない。

## 根拠と現在の判断

| 確認事実 | 根拠 | 計画への反映 |
|---|---|---|
| manualもautoと同じstrength4、seed42、8 steps、文章倍率0、同一LoRA SHA256。manualは左半分全体で、全体適用ほど頭身が変わらない | 提供PNG `krea2_female_slider_compare_manual_00002_.png` のprompt、左半分を作るSolidMask/MaskComposite/InvertMask。前回auto PNGとのメタデータ照合 | 自動マスク修正を先行させず、演算経路と適用範囲を検証 |
| 局所0/0.5/1は画像条件が同じ。0/1のログではnoise/latent/sigma/maskも一致 | 提供ログ２の診断2件、提供target_text PNG3枚 | 現在の1 tokenへの追加だけを繰り返す比較は不要 |
| Adapterはalpha/rankを掛け、入力dtypeへdown/upをcastする | `slider_fuse/lora.py:39`, `:43` | 既存演算を基準にする。FP32化やstrength補正を同時に入れない |
| Hookはimage suffixとtarget phrase行を個別に更新 | `slider_fuse/lora.py:207` | 画像scopeと文章scopeを独立に選択する診断分岐が必要 |
| manualと後処理は両人物の非空・排他マスクが前提 | `slider_fuse/masks.py:228`, `:76` | 全面適用のためにprotected空マスクを通常検査へ通す変更はしない |
| Phase1後の本生成は初期noiseと全sigmasから再開 | `slider_fuse/sampling.py:57`, `:196` | 診断manualではPhase1なし、全8 NFE。通常autoは従来どおり |
| runtimeはcore共有、独自hook拒否、解除/unloadの契約を持つ | `slider_fuse/lora.py:19`, `slider_fuse/sampling.py:100`, `:228` | backend比較は直列。標準patchと独自hookを二重に有効化しない |
| 既存probeは独自Adapter式との比較で、標準Loaderとの比較ではない | `tools/probe_krea2_slider.py:31`, `:53` | 標準Loaderとの比較を別途追加し、既存probeの合格で代用しない |
| 通常ノードの入力末尾・旧payload・workflow対応がテストされている | `tests/test_nodes.py:95`, `tests/test_workflows.py:86` | 診断専用ノードを採用し既存widgetsの並びを維持 |

2026-10-08の計画作成時検証: `python -B -m pytest -q -p no:cacheprovider tests/test_nodes.py tests/test_workflows.py` → **25 passed**。全suite・実GPU生成を今回実行したとは扱わない。

「画像面積19.85%だから実効strengthが約1/5」という解釈はしない。manual結果で自動maskの欠落説は後退したが、文章の全文適用で改善することも未証明。通常globalでは右人物も変化するため、人物間の構図・相対サイズへの影響も残る。

## Global Constraints / 要求

- 既存Sampler、Subjects、MaskPreviewの公開schema、optional既定値、旧workflowの数値動作を維持する。
- batch1、1024×1024、empty txt2img latent、CFG1、Euler/simple、denoise1、8 stepsを初回評価条件とする。
- checkpoint=`intorealismAsian_k2JAVFLASHV1.safetensors`、text encoder=`wen3vl_4b_bf16.safetensors`/krea2、VAE=`qwen_image_vae.safetensors`。
- upstream style=`krea2_darkbrush.safetensors` strength0.8。Slider=`Krea2/krea2_deaging_20261001T075826Z_c06cb035.safetensors` strength4、基準のみ0。
- promptは現行global reference workflowと完全一致。target/protectedはwoman/man、occurrence0。prompt変更・学習変更・precision変更・attention bias追加は対象外。
- 全文とはKrea2内部の結合系列にある全text行で、CLIP encoderへのLoRAではない。全文診断はprotected phraseも直接変更する。
- 全面とは4096 image tokensへの直接差分。人物保護を意味しない。scopeと実際の適用行数を画面説明・ログ・保存結果に明示する。
- 不正なscope、shape、非有限値、未知adapter形式を拒否する。診断が失敗しても通常方式へ黙って切り替えない。
- 既存依存のみを使う。新しいモデルやライブラリの自動取得はしない。実機検証用の未導入依存を合格扱いしない。
- 本変更で画質改善や通常Loaderとの数値一致を約束しない。原因を判別できる証拠が成果物。

## Review Focus

1. 画像全面時にprotected空を理由として通常マスク検査を弱めない。参照partitionと実適用maskを分けて試験する（Task 1/2）。
2. native/hookのcloneがcoreを共有したまま標準patchを持ち越さない。実行順序反転と例外後の再実行を試験する（Task 2/4）。
3. 全文モードなのに既存のprotected direct delta=zeroを出さない。実際のscopeから診断値を生成する（Task 2）。
4. 古いwidget並び、キャッシュ、結果ファイルの取り違えによって比較条件が変わらない。schema、省略入力、fingerprint、保存manifestを試験する（Task 3/5）。
5. 画像が似ていることやCPU doubleの成功を、実INT8の演算一致と呼ばない。反復誤差・同じ入力・環境情報と一緒に評価する（Task 4/6）。

## 採用する設計

### 選択肢

- **採用: 診断専用Samplerを追加し、内部の小さな共通処理を再利用。** 通常ノードの人物保護契約と、診断で意図的に全面/全文へ適用する契約を明確にできる。
- 通常Samplerに多数のoptional診断入力を追加する案は、既存UIの保護説明と矛盾しやすいため採らない。
- CLIだけで全比較を行う案は、通常のComfyUI操作で画像を評価したい用途に合わないため採らない。CLIは数値比較と結果集計を担当する。

### 内部インターフェース

`slider_fuse/lora.py`:

- `RoutingState`へ診断用 `image_scope="target_mask"`、`text_scope="target_phrase"` を追加。許容値は画像`target_mask|all`、文章`none|target_phrase|all`。
- 既存 `target_text_scale` の0..1の意味は維持。通常呼出しは既定scopeを使用し、従来の画像演算順序を維持する。診断ではnoneならscale0、それ以外はscale1に固定。
- `image_mask`は参照target maskを保持し、`effective_image_mask(output)`がscopeに応じtargetまたは同形のonesを返す。参照maskは書き換えない。
- 全文の実効indicesは`arange(cap_len)`。通常のtarget/protected位置検査は維持し、全文時も元のphrase位置は診断ラベルとして保持する。
- 全文呼出数を既存target_text_linear_callsへ偽装しない。診断に`text_linear_calls`と`selected_text_row_count`を新設。通常レポートの意味を維持。

`slider_fuse/sampling.py`:

- 既存 `sample_krea2(...) -> (latent, mask_bank, report)` の外部契約を維持。
- `sample_krea2_diagnostic(model, positive, negative, prompt_info, subjects, latent, lora_path, *, backend, image_scope, text_scope, strength, seed, trial_id=0, steps=8, cfg=1.0)` を追加。返り値は `(latent, mask_bank, DiagnosticPayload)`。通常関数の3番目のreport dict契約と混同しない。
- backendは`native|hook`。nativeは画像all/文章allだけ許可し、標準Loaderと同じComfyUIモデルLoRA適用APIを呼ぶ。nativeのstrength0はSliderの標準patchを追加しない。hookは各scopeに対応。
- まず標準Loader/KSamplerの実装を実装先ComfyUI checkoutで確認し、同じAPIと引数で参照経路を構成する。未知版に推測のfallbackを入れない。
- 診断はmanualのみ。Subjectsに左/右の非空・排他マスクを入力し、通常検証を通す。native/allでもこれらは参照partitionとして保持するが適用を制限しない。Phase1は0、本生成は8 NFE。
- load、入力検証、noise/sigma生成、txtfusion長確認、core_guard、try/finally復元を必要最小限で共通化する。サンプラー本体のコピーを作らない。native backendにはSliderHookを一切入れない。
- hookのscope/maskを設定した後に実行し、clearで消える診断情報は先にsnapshotする。nativeでもtext長・input/conditioning/sigmaの確認を行う。

`slider_fuse/diagnostics.py`（新規）:

- `DiagnosticPayload(report: dict, first_prediction: torch.Tensor, effective_image_mask: torch.Tensor)` を定義。reportはJSON可能な値だけ、tensorはdetach済みCPUコピー。maskは生成時の実効maskで、保存時にreportのhashと照合する。通常Samplerの返り値には使用しない。
- `DiagnosticRecorder`を診断呼出しだけで使用。通常実行ではtensorのCPU転送や統計計算を追加しない。
- 初回denoiser呼出しのinput/context/timestep hash、返却predictionのCPUコピー、最終latent hashを記録。異なるstep軌道上の後半predictionを同じ入力での比較と扱わない。
- 独自hookではblock番号順の最初のblockにある5 familyについて、最初のroute呼出し1回だけ統計を取る。phase1から採取しない。
- 統計はrank/alpha/alpha_over_rank、入力/adapter/base/output dtype、base RMS、strengthとmask適用後delta RMS、実際の`result-base` RMS、max abs、有限性、選択/非選択行数。textとimageを分ける。base RMS=0や選択行なしはratio=nullとして理由を記録。
- 実際の適用後差分も測ることで、BF16加算で消えた微小差分と、計算していない経路を区別する。計測目的のFP32 reductionは生成の演算dtypeを変更しない。
- nativeには独自adapterの統計を捏造しない。`adapter_stats=null`、モデル出力比較を記録する。

### 公開ノードと保存

`Krea2SliderFuseDiagnosticSampler`（新規）:

- inputs: model, positive, negative, prompt_info, subjects, latent, lora_name, strength, seed, steps, cfg, backend, image_scope, text_scope, trial_id（非負Int、既定0）。
- outputs: latent, mask_bank, diagnostics（診断専用Custom型 `KREA2_SLIDER_FUSE_DIAGNOSTICS`）。Custom型はJSON化するreportと初回prediction tensorを保持する。
- 通常Samplerの文字列diagnostics出力は変更しない。新ノードのfingerprintはLoRA内容SHA256を使い、各入力scope/backendも通常の入力キャッシュキーへ参加させる。
- `trial_id`は反復実験のキャッシュ無効化用。反復時に0→1と変え、Samplerの再計算と新しいrun_id/NFEを確認する。noise seed、LoRA強度、その他の生成演算へ混ぜない。通常の同一trialはキャッシュを許容する。
- categoryは`Krea2/Slider FreeFuse/Diagnostics`。説明とworkflowタイトルに「全画像/全文では保護対象にも作用」と記載する。

`Krea2SliderFuseDiagnosticSave`（新規output node）:

- inputs: images（VAEDecode出力）, latent, diagnostics, filename_prefix。
- V3 schemaに `hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id]` を指定し、executeでは `cls.hidden.prompt` / `extra_pnginfo` / `unique_id` を読む。API形式のpromptは必須、UIのextra_pnginfoは任意で、API-only実行でUI workflowがなくても保存できる。この構文はローカル公式ソースsnapshot `_io.py:1284/1310`、`comfy_extras/nodes_images.py:201` で確認済み。実装先のComfyUI版でも確認する。
- unique_idのSaveノードからlatentの診断Sampler、imagesのVAEDecodeと同じSamplerを逆追跡し、MODEL→style LoRA→UNETLoader、positive→専用Encode→CLIPLoader、VAE→VAELoaderの入力を取得する。診断用workflowの既知の接続形だけを受け付ける。prompt/unique_id欠落、未知の中継、別Sampler由来のimage/latent、複数候補、参照先欠落は保存前に説明付きで拒否する。prompt全体から最初のLoaderを選ぶ方法は使わない。
- 同じbasenameで画像PNG、JSON report、safetensors（`final_latent`, `first_prediction`）と実効画像mask PNGを保存する。既存PIL/safetensorsとComfyUIのoutputパス・連番規約を使用し、任意の外部保存先は追加しない。
- reportに`artifact_id`, `run_id`, `case_id`, `trial_id`, 各ファイル名、生成条件、scope、LoRA SHA256、入力hash、実効mask hash/被覆率、reference partition hash、NFE、復元状態、環境情報を保存する。case_idはbackend/scope/strengthから決定し、保存名だけから推測しない。
- filename_prefixはworkflowごとに固定し、seedと実行の連番を含める。古い出力を上書きしない。途中保存失敗を成功と表示せず、完成したmanifestだけを集計対象にする。
- 全面の実効maskは白、参照target/protectedは左/右のまま。`mask_bank`の先頭3マスクは参照partitionとして既存Previewに渡せる状態を維持し、実効maskは別fieldと保存画像にする。
- metadataにはcheckpoint/TE/VAE/styleのファイル名・設定をworkflow promptから取得。hash未取得の大きなモデルをSHA一致と偽装しない。実機probeではcheckpoint/Slider/styleのSHA256を記録し、同じ実機・ファイルで生成比較したことをmanifestで関連付ける。

## 評価ワークフロー

以下9ケースについてUI形式とAPI形式を各1つ作る。prefixは `krea2_slider_diag_<case>`、ファイル名も同じstem（APIは`_api.json`）。各workflowは1条件だけを生成し、同一coreの並列比較グラフを作らない。

| case | backend | image_scope | text_scope | strength | 目的 |
|---|---|---|---|---:|---|
| native_zero | native | all | all | 0 | 同じ診断SamplerでのSliderなし基準 |
| hook_zero | hook | all | all | 0 | hook自体の副作用検査 |
| native_global | native | all | all | 4 | 標準Loader相当の参照 |
| hook_all_all | hook | all | all | 4 | scopeを揃えた演算経路比較 |
| hook_all_none | hook | all | none | 4 | 画像全面・文章なし |
| hook_all_target | hook | all | target_phrase | 4 | 画像全面・対象1語 |
| hook_half_none | hook | target_mask | none | 4 | 今回manual相当 |
| hook_half_target | hook | target_mask | target_phrase | 4 | 左半分・対象1語 |
| hook_half_all | hook | target_mask | all | 4 | 左半分・全文 |

加えて標準`LoraLoaderModelOnly`/`KSampler`だけで生成する `krea2_slider_diag_standard_zero` と `krea2_slider_diag_standard_global` のUI/APIを作る。前者はstyleのみ、後者はstyle→Slider4。既存global referenceを元に、保存名と必要なlatent保存接続だけを変更する。独自hook/Samplerを通らない対照として、画像と標準SaveLatent出力を保存する。**標準2対照は最終latent/RGBとprompt記載条件の比較だけ**で、first_prediction、実noise/sigma hash、診断run_idを持つとは扱わない。

標準結果はcompare CLIの `--standard-latent <file> --standard-png <file> --standard-case <standard_zero|standard_global> --against <diagnostic-report>` で明示的に対を指定する。PNGとlatentの埋め込みprompt、case、seed、共通KSamplerノードへの接続、shapeを検査する。metadataがない、対応が異なる、未対応の保存形式なら拒否する。標準結果のファイルhashと抽出条件を集計出力へ保存するが、診断9条件と同じrun_id由来の保証は付けず `provenance_level=prompt_and_explicit_pair` とする。標準側の未取得の情報はnull/reason付きとし、独自の観測ラッパーを対照へ追加しない。

計11ケース×UI/API＝22 JSON。標準2ケースは診断Sampler自身による交絡を確認するための対照で、全scope比較を標準ノードで再実装するものではない。ノード追加後のUI保存/再読込も検査する。

全ケースの共通設定はGlobal Constraintsどおり。manual maskは今回PNG同様の左512px/右512px、上下1024px。seed42固定・control_after_generate=fixed、診断trial_id=0。全面ケースでも同じ参照maskを入力。全caseのscalar差分はbackend/scope/strength/保存名の許可リストで検証する。反復時のtrial_id差は生成条件差と区別する。

### 実行順と打ち切り条件

1. **基準の確認:** standard_zero ↔ native_zeroは最終latent/RGB、native_zero ↔ hook_zeroは加えて初回prediction/input hashを比較する。ゼロ強度で説明不能な差がある場合はサンプリング/観測/復元を先に調べる。
2. **通常経路の確認:** standard_global ↔ native_globalの最終latent/RGBを比較。差が大きければ標準Loaderの呼出しやサンプリング引数を見直す。
3. **演算経路の確認:** native_global ↔ hook_all_all。同一初回input/context/timestepでのpredictionと最終latentを比較。差が出た場合はTask 4のprobeで演算・量子化差を調べ、scope比較だけで原因を決めない。
4. **画像側の寄与:** hook_all_all ↔ hook_half_all。文章の条件を固定して画像の適用範囲だけを変える。
5. **文章側の寄与:** hook_half_none ↔ hook_half_target ↔ hook_half_all。画像の条件を固定する。hook_all_none/target/allで画像scopeとの相互作用も確認する。
6. 各段階の重要な診断ペアはtrial_idだけを変えて再実行し、run_id/NFEを確認して反復誤差を取得する。キャッシュ済み出力の再保存を反復実験と扱わない。native→hook→nativeと逆順のprobeで持越しも確認する。標準KSamplerの反復が必要な場合は実ComfyUIでキャッシュを無効化する正式な実行手段を確認し、実計算をログで検証する。
7. seed42で判別できた後、重要ペアのみ123/777で再確認。人物が左半分から出たseedはmask不適合として記録し、同じ条件での局所性評価から外す。全11ケースの多seed総当たりは初手で行わない。

### 数値と目視の評価

- 診断ペアでは比較前にモデル/LoRA/style/prompt/seed/steps/solver/conditioning/hash/shape/dtypeが一致するか検査する。不一致ペアは`invalid_comparison`。標準対照では前述のmetadata条件と最終latentを検査し、未観測のinput hashやpredictionを一致扱いしない。
- `first_prediction`と`final_latent`についてMAE、RMSE、max abs、relative L2を計測する。relative L2の基準norm=0はnull/理由付き。RGB MAEは補助として記録する。
- 反復誤差を併記し、`exact_equal`と`within_declared_tolerance`を区別する。GPU/INT8の全条件に通じる閾値は本計画では捏造しない。CLIの`--atol`と`--rtol`が両方指定された場合だけ、明示された許容値で一致判定する。未指定は`measured_only`として生値を出し、合格扱いしない。許容値は結果を見て都合よく緩めず、環境・dtype・反復誤差に基づく根拠を結果に記録する。
- ゼロ/非ゼロhookの独立したCPU明示式テストはFP32でrtol=1e-5/atol=1e-6。既存のscale0経路と領域外の直接差分は厳密一致で確認する。
- 目視は女性の顔、頭身、胴/脚の比率、右人物の同属性、構図、衣服/輪郭の破綻を同じ順番で確認する。任意の手動頭部/全身ROIを記録できるが、画像MAEや自動年齢推定を若返り成功判定にしない。
- native全体と女性局所は右人物の変更有無も異なる。画像全体のMAEを「女性だけへの効果不足」の定量値と呼ばない。

## 実装タスク

### Task 1: scopeを分離する最小の内部変更

**Files:** Modify `slider_fuse/lora.py`; Create `tests/test_diagnostic_routing.py`; retain `tests/test_lora.py`, `tests/test_target_text.py`。

**Interfaces:** 前述のRoutingStateの2 scopeと`effective_image_mask(output)`。既存Adapter.deltaを共用する。

- [x] 全6 scope組合せ×strength -1/0/4の明示式、画像/textの選択行、非選択行への直接差分0、invalid enum、元mask不変、text境界不一致の失敗テストを書く。
- [x] `python -B -m pytest -q -p no:cacheprovider tests/test_diagnostic_routing.py`で新規契約により失敗することを確認。
- [x] scope解決を実装。defaultは旧経路、noneはtext deltaなし、allは全text行。cache key/clearも新しいscopeに対応し、run間の設定持越しを禁止。
- [x] `python -B -m pytest -q -p no:cacheprovider tests/test_lora.py tests/test_target_text.py tests/test_diagnostic_routing.py`を成功させる。

### Task 2: 診断Sampler、native backend、観測

**Files:** Modify `slider_fuse/sampling.py`; Create `slider_fuse/diagnostics.py`, `tests/test_diagnostic_sampling.py`; extend `tests/test_runtime.py`。

**Interfaces:** `sample_krea2_diagnostic(...) -> (latent, mask_bank, DiagnosticPayload)`、`DiagnosticRecorder`。reportはJSON可能な値のみ、tensorは専用payloadへ分離。通常Samplerは引き続き3番目にreport dictを返す。

- [x] 全面でも参照partitionは排他/非空、Phase1=0、full sigma/同じnoise、nativeとhookの排他、strength0のno-op、非法native scopeの事前拒否をテストする。
- [x] reportがall scopeでprotected/outsideへの直接差分をzeroと誤表示しないこと、clear後も統計が残ること、計測on/offで出力とRNGが一致することをテストする。
- [x] 実装対象ComfyUIの標準Loader/KSamplerを読み、利用するAPI・commitを文書化してからnative経路を実装。モデルの巨大なdeepcopyや全weightの手動展開は行わない。
- [x] サンプリング共通処理を小さく抽出して診断経路を追加。標準patchと独自hookが有効なcoreを同時に使わず、finallyでowned hook/patch/cache/cloneを復元する。
- [x] 正常終了/例外/中断で復元、native→hook→nativeの独立性、通常auto/manual無変更を確認する。
- [x] `python -B -m pytest -q -p no:cacheprovider tests/test_sampling.py tests/test_runtime.py tests/test_diagnostic_sampling.py tests/test_diagnostic_routing.py`を成功させる。

### Task 3: 診断ノードと結果の保存

**Files:** Modify `nodes.py`, `__init__.py`; Create `tests/test_diagnostic_nodes.py`, `tests/test_diagnostic_artifacts.py`; implement serialization in `slider_fuse/diagnostics.py`。

**Interfaces:** 診断Samplerの3出力とDiagnosticSaveの4入力は前述どおり。保存payloadにfirst_prediction/final_latentを含む。保存側は生成時のlatent hashとの一致を確認。

- [x] 既存4ノードのschema/旧payload不変、新2ノードの登録、LoRA内容変更時のfingerprint変化、backend/scope validationをテストする。
- [x] trial_id変更がSampler入力の変更として再実行を要求し、同じseed/noise/sigmas/数値出力を保つテストを追加する。保存ノードだけの再実行や同じrun_idの複製を独立試行と数えない。
- [x] temp output配下で同名prefixの連番、JSON/safetensors/PNG間のartifact ID整合、途中保存失敗、latent取り違え、JSON有限値をテストする。
- [x] payloadの実効mask/hash一致、V3 hidden promptからの祖先追跡、複数Loaderの誤選択拒否、異なるSamplerへの接続拒否、prompt欠落拒否、extra_pnginfoなしAPI実行の成功をテストする。
- [x] 新規2ノードと保存処理を実装。ComfyUI出力ディレクトリの規約を再利用し、出力先のpath traversalや任意絶対パスは受け付けない。
- [x] `python -B -m pytest -q -p no:cacheprovider tests/test_nodes.py tests/test_diagnostic_nodes.py tests/test_diagnostic_artifacts.py`を成功させる。

### Task 4: 通常Loaderとの数値probeと集計

**Files:** Create `tools/probe_krea2_slider_parity.py`, `tools/compare_slider_diagnostics.py`, `tests/test_diagnostic_tools.py`; extend `tests/native_checks.py` and `tools/validate_comfy.py` as needed. Reuse `tools/comfy_environment.py`。

**Interfaces:**

- probe: `--comfy-root`, `--model`, `--lora`, optional `--style-lora`, `--style-strength`（既定0.8）、`--strength`（既定4）、`--repeats`（既定2、最小2）、`--cpu`, `--output`。固定入力tensorを生成して使い回し、repeatsで入力やseedを変更しない。
- compare: 診断reportパスの列、`--output`、任意の`--atol`/`--rtol`のペア。標準対照の入力modeは前述の `--standard-latent/--standard-png/--standard-case/--against` を一組として受け付け、通常report列modeと排他。異常条件は非0終了、許容値未指定の正常計測は`measured_only`。

- [x] FP32 tiny modelで通常重みpatchと全系列hookの同一入力出力が明示式に一致するテストを追加する。標準Loaderのmockだけを実機合格根拠にしない。
- [x] probeはまず既存の独自式probeに合格することを確認し、追加で標準Loader経路とhookの代表5 familyを同一の固定入力で比較。標準patch適用/解除を正式API経由で行い、native→hook→nativeの反復誤差を保存する。
- [x] nativeな量子化経路を単層呼出しでは再現できない場合は、その制約をエラー/未実施として報告。通常Loaderを単なる`base+Adapter.delta`へ置換して合格させない。実機Samplerのfirst_prediction比較を併用する。
- [x] 全LoRA key mapping、rank/alpha、precision、元のpacked weight/patch状態の復元、ComfyUI revision、Python/torch/CUDA/GPU、モデル/LoRA/style SHA256を記録する。
- [x] compareのcondition不一致拒否、shape/NaN/Inf、ゼロnorm、tolerance片側指定拒否、反復誤差、標準SaveLatentの明示的な形式変換をテストし実装する。最終latentの保存倍率が異なる形式は変換根拠を記録して比較する。
- [x] 標準対照modeでは埋め込みprompt照合、参照するSampler一致、PNG/latentの取り違え拒否、metadata欠落拒否をテストする。first_predictionが存在するという仮定を置かず、標準側は最終latent/RGBと取得可能な条件だけを報告する。
- [x] `python -B -m pytest -q -p no:cacheprovider tests/test_tools.py tests/test_diagnostic_tools.py`を成功させる。実native検証はComfyUI環境で行い、未実施を区別する。

### Task 5: UI/API評価workflowと説明

**Files:** Create 前述11ケースの `workflows/krea2_slider_diag_<case>.json` / `_api.json`; Create `tests/test_diagnostic_workflows.py`, `docs/diagnostic-parity.md`; update `README.md`, `docs/validation.md`。

**Interfaces:** Task 3のschema、固定case matrix、診断成果物の保存規約。

- [x] 11ケースの期待settings、共通prompt/model/LoRA/seed、左右mask、1変数比較の許可リスト、UI/API一致、リンク/slot型、全入力接続をテストとして定義する。
- [x] 現行global/manual workflowを基準に22 JSONを作る。通常比較templateのstrength2をコピーして取り違えない。各caseに目的・適用範囲を記載し、通常Loaderと独自Sliderの二重適用を構造検査する。
- [x] 標準2ケースはSaveImage/SaveLatent、診断9ケースはTask 3の保存ノードへ接続。成果物の持ち帰り方、実行順、判定限界、結果表の記入欄を説明する。
- [x] `python -B -m pytest -q -p no:cacheprovider tests/test_workflows.py tests/test_diagnostic_workflows.py`を成功させる。

### Task 6: 全体検証と実機評価

**Files:** 検証結果を `docs/validation.md` に追記。実画像/latent/モデルをリポジトリへ追加しない。

- [x] `python -B -m pytest -q -p no:cacheprovider tests`、Python構文検査、workflow JSON/link検査、`git diff --check`を実行する。lint/typecheck設定の有無を確認し、未導入なら追加依存を入れず適用不能と記録。
- [ ] 実ComfyUIで`python -B tools/validate_comfy.py <COMFY_ROOT>`、既存probe、新parity probeを実行。例外やskipを成功件数へ含めない。
- [ ] UIで保存/再読込してscope/backend/seedが維持され、PNG・JSON・latentが対応することを確認する。
- [ ] seed42を前述の段階順で実行し、差が出た段階で原因調査へ分岐。必要な反復と実行順反転を行う。
- [ ] 同条件の初回prediction一致/差、最終latent差、女性体格、右人物への影響をまとめ、次に変更すべき対象を1つに絞る。数値backend差が残るなら画像scope/textscopeを主因と断定しない。
- [ ] 実機未利用ならコード/CPU/UI静的検証の完了とGPU評価未実施を分けて納品する。INT8実機/画像品質の固定falseを、実行しただけでtrueへ変更しない。

## Acceptance Criteria

1. 通常4ノードのschema・旧API payload・既存workflowと通常出力の回帰試験が成功する。
2. 新診断2ノードが登録され、画像2 scope×文章3 scopeが別々の設定・実適用行数として観測できる。
3. strength0はnative/hook双方でSliderなしとなり、hookの非選択行への直接差分は0。全画像/全文hookはFP32 tiny modelの全系列LoRA明示式と一致する。
4. 全面診断は通常の非空protected検査を弱めず実行できる。全text時に保護text直接差分zeroを誤報しない。
5. 全診断caseで同一noise/initial latent/sigmas/contextを検査でき、初回predictionと最終latentを保存・比較できる。scalar統計にNaN/Infを出さない。
6. 通常Loader参照と独自hookが同時にSliderを適用しない。成功・失敗・中断後にowned hook/cache/patchを復元し、実行順で結果が変わらないことを検査できる。
7. 11ケース22 JSONがUI/APIで一致し、共通設定と許可された差分が自動検証される。
8. 診断9条件の保存画像・mask・JSON・tensorの関連がartifact IDとhashで確認でき、不完全な保存や異条件比較を拒否できる。標準2対照は明示指定したPNG/latentのpromptと共通Samplerを照合し、観測していない情報や同一run ID保証を付けない。
9. 実機結果の反復誤差と許容値の指定有無を区別し、CPU double・画像の類似・呼出回数だけで通常Loader同等性を合格扱いしない。
10. 最終報告で「実装検証済み」「実INT8計測済み」「画質/局所性評価済み」を別々に記載し、未実施項目を明示する。

## リスク・緩和策・終了条件

| リスク | 緩和策 |
|---|---|
| 標準patchと加算hookでは丸め/量子化順序が異なる | まずFP32明示式で論理を検証し、INT8は同じ入力・反復誤差・dtypeを保存して差を測る。bit一致を前提にしない |
| 診断機能自身が生成を変える | 通常呼出しでは記録off、観測on/off出力/RNG一致、標準KSamplerの2対照を用意 |
| ログや初回tensorでVRAM負荷が増える | 代表5 family×初回1回のscalar統計、first_predictionだけCPU保存。全層activation保存なし |
| MODEL clone間のcore共有で比較条件が混ざる | 一条件ずつ直列、正式patch API、finally復元、順序反転probe |
| 全文適用で男性にも作用する | 診断上の意図として表示し、女性効果と男性への影響を別々に評価 |
| ケースが増え実機試行が長くなる | 初期seed42、段階ごとに判定して停止可能。追加seedは重要ペアだけ |
| 実機ComfyUIが従来照合commitと違う | 使用commitを保存し、実装時にそのcheckoutの標準APIを確認。GPU未検証を残せる |

実装の終了条件は受入条件1〜9の実行可能な検証と、条件10に基づく残件の明示。実機評価の終了条件は通常経路との差・画像scope・文章scopeのどこで主要な差が生じるかを示せること、または差を判別できない具体的な境界と次の最小測定が明らかになること。診断段階で自動mask再設計、再学習、strength自動補正、生成品質改善を同時に実装しない。

## 実装時の最終状態（2026-10-08）

診断機能・保存・比較ツール・22 workflowを実装し、独立レビューの3指摘を修正。最終CPU suiteは237件成功、構文/compileとJSON/link・whitespace検査も成功。Task 6の実native/GPU/UI/画像評価は、ユーザーの最新指示によりユーザー側で実施。詳細は同名progress.mdとdocs/diagnostic-parity.mdを参照。
