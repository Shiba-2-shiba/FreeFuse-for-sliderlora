# 局所Sliderの全層診断・native予測混合 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:executing-plans`で依存順に実装する。独立したタスクの実装を明示的に分担する場合は`superpowers:subagent-driven-development`を使用する。チェックボックスは実装と検証の完了後に更新する。

**Goal:** 対象外への直接差分を全層・全stepで実測し、通常LoRAとベースの予測を空間的に混合する診断Samplerを追加して、対象人物への顔・体格の効果と保護人物の属性維持を比較できるようにする。

**Architecture:** 既存の画像側hookを比較基準として維持する。新しい診断ノードは単一のEulerサンプリングループ内で、同じlatent・sigma・conditioningからstyleのみのbase予測と標準LoaderでSliderも適用したnative予測を直列に計算し、出力空間で選択する。通常のcloneが同じcoreを共有することを前提に、モデル切替・復帰を独立した境界にする。

**Tech Stack:** Python >=3.10、既存のPyTorch／safetensors／Pillow、ComfyUI V3 API、pytest。新規依存なし。

**Spec:** ユーザーが承認した直前の提案「非選択行の実測ログ追加と、診断専用の予測混合Sampler／比較workflow」、および本書の要求・インターフェース・受入条件を実装仕様とする。保存先は既存計画と同じ`.omx/plans/`。

## 要求と対象範囲

- 主目標は対象人物へのLoRA効果と、保護人物の年齢・体格の維持。保護人物や背景の画素完全固定を合格条件にしない。
- 現行hookの全層・全stepについて、対象外画像行・保護文章行・その他文章行の直接差分を測定する。
- 新規`Krea2SliderFusePredictionMixSampler`、同じSaveノードでの成果物保存、比較CLI、UI/API workflowを提供する。
- 新方式は初期版ではmanual・二値maskのみ。`mix_scope=none|target_mask|all`で全黒／対象／全白を独立に検査する。Subjectsの参照partitionは全条件で維持する。
- 自動マスクの再設計、attention bias、独立baseline軌道による外側固定、最終RGB合成、soft mask、適用stepの変更、LoRA学習・倍率補正・生成演算のFP32化は本変更の対象外。
- 本計画の完了条件を「ローカル実装検証」「実ComfyUI/INT8検証」「画像品質の評価」に分ける。CPU成功だけで後二者を完了にしない。

## 根拠・ベースライン

基準HEAD: `9972b910a227ddfcd4accd6c3f67c751ed5f14fa`。計画作成時の作業ツリーはclean。

| 事実 | 根拠 | 設計への反映 |
|---|---|---|
| 直接LoRA差分はimage suffixのmaskと選択text indicesで制限される | `slider_fuse/lora.py:236`、`tests/test_diagnostic_routing.py:9` | 同じLinear入力に対する`result-base`と、別生成の出力差を区別する |
| 現在のadapter統計は最初のblock・最初の呼び出しの選択行だけ | `slider_fuse/sampling.py:199`、`slider_fuse/diagnostics.py:86` | 旧統計を維持して全層auditを追加する |
| nativeは標準Loader APIを使う | `slider_fuse/sampling.py:28` | Sliderを標準APIで一度だけ適用。出力差分へstrengthを二重に掛けない |
| cloneは同じcoreを共有する | `tests/native_checks.py:75`、ローカルComfyUI `comfy/model_patcher.py:431` | 同時常駐した2つの独立重みと解釈しない |
| `sampling_function`に入る前にcurrent_patcherの状態準備が必要 | ローカルComfyUI `comfy/samplers.py:258`、`:609`、`:1217` | `apply_model`の途中で重みだけを切り替える実装を避ける |
| 保存時は既知のノード接続と成果物hashを検査している | `slider_fuse/diagnostics.py:156`、`:216`、`:248` | 新ノードだけを明示的に許可し、検査を弱めない |
| 比較CLIは同条件・同一dtype・別run_idを要求する | `tools/compare_slider_diagnostics.py:51` | 予測の意味と軌道の同一性も検査に加える |
| native validatorは7件を固定検査する | `tools/validate_comfy.py:24`、`tests/native_checks.py:20` | 新nativeテスト追加時に件数契約も更新する |

2026-10-08の計画作成時に`python -B -m pytest -q -p no:cacheprovider tests`を実行し、**237 passed in 4.22s**。この計画作成で実GPU生成を再実行してはいない。

受領データの根拠は作業ルート直下の`新しいフォルダー/`の11 PNGと、`新しいフォルダー (2)/`の9 JSON・9 safetensors・2 latent・5実効mask。非公開のモデル・実行結果をリポジトリへ追加しない。以下の観測値を評価の出発点として保存する。

- native_zeroとhook_zeroの初回prediction／最終latentは厳密一致。標準zero/globalと対応nativeの最終latentも厳密一致。
- native_global対hook_all_allの相対L2差は初回`0.0168918768`、最終latent`0.0272933996`。許容判定は未設定。
- hook_zero対hook_half_noneの初回prediction MAEは左`0.2193168007`、右`0.1317844232`。文章側適用0でも右側出力は変わる。
- half_none対half_targetの最終latent MAEは左`0.0686817211`、右`0.1032779639`。この差を年齢・画質のスコアと解釈しない。
- half_none/half_targetの受領実効maskは64×64で左32列のみ白。ファイルhash、run_id、tensor hashを照合済み。
- 実行環境はRTX A4000、Python3.14.5、torch2.13.0+cu130、INT8 tensorwise base／BF16 hook演算。ComfyUIは`52f98af2e2e42c421070a3e147c161c47cdeaf22`。全trial_idは0で、同条件反復は未取得。
- ローカルの`.verification/ComfyUI`は`b26625f23a888367b92153b28d93e159e83e677b`。実機版との差がある。既存記録ではnative importに依存不足があり、native実行可否は別途確認する。

上流参考: [FreeFuse移植ガイド](https://github.com/yaoliliu/FreeFuse/blob/master/transfer_freefuse_to_more_arch_guide.md)、[MultiDiffusion](https://github.com/omerbt/MultiDiffusion)、[Blended Latent Diffusion](https://github.com/omriav/blended-latent-diffusion)。予測混合はこれらの空間制御を参考にした本プロジェクトの実験であり、Krea2 Sliderでの検証済み手法とは表記しない。

## Global Constraints

- 既存6ノードのID・入出力順・既存widget順・既定生成挙動を維持する。診断Samplerに追加する入力は末尾optionalのみ。
- batch1、静止画、empty txt2img latent、CFG1、Euler/simple、denoise1。最初の実験は1024×1024、8 steps、seed42。
- checkpoint=`intorealismAsian_k2JAVFLASHV1.safetensors`、CLIP=`wen3vl_4b_bf16.safetensors`/krea2、VAE=`qwen_image_vae.safetensors`。
- style=`krea2_darkbrush.safetensors` strength0.8を両branchで維持。Slider=`Krea2/krea2_deaging_20261001T075826Z_c06cb035.safetensors` strength4。ゼロ試験のみ0。
- promptは既存診断workflowと完全一致、target=`woman`、protected=`man`、occurrence0。manual左512px/右512px。
- 初期noiseとsigmasは1回だけ作る。2branchへ同じ入力を渡す。各stepで2回の完全な`comfy.sample.sample`を呼んだり、独自Euler式で置き換えたりしない。
- native branchに局所SliderHookを重ねない。上流MODELの未知wrapper/injection、複数GPU、ControlNet、reference latent等は既存同様に拒否する。
- model coreの排他、例外・中断時のcleanup/unload、packed weightの復帰を維持する。高水準のComfyUI load/unload APIを使い、量子化weightやbackupを直接書き換えない。
- 新しいmodel coreのdeepcopyやモデル2体ロードへの自動fallbackは行わない。安全な切替を実装先APIで実証できなければ、mix経路は説明付きで失敗させる。
- 標準経路との誤差許容値は観測後に都合よく変更しない。宣言値なしは`measured_only`。
- バージョンを`0.1.4`とし、`pyproject.toml:3`と`sampling.py:169`のreport版表記を同時更新する。既存出典・ライセンス表示を保持する。

## Review Focus

1. shared-core切替でSliderがbaseに残る、またはstyleが落ちる（Task 3: base→native→base、順序反転、例外注入）。
2. 全黒／全白で差分の減算・加算丸めにより端点一致が崩れる（Task 2: 選択元tensorからの直接選択）。
3. token gridとpredictionの空間サイズ、5D静止画、奇数サイズの対応がずれる（Task 2: patch反復後crop）。
4. policy宣言、同じ入力での直接差分、別軌道の出力差を混同する（Task 1/5: 記録種別・入力hash・比較ラベル）。
5. 古いreport／workflow／キャッシュと、新診断の重いtraceや不完全保存が混在する（Task 4/5/6: 後方互換、trial、manifest、保存失敗）。

## ファイル構成とインターフェース

| ファイル | 責務 |
|---|---|
| `slider_fuse/diagnostics.py` | 既存first-call統計、全層audit、step trace、schema version、保存・provenance |
| `slider_fuse/lora.py` | 演算後のaudit通知。生成演算は維持 |
| `slider_fuse/prediction_mixing.py`（新規） | 二値maskのprediction空間への対応、純粋な予測選択 |
| `slider_fuse/native_pair.py`（新規） | ComfyUI依存のbranch切替、単一ループのGuider、専用sample関数 |
| `slider_fuse/sampling.py` | 共通入力検証の最小抽出、既存診断のstep通知。新分岐で既存大関数をさらに肥大化させない |
| `nodes.py`、`__init__.py` | 新ノード、audit入力、登録7ノード |
| `tools/compare_slider_diagnostics.py` | legacy/v2結果・軌道・同入力予測の比較 |
| `tests/test_prediction_mixing.py`、`tests/test_native_pair.py`等 | 純粋関数とnative境界を分けた回帰試験 |
| `workflows/krea2_slider_mix_*.json`、`docs/prediction-mixing.md`（新規） | UI/API比較例と実行・評価手順 |

### 診断契約

- 既存診断Sampler末尾へ`diagnostic_level="summary"`を追加。選択肢は`summary|audit`。既定summaryは既存5投影のfirst-call統計・生成計算を維持する。
- `DiagnosticRecorder(selected_modules, *, level="summary")`に`begin_step(eval_index: int, sigma: torch.Tensor)`、`record_step(prediction, latent_input, sigma)`、`finalize() -> tuple[dict, dict[str, torch.Tensor]]`を追加する。
- audit時の全層記録は`eval_index`、sigma、module、call_indexとともに`linear_audit`へ保存。`record_linear`はsummaryの`_seen`判定より前にauditを処理する。
- 領域はimage target/protected/background、text target_phrase/protected_phrase/other、各scopeのunselected。RoutingStateへ`audit_partition: dict[str, torch.Tensor] | None = None`を追加して参照partitionを保持し、clearで解除する。新たな領域情報を生成演算へ使わない。
- 数値は`result.float()-base.float()`に基づくRMS・max_abs・changed_elements・element_count・nonfinite_count。RMSは各群の二乗和/要素数から計算し、RMS同士を平均しない。空群はnull＋理由。binary非選択行の直接差分はexact zeroを検査できる。
- 選択行ではcomputed_delta、actual_delta、computed非ゼロなのにresult==baseとなった要素数を記録。計測のFP32化は生成演算を変えない。演算中のRNGやtensorを変更しない。
- GPU上で小さな集計tensorを蓄積し、可能な範囲で最後にCPU転送する。auditの計測時間を通常生成の性能として報告しない。
- `DiagnosticPayload`の先頭3フィールドを維持し、`extra_tensors: dict[str, torch.Tensor] = field(default_factory=dict)`を末尾へ追加する。
- auditの`extra_tensors`は`trace_sigmas`、`trace_inputs`（step入口）、`trace_predictions`（Samplerへ渡した予測）。mixで両branchを実行した場合は`trace_base_predictions`と`trace_slider_predictions`も保存。全てdetach済みCPUコピー。存在しないbranchはtensorを捏造せずreportに理由を書く。
- 新reportは`diagnostic_schema_version=2`、`diagnostic_level`、`prediction_space="comfy_cfg1_denoised"`、`implementation_revision`とdirty状態、branch/model評価回数を追加。実装先のsource検証で同予測表現であることを確認する。legacyの`adapter_stats`とカウンターは意味を変えず維持する。
- step traceは同じstep番号でも入力hashが異なれば`trajectory_difference`。同じ入力・sigma・conditioningを確認できたものだけ`same_input_prediction_difference`と表記する。

### 予測選択契約

`prediction_mixing.py`に以下を追加する。

```python
prediction_mask(token_mask: torch.Tensor, prediction: torch.Tensor, *, patch: int) -> torch.Tensor
mix_predictions(base: torch.Tensor, slider: torch.Tensor, mask: torch.Tensor) -> torch.Tensor
```

- maskは有限な二値。predictionのshape/dtype/deviceが両branchで一致しない場合は拒否する。4Dと、時間軸1の5Dだけを許可する。
- token maskは`[1, ceil(H/patch), ceil(W/patch)]`。各tokenをpatch×patchへ反復してからH/Wにcropし、`[1,1,H,W]`または`[1,1,1,H,W]`のbool maskにする。奇数サイズを単純なresize比でずらさない。
- 数式は`base + M*(slider-base)`だが、二値maskの実装は`torch.where(mask, slider, base)`で直接選択する。これにより選択外のbaseと選択内のsliderをビット単位で保存し、端点で不要な丸めを起こさない。
- 入力tensorをin-placeで変更しない。画素画像を合成する関数ではない。

### native経路・新ノード契約

`native_pair.py`:

```python
class NativePairRunner:
    def __init__(self, base_patcher, slider_patcher): ...
    def predict(self, branch: str, x, sigma, *, positive, negative, model_options, seed) -> torch.Tensor: ...
    def close(self) -> None: ...

sample_krea2_prediction_mix(model, positive, negative, prompt_info, subjects, latent, lora_path,
    *, strength, seed, steps=8, cfg=1., mix_scope="target_mask", trial_id=0,
    diagnostic_level="audit") -> tuple[dict, dict, DiagnosticPayload]
```

- `NativePredictionMixGuider`を実装先の`comfy.samplers.CFGGuider`から派生させる。`predict_noise`でbranchを選択・直列評価し、同じComfyUI Euler samplerへ1つの予測を返す。`inner_sample`のnoise scaling、sigma schedule、latent入出力変換を独自再実装しない。
- `NativePairRunner.predict`はbranchを有効化してから、そのbranchのprepared conditioning/optionsで`comfy.samplers.sampling_function(..., cond_scale=1.)`を呼ぶ。これにより`current_patcher.prepare_state`も正しいbranchで実行する。既存の`model_function_wrapper`の中で共有weightだけを入れ替える方式は採用しない。
- lifecycleの唯一の所有者は`NativePredictionMixGuider.outer_sample`とする。これをoverrideし、`super().outer_sample`は呼ばない。外側の`CFGGuider.sample`と内側の`CFGGuider.inner_sample`はそれぞれ1回だけ使用する。base用`prepare_sampling`は1回、conditioningの`process_conds`は継承した`inner_sample`内で1回、Euler実行とlatent入出力変換も同じ`inner_sample`内で1回とする。nativeの同一アーキテクチャ・同一conditioningを確認し、ControlNet/追加モデル/動的conditioning hooksはこの共有前処理の対象外として拒否する。
- `outer_sample`は`prepare_sampling`成功後、Runnerでbaseを有効化してから`inner_sample`へ入る。noise/latent/sigmaのdevice・dtype移動とmodel optionsのcastは実装先の標準`outer_sample`と同じ順序を保つ。branchのactivation管理はRunnerだけが行い、Guider側で別途`pre_run`/`cleanup`を重複実行しない。
- branch切替の順序は、前branchの`cleanup` → 高水準`load_models_gpu([selected_patcher], ...)`によるclone切替 → 選択branchの`pre_run` → `inner_model.current_patcher is selected_patcher`確認 → `sampling_function`。同じbranchを継続する間はactivationを重ねない。必要メモリは実装先の推定APIで確保する。ポインターを手動代入してidentity確認を通すことは禁止する。`pre_run`がcurrent_patcherを設定し、`cleanup`が解除する根拠はローカルComfyUI `comfy/model_patcher.py:1443`と`:1314`。
- `outer_sample`のfinallyでRunnerのactive branchを1回cleanupし、成功したprepareに対応する`cleanup_models`を1回実行する。外側のsample関数がcore guard内で最終unloadを所有する。activation途中失敗も追跡し、元の例外を保持して復帰する。callerのstyle patch定義/optionsを維持し、終了時は既存Samplerと同じunload方針で共有weight/Slider patchを復帰させる。次のbase実行が同じ結果となることを検査する。
- Task 3はこの所有・順序契約を実装先APIで最初に確認する。成立しない版ではnative mixを実装済みの別方式へ黙って置き換えず、このタスクを互換性未解決として止める。同一coreのcloneを一緒に`load_models_gpu`へ渡さない。
- `base_patcher`は入力styleを保ったclone、`slider_patcher`は`apply_native_slider`でそこにSliderを一度だけ適用したもの。両方を事前に用意し、毎stepファイルを読み直さない。
- 非空の部分mask・非ゼロstrengthでは各stepにbase→sliderの2評価。両branchに同一の入力内容・sigma・conditioningを渡す。native-only／base-only端点は使用branchだけ評価し、追加評価をゼロと偽らない。
- strength0はbase経路だけ、scope noneはbaseだけ、scope allはnativeだけ。scope target_maskの保護領域は同入力base予測と厳密一致する。後続stepや最終RGBの固定は保証しない。
- `sampler_nfe=steps`と`branch_nfe={base,slider}`を別記。8stepの部分maskでは`{base:8,slider:8}`。端点とstrength0は片側8・他方0。`phase2_nfe`の既存意味を変更しない。
- reportには`backend="prediction_mix"`、`mix_scope`、`model_text_scope="all"`、`output_routing_policy`を記録する。内部native branchの全文LoRAと、Sampler出力の領域選択を混同しない。mixでは`adapter_stats`と`linear_audit`はnull＋`native_branch_not_instrumented`の理由を持ち、代わりに`prediction_mix_audit`で各stepの`mixed-base`の非選択領域exact zeroと`mixed-slider`の選択領域exact zeroを実測する。端点で実行しないbranchとの差はnullにする。
- mask_bankのtarget/protected/backgroundは参照partitionのまま、`effective_image_mask`だけをscopeで変更する。全黒・全白試験のためにSubjectsの非空条件を緩めない。
- 共通検証は`sampling.validate_run_inputs(model, positive, prompt_info, subjects, latent, *, strength, seed, steps, cfg, mask_mode, collect_step, collect_block, top_k_ratio, temperature, fill_holes_max_area=0, mask_dilate_radius=0, target_text_scale=0.) -> object`として現在の`sampling.py:124`から必要最小限を抽出し、検証済みcoreを返す。既存経路の検査順・例外・生成挙動を回帰試験で維持する。mix呼出しはmanual、collect_step1/block0、top_k_ratio0.3/temperature4000、後処理0、target_text_scale0で共通入力検証だけを再利用する。native branch内部の文章LoRAをこのscale0で無効化しない。mixには追加で未知のCFG/post-CFG処理、サンプラーcallbackによるモデル改変等を拒否する検査を置く。

新ノード`Krea2SliderFusePredictionMixSampler`の入力順:

`model, positive, negative, prompt_info, subjects, latent, lora_name, strength, seed, steps, cfg, mix_scope, trial_id, diagnostic_level`。

出力は既存診断と同じ`latent, mask_bank, diagnostics`。既存DiagnosticSaveへ接続する。fingerprintは既存LoRA内容hashを再利用し、trial_idは再計算だけを制御する。seedへ混ぜない。新ノードはDiagnosticsカテゴリで、人物の属性保護と画素固定の違い、モデル2評価と切替費用を説明する。

## 実装タスク

### Task 1: 全層・全stepの直接差分audit

**Files:** Modify `slider_fuse/diagnostics.py`, `slider_fuse/lora.py`, `slider_fuse/sampling.py`; extend `tests/test_diagnostic_recorder.py`, `tests/test_diagnostic_sampling.py`; create `tests/test_diagnostic_audit.py`。

**Interfaces:** 上記Recorder・Payload追加契約。後続タスクは`begin_step`／`record_step`／`finalize`を使用する。

- [ ] `test_audit_measures_excluded_rows_on_every_call`を追加。同じ層を2step呼び、summaryは1件、auditは2件、非選択差分0を確認。意図的に保護行を書き換えるtest doubleではchanged_elements>0となることを確認。
- [ ] `test_audit_separates_protected_and_other_text`、`test_empty_region_is_not_measured_zero`、`test_audit_reduction_is_weighted_by_element_count`を追加。未知dtype・NaNも合格扱いしない。
- [ ] `test_audit_does_not_change_outputs_rng_or_restore`を追加。summary/auditで同一出力・RNG・復元状態。2block×5投影×2stepの20記録とstep境界を検査。
- [ ] `python -B -m pytest -q -p no:cacheprovider tests/test_diagnostic_audit.py tests/test_diagnostic_recorder.py`で新仕様の失敗を確認してから実装する。
- [ ] Recorder、audit用partition、step通知とCPU traceを実装。従来の`adapter_stats`は最初の5投影のまま保持する。
- [ ] 上記試験と`tests/test_diagnostic_sampling.py tests/test_runtime.py tests/test_target_text.py`を実行して成功を確認。コミット単位: `feat: measure direct slider deltas across layers and steps`。

### Task 2: 純粋なmask対応・予測選択

**Files:** Create `slider_fuse/prediction_mixing.py`, `tests/test_prediction_mixing.py`。

**Interfaces:** `prediction_mask`、`mix_predictions`。Task 3/4はこの二値選択契約を使用する。

- [ ] `test_zero_full_and_partial_masks_preserve_selected_values_exactly`を追加。BF16/FP32、負値と大きさが異なる値で、全黒=base、全白=slider、部分maskの両側が選択元に`torch.equal`。
- [ ] `test_patch_grid_expansion_crops_odd_non_square_4d_and_5d`を追加。H=5/W=7/patch2の3×4 token gridの境界を明示的に検査。
- [ ] soft mask、NaN、batch2、動画、shape/dtype/device不一致を拒否し、入力を変更しない試験を追加。CPUだけで実行できないdevice試験は別native試験へ割り当てる。
- [ ] `python -B -m pytest -q -p no:cacheprovider tests/test_prediction_mixing.py`で失敗確認後に関数を実装し、全試験成功を確認。コミット単位: `feat: add exact binary prediction selection`。

### Task 3: ComfyUIのbranch切替と単一Sampler統合

**Files:** Create `slider_fuse/native_pair.py`, `tests/test_native_pair.py`; modify `slider_fuse/sampling.py`, `tests/native_checks.py`, `tools/validate_comfy.py`。

**Interfaces:** `NativePairRunner`、`NativePredictionMixGuider`、`sample_krea2_prediction_mix`、最小抽出の`validate_run_inputs`。

- [ ] 実装先のComfyUI revisionを記録し、`CFGGuider`、`sampling_function`、`ModelPatcher.clone`、clone切替、pre_run/cleanupの対応を確認する。前述の単一outer_sample所有、conditioning前処理1回、branchごとのcurrent_patcher identity、復帰順序が成立することを確認してからTask 3のnative統合実装へ進む。ローカルsnapshotと実機52f98afの同等性を推測しない。確認結果は`docs/prediction-mixing.md`の互換性欄に保存する。
- [ ] shared-coreのtest doubleで`test_base_native_base_keeps_style_and_restores_slider`と順序反転試験を追加。独立した2モデルを使ったmockだけでは切替検査を代用しない。
- [ ] `test_pair_passes_identical_inputs_and_runs_one_euler_loop`を追加。step内の2branch入力hash一致、初期noise生成1回、conditioning前処理1回、outer/inner sample各1回、8 sampler NFEに対して16 branch評価、trial変更でnoise不変を検査。各`sampling_function`直前の`current_patcher is selected_patcher`とcleanup→load→pre_run→predictの順序、最終cleanup回数もassertする。
- [ ] load、base評価、native評価、mix、cleanupでの例外・中断を各々注入し、guard解放、hooks/patches復帰、成功再実行を検査。cleanup例外で元の生成例外を隠さない。
- [ ] native endpointの同入力基準、input不変、未知wrapper拒否、次branchの計算で前branchの返却tensorが変更されないことを失敗試験に追加し、`python -B -m pytest -q -p no:cacheprovider tests/test_native_pair.py`を実行する。
- [ ] 実装先で確認した高水準APIを使用して切替とGuiderを実装。ComfyUIの単一Eulerループを利用し、量子化重みを直接操作しない。常時GPUに2体置く方式へ拡張しない。
- [ ] native小型CPUチェックへ`test_native_pair_switch_and_restore`、`test_native_pair_prediction_matches_individual_calls`、`test_native_prediction_mix_endpoints_and_partial_mask`の3件を追加。実Patcherのcurrent_patcher identity/activation順序/例外後復帰、個別forwardとの一致、端点と部分選択をそれぞれ検査する。validatorの期待件数は7→10へ更新し、skip/missing importを成功にしない。
- [ ] `tests/test_native_pair.py tests/test_runtime.py tests/test_diagnostic_sampling.py`とnative validatorを実行。nativeが環境で実行不能なら、CPU double成功とnative未確認を分け、mixの実機評価を未完了にする。Task 4/5の作成は継続可能だが、native gate未通過で実機対応済みと表示しない。
- [ ] コミット単位: `feat: mix native slider predictions with safe model switching`。

### Task 4: V3ノード・保存契約

**Files:** Modify `nodes.py`, `__init__.py`, `slider_fuse/diagnostics.py`, `pyproject.toml`, `slider_fuse/sampling.py`; extend `tests/test_diagnostic_nodes.py`, `tests/test_diagnostic_artifacts.py`, `tests/test_nodes.py`; create `tests/test_prediction_mix_nodes.py`。

**Interfaces:** 7ノード登録、schema v2のDiagnosticPayload、既存DiagnosticSaveとの接続。

- [ ] 既存6ノードの入力順・省略値維持、新ノードの入力/出力/既定値、trial・LoRA fingerprintを検査する失敗試験を追加。
- [ ] summary/audit双方、新旧Payloadの保存、v2追加tensorのhash・shape・dtype、別Sampler由来の画像/latent拒否、保存途中失敗を検査する試験を追加。
- [ ] `generation_config`／`diagnostic_provenance`で新ノードを明示的な許可集合に追加。Save→Decode→Samplerの同一性を保ち、不特定のノード型は許可しない。
- [ ] v2 manifestにtensorごとのhash/shape/dtype、trace順序、branch未実行理由を追加。JSONを最後に保存する既存契約を維持。`first_prediction`は実際にSamplerへ渡した最初のmixed予測で、branch予測と混同しない。
- [ ] バージョンと説明を更新。実測されていない範囲に`validated=true`を付けない。
- [ ] `python -B -m pytest -q -p no:cacheprovider tests/test_nodes.py tests/test_diagnostic_nodes.py tests/test_prediction_mix_nodes.py tests/test_diagnostic_artifacts.py`を実行。コミット単位: `feat: expose diagnostic prediction mix node and trace artifacts`。

### Task 5: 比較CLIと判定の区別

**Files:** Modify `tools/compare_slider_diagnostics.py`; extend `tests/test_diagnostic_tools.py`; create `tests/test_prediction_mix_comparison.py`。

**Interfaces:** 既存CLIを維持し、`--include-trajectories`を追加。未指定は初回／最終／RGBの既存比較を行う。

- [ ] legacy report、v2 report、v2間比較、legacy↔v2の共通項比較を検査。追加traceが無ければ理由付きunavailable。ファイル/ID/hash/生成条件の検査は維持する。
- [ ] `test_equal_step_different_input_is_trajectory_difference`、`test_same_input_branch_comparison_requires_hashes`、`test_endpoint_comparison_allows_scope_but_not_generation_changes`を追加。異なるstrength/seedを同条件としない。
- [ ] traceの領域別MAE/RMSE/max_abs、直接漏れの違反件数、branch NFE、反復差を出力する。mask座標はTask 2と同じ対応を使い、境界領域はtargetの内外1 tokenの帯と定義する。画像MAEを属性スコアにしない。
- [ ] `implementation_revision`や予測表現が不一致ならsame-input比較を拒否する。legacyには未観測項目があることを明記し、v2のaudit合格を付けない。
- [ ] tolerance未指定は`measured_only`、宣言された非選択行でのnonzero/非有限値は`direct_routing_violation`としてCLI終了コード1。nativeの非計測行を0へ埋めず、未計測を直接routing合格にしない。予測混合の実測結果はhookのLinear auditと別項目で判定する。
- [ ] `python -B -m pytest -q -p no:cacheprovider tests/test_diagnostic_tools.py tests/test_prediction_mix_comparison.py`を実行。コミット単位: `feat: compare routed predictions and trajectory diagnostics`。

### Task 6: 比較workflow・評価手順・統合検証

**Files:** Create `workflows/krea2_slider_mix_{zero,none,all,half}.json`と同名`_api.json`、`docs/prediction-mixing.md`, `tests/test_prediction_mix_workflows.py`; update `docs/diagnostic-parity.md`, `docs/validation.md`, `README.md`。

**Interfaces:** Task 4の新ノードと既存DiagnosticSave。1 workflow＝1条件。shared-coreの並列実行グラフを作らない。

| case | strength | mix_scope | 比較対象 | 意図 |
|---|---:|---|---|---|
| mix_zero | 0 | target_mask | native_zero | ゼロ強度の非回帰 |
| mix_none | 4 | none | native_zero | 全黒端点 |
| mix_all | 4 | all | native_global | 全白端点 |
| mix_half | 4 | target_mask | hook_half_none、native_global | 対象効果と保護側への影響 |

- [ ] 既存診断workflowのprompt/model/style/latent/mask接続を再利用し、差分を新Sampler入力・保存prefix・ノード表示だけに限定する。新workflowは`diagnostic_level=audit`、trial_id0。既存JSONのwidgets順を変更しない。
- [ ] UI/API設定一致、linkの型・入出力slot、seed fixed、Slider二重適用なし、VAEDecode/Saveの対応、全黒/全白でも参照mask維持を失敗試験に追加し、4組8 JSONを作る。
- [ ] 比較手順に「既存native_global/hook_all_all/hook_half_noneをaudit・trial0/1で各2回」「端点3条件」「mix_half」の順序を記載。全て新実装revisionで採取し、古いtrial0と新trial1だけで反復を代用しない。
- [ ] 画像評価票は対象の顔・頭身・体格、保護人物の年齢/体格、境界・構図を独立した項目にする。服のしわ等の小変化と属性の変化を同じ不合格理由にしない。明らかな輪郭破綻/保護人物の同属性変化は記録する。
- [ ] 候補が改善した場合だけstrength2とseed123/777へ展開し、各seedのbase/native参照も保存する。人物が左右maskを外れる場合はmask不適合と記録する。自動maskはこの段階の後に別評価。
- [ ] 保存4ファイルにtraceを含むsafetensorsをまとめて渡す手順を記載。ファイル名が変更された場合はhashによる対応確認が必要で、manifestを書き換えて成功を偽装しない。
- [ ] 対象workflowテスト、全CPU suite、Python構文、全workflow JSON、`git diff --check`を実行。リポジトリに未設定のlint/typecheckツールを新規依存として追加しない。
- [ ] 実ComfyUIで7ノード登録、UI保存/再読込、API実行、実INT8での切替/端点/部分maskを検証。未実行項目は明示する。コミット単位: `docs: add controlled prediction mixing evaluation workflows`。

## 受入条件と停止条件

| ID | 条件 | 証拠・担当 |
|---|---|---|
| A1 | 既存の通常生成・optional省略・既存workflowが回帰しない | Task 1/4/6のCPU suite、端点実機比較 |
| A2 | audit追加前後で同じ生成結果とRNG、全module/stepの対象外直接差分を測定できる | Task 1の合成故障検出・記録件数・不変試験 |
| A3 | 二値選択の端点と部分領域が選択元tensorと厳密一致 | Task 2のBF16/FP32・奇数4D/5D試験 |
| A4 | base→native→baseでstyle/原weight/patch状態が復帰し、例外後も再実行できる | Task 3のshared-core doubleと実native試験 |
| A5 | samplerは8 NFE、部分混合はbranch各8 NFE、端点は必要branchのみ8 NFE | Task 3 runtime試験・実機JSON |
| A6 | 新ノード・v2成果物・legacy読み込みが整合し、誤った対応は拒否される | Task 4/5のprovenance・hash・互換試験 |
| A7 | 同入力比較と別軌道比較を区別し、未指定toleranceを成功扱いしない | Task 5 CLI試験 |
| A8 | 4組8 workflow、実行手順、評価票、検証限界が揃う | Task 6の静的試験と文書 |
| A9 | 実INT8でゼロ/全黒がbase、全白がnativeに一致するか反復誤差とともに判定できる | 実機trial0/1の初回/最終/画像比較。許容値未宣言なら測定結果のみ |
| A10 | 対象効果・保護属性・境界品質が現在のhookより改善するか評価できる | seed42の評価票、改善候補だけ123/777で再確認。実装完了と採用判断を分ける |

- 対象外の直接差分が非ゼロなら、まずhookの実装/計測を修正し、画質の解釈を進めない。
- branch切替の復帰・入力同一性・端点が成立しなければ、mixed画質評価を停止する。独立モデル2体や手動weight置換へ自動変更しない。
- GPU反復差よりhook/native差が大きい場合は、既存`tools/probe_krea2_slider_parity.py:13`で代表投影の順序反転・復帰を調べる。FP32生成への変更は別の実験として扱う。
- 正常な直接routingでも保護属性が変わる場合は方式の限界として記録する。予測混合で改善しなければ、独立baseline軌道またはattention制御を次の設計判断とする。

## 検証コマンドと実行順

```powershell
python -B -m pytest -q -p no:cacheprovider tests/test_diagnostic_audit.py tests/test_prediction_mixing.py tests/test_native_pair.py
python -B -m pytest -q -p no:cacheprovider tests/test_prediction_mix_nodes.py tests/test_prediction_mix_comparison.py tests/test_prediction_mix_workflows.py
python -B -m pytest -q -p no:cacheprovider tests
git diff --check
# 実ComfyUIのPythonで、対応版source rootを指定
python -B tools/validate_comfy.py <comfy-root>
```

`python -B`のAST解析で全自作Pythonを検査し、全workflowを`json.loads`で検査する。AST/JSON成功をnative実行の代用にしない。

依存順は **Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6**。Task 2の純粋関数だけはTask 1と独立に作成可能。その他は共有interfaceとnative lifecycleが密接なので、主担当が直列で統合し、切替・保存境界に独立レビューを入れる。

実装時は隔離された作業ブランチ/既存worktreeの状態を確認し、タスクごとの検証後に小さくコミットする。実機でA9を確認するまでは「診断実験版」、A10の比較が終わるまでは既定方式への昇格や画質改善の断定を行わない。
