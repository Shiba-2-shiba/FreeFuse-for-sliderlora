# 次の比較：背景あり k1 / k3 と、同一 k3 mask の +2 / +4

推奨は、以下の **自己完結した cold workflow** です。過去の k3/+4 の重み・入力・mask の同一性を確認できなくても開始できます。完成 OFF 画像は追加生成しません。新しい mask アルゴリズムや通常 sampler の変更はありません。

## すぐ実行する

| 診断ケース | UI | API | 評価記録 |
|---|---|---|---|
| park、seed 444444 | [UI](../workflows/krea2_female_slider_attention_ablation_park.json) | [API](../workflows/krea2_female_slider_attention_ablation_park_api.json) | [index](../workflows/krea2_female_slider_attention_ablation_park.index.json) |
| 女性手前・強い重なり、seed 42 | [UI](../workflows/krea2_female_slider_attention_ablation_woman_front_strong_overlap.json) | [API](../workflows/krea2_female_slider_attention_ablation_woman_front_strong_overlap_api.json) | [index](../workflows/krea2_female_slider_attention_ablation_woman_front_strong_overlap.index.json) |

1. UI を読み込み、MODEL / CLIP / style LoRA / VAE / Slider のファイル名を環境に合わせます。Slider は各 sampler 内の一度だけです。上流 MODEL に追加しません。
2. まず mask の候補状態だけ確認したい場合は、3 行すべての `DiagnosticSave` と `SaveImage`（実効 mask・追加余白も含む）を無効にし、2 つの suite saver だけを実行します。2 collector の合計は 4 NFE。失敗候補は保存 report に残します。編集時は対応する出力を有効に戻し、失敗候補を成功扱いしません。
3. `bg_k1_s4`、`bg_k3_s4`、`bg_k3_s2` の 3 画像を生成します。k3 の +2 / +4 は **同じ Select → MaskPreview → Subjects** を参照します。mask を再計算して強度ごとに選び直す経路ではありません。
4. 実際に queue した API、2 suite の manifest と全 artifact、3 編集の画像・latent・診断 JSON を保管します。index の `evaluation_record_template` をコピーして、実測と人の判定を追記します。配布 index 自体は実行証明ではありません。

既存の 7 方式比較ファイルは変更していません。`multi_proto_bg` という内部 variant 名は互換性のため共通で、**実際の設定は `normalized_config.prototypes`** です。新 workflow の表示・出力名は k1 / k3 を明示します。既存 `centroid_control` は背景なしの 2 者競合なので、今回の背景あり k1 対照には使いません。

### 固定条件と実行量

- 過去と同じ prompt。park は 444444、強い重なりは 42
- Euler/simple、8 steps、CFG 1、collection step 2 / block 18、観測 step `[1,2]` × block `[16,18]`
- 元の空 latent と同じ seed を全 branch で共有。別の 2-step スケジュールに置換しない
- k1 / k3 の collection config は `prototypes` だけが異なる。背景 phrase、閾値、temperature、伝播設定は変更しない
- fill、mask dilation、selection dilation はすべて 0。equal-area 割当や一律膨張を追加しない
- 2 collector は `audit_tensors=true`。観測 tensor の照合に使うが、hash・転送の時間とメモリは追加で必要

**cold：1 ケース 2×2 + 3×16 = 52 NFE、2 ケース合計 104 NFE。**
2 collector は独立に同じ条件を走るため、同一 collection を共有した 50 NFE とは表示しません。mask-only を先に走らせ、同じ collector が cache されれば編集時に capture は再実行されません。cache が失われれば追加分を数えます。

**過去の k3/+4 を条件付き再利用：1 ケース 2 + 16 + 16 = 34 NFE、2 ケース合計 68 NFE。** 下記の事前確認を通せる場合だけです。過去の検証資料だけで再利用可能と認定してはいません。

以上は、有効な非空・部分 mask、非ゼロ強度、全指定出力が動く場合の model NFE 予定値です。VAE、hook、prototype、hash、転送、保存は含みません。実際の `phase1_nfe` / `total_model_nfe` と cache 状態を優先します。wall time は queue ごとに記録し、共有・重複する時間を足しません。VRAM は runtime の `memory.scope` と未測定理由を保持し、peak を合算せず、欠測を 0 にしません。

## 先に同一条件を確認する

### k1 対 k3：変えたものが k だけか

2 suite の次を照合します。seed だけの一致を同一入力の証拠にしません。

- 正規化 config は `prototypes` 以外が完全一致。prompt、背景 phrase / occurrence、token positions、conditioning 本体と metadata の hash が一致
- `initial_noise_sha256`、`initial_latent_sha256`、`full_sigmas_sha256`、`used_sigmas_sha256` が一致
- primary step 2 / block 18 の `tap_manifest` を選び、`tensor_sha256_before` / `tensor_sha256_after` の全項目、attention mask の hash / shape を一致確認。before と after 自体も一致していること
- 同一 queue の MODEL、style、CLIP、VAE、Slider の共有元を記録。別実行を比べる場合はファイル名だけで重みの同一性を認定しない
- `report.algorithm.variants.multi_proto_bg.prototype_indices` の各 role の配列長を **実現 prototype 数** として記録。k3 は上限で、seed pool や特徴の重複により 3 未満になり得る

不一致は `mismatch`、情報欠測は `unverified` と記録し、k だけの因果比較として採用しません。hash が同じで両 mask が同一だった場合は有効な null result です。生成の失敗や「比較不能」と置換しません。

### +2 対 +4：本当に同じ mask か

両編集診断の `reference_partition_sha256` の target / protected / background **すべて**と、`effective_image_mask_sha256` を比較します。source の target だけの一致では不十分です。fill / dilation / selection radius も 0 のまま確認します。全 sigma、初期 noise / latent、conditioning、Slider hash は一致、意図した差は strength だけです。

最初の入力は同じでも、編集後の denoising 軌道が異なるのは期待されます。後半 step の latent 一致を要求しません。異なる mask の k 比較に、既存の「同一 partition を要求する」診断比較 CLI を無理に通す変更もしていません。

## 保存済み k3/+4 を使う場合だけ

標準は cold graph です。元データや当時の asset identity が欠ける場合、現在の重みを hash して過去の値と見なすことはできません。その場合は 104 NFE の経路を使用します。

再利用には以下が必要です。

1. k3 `multi_proto_bg` の元 suite `manifest.json` と、hash 検証可能な全関連 artifact
2. 元 +4 編集の診断 JSON、画像、tensor を含む完全な bundle
3. 保存された target / protected の二値 grayscale PNG を、変更せず ComfyUI input へコピーしたもの。拡大・縮小・threshold・画像編集・alpha channel 使用は不可
4. **当時から保管した** checkpoint / style LoRA / CLIP / VAE / Slider の SHA-256 identity record と、現在読み込む各 asset の実パス。ファイル名だけでは不可

ツールは suite の config と成功状態、PNG の bytes / decoded tensor hash、partition / 実効 selection、初期 noise / latent / sigmas と conditioning、元の生成設定、artifact 整合性、asset hashes を確認します。歴史的 asset と実行の対応は、別途明示する人の provenance attestation に依存します。機械だけで過去の対応を証明できるわけではなく、suite の `weights_identity_verified=false` を書き換えません。

asset record の厳密な形式と実例は [reuse verifier の説明](attention-reuse.md) に従ってください。通常の cold generator は標準ライブラリだけで動きますが、reuse 検査は CPU の torch / Pillow / safetensors 等、既存 artifact 検証に必要な依存を使います。

```bash
python tools/build_attention_ablation_workflows.py --case park --seed 444444 --reuse-suite "archive/k3-suite/manifest.json" --reuse-edit "archive/k3-edit/bg_k3_s4.json" --input-dir "ComfyUI/input" --reuse-target "attention/park_k3_target.png" --reuse-protected "attention/park_k3_protected.png" --asset-record "retained/asset-binding.json" --output-dir "generated/park_reuse"
```

全引数が揃い、検査に通った場合だけ `_reuse` UI/API/index を作ります。古い +4 の検証済みモデル名・style 強度・画像サイズ等を引き継ぎます。graph は **新 k1 capture/+4 と、保存 k3 mask の +2 のみ**。k3 collector と +4 sampler は入りません。`LoadImageMask` は red channel で読み、保存 grid を維持します。元の +4 は新しい index で外部 evidence として対応付けます。

事前検査の `eligible_for_runtime_comparison` は「新実行まで同一と確認した」意味ではありません。queue 後に新 +2 と旧 +4 の partition・effective mask・初期入力・sigma・conditioning を必ず再照合します。新 k1 collector と旧 k3 collection の primary tensor audit が旧側に無ければ、観測 parity は `unverified` のままです。GUI でファイル名や設定を変更したら、その変更後の API と実測を証拠にします。

## 人による判定：成功条件を混ぜない

各画像に reviewer、evidence、notes を記録します。まず **構図 gate**、次に複数の独立した品質軸です。

### 構図 gate

- 2 人の人物、別々の頭、全身と両足が見える
- 強い重なり条件では、女性が手前で男性が後ろ、女性の体が男性の胴を大きく遮蔽する。男性の顔は見える
- 単なる横並びや肩の接触だけなら `composition_not_met`。prompt の文言や mask の交差だけで合格にしない
- 若返り自体は Slider の目的なので、「成人に見えない」だけで不合格にしない

不成立の画像も残し、その条件での「強い遮蔽に成功した画質改善」と数えません。構図に依存しない余分な顔等の失敗所見は記録できます。人の判定前は `passed=null` / `not_evaluated`、自動または補助的な目視結果を human approval に置換しません。

### 独立した評価軸

| 軸 | 記録すること |
|---|---|
| duplicate anatomy | 余分な顔・頭髪・腕・脚・靴、古い輪郭の残存、融合と場所 |
| target edit achievement | 意図した年齢感・サイズ等の変化が見えるか。数値年齢の推定はしない |
| natural body proportions | 頭身、首・肩・胴・腰・脚の連続性、服と身体の整合性 |
| male face / eyeglasses | 男性の顔・眼鏡の見た目を別々に比較し、可視性が足りなければ不明 |
| male height / build | 身長感と主要体格を別々に比較。camera / pose の差も注記 |
| small arm/hand adjustment | 自然な接触に必要な小さな腕・手の動きは許容枠として別記。顔・身長・体格の変化とは混ぜない |
| preservation vs OFF | 同条件の既存 OFF が検証できない場合は null / not evaluable。追加 OFF を生成しない |

+2 で自然になっても、編集が弱くなっただけかもしれません。自然さと編集達成を並べ、単一の採点へ潰しません。男性の見た目が候補間で似ていることから、元画像に対する identity 保持や画素不変を認定しません。

mask 精度には独立した可視領域 label と位置合わせが必要です。生成後に輪郭が変わった画像を無検討に初期 mask の正解にしません。label なしの precision / recall / core leakage は null と理由を残します。面積や mask 同士の IoU は記述統計です。

### baseline の表記訂正

受領 HTML の「baseline uncertain = 0」は保存値と一致しません。保存 baseline の `uncertain_mask` は residual routing background 全体です。新方式の confidence / margin 判定から来る unknown とは意味が異なります。「baseline の不確かさが 0」「新方式より自信がある」とは解釈しません。baseline +4 編集画像も OFF ではありません。

## 未見 seed での確認

42 / 444444 は診断用、123 / 777 も過去に結果を見ています。今回の調整に対する新しい holdout と呼びません。まず条件を凍結し、過去の実行 log にない seed を登録します。`confirmed_unseen_seeds` は意図的に空です。選んだ番号が未見であることを確認してから埋めてください。

新しい seed は `--case` と `--seed` を明示して生成します。凍結後に各診断ケースを新 seed で試し、試行数・失敗・構図 gate 未達をすべて記録します。結果を見て閾値等を変えた場合、その seed は以後 tuning 済みです。

```bash
python tools/build_attention_ablation_workflows.py --case park --seed 987654321 --trial-id 1 --output-dir generated/park_candidate_seed
```

上の番号はコマンド例で、未見と認定した番号ではありません。2 ケースの既定ファイルを再生成するには `python tools/build_attention_ablation_workflows.py`。`--seed` と `--case all` は誤った一括上書きを避けるため併用できません。trial_id を変更する際は graph の両 collector と全 sampler を一緒に変更します。

## CPU と実機の検証範囲

```bash
python -B -m pytest -q -p no:cacheprovider tests
python -B -m unittest discover -s tests -p test_attention_ablation_workflows.py
```

graph テストは schema・接続・widget 順序・UI/API 一致・再生成・k/強度の統制・同一 k3 mask 共有・予算・未評価値を確認します。CPU pass は実 GPU、native ComfyUI、INT8、UI reload、人体品質の成功を意味しません。

Windows evaluator の出力拒否は、`lstat()` で既存 entry（dangling symlink を含む）を拒否し、その後も exclusive create を保持します。target 未作成・既存 file / symlink / hardlink / 入力が不変なことを回帰検査します。Linux では実 symlink と Windows 挙動の限定 simulation を使います。事前確認と open は atomic な一体処理ではなく、競合下の race-free no-follow 保証ではありません。native Windows での再実行は別途必要です。
