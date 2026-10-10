# Attentionだけで行うmask候補の比較

実験用の追加経路です。既存のPredictionMix、core-exclusion、通常の診断比較条件は変更しません。完成したOFF画像は作らず、同じ8-step Euler/simpleスケジュールの先頭2 forwardから得た観測を7候補で共有します。別の2-stepスケジュールに置き換える実験ではありません。

CPUテストは接続・計算・保存契約の検査です。実GPU、INT8、ComfyUI UI再読込、生成画像の品質・人体改善は別途確認が必要です。

## 配布ファイル

- [mask-only UI](../workflows/krea2_female_slider_attention_validation.json) / [API](../workflows/krea2_female_slider_attention_validation_api.json)
- [任意の編集画像比較 UI](../workflows/krea2_female_slider_attention_validation_edited.json) / [API](../workflows/krea2_female_slider_attention_validation_edited_api.json)
- [ケース・候補・出力・評価記録テンプレート](../workflows/krea2_female_slider_attention_validation.index.json)

最初はmask-onlyを実行します。モデルとCLIP、style LoRAのファイル名を環境に合わせてください。mask-onlyにVAE、完成画像、外部画像、SAMは不要です。任意の編集画像比較には、従来と同じSlider LoRAとVAEを設定します。モデルの取得やインストールをこのツールは行いません。

## 1. mask-onlyを実行する

1. Encodeのprompt、Subjectsの`woman` / `man`、collectorのbackground phraseを確認します。背景phraseは実際のpromptに含まれ、人物phraseとtokenが重ならないものが必要です。公園は`park path`、屋内は`pale gray concrete wall`です。このphraseが背景全体を正しく表す保証はありません。
2. collector node 8はstyle付きbase MODELと元の空latentを受けます。Sliderを上流へ加えないでください。step数8、収集2、CFG 1を固定します。
3. suite saver node 10が、得られたmask・生float map・観測証拠・候補ごとの状態・runtime manifestを一度保存します。保存先の実ファイル名はruntimeのmanifestで確認します。
4. 全7候補の`status`を確認します。根拠不足、空target、候補計算失敗などは失敗として残します。自動的に成功候補へ置き換えたり、空maskを成功扱いしたりしません。

mask-only graphのSelect / MaskPreviewは、編集時の接続例として置かれています。terminal出力へ接続していないため、通常のmask-only実行では評価されません。失敗候補のSelectが例外になってsuite保存まで妨げるのを避けるためです。maskを確認するときは、まずsuiteが保存したPNGとmanifestを使用してください。

suite saverのprefixは安全なフラット名のみです。例は`attn_validation_park_s444444_t0_mask_only_suite`です。既存の編集画像・診断saverは`slider_attention_validation/park/seed444444/trial0/edited/...`の階層を使います。両者の対応はindex内の`output_prefixes`に記録されています。

## 2. 比較候補と対照関係

| 候補 | 比較で変えるもの |
|---|---|
| `baseline` | 既存のsingle-prototype attention基準 |
| `centroid_control` | 新方式のcalibration・seed選択規則を使った1-prototype対照 |
| `multi_proto` | `centroid_control`に対してprototype数を増やす |
| `multi_proto_bg` | `multi_proto`に明示的な背景competitorを追加 |
| `adaln_bg` | `multi_proto_bg`に対してfeature表現を変更 |
| `ensemble_bg` | `multi_proto_bg`にstep/block観測viewを追加 |
| `propagated_bg` | `multi_proto_bg`の主観測へ1回のQ/K伝播を追加 |

`baseline`対`centroid_control`はprototype数だけの比較ではありません。また`propagated_bg`はensembleへ伝播を重ねた方式ではありません。これらを混同して「何が効いたか」を解釈しないでください。

固定configは収集step 2 / block 18、観測step `[1,2]` × block `[16,18]`、top-k ratio 0.2、temperature 10000、prototype 3、seed confidence 0.65 / margin 0.15、mask confidence 0.55 / margin 0.10、context temperature 1、feature temperature 0.15です。伝播は1回、head `[0]`、chunk 128、affinity threshold 0.05、strength 0.5です。

これらは検証用の仮設定です。confidenceは校正済みの確率ではありません。top-kはseed候補の上限で、人物面積の割当ではありません。男女同面積や固定面積quotaは使いません。semantic headは現状すべてのheadの平均で、保存したhead/subword別の証拠は今後の検討用です。伝播head 0も暫定で、人物に最適なheadと確認された値ではありません。

baseに戻す不確かなセルと、背景competitorによる背景判定は区別して読みます。MaskPreviewのbackground slotはbase-routed remainderを含むため、それ自体を「正しい背景」のannotationとして使わないでください。

## 3. ケースと未使用seed

| ケース | 主seed | holdout seed |
|---|---|---|
| `park` | 444444 | 123、777 |
| `man_front` | 42 | 123、777 |
| `woman_front_strong_overlap` | 42、444444 | 123、777 |

過去のpromptをそのまま再利用し、mask方式と同時にpromptを変更しません。holdoutは今回の計画での未評価seedで、まだ実行結果があるとは主張しません。主seedの結果を見てconfigを決めた後は凍結し、holdoutを見て再調整した場合はそれを明記して新しいholdoutを設けてください。失敗seedもすべて残します。

```bash
python tools/build_attention_validation_workflows.py \
  --case park --seed 444444 --trial-id 0 \
  --output-dir generated/attention_park_seed444444

python tools/build_attention_validation_workflows.py \
  --case man_front --seed 42 --trial-id 0 \
  --output-dir generated/attention_man_front_seed42

python tools/build_attention_validation_workflows.py \
  --case woman_front_strong_overlap --seed 123 --trial-id 1 \
  --output-dir generated/attention_strong_holdout123
```

seedは0〜2^64−1、trial_idはUIと同じ0〜2^31−1です。生成ツールは指定先の所定5ファイルを置き換えます。手編集や別条件を残す場合は出力先を分けてください。個々のファイルはatomic replacementですが、5ファイル一括のtransactionではありません。生成は標準ライブラリだけで動作し、推論は行いません。

## 4. 必要なら編集画像を生成する

先にmask-onlyの候補状態を確認してから、edited graphを使います。配布edited graphは全7候補を有効にした比較です。失敗した候補はSelectでfail-closedになります。成功候補だけを実行する場合、その候補以外の画像診断save・実効mask save・余白mask saveをすべて削除または無効にしてください。画像のsaveだけを止めてもmaskのsaveが残ればSamplerが実行されます。suite saverは残します。

各候補は既存のmanual PredictionMixを通します。全branchは同じprompt / seed / MODEL / 空latent / Slider強度4 / 8 steps / CFG 1です。collectorの途中latentは使いません。対象・保護maskは同じ候補のSelect→MaskPreview slot 0 / 1からSubjectsへ渡します。各branchの上流MODELは元のstyle付きbaseで、SliderはSampler内の1回だけです。

穴埋め0、参照膨張0、最終選択余白0を固定します。最終MaskPreview slot 7の実効selection、slot 8の追加余白を保存します。生候補maskと最終実効selectionを取り違えないでください。

suiteの`run_id`を評価記録の`collection_id`へ転記し、実際にqueueしたAPI graph、suite manifest、各編集branchの画像・latent・診断JSONを対応付けます。indexは初期計画であり、UI変更後の実行内容の証拠ではありません。異なるmaskの比較に、従来の同一partitionを要求する診断CLIを無理に通さないでください。

## 5. 人による構図・品質判定

過去の女性手前・強い重なり条件は6回中6回、目的の構図が成立していません。新しいprompt成功例が得られたとは扱いません。今回もpromptの文言、maskの積、面積だけでは構図成立を確認できません。

完成OFF画像を追加生成する必要はありません。編集画像を生成した場合は、各候補の実画像で次を判定し、候補ごとに評価記録の`pose_gate`を更新します。mask-onlyだけの場合は`not_evaluated`のままです。

- 女性が手前、男性が後ろに立ち、女性が男性の胴体を大きく遮蔽している
- 男性の顔が見え、両者の全身・両足が確認できる
- 単なる横並びや肩の接触ではなく、意図した前後関係がある
- 余分な頭、融合した腕・脚など、評価を妨げる重大な人体破綻がない

不成立は`composition_not_met`として残し、「強い重なりで改善」と数えません。方法ごとに構図が変わった場合もその差を記録し、都合のよい候補だけでcase全体を成功扱いしません。

構図判定と別に、女性の顔・身長・胴脚・頭身、男性の顔・身長・主要体格、古い顔や服・脚の残存、背景への漏れ、手腕の接触・融合、首/胴/腰/脚の連続性を記録します。静的なprotected maskは人物領域の推定であり、解剖学的coreやidentityの正解ではありません。手腕だけの小調整や男性の不変を保証する機能でもありません。適切な既存の同条件参照画像があれば評価に利用できますが、未提供の参照を存在するものとして扱いません。

## 6. 保存maskのオフライン評価

`tools/evaluate_attention_masks.py`は保存target PNGを比較するツールです。モデル推論や完成OFF画像は不要です。入力は同じgridの二値PNGで、リサイズ・穴埋め・膨張・threshold変換は行いません。

```bash
python tools/evaluate_attention_masks.py \
  --candidate baseline=baseline_target.png \
  --candidate multi_proto_bg=multi_proto_bg_target.png \
  --output attention_mask_report.json
```

実際のsuite出力のファイル名を使ってください。annotationなしでは面積・連結成分などの記述統計を得られます。胴脚coverage、男性core侵入、背景false positiveを測るには、同じ可視領域・同じgridに対して独立に作成または人が検証したlabelが必要です。

```bash
python tools/evaluate_attention_masks.py \
  --candidate baseline=baseline_target.png \
  --candidate multi_proto_bg=multi_proto_bg_target.png \
  --target-full visible_target.png --target-torso visible_torso.png \
  --target-legs visible_legs.png --protected-core visible_protected_core.png \
  --background known_background.png --annotations-human-verified true \
  --output attention_mask_labeled_report.json
```

annotationを渡す場合は`--annotations-human-verified true|false`の明示が必要です。`true`は可視ground truthとの人による確認の申告であり、ツールによる独立検証ではありません。未確認の既存maskを使う場合は`false`とし、正解に対する精度とは解釈しません。自動protected maskとの重なりだけで「実際の男性への漏れ」と判定しないでください。

- annotationは必須入力ではありません。未提供・空annotationの指標は理由付き`null`になり、満点扱いしません。
- torsoとlegsは互いに排他的で、target-fullがある場合はその部分集合です。target / protected-core / backgroundも可視領域として排他的にします。
- 未label領域は未知であり、自動的に背景にはしません。同じ縦横サイズだけではsceneの一致・位置合わせを保証できません。
- カウント単位は保存maskのgrid cellです。元のtoken gridであることを確認した場合だけattention token数と呼べます。
- JSONは`variants.<name>.metrics.<metric>.value`と`reason`、入力のhashを記録します。自動順位・合否は付けません。exit 0は計算完了で、画質合格を意味しません。
- `--output`は既存ファイルを上書きしません。元PNG、runtime suite、評価JSONを一緒に残します。

## 7. NFE・時間・メモリの読み方

キャッシュなしmask-onlyはmodel forward **2 NFE**です。7方式ごとに2 NFEを繰り返しません。全候補が有効な部分maskなら、任意の編集画像は1候補につき16 NFE、全体は **2 + 7×16 = 114 NFE**が予定値です。空/全域selection、失敗、出力branchの削除、cacheで実行量は変わるため、実測reportを優先します。VAEやmask処理はこのNFEに含みません。

「追加model forwardがない」は「計算コスト0」ではありません。feature hook、prototype、affinity、転送、保存、auditには時間・メモリを使います。suiteの時間・VRAMは**全7候補と共有hook込み**です。候補ごとのalgorithm時間があっても共有作業を含まないので、各候補のend-to-end latencyや通常Samplerとの速度差にはできません。

runtime reportの次の情報を、indexの`evaluation_record_template`へ結び付けて残します。

- `run_id`、seed、trial_id、正規化config、全sigmaと使用prefix、初期noise/latent hash、実際にqueueしたAPI
- `phase1_nfe` / `phase2_nfe`と、各編集診断の`total_model_nfe`
- `elapsed_seconds` / `collection_seconds` / `algorithm_seconds`、save側の時間、queue全体のwall time
- `memory.peak_allocated_bytes` / `peak_reserved_bytes` / `scope` / `unavailable_reason`
- suite manifest・各artifactのhash・候補status・編集画像の診断JSON

時間項目は重なりがあり、そのまま合算できません。特にcollection時間には同期的な観測処理を含みます。peak memoryのscopeを保持し、nodeごとのpeakを足し算しません。CUDA測定が利用できない場合は未測定として残し、0 byteと記録しません。processの他の割当やbaseline常駐分を含む可能性もあり、純増VRAMとは区別します。

新しいcollectorの`audit_tensors=false`が既定です。feature tensor全体のhashは任意の重い監査で、`true`にした測定とは比較条件を分けます。監査を省略した記録をfull-tensor parity検証済みとは扱いません。

cold測定ではcollectorと全編集Samplerのtrial_idを同時に変更します。cacheされたsuiteのreportは以前の収集の記録であり、今回のqueueで新たに2 NFEを使った証拠ではありません。warm/coldを区別し、実行したnodeとwall timeを記録してください。保存済みruntime reportから、未実行の実GPU速度・VRAM削減・画質向上を推定しません。

## 開発検証

```bash
python -B -m unittest discover -s tests -p test_attention_validation_workflows.py
python -B -m pytest -q -p no:cacheprovider tests
```

workflowテストは共有collector、全7候補、VAE-free mask-only、失敗候補がsuite保存を妨げない接続、+4/8-step/余白0の編集対照、UI/API一致、再生成一致、ケース・holdout・人による判定・実測への対応を検査します。実機での成功条件の代わりにはなりません。

## 研究上の手掛かりと今回の実装範囲

- [FreeFuse v2](https://arxiv.org/abs/2510.23515v2)：初期denoisingで内部の意味対応を使い、外部segmentorなしで対象tokenと領域を対応させるroutingの出発点です。今回のbaselineは既存実装を保持します。
- [Seg4Diff](https://arxiv.org/abs/2509.18096)：MM-DiTの意味対応に強い層を調べる研究で、block/step別の観測を残す動機です。論文のmask教師によるfine-tuningを本実装が再現するわけではなく、Krea2のblock 16/18が最適と検証されたわけでもありません。
- [iSeg](https://arxiv.org/abs/2409.03209)：self-attentionによる反復refinementと弱い応答の抑制を参考にしています。今回のQ/K伝播は`relu(weight − threshold × row_max)`、行正規化、初期scoreへのrestartを使う限定的な設計です。iSeg論文の勾配降下によるentropy低減をそのまま再現したものではありません。
- [DiTF](https://arxiv.org/abs/2505.18584)：DiTのAdaLNと大きなfeature活性が対応付けへ与える影響を調べた研究です。feature tapを比較する根拠として参照します。今回の`adaln_bg`は論文全体のchannel modulation/discard方式の再現ではありません。

multi-prototype化、背景competitor、confidence/marginによるcalibration、固定viewのensembleは、これらを手掛かりとした本検証用の工学的提案です。論文で本設定の有効性が実証済みという意味ではありません。Krea2の体型Slider、接触人物、強い遮蔽での妥当性は、この比較とholdoutで別に確認します。
