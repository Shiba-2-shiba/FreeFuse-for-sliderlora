# Prediction Mixのmask再検討と三人状配置の切り分け

2026-10-12。状態：実装と診断2構図の追加検証を完了。下記は当初計画で、実行結果は [レポート](../../docs/attention-step-block-sweep-results/20261011T185947Z-paired-mix/report.html)と [実行記録](2026-10-12-prediction-mix-mask-validation.progress.md) を参照。採用gateを通過しなかったため条件付きholdout・動的化には進んでいない。ユーザーの「precision mix」は、このリポジトリのPrediction Mixとして扱う。

## 結論と要求

最優先は、**実際のPrediction Mixと同じ途中latent上で、人物の対応と形状を観測すること**。その上で、前半〜中盤の人物対応を手掛かりに、後半step3の輪郭を再分類する方式を第一候補にする。現行のON独立軌道の二値maskや、後半の保護ラベルをそのまま正解として固定する根拠はない。

目標はSlider +4による対象の顔・頭身・全身体型の変化、保護人物の顔・身長・主要体格の維持、自然な接触、余分な顔・人物・旧身体の残像がない二人構図。既存の8step/Euler/simple/CFG1、モデル・style・Slider・prompt・seedを初期検証では固定する。対象の手足と人物境界も評価し、顔の旧70セルだけを最適化しない。完成OFF画像は原因診断の対照として利用できるが、最終方式の必須入力にするかは別判断。[要求の出典](../../docs/superpowers/specs/2026-10-10-slider-mask-collection-design.md)、[完成OFF依存に関する既存方針](../../docs/final-core-exclusion.md)。

## 確認済みの事実と推論

| 優先 | 確認事項 | 確度と含意 |
| --- | --- | --- |
| 1 | OFF/ONの収集入力hashは両構図ともstep1だけ一致し、step2・3は不一致 | 事実・高。従来比較は重みの差と軌道の差を両方含む。実際のMix入力上のLoRA効果とは同一ではない |
| 2 | 公園ON step2/block18の旧穴70セル中32セルは保護seed。step3/block18でも32セル | 事実・高。現行実装はseedのスコアをone-hotで固定する。後段のmask閾値・保護を尊重する穴埋め・現行伝播ではこの誤seedを再分類できず、seed選択自体の見直しが必要 |
| 3 | 公園step3の前半blockは広く粗い範囲、後半は手足・輪郭が明瞭だが顔を保護へ誤割当 | 保存図とユーザー所見・高。人物対応と境界の推定を分ける価値がある。組合せ方式の成功自体は未検証 |
| 4 | 重なりON step3/block18・23・26は三頭状、同じstep3/block18のOFFは二人状 | 保存図・高。ONの重み、独立軌道、分類のどこが主因かは未確定。maskだけで実画像の人数は断定できない |
| 5 | 前回D（OFF/ON対象の和集合＋OFF保護）は、Cより目立つ欠損を減らした | 保存生成画像の補助目視。編集前後の占有範囲を含める必要性と整合するが、対応付けなしの単純和集合を汎用解とはしない |
| 6 | 全stepの予測選択は数値監査に合格した | 事実・高。選択演算の正しさと、選んだ領域・完成画像の正しさは別 |

新しい読み取り検査の実数・hashは [evidence.json](2026-10-12-prediction-mix-mask-evidence.json)。既存証拠は [late sweep検証](../../docs/attention-step-block-sweep-results/20261011T170454Z/verification.json)、[early supplement](../../docs/attention-step-block-sweep-results/20261011T173209Z-blocks1-13/verification.json)、[形状比較所見](../../docs/attention-step-block-sweep-results/20261011T173209Z-blocks1-13/mask-shape-review/observations.json)、[A/B/C/D比較](../../docs/mask-source-comparison-results/20261011T104616Z/report.html)。

コード根拠：

- `slider_fuse/native_pair.py:147`以降：同じx・timestepからbase/sliderを順に評価。`token_mask`はサンプリング前に決まり、全step固定。
- `slider_fuse/native_pair.py:256`：通常autoはbaseで予備収集。sweepのONは上流LoaderでSliderを適用した独立実験経路で、Mix内部の観測ではない。
- `slider_fuse/experimental_collection.py:190`・`:305`：各入力MODELで元scheduleのprefixを実行。途中latent・denoised予測の実tensorはsweep bundleに保存していない。
- `slider_fuse/experimental_masks.py:109`・`:167`：同じtapの文脈からseedを選び、seedをone-hotで固定。`:248`・`:252`では伝播後も確信ラベルとseedを維持する。
- `slider_fuse/masks.py:149`：保護セルを含む穴は既存の穴埋め対象から除外。
- `slider_fuse/prediction_mixing.py:30`：混合は二値maskによる同入力のbase/slider予測の選択。

## 推奨するmask方式の構成（検証前の設計仮説）

### 1. 観測する状態をMixへ合わせる

各stepの実際の入力をx_mix、sigmaを同一にして、baseとSlider +4の両forwardを観測する。

`p_mix = (1-M) * p_base(x_mix, sigma) + M * p_slider(x_mix, sigma)`

観測用に独立した全体ON軌道へ移ると、対象だけを編集しているMixの人物配置からずれる。両branchの入力hash・sigma・conditioningを揃えることを前提とする。base側も編集後のx_mixを見ているので、未編集人物の独立した正解画像とは扱わない。[実装](../../slider_fuse/native_pair.py)。

最初は保存済みのD maskをM0として固定したpilot Mixを使う。そこからmask候補M1を得て、元のnoise・全8stepからM1で再生成する。一度の修正で効果を比較し、改善が不明なまま無制限に反復しない。最初から毎stepのmask更新まで同時に導入しない。

### 2. 「誰の領域か」と「どこまでが身体か」を分ける

- 人物対応の手掛かり：前半〜中盤blockの安定した領域。初期候補はblock8/10。前半の二値mask全域を正解seedにせず、同じx_mix上で複数の観測が一致する内側の領域を使う。人物ごとの頭部・胴などの対応が必要で、左右座標や面積だけでは決めない。
- 輪郭・部位の手掛かり：step3付近の後半特徴。初期候補はblock18/23、block26は指定失敗例として残す。後半blockほど常に良いとは仮定しない。各blockのraw featureを直接平均せず、同じgrid上で役割を明示したスコア・親和性として利用する。
- 最終分類：前段の人物対応を持つseedから後半の特徴・画像token間親和性で領域を推定する。後半の誤ったprotected seed/高信頼ラベルは、検証せずに固定障壁として引き継がない。保護人物の確かな中心領域は維持し、割当が衝突するセルは不確定として記録する。
- 対象の顔が後半でprotectedになった場合、人物対応の証拠に基づき再分類する。単純な穴埋めで男性領域まで取り込む操作にはしない。

これは既存`propagated_bg`のパラメータ調整と異なる。現行伝播は、既に確信した誤分類を維持する構造である。新旧を比較する際は「seedだけ変更」「親和性による輪郭推定だけ変更」「両方」を分ける。[現行候補と伝播](../../slider_fuse/experimental_masks.py)。

### 3. 人物maskと編集に必要な領域を分ける

若返りで頭身や位置が変わる場合、新しい対象のシルエットだけを選ぶと古い肩・髪・服・脚が残り得る。同じMix入力上で対象として対応付けた「変更前側」と「変更後側」の支持領域を保持し、その和集合と局所的な境界余地を編集候補にする。旧人物と別人物を対応付けずに全て結合しない。

保護人物の確かな顔・胴の領域との衝突は別途判定する。OFF全領域を無条件に優先して対象を切り欠いたり、保護を一律縮小して男性まで編集したりする方針は採らない。曖昧な接触部と保護人物の主要部を区別する。[既存Dの結果](../../docs/mask-source-comparison-results/20261011T104616Z/report.html)、[既存core除外の評価方針](../../docs/final-core-exclusion.md)。

`abs(p_slider-p_base)`が大きい場所だけをmaskにする案は優先しない。差分には背景や別人物への影響も含まれ、人物の対応を証明しない。

### 4. 動的更新は第二段階

同じMix入力で得たM1の固定再生が有効だった場合だけ、maskをstepごとに更新する方式を検討する。前stepからの人物対応・変更量・低信頼時の保留を明示する。動的版では各stepのmaskと由来を保存し、各stepのmaskで予測選択を再構成する監査が必要。現在の固定mask監査を動的版の証拠に流用しない。

## 三人状配置の追加検証

### A. 既存traceの可視化から始める

既存A/C/Dとsweepの代表編集は、`trace_inputs`、`trace_base_predictions`、`trace_slider_predictions`、`trace_predictions`、sigma、実効maskを保持している。step1・2・3と最終stepについて、base/slider/mixのdenoised予測を並べ、最終画像へ何が残るかを確認する。[trace保存](../../slider_fuse/native_pair.py)、[保存仕様](../../docs/prediction-mixing.md)。

denoised予測は`comfy_cfg1_denoised`空間。`trace_inputs`のノイズlatentをそのままVAE復号して「三人」と数えない。観測predictionをVAEへ渡す前に、対象モデルの`process_latent_out`相当を適用し、既知の最終latent/画像との復号整合を確認する。前者は内部モデル空間、既存`final_latent`はsampler出口で変換済みなので二重変換しない。確認したローカルComfyUI sourceでは`model_base.py:258`がdenoisedを返し、`samplers.py:1238`がサンプル出口でlatent変換を行う。実行時は起動中backendの同じ契約を再確認する。

この段階は既存predictionのVAE復号なので新しいdiffusion model評価は0 NFE。ただしVAE費用は発生する。これだけでは、元の独立ON収集軌道の三人状mapの発生箇所までは確定しない。低sigma以外の不鮮明なpreviewは判定不能を許す。

### B. 重みと軌道を交差させる

同じ初期noise・sigma列・promptから、baseのみ、全体Slider +4、Dを使うMixの3軌道を作る。各stepの観測はモデル評価前の入力に揃え、step更新後のlatentと混ぜない。

| 固定する途中latent | base重みで観測 | Slider +4重みで観測 |
| --- | --- | --- |
| base軌道 x_OFF | 本来のbase観測 | 同じxにLoRAだけ追加 |
| 全体ON軌道 x_ON | ONで変わったxからLoRAを外して観測 | 従来のON収集に対応 |
| 実際のMix軌道 x_MIX | 同じMix入力のbase予測・map | 同じMix入力のLoRA予測・map |

step1は入力共通という陽性対照。step2・3で上の6条件を比較する。初期block集合は8/10/18/23/26。モデル評価をblock数倍にはしない。一つのforwardから複数blockを観測する。まず公園seed444444と重なりseed42だけを使う。

記録するもの：入力xとhash、sigma、base/slider/mixのdenoised、正しいlatent変換後のRGB、最終RGB、role/headスコア、seed選択前後、prototype添字・個数、mask、曖昧領域、分類前後のオーバーレイ。新しい特徴方式のoffline再計算に必要なattention output・Q/Kは、対象blockと条件を限定して保存量を見積もる。現bundleはこれらの大きなtensorのhashを持つが、実tensorの代わりにはならない。

全8stepのbase8 + globalON8 + 固定Mix16 = 32 NFE/構図。step2・3の不足する交差評価（slider@x_OFF、base@x_ON）を各1回追加すると4 NFE、合計36/構図、2構図で72 NFEが新規取得の基本予算。既存maskを再利用し、新たなmask収集分は含めない。observerが生成経路を変えないことを別に確認し、実際の再評価・cache・VAE費用を区別して記録する。

### C. 観測結果から対策を選ぶ

| 観測 | 支持する原因 | 次の対策・検証 |
| --- | --- | --- |
| 同じxのslider denoisedで余分な頭部が出て、base側には出ない | 現在のLoRA条件での即時予測の問題 | 強度4/2を一因子比較し、必要なら正のprompt表現だけ別比較。強度を下げた成功を+4の解決とはしない |
| x_ONではbase/slider双方で余分な配置が出るが、x_OFF/x_MIXでは出ない | 全体ONの独立軌道による配置変化 | 独立ON mask収集を主経路から外し、x_MIX上の観測へ移す |
| denoisedの二人配置は整合するがmapだけ三頭状 | 役割seed・prototype・token/head集約の誤対応 | 同一x・featuresを固定し、seed再割当、prototype更新、head集約を一因子ずつ比較 |
| base/sliderそれぞれでは二人だが、混合したdenoisedや後続画像で増殖 | 固定選択が別位置の旧像・新像を同時に残す | 対象対応付きの旧/新占有範囲と境界余地、後に追従maskを検証 |
| OFF完成画像も二人構図を満たさない | prompt/seed/基本生成の不成立 | その条件を構図不成立として保持し、正のpromptだけ整理した別対照を追加 |

原因は併存し得る。一枚の早期previewや連結成分数だけで分類を確定しない。現promptには既に「exactly two」「two separate heads」がある。CFG1収集はnegativeを使用していないので、negativeへ人物数の禁止語を追加することを主要対策にしない。[入力promptとnegative契約](../../docs/attention-step-block-sweep-results/20261011T170454Z/woman_front_strong_overlap/sweeps/mask/on/manifest.json)。

## 実装・実験の順序

1. **既存trace decoder**：まずA/C/Dのstep別branch画像を取得し、どの出力から第三像が見えるか記録する。`native_pair.py`のtrace空間と起動中ComfyUIのlatent変換を照合する。
2. **観測専用paired probe**：Bの交差実験を作る。`NativePairRunner`の共有core切替・入力clone・cleanupを維持し、`ExperimentalAttentionCollector`をbranchと入力hashで識別する。現collectorの(step, block)だけのキーでは両branchを区別できない。外部callback拒否やhook残留検査は緩めず、診断専用の観測境界を設ける。[関連実装](../../slider_fuse/native_pair.py)、[collector](../../slider_fuse/experimental_collection.py)。
3. **offline mask ablation**：現行k3、seed修正のみ、後半形状の再分類のみ、両者の組合せを比較する。後半protected seedの誤りをそのまま固定しない。計算可能なものを先に保存tensorで調べ、画像生成でない検査を生成成功と呼ばない。
4. **固定mask再生**：M0=Dを対照に、候補をまず2種類に絞り、同じseed・元noiseから+4で再生成。新候補2×2構図×16=64 NFE。M0の比較は新鮮で同一provenanceのものを使い、流用不可なら対照再取得分を別計上。明らかなmask構図不成立は代替せず失敗として残す。
5. **保持した条件と動的化判断**：両構図で構図・顔・全身・保護側が改善してから、選定に使わないseed123/777へ同じ設定を適用する。動的maskは固定候補の不足が画像で確認された場合に限り独立比較する。

この計画は追加検証の提案であり、未取得のbranch RGBやx_MIX attentionを観測済みとは扱わない。実行段階では起動中のComfyUIを優先し、backend機能の追加で再読込が必要な場合は、必要な変更を完成・CPU検証してから起動方法と両立する再読込手順を決める。

## 採用判定と停止条件

- **数値**：同一x・sigma・conditioningのhash一致、所定tapの欠落0、観測前後の予測とlatentの一致、全終了経路のhook除去を確認する。診断追加で変わった場合は品質比較へ進まない。
- **選択演算**：固定maskは全stepで内側=slider・外側=baseを再構成して一致。動的版なら各stepの実効maskを保存して同じ検査を行う。
- **構図**：二人の頭部・身体の対応、女性が手前の位置関係、男性の顔と両者の足元を確認し、余分な頭部・人物・旧身体の残像は不合格。maskの最大成分数や`status=ok`で代替しない。
- **mask**：同じ観測状態のRGBに対して独立に確認した可視の顔中心・胴・手足の内部領域を評価。対象内部の欠落、保護人物内部への侵入、背景への漏れ、境界を別々に測る。旧70セルは継続診断として残すが、正解labelに昇格させない。粗いgrid境界や不鮮明なpreviewは判定不能として残す。
- **編集品質**：対象の顔だけでなく頭身・胴脚比を確認。男性の顔・身長・主要体格、衣服・背景の境界も対照と比較する。自然な手・腕の小移動は、余分な顔や身体との融合と分ける。
- **候補選定**：診断2構図で顔被覆だけ改善し全身・保護・構図が悪化した候補は採用しない。maskの外形と顔割当の両方が改善し、+4の画像比較が成立した候補だけholdoutへ進める。未見seedで崩れたら既定値化せず原因別に戻す。
- **head選択**：今回の顔ROIで勝ったheadだけを選ぶ方法はtuningとして隔離する。全head平均を対照に、集約の正規化・頑健化を試す場合は選定用条件と評価用seedを分ける。

## 主なリスク

- 前半の広いmaskを正解として固定すると背景や別人物へ広がる。seedは内部の対応が確認できる部分に限定し、後半形状と衝突する領域を可視化する。
- 後半の輪郭は詳しくても身体全体が欠ける。block26の見た目の鋭さだけで選ばず、block18/23も対照に残す。
- OFF保護の固定やlate保護の強制固定は、対象の変形先を切る。maskの由来と対応を見て衝突を判定する。
- 自己更新maskは誤りを増幅し、step間でちらつく。最初は一回のpilot→固定mask再生とし、動的版の更新量・保留条件は別に定める。
- 外部segmenterは中間像の荒さや人物の重なりで誤る可能性がある。必要なら診断補助の候補にとどめ、新規依存の導入を今回の既定方針には含めない。

## 外部の一次資料との関係

[FreeFuse v3](https://arxiv.org/abs/2510.23515v3)はsubject LoRAの残差を対象領域へroutingする枠組みで、raw attentionの穴・曖昧な割当を課題としている。本件は属性Sliderの全予測を混合しており、同じblock/集約が有効とは限らない。

[DiffSeg](https://arxiv.org/abs/2308.12469)はself-attentionによる物体領域のまとまり、[SeeDiff](https://ojs.aaai.org/index.php/AAAI/article/view/32686)はcross-attention由来のseedとself-attentionによる領域拡張を扱う。意味対応と形状推定を分ける発想の参考になるが、Krea2の共同attention・deaging Slider・Prediction Mixで今回の提案が有効という検証結果ではない。
