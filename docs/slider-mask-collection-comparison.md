# マスク収集3方式の比較workflow

既存ノードだけでbase収集、Slider-on収集、hybridを比較する実験用workflowです。Sampler本体や診断schemaは変更しません。実機生成・画像品質・UI再読込は未検証です。

## ファイル

- [UI workflow](../workflows/krea2_female_slider_collection_comparison.json)：ComfyUIで開く形式
- [API workflow](../workflows/krea2_female_slider_collection_comparison_api.json)：API投入用の形式
- [比較条件と出力の対応](../workflows/krea2_female_slider_collection_comparison.index.json)：経路・収集元・出力を追うためのmanifest
- [仕様](superpowers/specs/2026-10-10-slider-mask-collection-design.md)：統合規則、制約、将来のSampler統合案

## 実行前に設定するもの

モデル、CLIP、VAE、style LoRA、Slider LoRAのファイル名を実環境に合わせます。weightsの取得・依存の追加はこのworkflowに含みません。

- node 4のprompt、node 7の画像サイズは全経路で共通です。
- seedは5つのSampler、node 8・11・21・31・42で同じ値にします。
- Slider強度は上流Slider Loaderのnode 10と、本生成node 21・31・42を同じ値にします。初期値は4です。
- 収集用node 8・11のSampler自身のstrengthは常に0、mix_scopeはnoneにします。node 11だけが、上流node 10でSliderを適用したMODELを受けます。
- 収集条件は両方step 2、block 18、top_k 0.2、temperature 10000、穴埋め0、参照膨張0、選択余白0です。
- 本生成は8step、CFG 1、Euler/simple、選択余白4を共通にします。
- キャッシュを使わず再測定する場合、5つのSamplerすべてのtrial_idを更新します。最後のSamplerだけを変えると収集maskが再利用される場合があります。

## 接続を保つポイント

node 8がbase収集、node 11がSlider-on収集です。収集後の8step生成は参考画像としてのみ保存し、本生成にはmask_bank由来のMASKだけを渡します。Preview node 12・13の出力slot 0がtarget、slot 1がprotectedです。

- 本生成21：baseのtargetとprotected
- 本生成31：Slider-onのtargetとprotected
- 本生成42：Slider-on targetからbase protectedを除いたtargetと、base protected

hybridのnode 40はMaskCompositeのsubtractです。destination=Slider-on target、source=base protected、x=y=0を保ちます。二値maskなので、減算後のclampで保護優先の排他的partitionになります。node 50は衝突、node 51はbase targetにだけ残る領域を可視化します。

本生成3つのMODELはすべて元のstyle付きbase、LATENTはnode 7の空latentへ接続します。Slider-on Loaderや収集後latentへつなぎ替えないでください。これが本生成への二重Sliderと途中latentの持込みを避ける条件です。

## 保存結果の読み方

本生成3経路は通常の診断4点セット、すなわち画像・実効mask・safetensors・JSONを保存します。診断JSONではmask_mode=manualが正しい状態です。収集maskを手動MASK入力経由で受け取るためで、各stepのinside=Slider/outside=base監査を検証できます。

収集元mask、hybrid、衝突、旧targetとの差は通常のPNGとして保存します。旧targetは比較用であり、自動消去や背景補完には使いません。手・腕だけを調整する機能も追加していません。

出力prefixはslider_collection_compare配下です。A_off・B_on・C_hybridが本生成、raw_T_off・raw_P_off・raw_T_on・raw_P_onが収集元の二値mask、conflict_T_on_times_P_offが衝突、hybrid_T_on_minus_P_offがhybrid target、old_T_off_minus_T_onが旧領域のみ、reference_off・reference_onが収集側の参考画像です。ファイル名のraw_は生similarity tensorを意味しません。

index manifestは配布graphの初期設定です。UIで変更した後の実行内容を証明しません。実際にqueueへ送ったAPI graphとPNG内のprompt/workflow metadataを保持し、そのrunのtrial_id・出力prefix・保存ファイルを対応付けてください。mask PNGは生成gridの二値領域を示し、生raw mapの保存ではありません。

Slider-on収集node 11にはDiagnosticSaveを追加しないでください。現行の保存側は全体style Loaderが1つであることを要求し、styleとSliderの2つが上流にある収集経路を拒否します。Sampler自身のstrength=0は、上流Sliderまで無効という意味ではありません。

本生成のmanual診断だけでは、maskが宣言した収集branchから得られたことまでは証明できません。生raw mapとhybrid統合過程もschema 3から完全再構成できません。正規化Previewも生raw tensorの代わりにはなりません。完全な収集由来の保存は、仕様に記した将来のschema 4案です。

## 比較と費用

キャッシュなしで全体を1回動かすと、収集用10 NFEを2経路、本生成16 NFEを3経路で合計68 NFEです。独立実行のbase/slider各26、hybrid36は、不要な出力・保存枝をすべて削除または無効化した場合だけです。配布graphでは、Aの本生成だけを残しても両収集のSaveImageが有効なら両方の収集が動きます。時間・VRAM・実機での成功を保証する値ではありません。

現行のcompare_slider_diagnostics.pyは参照partitionの一致を要求するため、3方式のJSONをそのまま比較するとinvalid_comparisonになります。検査を緩めて通さず、各runの監査と画像・maskの並列比較を分けてください。同一maskを再生する対照比較では従来の一致条件を維持します。

初回は男性手前の重なりseed 42、公園seed 444444を優先し、元prompt seed 42も確認します。対象の顔と全身体型、古い像・服の残存、男性の顔・身長・体格、手腕の接触、首・胴・背景の接続を別々に記録します。姿勢条件は基準画像の頭部cropを分けて扱います。

空maskやhybrid衝突後の空targetはエラーとして扱います。画像の良し悪しを確認する前に、失敗・キャンセル後もstyle付きbaseへ復帰することと、通常KSamplerの前後一致を実機で確認してください。

## 2026-10-11：k3方式で収集元を比較した実機追試

[8画像・mask・attentionの比較レポート](mask-source-comparison-results/20261011T104616Z/report.html)、[機械検証](mask-source-comparison-results/20261011T104616Z/verification.json)、[凍結条件](mask-source-comparison-results/20261011T104616Z/plan.json)を保存しました。起動中のComfyUI 0.37.0 / AMD Radeon AI PRO R9700 / INT8で、公園444444と女性手前の重なり42を実行しました。

前回attention ablationとの統制を保つため、配布workflowのclassic方式・選択余白4ではなく、既存nodeで **k3 multi_proto_bg・全radius 0** のAPIコピーを作りました。収集時のSlider有無と、以下のmask合成だけを変えています。本生成はすべてstyleのみのMODELからSlider +4を一度だけ適用します。

| 経路 | 対象 / 保護 |
|---|---|
| A_off | T_off / P_off |
| B_on | T_on / P_on |
| C_on_base_protected | T_onからP_offを除外 / P_off |
| D_union_base_protected | T_offとT_onの和集合からP_offを除外 / P_off |

| ケース | OFF対象セル | ON対象セル | 対象IoU | ON対象とOFF保護の衝突 | D対象セル |
|---|---:|---:|---:|---:|---:|
| park / 444444 | 1046 | 697 | 0.421697 | 15 | 1211 |
| woman_front_strong_overlap / 42 | 915 | 1350 | 0.510000 | 264 | 1236 |

全4suite・8編集bundleのfile/tensor hash、宣言した合成からのpartition再構成、全8stepの予測選択を検査しました。各ケースの全4経路で初回base予測とSlider予測が一致。Aは前回k3/+4とpartition・初回予測・最終latent・Slider hashまで一致しました。実測は各68、合計136 NFE、queue wallは公園229.0秒・重なり211.2秒です。

補助目視では公園B/Cに大きな顔欠損、重なりCに目・髪・腕の欠損が出ました。Dではそれらの目立つ欠損が解消し、重なりではAより頭身・体格の変化が強く見えました。ただし公園で全身縮小の明確な増強は確認できず、重なりDでも男性の顔・服に差が残っています。人の承認、独立labelによるmask精度、未見seedの汎化、完成OFFへの保持、独立no-hook parityは未検証です。

### 顔中心付近の抜けについての保存map診断

今回のmulti_proto_bgは **step 2 / block 18のみ** を使用します。4観測tapを平均するのは別候補ensemble_bgです。Q/Kの意味対応は全48 headを平均し、特徴量prototypeと合成してから3役割の競合・confidence・marginで分類します。

公園ONの最大の閉じた穴70セル（grid bbox [17,17,28,28]）は、62セルがprotected、8セルがuncertain backgroundでした。同一領域のblock 18のQ/K役割スコア平均は、step 1で対象0.7115 / 保護0.2805、step 2で対象0.3721 / 保護0.6197。step 2でも穴領域平均で対象役割が最大のheadは12/48存在します。最終prototypeスコア平均は対象0.1538 / 保護0.8455です。[全headの穴領域診断](mask-source-comparison-results/20261011T104616Z/park/head-hole-probe.json)と[閉じた穴の分類](mask-source-comparison-results/20261011T104616Z/closed-hole-analysis.json)を保存しました。

これは「blockに顔の情報が存在しない」とは異なり、step・head集約・prototypeの人物間競合が有力な原因候補です。穴はmaskの幾何から測定した領域で、独立した顔labelではありません。スコアは3役割で正規化したheuristicで、全tokenに対する本来のattention確率や校正済みconfidenceではありません。保存されたensemble maskにも抜けが残り、単純平均での解決は確認できません。

step 3は未観測です。この実験collectorは現状8-stepの先頭2 forwardに固定されており、同じk3方式のstep 1/2/3比較にはcollectorの対応とComfyUI再起動が必要です。通常auto samplerでは収集stepを変えられますが、そのclassic mask方式との比較をk3のstepだけの比較として扱いません。

### 2026-10-12：中断したstep/block検証の完了

追加済みの共有3-forward sweep collectorが起動中のComfyUIに読み込まれていることを確認し、再起動せずにstep1/2/3・block14〜27・OFF/ON・2構図の168条件を検証しました。[結果](attention-step-block-sweep.md)と[比較レポート](attention-step-block-sweep-results/20261011T170454Z/report.html)に記録しています。上の未観測・再起動必要という記述は前回試行時点のものです。

公園の旧70セルの穴はstep1/block18で49セルが対象へ戻りますが、生成画像の口・顎には継ぎ目が残ります。両構図のstep2/block18対照は前回Cと最終latentまで完全一致しました。11編集画像・188 NFEの保存監査は成功。マスク・予測の統制が成立したことと、顔・全身・保護側の画質が合格することは分けて扱います。
