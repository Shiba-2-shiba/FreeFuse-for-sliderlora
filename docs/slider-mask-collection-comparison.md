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
