# FreeFuse-for-sliderlora

Krea2用のComfyUI V3カスタムノードです。男女2人を一つの場面として描きながら、指定した人物の領域でSlider LoRAの予測を採用します。

**0.2.0：Prediction Mixを推奨経路にしました。** [推奨manual workflow](workflows/krea2_female_slider_prediction_mix_manual.json)は手動mask、[推奨auto workflow](workflows/krea2_female_slider_prediction_mix_auto.json)はSliderなしで収集したmaskを使います。通常利用はsummary、詳細検証はauditに切り替えます。[使い方・診断形式](docs/prediction-mixing.md)と[ユーザーによる手動実機確認](docs/real-machine-checklist.md)を参照してください。

旧版のmanual予測混合では、強い対象側の効果と保護人物の成人性維持を実機で確認しました。0.2.0もAMD R9700 / INT8環境でnative12件、auto/manualの同一mask比較、UI graph再読込、中断後の復帰を確認しました。seed42ではautoの選択余白4/8で効果を確認した一方、余白0では顔が崩れました。他条件への一般化は未確認です。[実機追試の条件と限界](docs/validation.md)を参照してください。対象外の完成画素は固定されません。旧hookノードと旧workflowは比較・互換用に保持しています。

重なり構図の追加評価では、男性が手前の条件でauto余白1〜4すべてに顔の断片が残りました。同条件の手動半分maskではその断片が消えています。余白の調整だけで自動maskの品質が保証されるとは扱いません。[重なり構図の比較記録](docs/validation.md)を参照してください。

U2はユーザー指定で対象外、U4の永続保存・UI再起動後再実行、U5の指定48条件評価、U6の標準KSampler復帰確認は完了しました。姿勢指定の頭部crop/増殖、公園の服と顔の融合などauto画質の未達は残しています。[最新評価と引き継ぎ](docs/validation.md)から状況と証拠を確認できます。

## 旧版の検討経緯

**0.1.4 診断実験版:** [全層auditとnative予測混合](docs/prediction-mixing.md)を追加しました。新しいPrediction Mix Samplerは通常LoRAとbaseの予測を二値maskで選び、画像側hookと比較できます。[zero](workflows/krea2_slider_mix_zero.json) → [none](workflows/krea2_slider_mix_none.json) → [all](workflows/krea2_slider_mix_all.json) → [half](workflows/krea2_slider_mix_half.json)の順で実機確認してください。部分maskは1stepに2経路を直列評価します。既存診断Sampler末尾の`diagnostic_level=audit`で全層・全stepの直接差分を保存します。通常Samplerの生成演算は維持し、実INT8/UI/画質の確認は未完了です。

**診断機能:** 通常Loaderと独自LoRAの演算差、画像全面/左半分、文章なし/対象語/全文を分けて比較する診断専用Samplerと保存ノードを追加しました。[実行順・保存結果・比較ツール](docs/diagnostic-parity.md)を参照してください。11条件のUI/API workflowがあります。全面/全文条件は保護対象にも作用します。実INT8・UI・画像品質の評価はユーザーの実機で行います。通常Samplerの入力・既定動作は維持しています。

**0.1.3:** [対象phraseへの文章側LoRA差分](docs/target-text-routing.md)を任意で追加しました。Sampler末尾の`target_text_scale`は既定0で従来動作を維持します。同じ画像側強度・seed・mask条件の[0](workflows/krea2_female_slider_target_text_0.json)・[0.5](workflows/krea2_female_slider_target_text_05.json)・[1](workflows/krea2_female_slider_target_text_1.json)と、[通常全体適用の参照版](workflows/krea2_female_slider_global_reference.json)を用意しました。効果の回復と男性への影響は実機で未確認です。

**0.1.2:** [保護付き小穴充填と最大成分だけの膨張](docs/mask-postprocessing.md)を追加しました。Sampler末尾の`fill_holes_max_area`/`mask_dilate_radius`は既定0です。最初は[穴埋め8・膨張0の比較版](workflows/krea2_female_slider_postprocess_fill8.json)を、[処理なし](workflows/krea2_female_slider_postprocess_off.json)と比較してください。新後処理の実機画質は未確認です。

**0.1.1:** 手動で効果が出る一方、自動maskが顔を覆わない実機結果を受けて、[同じSlider・強度での比較ワークフローと連続mapの診断](docs/auto-mask-investigation.md)を追加しました。自動maskの画質改善が確認された更新ではありません。

その後、`woman/man`、top_k_ratio0.2、temperature10000、既知のSlider強度4で、seed42のマスク改善と効果が報告されました。[候補設定と同条件の強度0/4比較](docs/auto-mask-candidate.md)を保存しています。単一例なので既定値は変更していません。

**実験版です。** ユーザーの実機からmanual/auto生成結果を受領し、manualの効果とautoの顔mask欠落を確認しました。同じLoRA・強度による統制比較、自動マスクの品質合格、ConvRot INT8のprobe結果は未確認です。CPUの数値・状態・接続テストと公式ソース照合を行っています。男性への直接LoRA差分は0にしますが、共有attentionを通じた間接的な属性変化や、領域外の画素変化は起こり得ます。

## 導入

ComfyUIの`custom_nodes`内で実行し、ComfyUIを再起動します。

```sh
git clone --branch dev https://github.com/Shiba-2-shiba/FreeFuse-for-sliderlora.git
```

Krea2とV3 APIに対応したComfyUI、通常のComfyUI環境にあるPyTorchとsafetensorsを使います。この拡張用にtorchや量子化ライブラリを更新する処理はありません。FreeFuse本体・Anima版・Slider学習ノードは実行時依存ではありません。SliderのファイルをComfyUIのLoRA検索対象（通常は`models/loras`）へ配置してください。

ネイティブ処理はComfyUI commit `3d9b2d551788d4fe80ede5743417077d1795cbd2`の公式ソースに照合しました。別commitでは後述のvalidatorと実機probeを実行してください。

## Prediction Mixを使う

1. [manual版](workflows/krea2_female_slider_prediction_mix_manual.json)を読み込み、モデル、Krea2 CLIP、VAE、styleとSliderのファイル名を環境に合わせます。
2. `Krea2 Slider Prediction Mix Sampler`の`lora_name`で局所用Sliderを指定します。同じSliderを上流の通常LoRA Loaderにも重ねないでください。
3. 左半分のtarget、右半分のprotectedが実際の人物を覆うことを確認し、同じseedのstrength0/2/4を比較します。
4. [auto版](workflows/krea2_female_slider_prediction_mix_auto.json)はSubjectsの両MASK入力を未接続にします。収集値は改善候補で、最適値ではありません。顔・全身がtarget maskに含まれるか確認してください。
5. 通常の保存はsummary。数値比較するときはauditにして、PNG・実効mask・safetensors・JSONを元の名前で一組に保存します。

旧`krea2_slider_mix_zero/none/all/half`はaudit付きmanual比較のままです。旧ノードを読み込んだだけで生成方式が切り替わることはありません。Prediction Mixでは文章側のSlider作用はnative経路に含まれるため、旧hookの`target_text_scale`は使用しません。

API形式：[manual API](workflows/krea2_female_slider_prediction_mix_manual_api.json)、[auto API](workflows/krea2_female_slider_prediction_mix_auto_api.json)。部分maskは1stepに2経路、autoはさらにcollect_step回の予備評価が必要です。

autoの`selection_dilate_radius=4`は、参照マスクを変えず選択範囲だけ背景へ広げる比較開始値です。通常Krea2で約64px。0/4/8を比較し、実効予測maskと追加余白のPreviewを確認してください。旧workflowの省略時は0です。

lowvram/offload環境では、経路を切り替えるたびの重み再適用とロードが負担になり、実測例の約2倍より遅くなる場合があります。生成速度は保証していません。

## 旧hook方式の実機確認

1. 通常のKrea2生成が動く環境で、[手動版ワークフロー](workflows/krea2_female_slider_manual.json)を読み込みます。
2. モデル、CLIP、VAE、darkbrushを自分のファイルへ合わせます。テンプレートは現在の比較環境の`intorealismAsian_k2JAVFLASHV1.safetensors`、`wen3vl_4b_bf16.safetensors`、`qwen_image_vae.safetensors`を指定しています。CLIPのtypeは`krea2`です。
3. **Krea2 Slider Fuse Samplerの`lora_name`で、学習済みのattention-target Sliderを選択します。** この選択欄は空で配布しています。上流の通常LoRA Loaderでは同じSliderを重ねないでください。darkbrushは全体用として上流に置けます。
4. まず`strength=0`、seed42で生成します。次に同じseedで`strength=1`に変えます。手動版は左半分が女性、右半分が男性のテスト用MASKです。実際の人物位置が異なる場合は、標準MASKノードで人物を覆うマスクへ変更してください。
5. 女性に効果が出るか、男性の同じ属性が変化していないか、人物の欠損・画像の継ぎ目がないかを比較します。
6. 手動版の確認後、[自動版ワークフロー](workflows/krea2_female_slider_auto.json)へ進みます。両人物MASK入力を接続せず、`mask_mode=auto`にします。女性・男性・背景MASKのプレビューを確認します。

API形式の接続例もあります: [手動API](workflows/krea2_female_slider_manual_api.json)、[自動API](workflows/krea2_female_slider_auto_api.json)。ファイル名・Slider選択を編集する必要があるテンプレートです。

## ノードと仕組み

| ノード | 接続・役割 |
|---|---|
| **Krea2 Slider Fuse Encode** | Krea2 CLIPと全体promptからpositive/prompt_infoを生成 |
| **Krea2 Slider Fuse Subjects** | 同じprompt_infoからtarget/protectedの2つのphraseを解決。任意の手動MASKは両方指定 |
| **Krea2 Slider Fuse Sampler** | MODEL、positive/negative、同じprompt_info、subjects、空LATENT、女性用Sliderを入力。LATENT/mask_bank/診断文字列を出力 |
| **Krea2 Slider Fuse Mask Preview** | mask_bankを女性・男性・背景MASKとして出力。標準MaskToImage/PreviewImageに接続 |

Mask Previewの第4・第5出力は、autoの生類似度mapを独立min-max正規化した`target_similarity`/`protected_similarity`です。明るさは校正されたconfidenceではありません。manualではこの2出力は黒で、診断にraw mapがないことを記録します。対応したプレビューを追加済みの比較workflowを使用してください。

0.1.2のPreviewは先頭5出力を維持し、さらに`original_target_mask`（処理前）と`added_target_mask`（追加領域）を末尾へ追加しています。生成に使うtargetは先頭の処理後maskです。

全体promptは1本で、男女・背景・照明をまとめて記述します。例えば`adult woman in a sage-green top`と`adult man in a blue T-shirt`を含む文章を使い、その語句をSubjectsへそのまま入力します。重複するphraseは0始まりの`occurrence`で指定します。性別の自動判定は行いません。targetはユーザーが指定した人物です。

positiveは専用EncodeからSamplerへ直接接続してください。途中の別ノードでconditioningのlistを作り直すと、token位置との対応を保証できないため拒否します。negativeはテンプレートのConditioningZeroOutを使用します。

自動モードでは局所Sliderを無効にした序盤のattentionから、女性・男性の概念mapを観測します。背景を含めて排他的なマスクを作り、同じ初期ノイズ・latent・全sigma列から生成を再開します。Phase 1の途中画像を引き継ぎません。モデルの全体attention、位置情報、全体用LoRAは維持します。

各対象Linearで画像側は`base(x) + strength × 女性mask × Slider差分`を計算します。`target_text_scale=0`ではテキスト位置への直接差分は0です。0より大きい場合は、同じKrea2内部Linearのtarget phrase行にだけ`strength × target_text_scale × Slider差分`を追加します。CLIP/text encoderへの適用ではありません。protectedとその他の文章行への直接差分は0です。通常の全体LoRAとは適用範囲が異なるため、全面IMAGEマスクでも通常Loaderの結果との完全一致は要求しません。

## 設定と制約

- 静止画1枚、batch1、空のtxt2img latent、Euler/simple、CFG1、denoise1が対象です。画像入力latent、noise mask、動画、ControlNet、reference latentは初期対象外です。
- 初期対応Sliderは学習ノードが出力する`lora_unet_blocks_N_attn_{wq,wk,wv,gate,wo}`のdown/up/alpha形式です。gateも適用します。`target=all`、DoRA/LoKr、text encoder LoRA、別形式、未対応キーを含むものは説明付きで拒否し、部分適用しません。負強度は設定できますが、単方向学習Sliderの逆方向品質は別途評価してください。
- ベース演算は通常のLinear/ネイティブ量子化処理を呼び、ベース重みのマージ・全体展開・再量子化を行いません。現在の対応検査は浮動小数点または`int8_tensorwise`（ConvRotを含む）です。他のquant_formatは初期版で拒否します。
- 自動maskはpatch grid単位で、人体の画素セグメンテーションではありません。1024pxの通常Krea2なら64×64 token gridです。性質の似た人物、接触・重なり、体格変化による人物移動では失敗する可能性があります。
- `collect_step=2`は1始まり、`collect_block=18`は0始まりです。temperature4000/top_k_ratio0.3は初期比較値で、実モデルでの最適値ではありません。空・拡散・非有限mapでは停止し、全画面maskへ置き換えません。
- maskの面積を男女50:50に強制しません。同点や低いforeground信号は背景へ回します。任意の後処理は小穴充填・最大成分膨張だけで、feather・attention biasは行いません。
- 通常Loaderの全体Slider二重適用を、ファイル名だけで完全には検出できません。指定された上流MODELには全体スタイル用LoRAだけを置いてください。
- 未知のtoken/attention patch、native hooks/injections、raw module hook、個別forward置換、model_function_wrapper、複数GPUを実行前に拒否します。同じmodel coreを使う本拡張の並列実行も拒否します。他のSamplerを同じcoreで同時実行する組合せは対応外です。
- 終了・中断・例外時は注入を解除し、専用cloneと同じcoreのモデルをunloadします。そのため次の生成でモデル再ロードが発生する場合があります。

## 実機用の検証ツール

**ComfyUIが使っているPython**で実行します。`comfy_root`は`comfy/ldm/krea2/model.py`があるソースのルートです。

```powershell
python -B tools/validate_comfy.py C:/path/to/ComfyUI
python -B tools/probe_krea2_slider.py --comfy-root C:/path/to/ComfyUI --model C:/path/to/krea2.safetensors --lora C:/path/to/slider.safetensors
```

validatorは実native Krea2の小型CPUモデルとV3 schemaを12件検証します。missing importやskipを合格にしません。probeは実チェックポイントとSliderを読み込み、全キー照合と投影種別ごとの代表層で明示的な差分計算との一致・対象への非ゼロ変化・外側差分・量子化データの保持・forward復元を確認します。ゼロ差分や、反復誤差以下しか変化しない場合は合格にしません。probeの合格も実画像の成功を意味しません。モデル・依存を自動取得する処理はありません。

生成ログの`[Krea2SliderFuse]`にはrun_id、キー一致/到達数、adapter_groups（1/0/0）、MASK範囲、実sigma/block、Phase 1/2の評価回数、時間、解除状態が出ます。CUDA peakはプロセス全体の値で、このノードがresetした値ではありません。診断文字列もSamplerの出力から取得できます。

0.1.1の診断にはSliderファイル名、mapのraw range/mean・normalized entropy・相関、maskの8連結断片数も含まれます。連続mapと二値maskを比較し、どの段階で顔が落ちるかを確認できます。

比較はseed `42 / 4444 / 4444444444 / 123 / 777`で、Sliderなし・通常全体適用・女性限定適用を同条件で行います。詳しくは[実機確認チェックリスト](docs/real-machine-checklist.md)を参照してください。

## 開発検証と現在の証拠

```powershell
python -B -m pytest -q -p no:cacheprovider tests
```

通常suiteはComfyUI境界のtest doubleを含むCPU検証です。`tests/native_checks.py`は専用validatorだけで実行され、test doubleによる代用はありません。

開発環境: Python3.10.11、torch2.10.0+cpu、CUDAなし。公式ComfyUIソースの検証用checkout `b26625f23a888367b92153b28d93e159e83e677b`では、既存`comfy_aimdo.storage`不足と`comfy_kitchen`のConvRot API版差によりimportが停止しました。依存は変更していません。ネイティブ6件、実INT8 probe、UI上の保存/再読込、実画像品質は未検証です。状態は[検証記録](docs/validation.md)に記載します。

実装は[計画書](.omx/plans/2026-10-07-krea2-female-only-slider-freefuse.md)に対応します。FreeFuseのアルゴリズム、Anima版の二段階生成・復帰設計を参考にしています。Apache-2.0の出典・変更点は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。

## ライセンスと出典

このリポジトリのソースは[Apache-2.0](LICENSE)です。[FreeFuse](https://github.com/yaoliliu/FreeFuse)と[FreeFuse-for-anima](https://github.com/Shiba-2-shiba/FreeFuse-for-anima)からの派生・改変箇所、固定コミット、元の権利表示は[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)と[NOTICE](NOTICE)に記載しています。独立した派生実装であり、上流の公式版・公認版を意味しません。

ソースの再配布では`LICENSE`・`NOTICE`・出典文書と改変表示を保持してください。ComfyUIはGPL-3.0の外部ランタイムであり、組み合わせたプログラムの配布ではGPLの義務も確認する必要があります。モデル・LoRA・データセットは同梱せず、それぞれの配布条件が適用されます。確認範囲と限界は[ライセンス監査記録](docs/license-audit.md)を参照してください。


## Slider ON/OFF mask collection comparison

An experimental three-way comparison uses existing nodes to collect masks from
style-only and style-plus-Slider trajectories, then runs the same manual
Prediction Mix sampler with base, Slider, and hybrid masks. The hybrid keeps
the estimated base protection mask and subtracts it from the Slider target.

- [Comparison protocol and workflow usage](docs/slider-mask-collection-comparison.md)
- [Design specification and future collection-only optimization](docs/superpowers/specs/2026-10-10-slider-mask-collection-design.md)
- [ComfyUI workflow](workflows/krea2_female_slider_collection_comparison.json)
- [API prompt](workflows/krea2_female_slider_collection_comparison_api.json)

This is a workflow-level experiment, not a new runtime collection mode. It uses
68 model evaluations for the shared 8-step three-way graph (cold cache), not the
18/20 evaluations proposed for a later collection-only implementation. Static
graph checks do not establish real-machine compatibility or image quality.

