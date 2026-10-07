# FreeFuse-for-sliderlora

Krea2用のComfyUI V3カスタムノードです。男女2人を一つの場面として描きながら、指定した人物（初期例は女性）にだけSlider LoRAの差分を適用します。男性もマスク推定に参加しますが、LoRAは持ちません。

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

## 最初の実機確認

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

Mask Previewの末尾2出力は、autoの生類似度mapを独立min-max正規化した`target_similarity`/`protected_similarity`です。明るさは校正されたconfidenceではありません。manualではこの2出力は黒で、診断にraw mapがないことを記録します。対応したプレビューを追加済みの比較workflowを使用してください。

全体promptは1本で、男女・背景・照明をまとめて記述します。例えば`adult woman in a sage-green top`と`adult man in a blue T-shirt`を含む文章を使い、その語句をSubjectsへそのまま入力します。重複するphraseは0始まりの`occurrence`で指定します。性別の自動判定は行いません。targetはユーザーが指定した人物です。

positiveは専用EncodeからSamplerへ直接接続してください。途中の別ノードでconditioningのlistを作り直すと、token位置との対応を保証できないため拒否します。negativeはテンプレートのConditioningZeroOutを使用します。

自動モードでは局所Sliderを無効にした序盤のattentionから、女性・男性の概念mapを観測します。背景を含めて排他的なマスクを作り、同じ初期ノイズ・latent・全sigma列から生成を再開します。Phase 1の途中画像を引き継ぎません。モデルの全体attention、位置情報、全体用LoRAは維持します。

各対象Linearで`base(x) + strength × 女性mask × Slider差分`を計算します。Krea2のtext/image混合列のうち、テキスト位置への直接差分は0です。通常の全体LoRAとは適用範囲が異なるため、全面IMAGEマスクでも通常Loaderの結果との完全一致は要求しません。

## 設定と制約

- 静止画1枚、batch1、空のtxt2img latent、Euler/simple、CFG1、denoise1が対象です。画像入力latent、noise mask、動画、ControlNet、reference latentは初期対象外です。
- 初期対応Sliderは学習ノードが出力する`lora_unet_blocks_N_attn_{wq,wk,wv,gate,wo}`のdown/up/alpha形式です。gateも適用します。`target=all`、DoRA/LoKr、text encoder LoRA、別形式、未対応キーを含むものは説明付きで拒否し、部分適用しません。負強度は設定できますが、単方向学習Sliderの逆方向品質は別途評価してください。
- ベース演算は通常のLinear/ネイティブ量子化処理を呼び、ベース重みのマージ・全体展開・再量子化を行いません。現在の対応検査は浮動小数点または`int8_tensorwise`（ConvRotを含む）です。他のquant_formatは初期版で拒否します。
- 自動maskはpatch grid単位で、人体の画素セグメンテーションではありません。1024pxの通常Krea2なら64×64 token gridです。性質の似た人物、接触・重なり、体格変化による人物移動では失敗する可能性があります。
- `collect_step=2`は1始まり、`collect_block=18`は0始まりです。temperature4000/top_k_ratio0.3は初期比較値で、実モデルでの最適値ではありません。空・拡散・非有限mapでは停止し、全画面maskへ置き換えません。
- maskの面積を男女50:50に強制しません。同点や低いforeground信号は背景へ回します。初期版は膨張・feather・attention biasを行いません。
- 通常Loaderの全体Slider二重適用を、ファイル名だけで完全には検出できません。指定された上流MODELには全体スタイル用LoRAだけを置いてください。
- 未知のtoken/attention patch、native hooks/injections、raw module hook、個別forward置換、model_function_wrapper、複数GPUを実行前に拒否します。同じmodel coreを使う本拡張の並列実行も拒否します。他のSamplerを同じcoreで同時実行する組合せは対応外です。
- 終了・中断・例外時は注入を解除し、専用cloneと同じcoreのモデルをunloadします。そのため次の生成でモデル再ロードが発生する場合があります。

## 実機用の検証ツール

**ComfyUIが使っているPython**で実行します。`comfy_root`は`comfy/ldm/krea2/model.py`があるソースのルートです。

```powershell
python -B tools/validate_comfy.py C:/path/to/ComfyUI
python -B tools/probe_krea2_slider.py --comfy-root C:/path/to/ComfyUI --model C:/path/to/krea2.safetensors --lora C:/path/to/slider.safetensors
```

validatorは実native Krea2の小型CPUモデルとV3 schemaを6件検証します。missing importやskipを合格にしません。probeは実チェックポイントとSliderを読み込み、全キー照合と投影種別ごとの代表層で明示的な差分計算との一致・対象への非ゼロ変化・外側差分・量子化データの保持・forward復元を確認します。ゼロ差分や、反復誤差以下しか変化しない場合は合格にしません。probeの合格も実画像の成功を意味しません。モデル・依存を自動取得する処理はありません。

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
