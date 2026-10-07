# 対象phraseへの文章側LoRA差分（0.1.3）

通常生成と比べて局所Sliderの効果が弱いという観測に対し、Krea2内部のtarget phrase行にも同じLoRA差分を追加する実験です。これまでの画像側だけの処理に、任意の文章側経路を加えます。学習済みSliderを作り直す必要はありません。効果の強さと男性への漏れの改善は、実機画像での確認が必要です。

## 設定

Sampler末尾の`target_text_scale`は0〜1です。既定0・旧workflowで省略した場合も0で、従来の画像側だけの適用を維持します。auto/manual双方で使用できます。

| 設定 | 画像側の係数 | target phraseの文章側係数（strength=4の例） |
|---|---|---|
| 0 | strength × target mask | 0 |
| 0.5 | strength × target mask | 2 |
| 1 | strength × target mask | 4 |

実際の文章側係数は`strength × target_text_scale`です。負のstrengthなら文章側も同じ符号になります。alpha/rankは画像側と同じ`Adapter.delta`内で一度だけ適用し、面積・文章行数で正規化しません。

対象はKrea2の結合されたtext/image列を処理する`blocks.N.attn.{wq,wk,wv,gate,wo}`です。CLIP/text encoderの重みは変更しません。target phraseの正確なtoken位置だけを選び、protected phraseとその他の文章行、画像mask外には直接差分を追加しません。phrase位置の重複・範囲外・runtimeの文章境界不一致は停止します。全文章への置き換えや位置の補正は行いません。

autoのPhase 1では画像側・文章側とも局所Sliderは無効です。同一モデル・prompt・seed・収集設定で比較すると、文章倍率を変えても収集の入力条件は変わりません。Phase 2では同じ初期noise/latent/full sigmasから再開し、既存の処理後target maskを使います。strength=0、空target mask、route以外のphaseでは文章側も適用しません。

画像側の直接計算式・倍率・maskは変えません。ただし、文章側の変化は共有attentionを通じて後続層の画像入力にも伝わるため、実際の画像差分と最終画素は変わり得ます。男性の外見変化や構図・照明も併せて評価してください。通常全体適用と同じ効果の強さや完全な人物保護を保証する設定ではありません。

## 比較ワークフロー

以下はUI形式です。同名の`_api.json`もあります。モデル・CLIP・VAE・LoRAファイル名は実機の環境に合わせてください。

- [文章倍率0](../workflows/krea2_female_slider_target_text_0.json)
- [文章倍率0.5](../workflows/krea2_female_slider_target_text_05.json)
- [文章倍率1](../workflows/krea2_female_slider_target_text_1.json)
- [通常全体適用の参照版](../workflows/krea2_female_slider_global_reference.json)

局所3版は、既知のdeaging Slider、画像strength4、seed42、8 steps、Euler/simple、CFG1、darkbrush0.8、`woman/man`、collect_step2/block18、top_k_ratio0.2、temperature10000、fill8/dilate1で揃えています。文章倍率と保存prefixだけを変えています。まず0→0.5→1で、女性の変化と男性への影響を比較します。良い条件は同じpromptで別seed・動作・重なりのある場面でも確認してください。

通常参照版は同じEncodeと生成設定を使い、標準`LoraLoaderModelOnly`で同じSliderを強度4で一度だけ全体適用し、標準KSamplerで生成します。男性も適用対象です。この参照版には局所Sampler・自動maskはありません。局所版のMODELに同じSliderの全体適用を追加しないでください。

## 診断

既存のnoise/sigmas hash・mask診断に加え、次を出力します。

- `target_text_scale` / `target_text_effective_strength`: 指定倍率と文章側係数。
- `target_text_positions` / `protected_text_positions`: Encodeで解決したtoken位置。
- `text_delta_policy`: 倍率0では`zero`、有効時は`target_phrase_only`。
- `other_text_direct_delta_policy` / `protected_text_direct_delta_policy`: ともに`zero`。
- `target_text_linear_calls`: 対象文章行の差分計算を実行したLinear呼び出し数。倍率0・strength0・空maskでは0。

`target_text_effective_strength`は設定値であり、最終画像の変化量ではありません。CPU検証は選択行への明示式との一致、その他行の直接差分0、収集同一性、manual/auto接続、例外時の復元・cache解放を確認します。実INT8/GPU、UI保存・再読込、画像上の効果と漏れは未確認です。
