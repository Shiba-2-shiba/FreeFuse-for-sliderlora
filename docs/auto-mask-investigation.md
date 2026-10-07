# 自動マスクの実機結果と切り分け

手動で効果が出たSliderを基準に、同じファイル・強度2・seed42へ揃えた比較を行う。今回は診断と比較条件の更新であり、自動マスクの画質改善を実機で確認したリリースではない。

## 提供結果で確認した事実

- auto画像とtarget previewは同じprompt metadataを持つ。previewは実際に`MaskPreview`のtarget出力へ接続されている。
- target maskは64×64、白477 token（11.6455%）、8連結の断片51個、最大断片242 token（白全体の50.73%）。
- auto画像で目視した女性の顔の概算範囲x300–405/y145–265には白tokenが0。頭部の概算範囲x275–425/y85–280の白率は約2.3%。ROIは目視の概算であり、人物検出や正確なセグメンテーションではない。
- 画像のmetadataでは、manualは`Krea2/krea2_deaging_20261001T075826Z_c06cb035.safetensors`のstrength2、autoは`Krea2/krea2_deaging.safetensors`のstrength4。ファイルfingerprintも異なるため、マスク方式だけの比較ではない。
- ユーザーが、manualで効果が出た前者のファイルを比較基準に指定した。

実装はtarget mask=0のimage tokenへ直接Slider差分を加えない。従って顔maskの欠落は、deagingの直接作用が顔へ届かないことと整合する。一方、LoRAの違いと間接attention伝播があるので、画像の無変化をこの原因だけに断定しない。

## 現時点で未確定の境界

元の類似度mapは`raw_maps`に保持されるが、0.1.0のPreviewは二値maskしか出力しない。そのため、attention観測/概念mapが顔を捉えなかったのか、独立min-max正規化とforeground閾値で顔が落ちたのかは未確定。

現在の背景scoreは実背景attentionから観測したものではなく、`1 - max(subject_score)`という初期ヒューリスティックである。mapの面積や位置の診断だけでは意味的な人物領域の正しさを証明できない。二値化の修正や空間割当への置換は、連続mapを確認してから検討する。

## 0.1.1で追加した診断

**Krea2 Slider Fuse Mask Preview**の先頭3出力は従来どおり。末尾に次を追加した。

- `target_similarity`: 女性の生mapを独立min-max正規化したグレースケール表示。
- `protected_similarity`: 男性の生mapを同様に表示。

これらはsegmentationでも校正済みのconfidenceでもない。弱いmapでも表示上は白黒の差が付くため、診断のraw range/meanも見る。manualには生mapがないので、この2出力は黒になり、診断の`raw_maps_available=false`で区別する。

Samplerの診断にmapのmin/max/mean/range、range_over_mean、normalized_entropy、target/protectedのPearson相関を追加した。maskには8連結の断片数、白token数、最大断片サイズ/比率を追加した。これらは観測用であり、mask生成やLoRA routingを変更しない。

## 比較ワークフロー

ComfyUIを再起動して更新したノードを読み込み、以下を順に実行する。全て同じbaseline LoRA、strength2、seed42、8 steps、CFG1、同じ全体prompt/model/styleを使用する。LoRAのファイル内容も揃える。

| ワークフロー | 変更する変数 |
|---|---|
| [compare_manual](../workflows/krea2_female_slider_compare_manual.json) | 左半分のmanual maskで既知の効果を再確認 |
| [compare_auto](../workflows/krea2_female_slider_compare_auto.json) | manualとの比較はmask_modeだけを変更 |
| [compare_auto_subject_words](../workflows/krea2_female_slider_compare_auto_subject_words.json) | autoからtarget/protected phraseだけを`woman`/`man`へ変更。衣服部分への偏りを検証 |
| [compare_auto_step4](../workflows/krea2_female_slider_compare_auto_step4.json) | autoからcollect_stepだけを2→4へ変更。収集時期によるmap変化を検証 |

各workflowに女性/男性の二値maskと連続mapのPreviewが接続されている。API版は同名の`_api.json`。

見る順序は、女性の連続mapが顔まで反応するか → 男性mapとの分離があるか → 二値maskで顔が消えていないか → 最終画像で効果が出るか。連続mapが顔を捉えているのに二値maskで失われるならforeground/割当を見直す。連続map自体が衣服や背景に偏っているならphrase・収集時期・blockを調べる。新たな左右固定割当を人物検出の成功として扱わない。

画像と対応するmask/mapプレビュー、`[Krea2SliderFuse]`のrun_id/diagnosticsを一緒に確認する。元の提供画像そのものはリポジトリに含めない。
