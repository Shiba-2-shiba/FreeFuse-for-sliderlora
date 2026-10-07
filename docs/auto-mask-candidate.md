# 自動マスクの改善候補: seed42

ユーザー提供の`krea2_female_slider_auto_00014_.png`と`ComfyUI_temp_jsbmx_00016_.png`は同じprompt metadataを持ち、後者はtarget maskの出力である。ユーザーは今回の設定でマスクが改善し、Sliderの効果がある印象と報告した。

この設定を、単一seedでの改善候補として保存する。全seedの推奨値・品質合格と扱わず、既定値は変更しない。提供PNGそのものは公開リポジトリに含めない。

## 記録した設定

| 項目 | 候補値 |
|---|---|
| target / protected phrase | `woman` / `man` |
| Slider | `Krea2/krea2_deaging_20261001T075826Z_c06cb035.safetensors` |
| Slider SHA256 (PNG fingerprint) | `6d3f75e1ab9b9a13fea094177fc80bd2dbc598a0b5bacc5ce05eddf6d79e4ecc` |
| strength | 4 |
| seed / steps / CFG | 42 / 8 / 1 |
| sampler / scheduler | Euler / simple |
| mask_mode | auto |
| collect_step / collect_block | 2 / 18 |
| top_k_ratio / temperature | 0.2 / 10000 |
| 全体用darkbrush | 0.8 |

全体prompt、checkpoint、CLIP、VAEは先の比較と同じ。最初の失敗例と比べ、Sliderファイル、phrase、top_k_ratio、temperatureが変わっている。複数の変更があるため、temperatureなど一つの要素を原因と断定しない。

## マスクで確認した改善

| 指標 | 先の失敗例 | 今回 |
|---|---:|---:|
| 白token数 / 4096 | 477 | 583 |
| 白率 | 11.65% | 14.23% |
| 8連結の断片数 | 51 | 35 |
| 最大断片 / 白全体 | 50.73% | 91.42% |
| 同一画像座標の旧顔ROI x300–405/y145–265 | 0% | 48.44% |

今回の生成画像の顔位置は変化している。目視した現在の顔の概算ROI x300–415/y180–310では白率66.67%、頭部の概算ROI x260–450/y120–325では57.69%。これらは目視ROIの概算であり、人体セグメンテーションの評価値ではない。異なる顔位置のROI値を同じ領域の比較として扱わない。

画像では左側の顔・体格に変化が見られ、背景・床・照明は一つの場面としてつながっている。右側に明瞭な同程度の変化は見られないが、strength0との統制比較がないため、男性への漏れなし・効果量・単調性を確定しない。

## 次の確認用ワークフロー

- [候補 strength4](../workflows/krea2_female_slider_candidate_strength4.json)
- [同条件 strength0](../workflows/krea2_female_slider_candidate_strength0.json)

同じSlider、seed、prompt、収集設定を使い、strengthだけを変える。保存prefixは比較用に区別している。両方とも二値maskと連続類似度mapのPreviewを接続してある。API版は同名の`_api.json`。

まずseed42で0/4を比較し、その後 `123 / 777 / 4444 / 4444444444` でも同条件で確認する。自動maskが顔まで届くこと、女性に効果があること、男性の同じ属性が大きく変わらないこと、画像全体の調和を確認する。別seedでmapが拡散・空になる可能性は残る。

次はこの組合せの再現性とmaskの安定性を確認し、既定値や生成方式の調整を判断する。
