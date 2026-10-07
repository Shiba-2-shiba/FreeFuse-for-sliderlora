# 実機確認

確認対象: Krea2、attention-target Slider 1本、男女各1人、Euler/simple、8 steps、CFG1、1024×1024。

## 接続・互換性

- [ ] 通常生成が動くモデル・CLIP(type=krea2)・VAEを選んだ。
- [ ] 4ノードが検索で見つかり、workflowを保存して再読込できる。
- [ ] `tools/validate_comfy.py`のnative6件がskipなしで成功した。未実施/失敗を成功と混同しない。
- [ ] `tools/probe_krea2_slider.py`で実Sliderの全キーが対応し、代表層のoutside誤差とpacked weight保持が合格した。
- [ ] Samplerに選ぶ女性Sliderを通常LoRA Loaderでも全体適用していない。
- [ ] 最初は手動workflowでMASKと実人物位置が対応することを確認した。

## 比較表

prompt/model/CLIP/VAE/darkbrush0.8/解像度/stepsを同じにし、次の条件を比較する。Aは通常KSamplerでSliderなし、Bは通常LoRA Loaderで同じSliderを全体適用、Cは本Samplerで女性限定。まず+1を使い、必要なら単体で確認済みの強度へB/C双方を揃える。

| seed | A: 人物数・構図 | B: 女性/男性への効果 | C: 女性への効果 | C: 男性への同属性漏れ | C: 欠損/継ぎ目/縮尺/照明 | C: mask位置 |
|---|---|---|---|---|---|---|
| 42 | | | | | | |
| 4444 | | | | | | |
| 4444444444 | | | | | | |
| 123 | | | | | | |
| 777 | | | | | | |

Bでも男性が変化しない属性だけでは、男性保護の有効性を評価できない。Aで2人が成立しないseedはそのまま記録し、基礎生成と局所適用の失敗を分ける。初期採用目安はCの4/5seed以上で「男女各1人・女性の効果・男性への明瞭な同属性漏れなし・新たなタイル状分断なし」。5seedから一般的な成功率は推定しない。

## 動作継続・復帰

- [ ] 同じseedのstrength0と通常生成を比較した。
- [ ] 本Sampler→通常KSampler→本Samplerの順に実行し、通常生成へSliderが残留しない。
- [ ] 中断後の通常生成と再実行が動く。
- [ ] debugログでmatched_modulesとreached_modulesが一致する。
- [ ] adapter_groupsがtarget1/protected0/background0で、owned_hooks_removedがtrue。
- [ ] 自動版で両人物と背景のMASKを確認した。女性のeffectと同時に人物がmask外へ動いていない。
- [ ] 初回ロードとwarm runの時間、Phase1/2評価回数、VRAMを別記した。

PNGの生成metadataと、対応するrun_idのログを一緒に保存する。失敗時はComfyUI/extensionのcommit、モデルのquant_format、Slider形式、manual/auto、収集step/blockを添える。

## 0.1.2のmask後処理

- [ ] [処理なし](../workflows/krea2_female_slider_postprocess_off.json)と[穴埋め8](../workflows/krea2_female_slider_postprocess_fill8.json)を同条件で比較した。
- [ ] original_target_mask、処理後target_mask、added_target_maskを確認した。
- [ ] protectedが処理前後で同じで、mask_postprocess.protected_changed_token_count=0/partition_valid=trueを確認した。
- [ ] 穴埋め後も女性への効果があり、男性の同属性変化・背景破綻が増えていない。
- [ ] 膨張1を試す場合は最大成分だけが広がり、小断片が拡大しないことを確認した。
- [ ] 旧manual/auto workflowは設定省略または両方0で動き、UI保存・再読込後も設定が維持される。

## 0.1.3の対象文章行への適用

- [ ] [比較版](target-text-routing.md)の`target_text_scale=0 / 0.5 / 1`を同じ画像strength・seed・promptで生成した。
- [ ] target/protected mask、initial_noise/full_sigmas hash、収集条件が3版で同じことを確認した。
- [ ] 女性への効果と男性の同属性変化、構図・照明・人物欠損を比較した。
- [ ] 通常全体適用の参照版で同じSlider・強度を一度だけ使用した。
- [ ] 診断のtarget/protected token位置とtext policy、実行回数を確認した。
- [ ] 旧workflowの省略時は0、UI保存・再読込後は指定倍率が維持される。
