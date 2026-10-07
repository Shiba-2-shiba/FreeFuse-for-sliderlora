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
