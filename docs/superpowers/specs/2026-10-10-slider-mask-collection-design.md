# Sliderマスク収集3方式の比較設計

2026年10月10日　仕様と比較workflow　実機未検証

## 今回の範囲

既存ノードを組み合わせ、収集元だけを変えた3方式を比較する。今回の追加は仕様とworkflowで、Sampler本体は変更しない。後述のcollection_modeとschema 4は将来の統合案として区別する。

目標は対象女性の顔と全身体型が一貫して変化し、古い顔・身体・服との融合や増殖を減らすこと。男性の顔・身長・体格を維持し、接触に必要な手・前腕の軽微な変化は目視評価で許容する。腕だけ保護を解除する機構は追加しない。

基準はGitHub devの`4c53e66752d8beafb40c64c1402d74089bba60ae`。主要7ファイルは添付コードと同一。コードの経路は確認したが、このworkflowの実機動作・画像品質は未検証。

## 3方式の意味

baseは全体styleを含み、対象Sliderを加えていないモデル。sliderはbaseへ通常Loader相当のSliderを1回適用する。on収集と本生成は同じ強度を使う。

両経路は同じ初期noise・latent・conditioning・seed・全sigmasから開始する。先頭からcollect_step回Eulerで評価し、指定blockを既存hookで観測する。collect_step=2は第2回評価中で、更新後の画像ではない。Slider-onは最初からSliderが有効な独立軌道とする。

- **base**：baseのtarget/protected raw mapから、現行方式でT0/P0/B0を得る。
- **slider**：Slider-onの両raw mapから、同じ方式でT1/P1/B1を得る。
- **hybrid**：両経路で二値partitionを独立に生成し、P=P0、T=T1 AND NOT P0、B=NOT (T OR P)で統合する。

初回は穴埋め0・参照膨張0・選択余白4を3方式で固定する。収集元と余白の最適化を同時に変えない。

hybridは推定P0を優先するため、体格変化を妨げる可能性も評価する。異なる軌道のraw mapを直接組み合わせる再分類はP0を維持できないため採らない。

T0、T1、衝突領域T1 AND P0、旧領域のみT0 AND NOT T1、統合target、実効maskを保存・表示する。旧targetは記録用とし、和集合化・自動消去・背景補完はしない。既存original_target_maskの「後処理前target」という意味も維持する。

## 現行ノードでの比較workflow

同じgraph内で2つの収集Samplerを共有し、3つの本生成へ接続する。

1. style付きbaseを用意し、別経路だけにLoraLoaderModelOnlyでSliderを強度4で追加する。
2. 2つのKrea2SliderFusePredictionMixSamplerを収集用に置く。両方auto、Sampler自身のstrength=0、mix_scope=none、選択余白0とする。MODELは一方をstyle付きbase、他方をstyle＋Sliderへ接続する。追加Sliderが有効なのは後者の上流MODELだけ。
3. 各Samplerのmask_bank出力slot 1をMaskPreviewへ接続する。targetはslot 0、protectedはslot 1。base方式はT0/P0、slider方式はT1/P1を使う。
4. hybridのtargetはMaskCompositeのdestination=T1、source=P0、x=y=0、operation=subtractで作る。標準ノードは減算後に0〜1へclampするため、二値maskではT1 AND NOT P0と一致する。protectedはP0を使う。
5. 3組を別々のSubjectsへ渡し、3つの本生成Samplerをmanual、strength=4、mix_scope=target_mask、選択余白4にする。MODELはすべて元のstyle付きbase、LATENTは同じ空latentへ接続する。収集Samplerの出力latentは使わない。

現行の入力検証は上流の標準weight LoRAを許可し、strength=0ではSampler側Sliderを適用しない。本生成のMODELは元のbaseを使う。

収集用Samplerはmask収集後に不要な8step生成も実行する。収集2＋不要生成8を2経路、本生成16を3方式で、合計68 NFE。単独実行のbase/slider各26、hybrid36は、不要な出力・保存枝をすべて外すか無効化した場合だけ。配布graphは収集mask保存が両経路を実行させる。モデル評価数であり、時間の保証ではない。cache利用時は実費が変わる。新しい反復測定では収集用を含め各trial_idを更新する。

## 保存と比較で守ること

本生成の3経路にはDiagnosticSaveを使える。収集用のSlider-on経路には使わない。現行の診断provenanceは「全体style Loaderが1つ」を要求し、上流にstyleとSliderを持つ経路を拒否するため。

収集元T0/P0/T1/P1、hybrid、衝突、旧領域の二値maskはMaskToImage→SaveImageで保存する。配布manifestは初期設定であり、UI編集後の実行証明ではない。実際にqueueへ送ったAPI、PNG metadata、trial_id・prefix・出力対応を残す。二値PNGや正規化Previewを生raw tensorと扱わない。

本生成の診断はmanualとして保存され、同入力のinside=Slider/outside=baseを監査できる。maskの宣言した収集元、生raw map、統合過程はschema 3単体では証明・再構成できず、将来のschema 4監査とは区別する。

現行比較CLIはpartition一致を要求し、3方式の診断JSONの比較を拒否する。既存検証は緩めず、画像・mask比較と各runの監査を分ける。同一maskのmanual再生では全step予測・latent一致を確認する。

## 最初の評価と復帰確認

男性手前の重なりseed42、公園seed444444を優先し、元prompt seed42を維持確認に加える。各ケース内でprompt、強度4、8step、収集step2/block18、top_k0.2、temperature10000と前述の余白を固定する。姿勢seed444444は基準の頭部cropを別記し、顔の効果を採点しない。有望なら強度2、既存のもう一方のseed、未使用seedへ広げる。

maskは全step固定で追跡しない。直接選択監査が成功しても、選択外の完成画素の不変は保証されない。推定maskと実人物の保護は別判定とする。

共有coreの復帰は実機確認が必要。on収集と各本生成の成功・失敗・キャンセル後に、style付きbaseの予測・patch構成、残留hook、標準KSamplerの前後一致を確認する。clone破棄だけで復帰済みと判断しない。空・非有限map、空target/protected、hybrid衝突後の空targetはエラーとし、baseや全画面maskへ黙って戻さない。

## 将来のSampler統合案

比較で有効性を確認した後の候補。今回のworkflowを動かす条件ではない。

- 末尾にcollection_modeを追加し、baseを既定値にする。既存node ID・出力・入力順・manual動作を維持し、manualとslider/hybrid併用は拒否する。
- 収集後は元のnoise・latent・全sigmasから再始動する。非ゼロならnone/allでも指定方式を収集する。strength=0だけbase収集1回へまとめ、両源の共有を明記する。収集2・本生成8の部分混合はbase/slider各18 NFE、hybrid20。none/allは各10、hybrid12。強度0は全方式10。
- shared-core guard内でpatcher・current_patcherを確認し、on収集後にstyle付きbaseを再有効化する。全終了経路でunload・hook除去・resetを試み、元の例外を保つ。
- 新規出力はschema 4とし、schema 1〜3の検証も保持する。各源のbranch・強度・NFE・観測step/block/sigma・token位置・初期入力hash・full/prefix sigmas・両raw map・統合規則を保存する。readerはpartitionから最終maskまで再構成し、不正参照・欠落・NFE改変をsummaryでも拒否する。
- 専用比較で許す差は収集方式・由来・回数・mask関連だけ。モデル・style・Slider/強度・prompt/conditioning・seed・初期入力・sigmas・観測設定・後処理・余白・scope・環境・実装を一致させる。明示したbase runを領域評価の共通参照とし、既存の同条件比較と--same-mask-referenceは緩めない。

統合時はmask_collection.py、native_pair.py、masks.py、diagnostics.py、nodes.py、比較ツールを変更する。attention・二値混合の数式は維持する。CPU検証、native復帰、UI互換・再読込は別々に確認する。新モデル・依存は追加しない。

## 根拠

- [収集と再始動](https://github.com/Shiba-2-shiba/FreeFuse-for-sliderlora/blob/4c53e66752d8beafb40c64c1402d74089bba60ae/slider_fuse/native_pair.py)
- [現行診断の保存とprovenance](https://github.com/Shiba-2-shiba/FreeFuse-for-sliderlora/blob/4c53e66752d8beafb40c64c1402d74089bba60ae/slider_fuse/diagnostics.py)
- [ノード入力と出力slot](https://github.com/Shiba-2-shiba/FreeFuse-for-sliderlora/blob/4c53e66752d8beafb40c64c1402d74089bba60ae/nodes.py)
- [評価環境のMaskComposite](https://github.com/Comfy-Org/ComfyUI/blob/15ef24d1c0333a3eba56c5cd153d8db65363ff8f/comfy_extras/nodes_mask.py)
