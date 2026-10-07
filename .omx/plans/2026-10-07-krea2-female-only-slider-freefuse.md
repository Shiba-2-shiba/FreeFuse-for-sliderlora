# Krea2 女性限定 Slider LoRA / FreeFuse 実装計画案

作成日: 2026-10-07

状態: 提案。今回は計画書だけを作成し、実装・依存追加・モデル取得・生成は行わない。

## 目的と前提

男女各1人が同じ空間にいる構図・照明・背景を維持しながら、指定した女性にだけ既存のSlider LoRAを適用する。男性は領域推定には参加するが、LoRAを持たない非適用対象とする。対象は性別を自動判定する分類器ではなく、ユーザーがpromptのphraseで指定する。

ユーザー指定の実装先は `FreeFuse-for-sliderlora`。初期版はKrea2推論専用の独立したComfyUI拡張とする。学習側、既存Regional Attention、FreeFuse、Anima版のソース変更は不要な設計にする。

今回の成功は「モデル内部で男性に直接LoRA差分を加えないこと」と「生成画像で男性の属性変化・場面の分断が抑えられること」の両方で判定する。前者から後者を保証したとは扱わない。

## 調査済み根拠

以下の参照は現在のローカルソースに基づく。実機検証済みという意味ではない。

| ID | 根拠 | 計画への反映 |
|---|---|---|
| S1 | `../../../Confyui-krea2-slider-node/krea2_slider_node/regional_attention.py:63` — 同一owner内の画像attentionだけを許可 | 現行strict分離は新方式に重ねない |
| S2 | `../../../FreeFuse/freefuse_comfyui/freefuse_core/krea2_support.py:57` — Krea2のtext-first配置、txtfusion長、Q/K/V・RoPE・GQA観測 | Krea2固有の収集処理の基礎にする |
| S3 | `../../../FreeFuse/freefuse_comfyui/freefuse_core/token_utils.py:837` — Qwen3-VLのtemplate除去後位置対応 | 実際のconditioningとphrase位置を照合する |
| S4 | `../../../FreeFuse/freefuse_comfyui/freefuse_core/bypass_lora_loader.py:679` — text maskが1、image maskが領域値 | 専用実装はtextへの直接差分を0にする |
| S5 | `../../../Confyui-krea2-slider-node/krea2_slider_node/lora.py:32` — 既定targetはblocks内attention Linear、標準down/up/alpha形式は同ファイル73行以降 | 初期対応キーをwq/wk/wv/gate/woに限定 |
| S6 | `../../../Confyui-krea2-slider-node/krea2_slider_node/model.py:187` — attentionにはwq/wk/wv/gate/woが存在 | gateを取りこぼさず、5種類を検証 |
| S7 | `../../../FreeFuse-for-anima/anima_freefuse/sampling.py:31` — 同じnoise・latent・sigma列のprefix/fullによる再開始 | 二段階samplingの実行契約を参考にする |
| S8 | `../../../FreeFuse-for-anima/anima_freefuse/lora.py:114` — 専用clone、形式検査、PatcherInjection | clone・注入解除の設計を参考にする。Animaの層分類は転用しない |
| S9 | `../../../FreeFuse/freefuse_comfyui/freefuse_core/krea2_support.py:34` — raw hooksはclone間で共有されるmoduleに作用 | cloneだけで安全と判断せず、解除・例外復帰・同時実行拒否を検証 |
| S10 | `../../../FreeFuse/freefuse_comfyui/freefuse_core/attention_bias.py:65` — biasはtext/image対応を補助し、image/imageは中立 | 初期版はbiasなし。必要時の後続比較として分離 |
| S11 | `../../../FreeFuse-for-anima/IMPLEMENTATION_STATUS.md:5` — Anima版の実機画質・互換性は未検証 | 再利用コードの存在を成功実績と混同しない |
| S12 | `../../../ログ１.txt:22` — regional_lora_count=0、後続行でstrict・64×64 token grid | 提供画像はprompt分離の証拠。局所Sliderの品質証拠には使わない |
| S13 | `../../../FreeFuse-for-anima/anima_freefuse/masks.py:35` — 人物間でほぼ等面積となる割当 | 面積強制を初期版へそのまま持ち込まない |
| S14 | `../../../Confyui-krea2-slider-node/docs/training-direction.md:3` — 単方向学習が既定、負強度は逆方向の保証なし | 最初は効果確認済みの正強度1本で評価 |

FreeFuse論文: https://arxiv.org/html/2510.23515v3 （主体LoRA差分routingの参考。Krea2 Slider専用の成功率を示すものではない。）

## 方式の選択

| 案 | 利点 | 問題 | 判断 |
|---|---|---|---|
| Krea2版FreeFuseから必要部分を取り込み、専用拡張にする | 既存Krea2観測を使い、女性だけの適用契約を明確化できる | text側差分・モデル管理・マスク割当を専用化する必要 | 推奨 |
| 既存FreeFuseノードを外部依存として組む | 初期配線は少ない | text全面適用やadapterと人物の結合を専用ノード側だけで制御しにくい | 比較用に留める |
| Anima版全体をKrea2へ置換する | V3ノード・二段階処理の構成を利用できる | T5/cross-attentionの前提が異なり、主要処理を置換する必要 | ライフサイクルとUI構成だけ参考 |

取り込むコードは必要なKrea2観測・類似度計算に絞り、ライセンスと出典を保持する。隣接リポジトリからの実行時importは行わない。新しい外部依存は追加せず、ComfyUI環境の既存パッケージを使う。

## 初期対応範囲

- Krea2、静止画1枚、batch size 1、txt2img、denoise=1。
- Euler / simple / CFG=1。現在の8-stepワークフローを基準とし、step数自体は設定可能。
- 女性1人・男性1人・背景。LoRAは女性に適用する1ファイルのみ。
- 通常の加算型Linear LoRA。初期対応は `blocks.N.attn.{wq,wk,wv,gate,wo}`。
- `target=all`等の対象外キーを含むLoRAは黙って部分適用せず、キー一覧付きで停止。
- 現在使用中のConvRot INT8/mixed precisionを最初の実機対象にする。BF16の小型数値参照も用意する。未知の量子化方式へ自動フォールバックしない。
- darkbrush等、標準LoRA Loaderで適用された全体用LoRAを含むMODELを想定する。Phase 1/2とも同じ全体用LoRAを維持する。初回診断は全体用LoRAなしでも行う。
- 女性用Sliderを上流の通常Loaderでも重ねて全体適用しない。上流パッチのファイル同一性を確実に判定できない場合、この制約と検出限界をUI・診断に明示する。
- 手動マスクと自動マスクの2モード。手動モードでは両人物のマスクを指定し、自動と手動の混在は初期対象外。
- attention bias、領域ごとのprompt分割、領域ごとのlatent crop、RoPE位置変更は初期版に含めない。
- 複数Slider、男性にも別LoRA、動画、img2img、ControlNet、DoRA/LoKr、別architecture、同一model coreの並列samplingは後続対象。

## 処理と不変条件

1. 全体promptは1本。男女、二人の関係、背景、照明を一緒に記述する。従来の矩形内に身体を収める命令は標準テンプレートに引き継がない。
2. 女性・男性のphraseを全体prompt内から選び、実際のKrea2 conditioningに対応するtoken位置を取得する。年齢や胸サイズなどSliderの属性語を人物IDとして使わない。
3. 自動モードのPhase 1では局所Sliderを無効にし、同じモデル・全体用LoRA・promptで序盤のattentionを観測する。女性と男性はLoRAの有無に関係なく収集対象になる。
4. 女性・男性・背景のマスクを生成する。Phase 2は同一初期latent・noise tensor・全sigma列から再開始する。Phase 1の途中latentを引き継がない。
5. 対応する各Linearで `base(x) + strength * mask * LoRA_delta(x)` を計算する。alpha/rankは標準LoRA仕様通りに一度だけ掛ける。
6. Krea2の `[text, image]` 列に対し、最終routing maskは `[textの0列, 女性mask]`。男性・背景・text位置への直接差分は0。attentionの参照先には制限を追加しない。
7. 終了・例外・中断でhook、routing状態、観測cacheを解除する。cloneは元moduleを共有する前提で、既存パッチを復元する。本拡張の実行間では、process内のregistryを実 `diffusion_model` のidentityで管理し、hook登録・forward変更より前に排他的に取得する。同じcoreの2回目の実行は変更前に拒否し、`finally`で自分が所有するhandleのみを解除、元forwardを復元して占有を解放する。別coreは独立に扱う。このregistryへ参加しない外部Samplerの並列実行まで防げるとは扱わず、その組合せは対応外とする。

女性maskは「身体をその領域へ閉じ込める枠」ではなく、LoRA差分の適用領域である。全体attentionを維持するため、男性への間接的な変化は起こり得る。画像の完全な外側不変を仕様にしない。

## ノード案

V3 `ComfyExtension + comfy_entrypoint()` で登録する。初期UIは以下の4ノードと標準VAE/Previewノードで構成する。

| ノードID | 主な入力 | 出力と責務 |
|---|---|---|
| `Krea2SliderFuseEncode` | CLIP、全体prompt | positive、prompt_info。encodeとtoken位置情報を同じ入力から作る |
| `Krea2SliderFuseSubjects` | prompt_info、target_phrase、protected_phrase、各occurrence、任意の両人物MASK | subjects。target/protectedの2主体を登録し、LoRAとは切り離す |
| `Krea2SliderFuseSampler` | MODEL、positive/negative、prompt_info、subjects、LATENT、lora_name、strength、seed、steps、mask_mode、収集設定 | LATENT、mask_bank、diagnostics。LoRA検査・マスク取得・二段階samplingを所有 |
| `Krea2SliderFuseMaskPreview` | mask_bank | 女性MASK、男性MASK、背景MASK。標準MaskToImage/PreviewImageへ接続 |

内部のsubjectsは、安定ID `target` / `protected`、phrase、解決済みtoken位置、任意manual_maskを持つ2レコードとする。Samplerが作る適用対応は必ず `target → [選択したSlider 1本]`、`protected → []`、`background → []`。人物mapの収集対象はこのadapter一覧から作らず、常にsubjectsの両レコードから作る。診断でもこの3者のadapter件数を1/0/0として確認する。

初期例のphraseは `adult woman in a sage-green top` と `adult man in a blue T-shirt` のように主語と服装を含める。人物の左右は必要に応じpromptで指定するが、左半分=女性という固定割当は行わない。

負conditioningはCFG=1での現在の配線に合わせ、Encode出力から標準ConditioningZeroOutを使うテンプレートとする。token対応を壊すconditioning加工や別prompt_infoの混入は検出・拒否する。

収集の最初の比較候補はcollect_step=2、collect_block=18（stepは1始まり、blockは0始まり）、temperature=4000、top_k_ratio=0.3。既存FreeFuseを出発点にした仮値であり、Krea2 Sliderの推奨値として確定しない。実ブロック数・総step数で範囲検証する。

## 自動マスクの初期アルゴリズム

- Krea2用FreeFuseAttnの概念map計算を取り込み、実image token数とpatch gridから女性/男性mapを復元する。T5処理は持ち込まない。[S2/S3]
- バッチごと・主体ごとにmin-max正規化。NaN/Inf、空map、変動幅が相対1e-6以下のmapは無効として停止する。
- 初期の背景scoreは `1 - max(target_score, protected_score)` とし、人物の最大scoreが背景scoreを超える位置をforeground候補にする。これは比較可能な初期ヒューリスティックであり、真の人物確率とは扱わない。
- foreground内で二人のscoreを比較する。差が1e-6以内の同点は非適用領域へ退避する。男女の面積を50:50に強制しない。[S13]
- 得られるmaskはhardかつ排他的。女性maskと男性maskの積は0、背景を含む和は1。初期版はdilation/featherによる領域拡張を行わない。
- 女性または男性のmaskが空なら停止する。片方のLoRAがないことを理由に男性mapを省略しない。自動mask不成立時に全画面maskへ置き換えない。
- raw map、coverage、bbox、grid、token位置、block/stepを診断出力へ残す。数値検査合格と人体位置の正しさは別に評価する。
- 手動モードではMASKを実patch gridへ縮小し、重なり・範囲外・空領域を検査する。マスクには同じbaselineで確認した人物位置を使う。

## 予定ファイルと責務

すべて新規拡張内に作成する予定。今回はこの計画書以外は作成しない。

| ファイル | 責務 |
|---|---|
| `__init__.py`, `nodes.py` | V3登録、4ノードのschema、入力検査 |
| `slider_fuse/conditioning.py` | PromptInfo、2主体のphrase/occurrence、token照合 |
| `slider_fuse/attention.py` | Krea2の観測、Q/K/V・GQA・RoPE、収集stepとgrid |
| `slider_fuse/masks.py` | 手動/自動mask、背景、排他割当と診断 |
| `slider_fuse/lora.py` | キー/形式検査、image-only差分、量子化baseの保持 |
| `slider_fuse/sampling.py` | 同じnoise/sigmaの二段階処理、clone・hook lifecycle |
| `tests/test_conditioning.py`, `test_attention.py`, `test_masks.py`, `test_lora.py`, `test_sampling.py`, `test_nodes.py` | 下記各段階の数値・状態・接続検証 |
| `tools/probe_krea2_slider.py`, `tools/validate_comfy.py` | 実環境probeと必須integration。生成・ダウンロードを勝手に開始しない |
| `workflows/krea2_female_slider_manual.json`, `krea2_female_slider_auto.json` | 手動/自動の接続済み比較テンプレート |
| `README.md`, `THIRD_PARTY_NOTICES.md`, `licenses/` | 制約、検証状況、source commitと出典 |

## 実装順序と合格条件

### P0: 実モデルと実Sliderの契約確認

- [ ] `tools/probe_krea2_slider.py`で実ComfyUIのcommit、MODEL class、演算dtype/量子化、5種類の対象module、latent→patch grid、LoRA全キー・rank/alphaを記録する。[S5/S6]
- [ ] 対象外キー・未対応adapter・不一致shapeは実行前に拒否する。1キーも黙って捨てない。
- [ ] 現在のConvRot INT8で「元Linear出力 + 非量子化LoRA差分」が扱えるか、少数moduleの入出力で確認する。重み全体の展開・再量子化はしない。
- [ ] 合格: 実Sliderの全キー・shape・rank/alphaを対象moduleへ対応づけられ、代表moduleのConvRot INT8 probeでbase量子化状態を変更せず差分計算できる。全対象moduleへの実注入はP1で検証する。実機未実施ならP0は未合格として残す。

### P1: 手動マスクによる女性限定LoRA

- [ ] `lora.py`と`sampling.py`の最小経路を、数値・復帰テストを先に用意して実装する。[S4/S7/S8/S9]
- [ ] 正/負/0の強度、alpha/rank、wq/wk/wv/gate/wo、text-firstの長さをsynthetic FP32データで検証する。
- [ ] P0で対応づけた実Sliderの全対象moduleへ注入されることを、一致キー件数と到達したmodule集合の一致で検証する。protected/backgroundのadapter件数は0を必須とする。
- [ ] all-zero maskおよびstrength=0でbase出力一致。非対象位置の直接差分=0、女性位置の差分=明示計算したLoRA差分×強度。
- [ ] all-one IMAGE maskは、text部分を0にした同じscopeの参照計算と一致させる。通常の全token LoRA Loaderとの一致は要求しない。
- [ ] 例外・中断後に元forward/hook状態へ戻り、局所適用→通常生成→局所適用の順で状態が混ざらない。
- [ ] 同一coreを共有する2つのModelPatcher cloneで、本拡張の2実行目がhook登録前に拒否されること、別coreでは競合しないこと、失敗後に占有が解除され再実行できることをテストする。既存hookと元forwardの同一性も確認する。
- [ ] 合格: 手動マスクで1本の実Sliderの効果が女性に見え、通常の全体LoRA適用より男性への属性変化が減る。直接差分の数値合格と画像判定を別記する。

### P2: Krea2 attentionの非改変観測

- [ ] `conditioning.py`と`attention.py`を実装し、実Encodeのtoken列からphraseを解決する。template二重付与・prefixズレ・繰返しphrase・一致しないphraseをテストする。[S2/S3]
- [ ] 実ブロックのcap_len/image_len/patch gridと収集情報を照合する。4D入力とT=1の5D変換、非正方形を検証する。
- [ ] observe-onlyで同じ入力のmodel予測を観測なしと比較する。FP32 fixtureはrtol=1e-5/atol=1e-6、実量子化モデルは同一設定の反復誤差を先に記録し、差分も併記する。
- [ ] RNGを進めない、prompt/latent/attention permissionを変更しない、収集hookを解除する。
- [ ] 合格: 実conditioningの位置対応が確定し、観測による予測変化が測定した数値誤差内。単なるhook呼出回数では合格にしない。

### P3: 自動マスクと二段階sampling

- [ ] `masks.py`を上記初期アルゴリズムで実装。女性/男性の大小関係、非正方形、同点、空map、NaN/Inf、手動重なりをテストする。
- [ ] Phase 1では女性Sliderの差分が0、全体用LoRAは有効。Phase 2へ渡すnoise/初期latent/全sigma列は同一データのcloneとし、各hashを記録する。[S7]
- [ ] stepはcallback回数の推測に頼らず実評価時点へ対応づけ、収集block/実sigma/stepを記録する。短縮scheduleの再計算は行わない。
- [ ] 同じseedの強度比較で収集maskが不変となることをテストする。mask_bankを別サイズ・別prompt・別modelの実行へ使い回さない。
- [ ] 合格: 男性がLoRAなしでも有効なmaskを持ち、3maskが排他的に全tokenを覆う。自動mask不良は説明付きで停止する。

### P4: V3ノード・配線・ユーザー向け診断

- [ ] 4ノード、標準loader/negative/VAE/SaveImageと接続した2ワークフロー、出典文書を作成する。
- [ ] `/object_info`登録、UI保存/再読込、API実行、異なるprompt_info混入、ファイル変更時のキャッシュ無効化、strength変更時の再実行を検証する。
- [ ] 診断はLoRA読込/一致キー件数、text/target/protectedの直接差分、mask coverage、Phase 1/2のNFE、時間・VRAM、解除状態を出力する。全tensor dumpを常時出力しない。
- [ ] 合格: manual/autoとも実ComfyUIで接続・生成・保存が完了し、生成PNGと設定・run_idを対応づけられる。

### P5: 画質評価と初期版の判定

- [ ] seed `42, 4444, 4444444444, 123, 777` を固定し、1024×1024、Euler/simple、8 steps、CFG1、同じ全体promptで比較する。
- [ ] 1本のSliderについて A: Sliderなし、B: 通常Loaderによる全体適用、C: 女性限定FreeFuse の3条件×5seed=15枚を作成する。まず正強度+1。+1で効果が見えない場合は、単体で確認済みの強度へ全B/C条件を揃えて再評価する。
- [ ] 既存darkbrushはA/B/Cで同じ0.8。Bには局所注入を重ねない。CのmaskはseedごとにSliderなしで作る。
- [ ] 全体適用Bで男性にも属性変化が観察できるSliderを局所性の試験に使う。Bでも男性が変わらないSliderだけで男性保護の成功を主張しない。
- [ ] 通常生成Aで男女2人の構図が成立しないseedは基礎生成の失敗として残す。都合のよいseedへ差し替えず、人物数の問題と局所適用の問題を別記する。
- [ ] 各seedで、男女各1人・頭部/四肢の欠落なし、女性の目的属性変化、男性の同属性変化、矩形継ぎ目/縮尺/照明不整合を記録する。数値の外側画像差分は補助指標で、意味的な属性漏れとは区別する。
- [ ] 初期採用目安: Cの4/5seed以上で「男女各1人・女性の効果・男性への明瞭な同属性漏れなし・新たなタイル状分断なし」を同時に満たす。5seedは探索用であり母集団の成功率推定には使わない。
- [ ] 初期設定が得られたら強度0と負強度を別途比較する。負強度の単調性は初期合格条件に含めない。[S14]
- [ ] 時間/VRAMは初回ロードとwarm runを分け、Phase 1/2の実NFEを報告する。追加の序盤推論とQKV観測があるため通常生成と同速とは想定しない。

## リスクと失敗時の分岐

| 観測された問題 | 次の対応 |
|---|---|
| 手動maskでも男性へ強く漏れる | 自動maskを調整せず、text位置0・wq/wk/wv/gate/woの直接差分・上流の二重LoRA適用を先に確認。間接伝播が支配的なら限定層/適用時期の比較を次案として提示 |
| 手動maskは成功、自動だけ失敗 | raw map、phrase位置、block/step、背景scoreを比較。manualを有効な代替として維持 |
| image-onlyでは女性の効果が弱い | 学習時のtext+image作用との差を比較し、対象phrase限定のtext差分を後続実験として検討。初期版で黙ってglobal text作用を戻さない |
| 体格変化で人物が収集maskから動く | mask/人物のずれを記録。再収集や膨張を無条件に追加せず、初期版の適用限界として報告 |
| ConvRot INT8の実経路で不一致/OOM | 対応層の小型BF16参照で原因を切り分け、全モデルBF16化を必須回避策にしない。ユーザー環境のINT8合格が未了なら明記 |
| パッチ競合やclone間のhook残留 | `finally`による復帰とinjection lifecycleを修正。未知の既存attention overrideは検出して拒否し、成功扱いにしない |
| UI配線・数値テストは成功、画質基準は不合格 | 実装済みと画質検証済みを分けて報告し、既存方式への置換を完了扱いにしない |

## 検証と停止条件

実装時のローカル検証は `python -B -m pytest -q -p no:cacheprovider tests`。テストはComfyUI不要のcoreと、実ComfyUI必須のintegrationを区別する。後者の未収集/skipを成功件数へ含めない。既存環境にlint/typecheck設定がなければ、新依存を追加する前に構文・import・schema・対象テストで確認範囲を明示する。

`tools/validate_comfy.py`は実ComfyUIのパスを明示引数として受け、V3登録と実native Krea2の小型forward/注入復帰を確認する。P0/P5はモデルとSliderがある実GPU環境で実施し、利用できなければ未検証のまま記録する。

今回の停止条件は、この計画案の提示まで。実装開始後の停止条件はP0〜P5の証拠が揃うこと、または利用可能な環境では確認できない項目を明確にした実装成果の報告。未確認の画質・量子化互換性を達成済みと記載しない。
