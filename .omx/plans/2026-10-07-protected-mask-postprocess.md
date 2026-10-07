# 自動maskの保護付き穴埋め・限定膨張: 追加実装案

状態: ユーザー承認済み。0.1.2の実装・検証証跡は同名の`.progress.md`に記録する。以下は承認された追加実装案。

## 目的

改善候補の自動target maskに残る小さな内部穴を埋め、必要時だけ輪郭を少し拡大する。既存のprotected maskを変更せず、3領域の排他性を維持する。処理するのはLoRA差分のrouting maskであり、モデルの全体attention・latent・位置情報へ制限を追加しない。

実装先は既存 `FreeFuse-for-sliderlora`。既存のFreeFuse・学習ノード・Anima版は変更対象外。CPU検証後にcommit/pushし、画質・INT8・UIの実機確認はユーザーが行う既存の進め方を引き継ぐ。

## 観測と根拠

今回の3PNGはprompt metadataが一致し、64×64のbinary maskを重ねると全4096tokenで和が1、重なり0、未割当0となる。

| ファイル | 白token数 | 見た目から推定した役割 |
|---|---:|---|
| `ComfyUI_temp_jsbmx_00016_.png` | 583 | target（前回までに確認済み） |
| `ComfyUI_temp_fxezv_00016_.png` | 446 | protectedの候補: 右側人物が白 |
| `ComfyUI_temp_uuhis_00016_.png` | 3067 | backgroundの候補: 背景が白 |

後者2ファイルとPreview出力の対応は、画素の見た目だけでは完全には証明できない。PNG metadataは全graphを記録しており、このPNGを書いたPreview node IDを記録していない。ユーザーへ対応を確認中。実装はファイル名・色の見た目から役割を推定せず、内部 `mask_bank['masks']['target'/'protected'/'background']` を使う。

以下の数値予測は `fxezv=protected / uuhis=background` の対応を前提とする。対応が逆なら候補穴は全て保護領域で、初期の厳格保護では追加0tokenとなる。この場合はmask役割・生成段階の診断を先に見直し、保護領域の上書きで解決したことにしない。

| 操作 | 追加candidate | protectedへの侵入 | 実際に許可できる追加 |
|---|---:|---:|---:|
| 4token以下の閉じた穴 | 21穴 / 29token | 0 | 29 |
| 8token以下の閉じた穴 | 23穴 / 44token | 0 | 44 |
| target全体を半径1token膨張 | 510token | 11 | 499 |
| 最大target成分のみ半径1token膨張 | 229token | 0 | 229 |

全23穴は、この対応ではbackgroundに属する。8tokenの穴は顔付近、7tokenの穴は胴体付近にある。target最大成分は533tokenで、残り50tokenは小さな断片。このため膨張は最大成分だけを対象にする。今回の1tokenは約16画像pixelで、膨張1でも変更量は大きい。

| 既存ソース | 根拠・接続箇所 |
|---|---|
| `../../slider_fuse/masks.py:94` | 自動maskを生成。現在は人物mapの正規化・割当のみ |
| `../../slider_fuse/masks.py:116` | target/protectedの排他割当、backgroundは補集合 |
| `../../slider_fuse/masks.py:39` | 8連結成分の診断が既存。最大成分取得に必要な考え方を再利用 |
| `../../slider_fuse/masks.py:22` | mask coverage・bbox・成分数の診断を作成 |
| `../../slider_fuse/sampling.py:52` | Phase1収集→make_masks→state.mask→Phase2という境界 |
| `../../slider_fuse/sampling.py:187` | 自動modeのmake_masks callback。後処理をここへ挿入する |
| `../../nodes.py:55` | SamplerのV3入力定義。温度の後ろに任意設定を追加する |
| `../../nodes.py:97` | Previewは現在5出力。追加出力を末尾へ付け、既存slot番号を維持する |

## 推奨方式と初期設定

優先順位は「閉じた小穴の充填」→「最大成分だけの膨張」。closing/opening/blur/featherは初期対象外にする。closingは狭い外側の隙間もつなぎ得て、今回の内部穴に必要な修正範囲より広い。単純な全target膨張は小断片も広げ、今回11tokenのprotected侵入候補を作る。

追加するSampler入力:

| 入力 | 型・範囲 | 既定値 | 比較用候補 |
|---|---|---:|---:|
| `fill_holes_max_area` | Int、0..64、単位token面積 | 0 | 4 / 8 |
| `mask_dilate_radius` | Int、0..1、単位token半径 | 0 | 0 / 1 |

既存workflowの値・結果を維持するため両方0を既定とする。今回のデータに基づく最初の改善候補は **穴埋め8、膨張0**。頭部の8token穴も補え、targetは583→627、protectedは446のまま、backgroundは3067→3023となる予測である。画質改善は実機比較で判定する。

両入力をV3で **optional=True、default=0** とし、既存temperatureの後へ追加する。Python execute/sample関数にも末尾のdefault付き引数として追加する。旧JSON/APIのwidget位置を変えず、APIで新2キーを省略したpayloadがvalidate/executeできることを試験する。Python defaultsだけを互換性の証明としない。初期はauto専用。manualで非zeroを指定した場合は説明付きで拒否し、黙って無視しない。

## 処理契約

### 1. 入力検査

target/protected/backgroundは同じdevice・同じfloating dtype・同じ `[1,H,W]`、finite、binary0/1。gridとshapeを照合し、和が1・重複0を確認する。設定の負数・bool・範囲外は拒否する。空maskを全画面へ置換しない。

### 2. 小穴充填

targetの補集合を**4連結**で成分分解する（前景側の既存診断は8連結）。判定順は、外周接続を除外 → 上限超過を保留 → protectedを含む成分を保留 → 全tokenが元backgroundの穴を充填、と固定し、診断数を二重計上しない。外周へ斜めにつながるだけの成分は、4連結では閉じた穴として扱う。このtopologyを境界試験で明示する。

protectedを含む穴は成分全体を保留する。穴の一部分だけを移して、埋まったと診断しない。人物間の隙間など外部につながるbackgroundも保留する。入力targetは削らず、既存小断片も穴埋めだけで削除しない。

### 3. 任意の限定膨張

穴埋め後targetの8連結最大成分だけを使う。同面積の場合は最小のrow-major開始indexで決定し、再現性を保つ。半径1は3×3のsquare neighborhood、stride1/padding1で、canvas外へwrapしない。

膨張候補のうち、**元background**に属するtokenだけを追加する。既存targetの小断片は保持するが拡大しない。protectedへ入る候補は除外し、除外数を診断する。protectedの収縮・再割当は行わない。

protectedは現在の推定maskで、実際の男性全体を保証するsegmentではない。protectedとの数値的非重複は保証できても、男性の未捕捉部分や間接attentionへの影響は実画像で確認する必要がある。

### 4. 所有権と不変条件

- 処理済みtargetは元targetを全て含む。
- 追加領域は元backgroundの部分集合で、protectedと重ならない。
- 処理済みprotectedは元protectedと完全一致する。
- 処理済みbackgroundは `1 - target_after - protected_before` で再計算する。
- 3maskの和は1、重複0、shape/device/dtypeは既存契約を維持する。
- 元bankと生類似度mapは変更しない。両設定0はmask値・shape/device/dtypeと推論結果を元と完全一致にし、診断付きの新bankだけを返す。bankのobject identityは保証対象にしない。
- samplerは**処理済みtarget**をstate.maskへ渡す。生成後のPreviewだけを直して効果が変わったと扱わない。

## ファイル・インターフェース案

| ファイル | 変更内容 |
|---|---|
| `slider_fuse/masks.py` | 純関数 `postprocess_masks(bank, *, max_hole_area=0, dilate_radius=0) -> bank`。穴判定、最大成分膨張、保護判定、診断を所有 |
| `slider_fuse/sampling.py` | make_masks内でgenerate→postprocessを呼び、Phase2と返却bankを同じ処理結果にする。manualの設定検証も実装 |
| `nodes.py` | Samplerの2入力を末尾へ追加。Previewは現5出力を維持し、末尾へ`original_target_mask`と`added_target_mask`を追加 |
| `tests/test_masks.py` | 下記不変条件・境界・拒否条件の数値試験 |
| `tests/test_sampling.py`, `tests/test_runtime.py` | Phase2が処理後maskを使うこと、同じnoise/sigma、manual0の既存挙動を検証 |
| `tests/test_nodes.py`, `tests/test_workflows.py` | 新入力default0、旧widget/出力順、新UI/API一致を検証 |
| `workflows/` | 候補条件を固定したoff/fill4/fill8/fill8+dilate1のUI/API比較例 |
| `README.md`, `docs/` | token単位、保護範囲、設定と実機判定の説明 |

新依存は追加しない。64×64 grid程度の連結判定はCPUの標準Python/PyTorch、膨張は既存PyTorch poolingで実装する。成分診断のhelperを整理する場合は現成分数・bboxの回帰試験を維持し、汎用画像処理クラスを新設しない。

bankには`original_masks`（処理前3maskの独立snapshot）、`added_target_mask`、`postprocess_diagnostics`を保存する。既存`masks`と`diagnostics`は処理後を表す。raw_maps/map_diagnosticsは観測値のまま保存する。無効時のPreview originalは現maskと同じ値、addedは黒で、処理の有無が分かる診断を返す。

Samplerは`report['mask_postprocess'] = bank['postprocess_diagnostics']`として診断文字列/ログへも転記する。既存コードはreportをserializeするため、bankへの追加だけで表示済みと扱わない。サンプリング後の返却では`masks`に加え`original_masks`/`added_target_mask`もCPUへdetachする。旧cache/manual/no-opで追加fieldがないPreview入力は、original=現在target、added=zeros_like(target)にフォールバックする。

`filled_token_count`は穴埋めで新たに増えた数、`dilated_token_count`は**穴埋め後**から新たに増えた数。`added_target_mask = filled_mask ∪ dilated_mask`、`after_white - before_white == added_target_mask.sum() == filled_token_count + dilated_token_count`を検証する。後処理0の診断はenabled=false/追加0とし、計算していない候補穴数などはnullで区別する。

診断項目: 設定、before/after白数・成分数、穴候補数/充填数/大穴保留数/保護により保留した数、filled_token_count、dilated_token_count、膨張のprotected除外数、protected_changed_token_count（必ず0）、partition_valid。

## 実装順序と検証

1. **純関数・小穴充填**: 閉じた1/4/8token穴、上限超過、外周への通路、外周へ斜めにだけつながる補集合、protected単独/混在穴、形状非正方形、NaN/Inf、重複・非binary、設定0の回帰試験を先に作る。入力非改変・protected完全一致・partitionを確認する。
2. **限定膨張**: 最大成分のみ拡大、小断片維持、前景が斜めに接続した8連結成分、同面積tie、四辺・角でwrapしないこと、protected候補の除外、穴埋めとの処理順を検証する。穴と膨張の候補が重なるfixtureで追加数の和を確認する。半径0は完全一致。
3. **Sampler接続**: generate→postprocess→state.maskという実行順を、既存のComfyUI境界test doubleで検証する。生のmaskと異なる処理後maskが実際のSliderHookへ届くことを確認し、manual0/auto0・hook復帰の現78件を回帰確認する。
4. **UI/診断/例**: 追加入力を末尾に置き、旧5 Preview outputsの順序と旧JSON/APIの省略入力を保つ。reportに後処理診断が入ること、返却snapshot/追加maskがCPUであること、manual/no-opと旧bankのPreview fallbackを試験する。新Preview2出力を配線し、診断だけの変更でないことを確認する。
5. **実機比較**: 既知のSlider、strength4、seed42、woman/man、top_k0.2、temperature10000、step2/block18、8steps/CFG1を固定し、off→fill4→fill8→fill8+dilate1を比較。その後123/777/4444/4444444444で再現性を確認する。変えるのは後処理だけ。

対応確認後の今回3maskについて、read-only CPUシミュレーションの受入値はoff583/fill4=612/fill8=627/fill8+dilate1=813、protected446固定、sum1。最後の組合せはfilled44+dilated186=追加230。膨張の229追加は**未穴埋めの最大成分**だけの観測値であり、44+229と単純加算しない。実装fixtureにする際は画像を公開せず、許可された匿名化maskデータまたは同構造の合成fixtureを使う。

実装の合格は数値不変条件と接続確認。画質の採用条件は、女性の効果が維持・改善し、男性の同属性変化・背景破綻・人物間の誤適用が増えないこと。直接差分0と最終画像の外側完全一致を混同しない。条件が悪化したら後処理0へ戻せる。

## 今回の停止条件

計画作成時は案の提示までを範囲とした。その後のユーザー承認により実装・commit/pushへ進む。実機品質の結果がない状態で改善済みと記載しない。
