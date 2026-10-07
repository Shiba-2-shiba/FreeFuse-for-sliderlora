# 対象phraseのテキストtokenへSlider差分を追加するリファクタリング

ユーザー承認済みの提案: 女性mask内の画像への適用を維持し、女性を示すKrea2内部のテキストtokenにだけLoRA差分を加えて、通常適用との効き方の差を検証する。実機確認はユーザーがcommit/push後に行う。

## 範囲と挙動固定

- `slider_fuse/lora.py`, `sampling.py`, `nodes.py` と対象テスト、比較workflow、説明文書に限定する。
- 現120件のCPU suiteが成功。既定値0で既存の画像差分の式・順序・数値、mask生成、noise/sigma、hook復帰を維持する。
- 新入力 `target_text_scale` はoptional Float、default0、範囲0..1、Sampler末尾に追加。bool/NaN/Inf/範囲外を拒否する。
- `strength`は画像側の既存強度。対象textの実効強度は `strength * target_text_scale`。例: strength4/scale0.5なら対象textへ2。負強度も同じ符号で適用する。
- CLIP/text encoderやtxtfusionの重みを変更しない。既存の `blocks.N.attn.{wq,wk,wv,gate,wo}` が処理するtext行に差分を追加する。

## 既存実装からの根拠

- 学習側`../Confyui-krea2-slider-node/krea2_slider_node/lora.py:21`は、渡された全行に標準加算型LoRAを適用する。`model.py:398`はcontextとimageを連結している。
- 現`slider_fuse/lora.py:173`はimage suffixのみを計算し、binary maskを掛ける。alpha/rankとstrengthは各1回で、面積による減衰はない。
- 0.1.2のsubjectには同じPromptInfoと解決済みのtarget/protected positionsがあり、重複語句はoccurrenceで解決する。これを直接使用し、再tokenizeや全文章へのfallbackを追加しない。

## 機能・安全性契約

1. scale0では全text行への直接差分は0。現在のimage-only pathをそのまま実行する。
2. scale>0ではtarget phraseの一意なpositionsだけを使う。positionsは非empty・int・重複なし・0..cap_len-1。protected positionsとの重複を拒否する。invalid/missing positionsは説明付きで停止し、clamp・全文章への拡張・無視で隠さない。
3. protectedとその他text行、target mask外のimage行への直接差分は0。画像側の直接式は `strength * Adapter.delta(current_image_input) * mask` で、target_text_scaleを画像係数へ掛けない。同じhook入力なら画像差分は同じだが、文章側の変化が共有attentionを通じて後続層へ伝わるため、後続の画像入力・差分値・最終画素はscaleによって変わり得る。
4. Phase1は局所LoRAがoffのまま。target textへの作用はPhase2だけ。同じnoise/latent/sigmaと、強度比較で同じmaskを用いる。
5. strength0、空image mask、phaseがroute以外ならtextもimageもoff。実runtimeでは空target maskを既存の検査で拒否する。
6. target text経由の情報は共有attentionで伝播し得る。protectedの直接差分0から、最終画像で男性への影響0を保証しない。通常の全体LoRAとの画質同値性も保証しない。
7. 正しいtext/image境界は実txtfusionとEncodeの長さ一致で確認する。収集block・text scale・画像maskの意味を混同しない。

## 最小構成とリファクタリング

- `RoutingState`にscale、target/protected positions、device別のLong index cache、適用呼出数を保持する。状態設定で位置を検査し、clear時に設定とcacheを消す。reached/callsの既存診断用途は維持する。
- `SliderHook`の画像計算はそのまま。textが有効な時だけ、選択行へ同じAdapter.deltaを計算し `strength*scale` を掛けて加算する。全sequenceの差分を常時計算する新しいpathは作らない。
- 同じalpha/rank、adapter precision/cast/cache、注入解除を再利用する。独自のadapter class、外部依存、mask面積補正は追加しない。
- Samplerはsubject positionsを状態へ渡し、mask収集前に検査する。scale0でも渡されたpositionsの整合性を確認する。
- レポートにscale、実効text強度、対象positions、`text_delta_policy=zero/target_phrase_only`、other/protected text direct policy、実際のtext適用呼出数を出す。autoのfinallyで設定が消えるため、reportは入力値とコピーしたposition一覧を使う。
- 診断・比較用workflowsはscale0/0.5/1を用意する。seed42、同じLoRA、image strength4、fill8/dilate1、同じprompt/model/styleを固定する。
- 通常適用の参照workflowは標準LoRA LoaderとKSamplerで同じ条件を使う。局所Sliderを重ねない。全体参照は男女両方に効き得ることを明示する。

## cleanup/fallback inventory

| 対象 | 分類・対応 |
|---|---|
| optional引数のdefault0 | 旧API/JSON互換の根拠付き境界。schema optionalと旧payload試験を維持 |
| 旧Preview bankの追加fieldなし | 根拠付き互換。現mask/黒出力の既存試験を維持。今回変更しない |
| 不明hook/shape/key・空maskの拒否 | 明示的fail-closed。今回も弱めない |
| 不正text positions | fallbackを新設せず停止。全text適用で救済しない |
| 重複LoRA計算/新階層 | Adapter.deltaを再利用し、責務をRoutingState/Hookへ置く |
| 一般的なUI美化・別architecture | 今回対象外 |

writerは主担当、planのcriticと最終code-reviewerは独立したread-only担当とする。通常の直接実行内のbounded cleanup helperとして進め、別のConductor/Team/runtimeは起動しない。

## テストと順序

1. 既存120件を基準に、scale0の厳密なimage-only一致、0.5/1の明示式、正/負/0のstrength、他text・保護text・非対象imageの差分0、空mask/off/collectをテストする。
2. missing/duplicate/out-of-range/bool/overlap positions、不正scale、cache/設定のclear、base storage/forward復元をテストする。
3. Phase1でtextがoff、Phase2で指定positionsだけが作用し、maskがscale比較で同じになることを既存runtime doublesで確認する。autoのclear後もreportが正しいことを検査する。
4. Node schema optional0/末尾追加、旧payloadの省略、JSON/UI/APIの一変数比較を検証する。Native6件のschema検査にも新入力を含めるが、依存不足を成功扱いしない。
5. CPU tests/構文・リンク・staged whitespaceと独立レビューで確認し、commit/push。GPUでの効果回復・漏れ・通常適用との差は実機確認待ちとして記録する。

品質チェック: pytest・AST/compile・workflow参照/型・source差分の静的確認。既存lint/typecheck設定や依存はないため、追加インストールせず適用不能を明記する。後処理や学習側の再設計を同時に行わず、text寄与という1変数を検証する。
