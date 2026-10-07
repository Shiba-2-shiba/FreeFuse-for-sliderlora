# ライセンス監査記録

確認日: 2026-10-07。修正前の対象は `dev` commit `49b42d7eee8f0a7b951e698deb491d2a892e2c91`。

目的は、FreeFuseを参考にした現リポジトリの出典と再配布条件を照合し、確認できた表示の不足を修正したうえで`main`へ統合すること。法的な無侵害の保証とは区別する。

## 調査結果

| 優先度 | 結果 | 確度・根拠 |
|---|---|---|
| 1 | 両派生元の指定コミットはApache-2.0。現リポジトリの`LICENSE`は両方と一致 | 高・取得したライセンス本文を改行正規化後に比較 |
| 2 | 派生ファイルの出典・改変表示が不足していた | 高・`sampling.py`、`lora.py`、`nodes.py`、`__init__.py`の既存ヘッダーには表示がなかった。`conditioning.py`も変更点を明示していなかった |
| 3 | Anima版の元`NOTICE`は既に出典文書内に保持されていたが、上流URLとファイル対応表が不足 | 高・元ファイルを取得し既存の引用を照合。元の`NOTICE`をルートにもそのまま追加 |
| 4 | ComfyUIとの結合・配布にはGPLの検討が必要 | 高・ComfyUIのGPLv3本文と本拡張のPython API呼出し・モデル共有を確認。配布形態ごとの判断はこの監査の範囲外 |
| 5 | 確認した配布対象にモデル、画像、データセット、依存バイナリはない | 高・修正前のGit追跡対象75ファイルを確認。JSON内のモデル名は参照のみ |

## 確認した一次資料

- [FreeFuse `f5570195e84d3bc8e7f6702fcb7b012b89372b8e`](https://github.com/yaoliliu/FreeFuse/tree/f5570195e84d3bc8e7f6702fcb7b012b89372b8e): ルートと`freefuse_comfyui/`の`LICENSE`、再帰的なファイル一覧、`krea2_support.py`、`attention_replace.py`、`token_utils.py`、`bypass_lora_loader.py`。ファイル一覧に`NOTICE`はなく、照合した派生元コードに個別の著作権表示はなかった。`LICENSE`末尾の`[yyyy]`等は標準の適用例であり、実作者名として置き換えていない。
- [FreeFuse-for-anima `1b924b5dd1f7266fa6e5869331e67a2033e7ec2f`](https://github.com/Shiba-2-shiba/FreeFuse-for-anima/tree/1b924b5dd1f7266fa6e5869331e67a2033e7ec2f): Git認証で固定コミットを取得し、`LICENSE`・`NOTICE`と`sampling.py`・`lora.py`・`conditioning.py`・`nodes.py`・`__init__.py`を照合。匿名HTTPの404を、ライセンスが存在しない証拠とは扱っていない。
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0): 第4条のライセンス添付、改変表示、権利表示とNOTICE保持、第6条の商標使用範囲。
- [ComfyUI `3d9b2d551788d4fe80ede5743417077d1795cbd2`](https://github.com/Comfy-Org/ComfyUI/tree/3d9b2d551788d4fe80ede5743417077d1795cbd2): `LICENSE`とネイティブKrea2モデルを照合。
- [FSFのApache-2.0とGPLv3の互換性説明](https://www.gnu.org/licenses/license-list.html#apache2)、[プラグインの説明](https://www.gnu.org/licenses/gpl-faq.html#GPLPlugins)、[プラグイン配布時の説明](https://www.gnu.org/licenses/gpl-faq.html#GPLAndPlugins)。Apache-2.0の部品をGPLv3と組み合わせられることと、組合せ全体がApache-2.0だけで配布できることは同一ではない。

取得した比較用ファイルはGit追跡対象外の`.verification/license-audit/`に置き、配布物へ含めない。

## 派生コードの対応と修正

| 現ファイル | 派生元・参考にした内容 | 本拡張の変更 |
|---|---|---|
| `slider_fuse/attention.py` | FreeFuseのKrea2観測と二段階の概念類似度 | 2役割の独立した収集、厳格なtoken/grid/sigma検査、分割集計、attention biasなし |
| `slider_fuse/conditioning.py` | FreeFuseのKrea2 template/prefixとAnima版のconditioning同一性 | native templateを1回適用、prefix単位の復号、phrase occurrence、役割の重複拒否 |
| `slider_fuse/sampling.py` | Anima版のnoise/latent/sigmaを共有する二段階再開始 | Krea2、sigmaでの観測選択、単一Slider、共有core保護、診断と解除 |
| `slider_fuse/lora.py` | Anima版の状態管理と可逆的なforward注入 | Krea2キー検証、down/up加算、画像と任意の対象文章行、元のforward属性の復元 |
| `nodes.py`、`__init__.py` | Anima版のV3ノード構成と登録 | Krea2の2人物、単一Slider、診断、遅延import |

各ファイルの冒頭にApache-2.0識別子、出典コミット、改変内容、権利表示への参照を追加した。処理本体は変更していない。既存の`LICENSE`、出典、Anima版の元`NOTICE`は保持した。推論の性能や画質を改善した変更としては扱わない。

## 推論と残る限界

上記資料から、確認したFreeFuse/Anima由来のコードはApache-2.0の条件を守って利用・改変・再配布できると判断する。元の権利表示の保持、変更の明示、追跡可能な出典を今回補強した。

ComfyUIのモデル実装全体を同梱してはいないが、観測処理はその内部モジュールを呼び出してtensorを共有する。外部依存や別リポジトリであることだけを理由にGPLの適用を否定しない。ComfyUIを含む配布・実行パッケージを作る場合は、GPLのライセンス・対応ソースの提供等とApacheの表示保持を合わせて満たす必要がある。

確認対象は追跡された現在のソースと指定した上流コミット。全世界の特許、商標登録、上流投稿者の権利帰属、将来のモデル・LoRA・生成物・第三者パッケージの条件を網羅した調査ではない。モデル等を後から同梱する場合や、独自の配布契約を付ける場合は別途確認が必要であり、「権利的な問題が一切ない」とは断定しない。

## 統合と検証

作業開始時、ローカルと`origin`には`dev`だけが存在し、`main`はなかった。そのため既存`main`との競合解決を伴うマージではなく、修正済み`dev`の履歴を引き継ぐ`main`を新設する。リモートの変更と既定ブランチの変更は、このローカル統合とは区別する。

修正後のCPU suiteは165件成功、失敗・skipなし。23のPythonファイルは構文検査を通過し、モジュールの説明文を除いたASTが修正前と一致した。3つの上流Apacheライセンス本文の一致、元Anima NOTICEの全文保持、6ファイルの改変表示、32ワークフローJSONの構文、25のローカル文書リンクを確認した。`git diff --check`も成功した。専用lint/typecheckの既存設定はないため、新しい依存を追加せず構文・AST検査を使用した。

独立した補助エージェントのレビューは、設定されたモデルがこの実行環境で非対応のため起動できなかった。上流資料と差分の直接照合で代替した。実GPU生成・INT8・画像品質の未検証状態は[既存の検証記録](validation.md)のとおりであり、ライセンス監査で合格へ変更しない。
