# 中断試行の引継ぎ

2026-10-12（日本時間）、起動中の ComfyUI `http://127.0.0.1:8000` を使用して同じ凍結計画を続行した。

続行結果は [20261011T170454Z/report.html](../20261011T170454Z/report.html)、実行履歴は [summary.json](../20261011T170454Z/summary.json)、機械検証は [verification.json](../20261011T170454Z/verification.json) を参照。

元の prompt ID `4e48a80c-b7cc-40ba-b25e-b9d884fdac55` は現サーバーの履歴・キューに存在しなかった。公園 OFF の42 tapを保存した完成バンドルのみ、ComfyUI出力に残っていた。既知の収集コストは3 NFEで、ONの中断までの追加コストは不明。

古いジョブIDを待ち続けたり、異なる起動時の片側収集を無条件に比較へ混ぜたりしないよう、旧ファイルを保持し、続行フォルダーでOFF/ONを再収集した。[continuation.json](../20261011T170454Z/continuation.json) に試行の関係と保存済み出力を記録した。今回のNFE集計には中断済み試行を含めない。

ComfyUIの再起動やモデル設定の変更は行っていない。
