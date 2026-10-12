# Planning ledger — Seed Lock / Base Bootstrap

Status: implementation and bounded validation complete; Tasks 1–5 complete. No default adoption.

Plan: [2026-10-12-seed-lock-bootstrap-validation.md](2026-10-12-seed-lock-bootstrap-validation.md)

## 2026-10-12 再開

依頼は `.omx` から現状を把握して実装計画の続きを作ること。通常のplanning laneで計画を改訂し、コード実装・ComfyUI生成・再起動・既定値変更は行っていない。旧sessionのhook/stateを再開権限とみなさず、runtime stateも変更していない。

### 引き継いだ事実

- 直前のpaired-mask-attribution計画はTasks 1〜5完了。progress冒頭には古いexecuting表記が残るが、末尾の完了記録と成果物が存在する。
- 保存verificationは `source_integrity=true`、`attribution_recalculated_equal=true`、`verified_raw_variants=32`、`diffusion_model_nfe=160`。delivery記録は10 trace bundle、66 report links、broken links 0。
- CPU `745 passed / 2 skipped / 85 subtests passed` は前回記録で、今回CPU回帰を再実行したという意味ではない。
- 顔・二人配置は診断条件で維持。重なりの背景選択増加と全身改善の未確定により、既定採用とholdoutは保留。seed/bootstrapはこの後続検証であり、前計画の未完了作業ではない。
- 開始時の最新草案SHA-256は `a8f7eb5682d5f719051bb3df2aa915937c63efeb321ba597e9d179248430f365`。変更前のコード106ファイルのhashとgit statusを[文書検証記録](2026-10-12-seed-lock-bootstrap-validation.planning-checks.json)へ保存した。

### 今回確定した実装上の決定

1. 現行R0を独立oracleとして残す。prototype用poolと固定セルを別fieldで持ち、旧bank互換のseedsには固定セルだけを渡す。空supported poolでR0/R1まで中断しない。
2. R*の選定はraw branchと保護減算後の最終maskを分け、同じpilotのlabelと比較する。固定セルを両branch分重複計上しない。
3. A経路でDを挿入する既存 `common()` は使わない。base collectorのconfig、style適用、mask未接続、Slider branchを作る段階を明確にした。実運用入力のD非依存と、Dデータによる方式選定を区別する。
4. 衝突schema 1は保持し、schema 2の7分類を追加する。片側対象は明示された反対branchの対象／背景確認があるときだけ診断解除する。unknownや遮蔽を非保護とみなさない。
5. gateの比較対象を同じ初期化のR0に固定。R*選定後のgate失敗をR0成功で置き換えない。R*未選定時のみA-R0の経路確認が可能。
6. freeze入口、case_id、run identity、trace復号の順序、annotation/review待ちからの再開、正当なholdout対象外でもreportを作る条件を定義した。
7. 5タスクそれぞれに具体的なテスト名・assertion・保存証拠を割り当てた。通常最大12画像/196 NFE、局所解除・holdoutを含む上限26画像/428 NFEは維持。失敗試行費用は別記する。

### レビューと文書検証

- native `explore` に独立した実装契約確認を依頼したが、role既定の `gpt-5.3-codex-spark` が当アカウント非対応で起動しなかった。未実施の独立レビューを実施済みと扱わず、同じmodelの再試行はしていない。
- leaderが現行scorer、seed/prototype計算、candidate builder、collector graph、paired runner、collision evaluator、既存テスト、保存configを読んで契約を確認。執筆後は別の自己点検で4要求のtask対応、失敗分岐、参照先、NFE、code変更なしを確認した。
- 文書検証はリンク35件・リンク切れ0、実装タスク5件・受入テスト11件を確認。コード106ファイルのhashとgit statusは開始時と同じ、`git diff --check` はexit 0。26画像/428 NFEの算術も確認した。
- 文書変更だけのため実装テスト・GPU生成は未実行。検証結果はplanning-checks.jsonに保存済み。画質改善・実装成功はまだ主張しない。

## 次回の開始位置

実装依頼を受けたら本計画のTask 1から開始する。最初に既存変更とsource hashを確認し、新しい `<UTC>-seed-bootstrap` rootへ実験条件を固定する。CPUのRED→GREENと保存2case×2branchのR0完全再現が成立してから、起動中ComfyUIを確認してTask 2へ進む。

ユーザーの追加回答を必要とする未決事項はなし。画質が改善するか、R*が選定できるか、holdout条件を満たすかは実装後の検証対象であり、計画上の未決定とは区別する。

## Implementation run — user authorized current live ComfyUI

Status: executing Tasks 1–5.

Ruling: Work in the active custom-node checkout; a separate worktree is not loaded by the requested backend. Preserve existing dirty/untracked changes, with no unrelated commits or cleanup.
Ruling: Use the existing plan ledger and task-specific test logs on Windows rather than Bash-only SDD bookkeeping scripts; keep the same per-task RED→GREEN completion evidence.
Pre-flight: Task 1 scoring/selection feeds Tasks 2/4/5; Task 2 bootstrap/pilot identities feed Tasks 3/4; Task 3 schema2/release review feeds Task 4; Task 5 requires frozen selection, all completed-run audits, and explicit holdout eligibility.
Ruling: Native executor could not launch because the installed role resolves to unsupported gpt-5.4 despite the supported model override. Implement the bounded collision slice inline; no repeated launch attempts.
Live preflight: http://127.0.0.1:8000 is reachable, ComfyUI 0.37.0 / Python 3.12.11 / C:\ComfyUI custom_nodes. Existing observer/retained-mask/collector nodes will be checked before submission.

Result root: docs/attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap
Task 1: complete — 97 targeted tests pass; 2case×2branch R0 scores/seeds/prototype indices/masks exactly match legacy and saved evidence. Background seeds 14/126 and face9→8 (23,15) reproduced. R1–R4 fail face/limb/protection non-worsening criteria; R*=not_selected frozen.
Task 3 CPU: 31 targeted tests pass after expected missing-method RED. Seven-class partition and explicit opposite-branch diagnostic release validation added; live annotation/release pending.
Task 2: base-only two-forward bootstrap completed in both cases, 4 measured NFE. New B_T/B_P exactly match old A, B_P matches old D. Graph/observer/sweep tests:26 passed.
Ruling: Windows sandbox user cannot read files generated under C:\ComfyUI\output. The requested escalated runner reuses completed prompt IDs/history and only retrieves missing evidence; no duplicate diffusion generation.
Task 2 pilots: running D/A passive observed Mix on the same live :8000 backend, no restart.

Task 2: complete — both D/A pilots in both cases match all source traces and final latents exactly. D observer frames and every tap tensor hash match prior D observations. Additional pilot cost64 NFE; total bootstrap+pilot68 NFE. New A RGB labels frozen separately; blurry footwear and overlap left arm remain null.
Task 3: park schema2 finds one base-only face collision (23,15) that old schema1 left unresolved; opposite Slider RGB visibly shows the target head/hair at this cell. Freeze one-cell diagnostic release before generation. Overlap has no independently releasable target collision; release not_applicable.
Task 4: D/A×R0 four fixed replays complete, all input/mask/first-branch/routing checks pass. Total132 diffusion NFE,8 completed diagnostic images so far.

Task 3: complete — one-sided collision diagnosis and explicit one-cell release executed in park; overlap release not_applicable. Both target/protection differences equal L. Raw base face returns8/9→9/9, no clear additional final-face quality improvement.
Task 4: complete — 9 diagnostic images,148 NFE including bootstrap/pilots/fixed replays/one release. All numerical routing/source/decoder checks pass. A-R0 qualifies for limited face/person-count path-only holdout; G4 fail and G5 unknown prohibit broad adoption.
Task 5: complete — frozen2prompt×seed123/777 executed once,8 images/136 NFE. All4 final faces intact and no separate third head/person. Overlap123 has a pale old-body-like fragment behind left shoulder (G2 fail) and inconclusive contact boundary (G3 unknown). Other3 limited cases passG0–G3. New RGB-specific annotations and final assessments saved; no reselection or tuning.
Final numerical audit:284 diffusion NFE,17 completed generation images,80 reconstructed raw banks,17 decoded trace bundles with exact finalRGB parity. Five fresh on-disk model/style/clip/vae/slider hashes match prior runtime-path record. No restart and no extra globalON collection.
Final review: separate author self-review; installed native roles resolve to account-unsupported model, so independent review is unavailable. Review focus checked: immutableR0 oracle, fixedK/prototype comparisons, A graph purity, unknown/conflict release rejection, frozenholdout cases/selection, and phase/cache/NFE accounting.
Final review fix: completedholdout resume was re-entering phase orchestration after annotations appeared. RED→GREEN regression now returns the original generation record only after verifying completed-run identities/hashes. Assessment is a separate immutable artifact; it never resubmits generation.
Supplementary pool attribution:240 role/branch/variant rows separately count prototype-pool vs fixed-lock labelled matches/errors/unlabelled cells, tied to audited tensor/annotation hashes. These post-selection counts are not used for retuning. Composition fields added_to_pilot/removed_from_pilot in new scoring bundles compare against base bootstrapB_T; they are not D-union deltas.
Final CPU:778 passed,2 Windows platform skips,85 subtests passed in32.08s. Logs:test-results/seed-bootstrap-final.log. No configured projectlint/typecheck; delivery checks separately parse changedPython, check report links, preserve existing unrelatedcode, and query finalqueue.
Reports:docs/attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/report.html;verification.json;holdout-assessment.json;seed-pool-attribution.json.
All planned questions have measured conclusions or negative/unknown results. No required implementation or generation remains.

Final review safety fix: reserve a canonical caller UUID and persist submission-intent before POST. Ambiguous response loss now queries history/queue and fails closed rather than re-POSTing. Installed ComfyUI server.py1093 confirms callerprompt_id support. Expected duplicate-POST regression observedRED→GREEN; knownhistory recovery also tested. Existing completed prompt IDs and all data reused;0 new diffusion NFE.
Final review metadata fix: futurefreeze reads actual token_positions instead of an obsolete subject_token_positions key. Actual roles in the existing sampler/collector/tap reports were already correct; runtime-role-positions.json supplements the original immutable descriptive field. No conditioning or mask changed.

Final verification revision2: preserves original report/verification and re-audits unchanged data with final code. Authoritative delivery:report-r2.html andverification-r2.json. Final CPU after safety fix:780 passed,2 skipped,85 subtests passed in38.00s;test-results/seed-bootstrap-final-r3.log. All counts unchanged:284 NFE/17 images/80 rawbanks/17 traces. Saved queue.after.json reports0 running/0 pending.

Delivery checks complete:10 changedPython files parse;225 reportlinks valid,0 broken; finalcode matches auditedSHA; unrelated existingPython sources unchanged; gitdiff--check exit0; queue0/0. delivery-checks.json retained. No required work pending.

Visual summary delivery: user requested understandable validation images and mask images. docs/build_seed_bootstrap_visual_summary.py creates visual-summary-v2/index.html and26 PNG comparisons from existing evidence. Six finalA-R0 cases and six effective masks share scene/seed order. Includes D/A comparisons with two sliders, aligned base/Slider overlays, raw union/protection/collision/effective-target flow, R0–R4 raw face crops, one-cell release and remnant close-up, plus all17 completed images. Japanese legends distinguish fixed protection from raw protected-role classification and uncertain regions. Preserve aspect ratio; raw mask-only sheets use nearest-neighbor.
Visual summary checks:104 links valid,0 broken;17 original image hashes and all audited implementation hashes unchanged;71 mask/reference inputs checked against verification-r2. New diffusionNFE0. Original report/data retained. Authoritative visual guide:docs/attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/visual-summary-v2/index.html.
