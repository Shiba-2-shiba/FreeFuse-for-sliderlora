# Execution ledger — plan: .omx/plans/2026-10-12-paired-mask-attribution-validation.md

Status: executing Tasks 1–5. User authorized implementation and validation on the currently running ComfyUI.

Ruling: Work in the active custom-node checkout, preserving existing dirty files. A separate worktree would not be the package loaded by the requested live backend; no branch/commit or unrelated cleanup is required.
Ruling: Existing retained-mask and sampler nodes suffice. New composition/evaluation functions run offline; avoid restarting the user's current ComfyUI unless live capability evidence requires it.
Pre-flight: Task 1 produces experiment/source identity consumed by Tasks 2–5. Task 2 produces candidates/raw maps consumed by Tasks 3–5. Task 3 produces frozen annotation identities and collision release consumed by Task 4. Task 5 requires D parity before factorial/release submission.
Ruling: Agent annotations are conservative visible interior cells, human_verified=false. They support descriptive diagnostics, never human ground-truth approval.

Result root: docs/attention-step-block-sweep-results/20261011T235655Z-mask-attribution
Task 1: complete — 60 retained Mix taps, source masks/diagnostics/previews and old candidates verified; baseline evidence and immutable experiment saved. Path-string reader issues reproduced and fixed with regression tests.
Task 2: complete — all 32 raw variant/branch/block/case banks recomputed. C1 matches prior combined-union masks exactly. S0/C0 remove 298/301 pilot cells in park and 254/250 in overlap; union variants remove none.
Task 3: complete — four RGB-specific sparse interior label sets frozen, explicit missing overlap-left-arm label and no inferred target_full. Metrics recalculated for all raw/final masks with null and unknown semantics.
Task 4: complete — park collision: 1 common-background + 15 unresolved; overlap: 9 protected + 19 common-background + 25 unresolved. Both have 0 confirmed-target cells. S0-release is not_applicable in both cases; no protection was changed to manufacture a release experiment.
Ruling: Add docs/prepare_mask_attribution_annotations.py as a reproducible source-image-bound record of assistant-labelled cells. This supports Task 3 and is not an automatic segmenter.
Ruling: The ComfyUI venv client encountered OPENSSL_Applink during HTTP preflight before submission. Use the existing system Python for REST orchestration; the same running backend performs generation. Server not restarted.
Task 5: in progress — fresh D in both cases exactly matches all saved input/base/Slider/mixed trace tensors and final latent. Factorial generation submitted to current :8000 backend. CPU suite: 739 passed, 2 skipped, 85 subtests passed; -p no:cacheprovider and a fresh task-specific basetemp avoid old Windows temporary/cache permissions.

Task 5: complete — 10 new live images, 160 diffusion NFE; release not_applicable in both cases. Ten retained trace bundles decoded on the same backend, 0 extra diffusion NFE, final RGB exact parity. All 32 raw banks reconstructed from saved scores; candidate composition, masks/PNGs, per-run routing/provenance, cache history, source integrity and D parity audited.
Ruling: Attribution revision 2 adds an explicit absolute common-background indicator and stronger candidate identity checks; preserve revision 1, recompute every candidate with the unchanged annotation revision, and link revision 2 in the final report.
Final review: configured native code-reviewer could not launch (unsupported gpt-5.4 on this account). A separate author self-review was performed using the requesting-code-review checklist. Explicit sampler/scheduler/CFG/step/radius validation and independent target_full delta attribution were added with RED→GREEN regression tests. JSON path-string boundaries and two-branch common-background semantics also have RED→GREEN tests.
Final CPU regression: 745 passed, 2 Windows platform skips, 85 subtests passed in 36.99s. Test commands/logs retained in test-results/attribution-* and the result root/cpu-final.log. No configured project lint/typecheck; separate syntax inspection is recorded with final delivery checks.
Quality gate: not passed. Union-free candidates keep face/two-person layout and improve park footwear, but whole-body improvement is not consistent; overlap common-background support is worse than D and some sparse labelled limb cells remain missing. No default adoption, holdout, dynamic mask, or extra GPU experiments pending under the plan.
Reports: docs/attention-step-block-sweep-results/20261011T235655Z-mask-attribution/report.html; verification.json; attribution-r2.json; assistant-visual-observations.json.
Delivery checks: 10 Python files parse successfully; 66 report links checked, none broken; git diff --check clean; current :8000 queue empty. delivery-checks.json and queue.after.json saved. Tasks 1–5 complete; no required work pending.
