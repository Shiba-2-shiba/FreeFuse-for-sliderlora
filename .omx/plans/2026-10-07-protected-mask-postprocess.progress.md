# Implementation ledger — protected-mask-postprocess

User approved the proposal. Continue the standing commit/push-to-dev workflow; real GPU/image/UI validation remains user-owned.

- Baseline: HEAD5f0effe, clean tracked tree,78 CPU tests passed. The approved plan was the only untracked file.
- No filename-based ownership assumption in implementation: use target/protected/background keys inside the bank. Numerical predictions for supplied PNGs remain conditional on their labeled correspondence.
- Architecture: optional zero-default inputs appended, CPU connected components (4-connected holes/8-connected target), background-only additions, fixed protected mask, postprocessing beforePhase2. Preserve first5 Preview slots and append original/added masks.
- Execution: direct under executing-plans/test-driven-development; final independent code review. Dedicated postprocessor test module keeps existing mask regression cases intact.

## Verification during implementation

- RED: missing postprocess_masks import, then41 targeted mask/core tests green after implementation.
- RED:11 schema/runtime/default/preview cases failed before integration; GREEN full115 tests after adding two optional inputs, Phase2 wiring and7-output Preview.
- RED:4 workflow cases failed before artifacts existed; GREEN12 workflow checks with fixed candidate settings and off/fill4/fill8/fill8+dilate1 controls.
- Provided-mask replay (conditional labels): target583/612/627/813, protected446 unchanged, filled44+dilated186 at combined setting, partition true. Reversed protected/background labels blocked all23 protected holes instead of stealing them. No provided PNGs/weights published.
- Full local suite119 passed before final review. New GPU/UI/quality gates remain user-owned; no inference kernel or attention behavior changed.
- Supplemental regression: BF16 mask sums rounded303 addedtokens to304. Watched the case fail, switched diagnostics to integer boolean-cell counts, then29 purepostprocess cases passed. Mask values/inference unchanged.
- Fresh full suite120 passed. Native validator remains exit1/tests_run0: missing comfy_aimdo.storage and existing kitchen ConvRot API mismatch. Environment dependencies unchanged.
- Final independent installed code-reviewer:24 files,0 findings, APPROVE. Confirmed ownership/topology, BF16 counts, Phase2 routing, CPU snapshots, oldAPI defaults, first5outputs and four controlled workflow pairs.
- Final index check: BF16 correction and regression staged; no unstaged source changes. All12 UI/API workflow pairs and local documentation links validated; staged whitespace clean. No deferred minor findings.
