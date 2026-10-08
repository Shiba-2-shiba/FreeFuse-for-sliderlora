# Execution ledger — plan: .omx/plans/2026-10-08-slider-diagnostic-parity.md

User authorized implementation in this repository on 2026-10-08.
Baseline: 165 tests passed; branch dev; only the approved plan was untracked.

Pre-flight: Task 1 scope/state -> Task 2 runtime -> Task 3 payload/save -> Task 4 comparison -> Task 5 workflows: interfaces match after plan review corrections.
Ruling: implement in the requested dev checkout, preserving the existing user-visible location and untracked plan; no worktree, commit, or push is required by this plan. Cost: changes remain uncommitted for review.
Ruling: use portable progress bookkeeping in this file rather than Bash-only skill scripts on Windows. No workflow/runtime ownership state is touched.

- Task 1: complete. RED: 22 scope tests failed for missing arguments; GREEN: 70 routing/LoRA/target-text tests passed.
- Task 2: complete. RED: 11 sampling/recorder tests failed for missing implementation; GREEN: 72 scope/sampling/runtime/recorder tests passed.
- Task 3: complete. RED: 12 schema/save tests failed for missing functionality; GREEN: 21 node/artifact tests passed.
- Task 4: complete for local delivery. RED: 9 missing tool failures (3 tolerance tests already passed); GREEN: 26 sampling/tool tests passed. FP32 whole-model and real-native explicit merge check added; actual native/INT8 execution delegated to user.
- Task 5: complete. RED: 12 missing workflow failures; GREEN: 28 workflow tests passed. Created 11 cases, 22 UI/API JSON files and the evaluation guide.
- Task 6: complete for the user-authorized local delivery. Final full suite: **237 passed, 0 skipped/failed**. Python syntax/compile 33 files, workflow JSON 54 files (22 new), git diff --check passed. No lint/typecheck configuration or extra dependencies added. User owns native/GPU/UI/image evaluation.

Ruling: standard KSampler widget order differs from diagnostic Sampler; verify control_after_generate at the correct index for each schema. No existing workflow changed.
Ruling: comparison rejects strength0-vs4 numeric pairs as different conditions; zero-strength images serve as visual effect baselines, while numerical parity compares equal strengths. Cost: intentional strength-response measurement needs a separately declared comparison mode.
Ruling: native CPU/schema/INT8/UI/quality execution is user-owned per the latest instruction; no real-machine execution or dependencies installation attempted.

Final independent review: 0 critical/high; three medium findings including a provisional error-path finding. All addressed in one fix pass:
- Native malformed-LoRA error masking: test_native_invalid_lora_preserves_original_error_and_allows_clean_rerun RED→GREEN; adapters initialized before try/finally.
- Standard comparison status omitted: standard latent version/provenance test RED→GREEN; top-level measured_only/within_tolerance/mismatch derives from final-latent metrics.
- Unknown ComfyUI revision accepted: invalid-comparison revision test RED→GREEN; unavailable reason recorded, namespace package root supported, comparisons without identifiable revision rejected.
Final full-suite verification after these fixes: 237/237. No minor findings deferred.
Ruling: no separate architecture review is needed for this approved bounded diagnostic design; final code review covered lifecycle, correlation, V3 surfaces and probe semantics. Real-machine conclusions remain unclaimed.
Delivery: code, tests, 22 workflow JSONs and docs remain uncommitted in the requested checkout. No push, merge, dependency install, or OMX runtime mode was invoked.

Ruling: reuse sample_krea2 via a private _diagnostic option rather than duplicate the sampler; ordinary calls still return the existing dict and have no measurement work. Cost: internal changes require existing runtime regression coverage.
Ruling: tests account for TinyKrea's patch-grid output and create a new Slider file per trial on Windows because live safetensors mappings prohibit overwrites; real output is measured without assuming test-double shape.
User update: real machine/GPU/UI execution will be performed by the user; implementation deliverable is CPU/static validated and documented for that evaluation.
