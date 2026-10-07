# Implementation ledger — 2026-10-07-krea2-female-only-slider-freefuse.md

User approved implementation and commit/push to Shiba-2-shiba/FreeFuse-for-sliderlora; real GPU/quality checks will be performed by the user after push.

- Pre-flight: target is a new empty repository; remote has no refs. Initialized development branch `dev` in the user-specified dedicated directory. No additional checkout is necessary.
- Ruling: real GPU gates P0/P1/P2/P4/P5 remain explicitly unverified; implement their probes/workflows and CPU contracts now because the user owns real-machine validation. No success claim for INT8 image quality.
- Pre-flight interfaces: subjects contain target/protected without adapters; sampler binds exactly one slider to target. Image-only mask must use runtime cap_len and observed image grid. Phase 1 and Phase 2 reuse initial noise/latent and the full sigma schedule. Shared native module ownership is acquired before mutation.
- Execution: direct implementation using executing-plans and test-driven-development; one independent code review before publication. Local source references remain provenance, never runtime imports.

## Implementation evidence

- P0 tooling: strict canonical down/up/alpha parser, all matched keys, representative native linear/storage probe, real ComfyUI validator provided. GPU gate remains user-owned.
- P1 core: watched absent modules fail, then 23 manual-mask/LoRA tests pass; signs, alpha/rank, all-zero/all-one IMAGE references, text/direct outside deltas and shared-core guard are covered.
- P2/P3 core: absent modules failed, then 35 tests passed after conditioning, sigma-selected observation, unequal-size masks and two-phase restart. Native source lookup confirmed raw `clip.tokenize(prompt)`, post-strip text length, and sampler sigma in `model_function_wrapper`; no inverse model_sampling.sigma conversion is used.
- P4: node/tool/workflow tests failed before their artifacts existed, then 58 CPU tests passed. Four V3 nodes; manual 22 nodes/25 links and auto 18 nodes/20 links, plus API examples.
- Static check: 21 Python files parse; user documentation links resolve; verification checkout and weights are ignored.
- Native validator: actual official checkout b26625f23a888367b92153b28d93e159e83e677b, exit1, 0 tests. Existing environment lacks comfy_aimdo.storage and a compatible kitchen ConvRot API. No dependencies changed; no GPU/INT8/quality success claim.
- API reference: pinned user source 3d9b2d551788d4fe80ede5743417077d1795cbd2, verified by official-source researcher; current verification checkout is newer and is not distributed.
- Publication: explicit user choice is commit + push to the designated repository, then user performs real-machine validation. No additional integration-choice prompt is needed.

## Final review and verification

- Independent installed code-reviewer examined the whole initial tree and identified two High findings, no deferred minor findings.
- Final: fixed foreign injection/active-skip state/raw hooks/per-instance forward composition — six regression cases failed before validation was added, then passed. Rejection happens before clone/sampling/unload, preserving foreign state.
- Final: fixed ineffective native-probe false positives — zero/suppressed/wrong-scale cases added before comparison implementation; explicit expected output plus nontrivial target change is now required. Positive alpha/rank reference passes.
- Final suite: `python -B -m pytest -q -p no:cacheprovider tests` → 70 passed, 0 skipped, 0 failed. Includes integer storage and BF16 compute tests; these are not ConvRot GPU evidence.
- Native gate remains failed/unexecuted due to the documented environment dependency gap. GPU/INT8 image generation and quality are handed to the user, as instructed.
- Python AST and local documentation/workflow references verified. Staged whitespace check passes. External verification checkout, caches and all weights are excluded from the initial release.
