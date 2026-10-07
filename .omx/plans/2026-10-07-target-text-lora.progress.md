# Target text routing implementation ledger

- User approved refactor and standing commit/push workflow. Baseline HEAD `9b6c039`, 120 CPU tests passed; tracked tree clean.
- Cleanup plan written before code: bounded LoRA/state/sampler/schema change; keep grounded legacy defaults and Preview fallbacks; no global text or clamped position fallback, no new dependencies/classes.
- Independent critic found one wording defect: later image delta values can change through joint attention. Repaired invariant to fixed direct formula/coefficient/mask under identical hook inputs, not cross-runtime equality.
- RED: 30 direct routing/state/validation cases failed because `configure_target_text` was absent. GREEN: these 30 plus 18 existing LoRA tests passed after implementation.
- RED: 10 node/settings/runtime failures from missing Sampler integration. GREEN: all 45 targeted tests passed after optional schema input, state configuration, and diagnostics were added.
- RED: 4 comparison tests failed from absent workflow artifacts. GREEN: all 16 workflow tests passed after 3 local UI/API variants and one standard global reference were added.
- Full suite: 165 passed, 0 skipped/failed (Python 3.10.11 / torch 2.10.0+cpu). Includes manual 0/0.5/1 and exception after active text routing with exact forward restoration and cache release.
- Cleanup resolution gate: default0 and old Preview bank fallback remain grounded compatibility; invalid scale/positions explicitly fail. No dead branch or duplicate adapter needed deletion; reuse `Adapter.delta`, preserve image formula and precision. New work stays inside existing state/hook/sampler boundaries.
- Native validator re-run against `.verification/ComfyUI` SHA `b26625f`: exit1, tests_run0, missing `comfy_aimdo.storage`; existing ConvRot API import mismatch also logged. No dependency updates. Native/GPU/INT8/UI/image quality remain unverified.
- Final independent code reviewer launched read-only with fresh context. Extra architect launch hit host agent thread limit; no separate architectural approval claimed.
- Static checks: 23 Python sources parse/compile; 32 UI/API JSONs have valid links/types/output slots; 39 local documentation links resolve. Standard KSampler input/widget order and denoise1 checked against local official ComfyUI source. `git diff --check` passed.
- Quality gates: CPU regression tests PASS; AST/compile and workflow/doc static checks PASS. No configured lint/typecheck/static-security tool in this repository, so those gates are N/A; no dependencies added. Product UI flow unchanged except the appended experimental control; no visual redesign.
- Remote pre-push check: GitHub `refs/heads/dev` is still baseline `9b6c0392bf7e1155e507f6a75a64e38a6f9efa14`.
- Independent code review `/root/target_text_code_review`: 24 changed files reviewed, 0 findings, APPROVE for the bounded code review. Reviewer independently ran 165 passing tests, Python compilation, whitespace and 8 new JSON checks. No deferred findings/rulings; no extra fix pass required. Native/GPU validation gap retained.
- Ready for the authorized `dev` commit and push: implementation, comparison artifacts, docs, tests and review complete; real-machine evaluation remains with the user.
