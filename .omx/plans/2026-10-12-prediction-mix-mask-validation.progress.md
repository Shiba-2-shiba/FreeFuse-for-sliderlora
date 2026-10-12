# Execution ledger — plan: .omx/plans/2026-10-12-prediction-mix-mask-validation.md

Result root: C:\ComfyUI\custom_nodes\FreeFuse-for-sliderlora\docs\attention-step-block-sweep-results\20261011T185947Z-paired-mix

Ruling: Work in the active custom-node workspace so the requested running ComfyUI loads this implementation; preserve pre-existing dirty changes and do not commit unrelated work.
Ruling: Diagnostic observers are opt-in private backend arguments; existing production node inputs and routing remain compatible. Fresh paired runs must match retained D predictions before mask quality experiments.

Tasks: trace decoder; passive branch observer and cross replay; offline seed/shape ablations; two-candidate fixed replay; quality gate and conditional holdout; final review.

Implementation: observer/seed correction RED→GREEN, 37 targeted tests pass; decoder transform RED→GREEN, 46 targeted tests pass. Full regression: 709 pass, 2 skipped, 85 subtests; known pre-existing Windows symlink spelling test fails.
Ruling: cross replay also recomputes the original-weight prediction at both steps as an exact parity check. Adds 8 NFE across two cases above the 72 NFE matrix budget; experimental total is 80 NFE.
Ruling: first backend load preceded the final retained-mask node registration; reload the completed node package before submission, preserving the first restart record.

Implementation: 76 seed/shape and existing-mask tests pass. Passive observer package and retained-mask loader registered on same port, same runtime argv. Trace decode submitted.
Ruling: First candidate replay holds the audited D protection fixed and records collisions, isolating target reconstruction before changing the protection policy. Two candidates are combined anchors/refinement at late blocks18 and23; baseline/seed-only/shape-only/combined saved as offline ablations.

Tasks 1-4 complete: six retained traces correctly decoded; all final RGB references identical. 80 NFE paired/cross experiment, 200 taps, both D trajectories exactly match stored predictions/latents. 64 NFE fixed candidate replay; 4 candidates generated and exact routing audited. Total 144 new diffusion NFE, 10 final images.
Task 5: conditional gate evaluated — no consistent full-body improvement across both diagnostic cases; block23 overlap forearm has a clear cutout. Holdout/dynamic-mask expansion not activated under the plan gate.
Final review: configured native code-reviewer failed (unsupported gpt-5.4 on this account); separate author self-review performed using requesting-code-review checklist. Fixed incomplete mask acceptance and stale file cache; reader now returns owned tensor clones to release Windows mapping. Reader RED→GREEN. Existing Windows symlink-target test normalized extended-path prefix without changing its behavioral checks. Full suite 713 passed, 2 platform skips, 85 subtests.

Final completion: 144 NFE, 200 verified taps, 10 final images and paired previews; all 4 candidate routing audits pass. Final code loaded under identical backend arguments. Live retained-mask smoke matches candidate PNG exactly; queue empty. Report links checked. Tasks complete within conditional gates; no pending required generation.
