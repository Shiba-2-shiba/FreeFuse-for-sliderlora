# Auto mask follow-up — 0.1.1

- User evidence: manual slider visibly works; auto does not; target preview provided. Manual-success Slider selected by user as comparison baseline. Standing workflow: commit/push before user-owned GPU validation.
- Read-only inspection confirmed target mask64×64,477 foreground tokens,51 eight-connected components; approximate faceROI coverage0 and headROI2.3%. Preview metadata traces to target slot0. Manual/auto LoRA filenames, SHA fingerprints and strengths differ.
- Independent debugger compared collector tensor layout with FreeFuse Krea2 and found no layout mismatch. Binary preview cannot distinguish bad continuous similarity from destructive binarization; raw map diagnostics are necessary before a segmentation rewrite.
- Decision: keep inference/mask algorithm unchanged, expose raw similarity previews and strength statistics; produce manual/auto controls using the successful timestamped LoRA atstrength2, plus isolated phrase and collect-step variants. User images/weights are not published.
- RED: four preview/statistics tests failed before implementation; four comparison-workflow cases failed before artifacts existed. GREEN: targeted preview/mask/runtime28 tests and workflow8 tests passed.
- Full local suite: 78 passed,0 skipped/failed. No inference/mask algorithm change; original70 contracts remain green. Native/GPU probes are still user-owned.
- Independent code-reviewer: 19 files,0 findings; verified nonmutating diagnostics, preserved first3outputs, positional/named/API widget parity and isolated comparison variables. Staged whitespace check clean.
