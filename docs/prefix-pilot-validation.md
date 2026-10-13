# Experimental paired-prefix mask pilot

This opt-in experiment replaces only the **mask-discovery pilot** in the existing
seed-locked A-R0 path. Production sampler defaults and existing workflows are
unchanged. CPU tests establish the implementation and artifact contracts; they
are not native GPU equivalence, image-quality, or wall-clock speed evidence.
The comparison is a fresh, same-runtime cold paired experiment. Archived raw
pilot tensors are not included in this checkout, so it makes no claim of exact
equality to an older archived run.

## 最短の確認手順

1. まず [公園の compare UI](../workflows/krea2_prefix_pilot_park_compare.json) と [強い重なりの compare UI](../workflows/krea2_prefix_pilot_strong_overlap_compare.json) を読み込み、モデルのファイル名を使用環境に合わせて実行します。両ルートの seed・モデル・prompt・8-step 設定は揃えたままにします。
2. 新しい pilot saver の manifest を2個と、最終 replay の DiagnosticSave report を2個保存し、下記の compare コマンドへ渡します。bootstrap の保存 manifest と取り違えないでください。
3. `full_pipeline_parity: true` を確認してから、[公園の fast UI](../workflows/krea2_prefix_pilot_park_fast.json) または [強い重なりの fast UI](../workflows/krea2_prefix_pilot_strong_overlap_fast.json) を使用します。

完成画像まで含む compare は1条件あたり計画上56 NFE、fast は24 NFEです。実機GPUでの一致・実測速度・画質の改善は未確認です。UIファイルは構造とAPI対応をCPU検証したもので、実ComfyUIでの読み込み・再保存確認はまだ行っていません。

## Bounded path

1. Obtain the shared base-only bootstrap B with two model forwards on the
   original eight-step schedule, observing steps 1/2 and blocks 16/18.
2. Run a paired base/Slider pilot with the same original noise, empty latent,
   conditioning, CFG 1, strength 4, manual B masks, and Euler/simple schedule.
   `full_reference` executes all eight updates (8 + 8 branch forwards).
   `prefix` executes only the first three updates (3 + 3 branch forwards), using
   the first four values of that same nine-value schedule. Its terminal sigma
   is positive. The model still receives the full original schedule.
3. Both modes retain the original observer instrumentation: steps 1/2/3 and
   blocks 8/10/18/23/26 for both branches, totaling 30 taps. The minimum
   extraction evidence is the 18 taps at blocks 8/10/18; the first strict
   experiment nevertheless requires all 30 to avoid changing instrumentation.
4. Reconstruct the existing A-R0 mask from step-3 early four-view anchors and
   late block-18 features. Combine the paired target union minus the shared
   bootstrap protection. Do not add the bootstrap target, change thresholds,
   introduce propagation/release variants, or substitute historical masks.
5. Restart the ordinary eight-step Prediction Mix sampler from the original
   seed and original empty latent, using the generated R0 target/protected
   masks. Decode only this final replay with the VAE.

The experimental pilot returns only a mask bank and an evidence probe. Neither
mode exposes a LATENT output. Its saver has no VAE input and saves no partial
latent, pilot RGB, or pretend completed DiagnosticSave report. A three-step
prefix must never be resumed or decoded as the final image.

## Ready-to-load workflows

Four API workflows and their matching ready-to-load UI workflows are included:

- `workflows/krea2_prefix_pilot_park_fast_api.json`
- `workflows/krea2_prefix_pilot_park_compare_api.json`
- `workflows/krea2_prefix_pilot_strong_overlap_fast_api.json`
- `workflows/krea2_prefix_pilot_strong_overlap_compare_api.json`

Load these UI counterparts through ComfyUI's workflow loader:

- `workflows/krea2_prefix_pilot_park_fast.json`
- `workflows/krea2_prefix_pilot_park_compare.json`
- `workflows/krea2_prefix_pilot_strong_overlap_fast.json`
- `workflows/krea2_prefix_pilot_strong_overlap_compare.json`

`python tools/build_prefix_pilot_ui.py` regenerates only these four UI files from
the corresponding API graphs. UI/API literals and typed links are tested for
parity; the source API files are preserved.

The park case retains seed 444444. Strong overlap retains seed 42. Both retain
1024×1024, batch 1, eight steps, CFG 1, strength 4, the existing prompts/role phrases,
and the frozen collector settings. `fast` builds one cold bootstrap, one prefix
pilot, and one ordinary full replay. `compare` builds one shared cold bootstrap,
both pilots, and two ordinary full replays. Only replay outputs are decoded.
These are experimental workflows; no run is submitted by the builder.

The assets must exist under your ComfyUI model paths. To adapt asset filenames,
copy the frozen case configuration, change its loader/Slider filenames, and
regenerate both routes together. Keep all experimental controls and identity
consistent; changing a model invalidates comparison against an earlier model.
No historical retained masks are needed for a cold run.

## Offline graph builder

Run from the extension repository with its Python dependencies installed:

```sh
python docs/run_prefix_pilot_validation.py build \
  --case-config docs/attention-step-block-sweep-results/20261012T033137Z-seed-bootstrap/experiment.json \
  --case-id park__seed_444444 \
  --mode compare --with-replays --trial-id 1 --prefix park_prefix_validation \
  --output park_prefix_validation_request.json
```

The output is a ComfyUI API request object with a `prompt` graph. It is written
exclusively: an existing output is never overwritten. Submit it through your
normal ComfyUI API tooling, or use the corresponding UI workflow. Change
`trial-id` to remeasure and keep each run's output artifacts.

For the lower-forward standalone path, use `--mode fast`; its final replay is
always included. In `compare` mode, omit `--with-replays` for pilot-only evidence
with no VAE loader or decode nodes. That result cannot establish full-pipeline
parity.

`--case-config` also accepts one case object exported from the frozen experiment,
in which case omit `--case-id`. An optional `--bootstrap-manifest PATH` uses a
verified retained bootstrap mask accepted by `Krea2SliderFuseRetainedMask`
(anchored-mask or DiagnosticSave schema). Otherwise the builder creates B fresh.
An ExperimentalMaskSave suite manifest is not itself a RetainedMask manifest.

The graph copies only explicit source nodes 1/2/3/4/5/7 (checkpoint, style, text
loader, encoding, zeroed negative conditioning, empty latent). Node 9 VAE is
included only for final replays. Historical OFF outputs, prior target masks,
other samplers, and inherited output nodes cannot execute through this graph.
There is no dependency on machine-specific live runners or Windows helper paths.

## Retained evidence and exact comparison

Each `Krea2SliderFusePrefixPilotSave` call creates a new uniquely named bundle:

- `manifest.json`, with a companion `manifest.sha256`
- `evidence.safetensors`, checked by file SHA256 and every tensor's SHA256,
  shape, and dtype
- report/provenance, actual-input observation frames, all tap algorithm settings,
  evidence maps and banks, original B masks, rebuilt R0 evidence, returned masks,
  selection/effective masks, initial noise/latent and full/used sigma schedules
- every actually executed input, sigma, native base prediction, native Slider
  prediction and mixed prediction

Publication is exclusive and rolls back handled errors. It is not crash-atomic;
interrupted/incomplete bundles fail loading. Hashes detect changed files and
metadata relative to the retained checksums; they are not signed attestations.
Keep all files in the bundle together. The saver does not overwrite prior runs.

Compare the two new saver manifests offline:

```sh
python docs/run_prefix_pilot_validation.py compare \
  --reference /path/to/full_reference_bundle/manifest.json \
  --prefix /path/to/prefix_bundle/manifest.json \
  --output pilot_parity.json
```

A successful pilot-only result is explicitly `pilot_parity_only` with
`full_pipeline_parity: null`. Every comparison is exact, including dtype and
shape; there is no tolerance fallback. The checker requires:

- identical original noise/latent, full sigma schedule, conditioning, controls,
  runtime environment, actual Slider checksum and implementation-source checksum
- exact equality of the first three actual inputs/sigmas and all three branch
  prediction streams; the prefix's four used sigmas must be the original prefix
- all 30 original observer taps, complete per-frame before/after mutation audit
  records, raw late features, and all saved evidence and algorithm metadata
- bootstrap masks bound to the original routing hashes, freshly rebuilt R0
  masks/evidence matching what was saved, and identical output/selection masks
- finite tensors and metadata, complete expected counts, and checked file hashes

The builder supplies case ID, frozen asset hashes and canonical source-graph
hash as `source_identity_json`. These are **supplied frozen provenance**, not
runtime verification of checkpoint/style file bytes. The actual Slider file and
implementation source are hashed by the sampler. The environment report records versions, ComfyUI revision and GPU information;
it does not checksum dirty ComfyUI runtime files such as samplers.py. Preserve
the same installed runtime between paired runs. Recorded R0 CPU thread settings
must agree between bundles; the code does not change global PyTorch threading.
The result states the supplied-identity limit;
a matching supplied identity must not be presented as proof of unchanged base
model/style assets.

To establish complete-path equality, supply both final replay reports from the
ordinary DiagnosticSave nodes:

```sh
python docs/run_prefix_pilot_validation.py compare \
  --reference /path/to/full_reference_bundle/manifest.json \
  --prefix /path/to/prefix_bundle/manifest.json \
  --reference-replay /path/to/replays/reference/replay.json \
  --prefix-replay /path/to/replays/prefix/replay.json \
  --diagnostic-root /path/to/replays \
  --output complete_path_parity.json
```

Both distinct replay artifacts are required. The optional diagnostic root is
an explicitly chosen allowed directory; without it the checker uses their shared
parent. It never silently constrains external replay artifacts to this repository's
`docs` directory. Each report/file/tensor is checked with `checked_diagnostic`,
bound to its pilot's seed/noise/latent/schedule, conditioning, native first branch
predictions, code/runtime, role positions, masks and zero-radius selection.
All eight replay traces, final latent, and decoded RGB pixels must agree exactly.
A single report, duplicate report, missing image hash, changed file, absent trace,
nonfinite value, or mismatch fails closed with exit status 2. Success exits 0.
Even successful complete-path equality is equivalence evidence, not a human
image-quality assessment or a timing measurement.

## Nominal diffusion-forward budget

These are uncached model-forward counts; they exclude VAE decoding, observation,
mask reconstruction, saving, and all other non-model work:

- Reference route: bootstrap 2 + full paired pilot 16 + final paired replay 16 = 34
- Prefix route: bootstrap 2 + prefix paired pilot 6 + final paired replay 16 = 24
- Cold paired validation with shared B: 2 + 16 + 6 + 16 + 16 = 56
- Cold pilot-only comparison: 2 + 16 + 6 = 24

A retained B removes its two forwards from the queued graph. ComfyUI caching can
also avoid executing nodes; a cached request is not a fresh timing trial. The
34→24 reduction is a nominal ten-forward difference, not a measured latency
speedup. Keep observation/instrumentation, hardware, model versions and cache
state consistent for later timing work.

## CPU verification record (2026-10-13)

- Python 3.12.14, PyTorch 2.14.1+cpu, pytest 9.1.1; OMP/MKL thread limits 4.
- All six added test files: **130 passed**. This includes actual PyTorch attention
  observations at a mocked ComfyUI boundary, failure/interrupt cleanup, source
  nonmutation, producer-to-artifact integration, exact rejection gates, UI/API
  structure, and three 4096-cell cases against the actual historical R0 scorer.
- Complete repository suite: **888 passed, 23 failed, 1 skipped, 85 subtests
  passed**. All 23 failures match the clean baseline: an author-specific Windows
  runtime helper is imported at module load by existing live-validation scripts.
  They occur in `test_audit_seed_bootstrap_validation.py` (3),
  `test_mask_attribution_validation.py` (9), and `test_seed_bootstrap_validation.py`
  (11). Those pre-existing scripts were not changed for this experiment.
- Compilation and whitespace checks passed. CUDA and native ComfyUI are absent
  from this test environment. Real-checkpoint/GPU equality, ComfyUI UI reload,
  memory use, image quality and wall-clock performance remain unverified.
