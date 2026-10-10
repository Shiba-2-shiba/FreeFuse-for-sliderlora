# Optional saved k3-bg reuse

The recommended path is the **cold graph: 104 expected model NFE for both cases**.
Use saved reuse only when the original `multi_proto_bg` k=3 suite, its complete
+4 diagnostic bundle, unchanged input masks, and retained historical asset
identities are available. Reuse expects **68 new model NFE for both cases**
(34 per generated case). Neither number is a measured runtime or memory result.
No additional completed OFF image is generated.

## Trust boundary: historical asset identity

Schema-1 collection reports explicitly say `weights_identity_verified: false`.
The verifier preserves this fact and refuses a retrospectively upgraded claim.
It cannot infer historical checkpoint/style/CLIP/VAE identities from filenames,
map hashes, or hashes computed from the files currently installed.

A separately retained historical hash record is required. If none exists, use
the cold graph. **Do not create a new record from today's asset files and call it
historical.** The tool verifies the retained file's supplied SHA-256 and matches
current asset bytes against its recorded identities. It cannot independently
establish the record's age or historical assignment. A separate, explicit human
provenance attestation must explain that assignment and identify the operator.
The historical Slider additionally has a machine-recorded diagnostic
`lora_sha256`, which must match.

The result prominently reports
`historical_assignment: human_attested_not_independently_verified`.
This is conditional evidence, not retroactive machine proof of loaded weights.

## Required inputs

- The original runtime suite `manifest.json`, `maps.safetensors`, and every
  artifact named by its file-hash manifest, in the original bundle directory
- The original +4 diagnostic JSON with all of its original image, effective-mask
  PNG and tensor files; the unchanged strict diagnostic reader validates them
- Byte-identical copies of the suite's `multi_proto_bg.masks.target.png` and
  `multi_proto_bg.masks.protected.png` already placed inside ComfyUI's input
  directory. Supply their input-relative names, such as `reuse/target.png`
- All five current local asset files: checkpoint, style LoRA, CLIP, VAE, Slider
- A retained historical asset record and a new verification wrapper described
  below. The verifier never creates either record

The PNGs must be lossless binary grayscale (mode 1/L or identical-channel RGB),
without alpha, transparency, palette or animation, on the original token grid.
Both file bytes and decoded tensor hashes must match the suite. The graph uses
`LoadImageMask` with `channel: red`, without inversion. The verifier does not
copy, resize, threshold, fill, dilate, or repair masks. Renaming the copies within
ComfyUI input is fine; re-encoding them is not.

## Exact asset JSON schema

The **retained historical record** contains these five roles. Each `name` is the
exact original generation-config name, including its relative subdirectory if
present. Every digest is a lowercase, 64-character SHA-256. Placeholder digests
below must be replaced with genuinely retained historical evidence.

```json
{
  "schema_version": 1,
  "assets": {
    "checkpoint": {"name": "original-checkpoint.safetensors", "sha256": "<historically recorded SHA-256>"},
    "style_lora": {"name": "original-style.safetensors", "sha256": "<historically recorded SHA-256>"},
    "clip": {"name": "original-clip.safetensors", "sha256": "<historically recorded SHA-256>"},
    "vae": {"name": "original-vae.safetensors", "sha256": "<historically recorded SHA-256>"},
    "slider": {"name": "original-slider.safetensors", "sha256": "<historically recorded SHA-256>"}
  }
}
```

The **verification wrapper**, supplied using `--asset-record`, points to that
separate retained record and the current files. Paths may be absolute or
relative to the wrapper JSON's directory. Windows paths should use `/` or escaped
`\\` inside JSON. Digesting the retained record and the two reports now binds
these exact evidence files; it does not prove when their contents were recorded.

```json
{
  "schema_version": 1,
  "historical_record": {
    "path": "retained-historical-assets.json",
    "sha256": "<SHA-256 of that retained JSON file>"
  },
  "current_paths": {
    "checkpoint": "C:/ComfyUI/models/diffusion_models/original-checkpoint.safetensors",
    "style_lora": "C:/ComfyUI/models/loras/original-style.safetensors",
    "clip": "C:/ComfyUI/models/text_encoders/original-clip.safetensors",
    "vae": "C:/ComfyUI/models/vae/original-vae.safetensors",
    "slider": "C:/ComfyUI/models/loras/original-slider.safetensors"
  },
  "historical_binding": {
    "kind": "human_provenance_attestation",
    "attested_by": "Operator name",
    "statement": "Explain how the separately retained hashes identify the assets actually used by these original collection and +4 runs.",
    "suite_manifest_sha256": "<SHA-256 of original suite manifest.json>",
    "edit_report_sha256": "<SHA-256 of original +4 diagnostic JSON>"
  }
}
```

Current-file paths do not bind ComfyUI's loader resolution. Before execution,
ensure the graph's checkpoint/style/CLIP/VAE/Slider names resolve to those exact
verified files, and keep them unchanged. Duplicate asset names in multiple model
folders require particular care.

## Generate an optional reuse graph

Run from the repository root using an environment with the project's PyTorch,
Pillow and safetensors dependencies. The cold generator remains stdlib-only.
The verifier itself is a Python function, invoked by the generator's CLI:

```text
python tools/build_attention_ablation_workflows.py --case park --seed 444444 --output-dir verified-reuse-park --reuse-suite /evidence/original-suite/manifest.json --reuse-edit /evidence/original-plus4.json --input-dir /ComfyUI/input --reuse-target reuse/park-target.png --reuse-protected reuse/park-protected.png --asset-record /evidence/asset-verification.json
```

All six reuse options are required together. Generate one case at a time;
`--case all` is rejected for reuse. Repeat with the appropriate original
artifacts for `woman_front_strong_overlap` and seed `42`. Use the matching
seed/trial/prompt and unchanged generation settings from the historical run.
Generation fails before output publication if validation fails. It never falls
back silently to a weaker comparison.

The verifier returns the original `generation_config`, `frozen_suite_config`,
verified `masks.target.filename` / `masks.protected.filename`, tensor/file hashes,
source report hashes, and asset verification evidence. The generator compares
historical generation settings against its planned graph and requires the same
frozen k=3 configuration. The reuse graph contains one new k=1 collector, the
k=1/+4 edit, and the saved-k3/+2 edit. Historical k3/+4 remains external evidence.

## What still needs checking after execution

A passing result means only `eligible_for_runtime_comparison`, with
`new_execution_hashes_verified: false` and `observation_parity: unverified`.
It verifies the original noise, latent, full eight-step schedule, prefix,
conditioning, saved partition and effective mask against the historical report.
It does not establish that a new execution has reproduced them.

- Compare new initial noise/latent/full-sigma/positive-conditioning hashes and
  generation configuration against the retained originals
- Match target/protected/background token positions, background phrase and its
  occurrence. The same prompt alone does not prove the same phrase mapping
- Match recorded runtime source hashes, revisions and environment, especially
  ComfyUI revision, or document an explicit compatibility review. Missing source
  identity remains unverified. Conditioning metadata absent from the original
  edit report likewise remains unverified even when the positive tensor matches
- Confirm the new graph actually loaded the verified asset files
- Preserve and compare exact saved mask hashes and radius/fill settings
- Where retained primary-observation hashes exist, compare them with the new
  collector before making a strong prototype-count causal claim. Historical
  `audit_tensors: false` does not become exact observation parity
- Independently review images, visible body/face changes, and the human pose
  gate. Mask equality or zero direct routing leakage proves neither image
  quality nor protected-person invariance

The ordinary diagnostic comparison CLI is unchanged. This verifier does not
bypass its mismatch checks. Verification is read-only CPU/offline testing; it is
not evidence of Windows, GPU, model-runtime or image-quality validation. Files
must remain stable during verification and execution; this is not an atomic
snapshot or protection against concurrent file replacement.
