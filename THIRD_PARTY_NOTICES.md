# Third-party notices

This project's source is distributed under Apache-2.0; the full license is in
[`LICENSE`](LICENSE). Keep `LICENSE`, `NOTICE`, this file and the per-file
attribution/modification notices when redistributing the source or adaptations.
External runtimes and model weights retain their own licenses.

## FreeFuse

- Source: https://github.com/yaoliliu/FreeFuse
- Verified source revision: [`f5570195e84d3bc8e7f6702fcb7b012b89372b8e`](https://github.com/yaoliliu/FreeFuse/tree/f5570195e84d3bc8e7f6702fcb7b012b89372b8e).
- Source files: `freefuse_comfyui/freefuse_core/krea2_support.py`, `attention_replace.py`, `token_utils.py`.
- License: Apache-2.0. Both the root [`LICENSE`](https://github.com/yaoliliu/FreeFuse/blob/f5570195e84d3bc8e7f6702fcb7b012b89372b8e/LICENSE) and [`freefuse_comfyui/LICENSE`](https://github.com/yaoliliu/FreeFuse/blob/f5570195e84d3bc8e7f6702fcb7b012b89372b8e/freefuse_comfyui/LICENSE) match this project's license text. No `NOTICE` file occurs in the verified source tree.
- `slider_fuse/attention.py` adapts Krea2 observation and two-stage concept similarity. Changes include strict token/grid/sigma checks, target/protected groups independent of adapters, chunked similarity reduction, and no attention bias. The text template and prefix alignment follow the source/native encoder contract. No neighboring repository is imported at runtime.
- `slider_fuse/conditioning.py` adapts the Krea2 template/prefix contract from `token_utils.py`. Changes apply the native template once, decode whole prefixes, resolve explicit phrase occurrences and reject overlapping roles. `bypass_lora_loader.py` informed the comparison with upstream text/image routing; its multi-adapter loader is not included.

## FreeFuse-for-anima

- Source: https://github.com/Shiba-2-shiba/FreeFuse-for-anima
- Verified source revision: [`1b924b5dd1f7266fa6e5869331e67a2033e7ec2f`](https://github.com/Shiba-2-shiba/FreeFuse-for-anima/tree/1b924b5dd1f7266fa6e5869331e67a2033e7ec2f).
- Source files: `anima_freefuse/sampling.py`, `anima_freefuse/lora.py`, `anima_freefuse/conditioning.py`, `nodes.py`, `__init__.py`.
- License: Apache-2.0 (same license text in `LICENSE`).
- Its two-phase restart, reversible injection, exact-conditioning linkage and V3 layout informed this implementation. Krea2-specific image-only routing and subject registration replace the Anima T5/cross-attention contracts. Equal-area mask assignment and global text K/V LoRA are not used.
- The corresponding modified files are `slider_fuse/sampling.py`, `slider_fuse/lora.py`, `slider_fuse/conditioning.py`, `nodes.py` and `__init__.py`. Their headers identify the source and changes. The original `NOTICE` is reproduced verbatim in this project's `NOTICE` and below with Markdown formatting.

Original Anima NOTICE:

> FreeFuse-for-anima
>
> The spatial multi-LoRA concept and two-phase generation design were adapted from FreeFuse (https://github.com/yaoliliu/FreeFuse), licensed under Apache 2.0. The original license text is included in LICENSE.
>
> ComfyUI source code is referenced through its public extension APIs and is not copied into this project. The Anima ConceptAttention Survey repository informed the token-alignment audit; no source files from that repository were copied.

## ComfyUI API reference

Official sources at `3d9b2d551788d4fe80ede5743417077d1795cbd2`:

- [Krea2 tokenizer](https://github.com/Comfy-Org/ComfyUI/blob/3d9b2d551788d4fe80ede5743417077d1795cbd2/comfy/text_encoders/krea2.py)
- [Qwen3-VL template](https://github.com/Comfy-Org/ComfyUI/blob/3d9b2d551788d4fe80ede5743417077d1795cbd2/comfy/text_encoders/qwen3vl.py)
- [Native Krea2](https://github.com/Comfy-Org/ComfyUI/blob/3d9b2d551788d4fe80ede5743417077d1795cbd2/comfy/ldm/krea2/model.py)
- [Patcher lifecycle](https://github.com/Comfy-Org/ComfyUI/blob/3d9b2d551788d4fe80ede5743417077d1795cbd2/comfy/model_patcher.py)
- [Sampling API](https://github.com/Comfy-Org/ComfyUI/blob/3d9b2d551788d4fe80ede5743417077d1795cbd2/comfy/sample.py)

ComfyUI is an external runtime dependency under [GPL-3.0](https://github.com/Comfy-Org/ComfyUI/blob/3d9b2d551788d4fe80ede5743417077d1795cbd2/LICENSE). The extension calls its Python APIs and shares model/conditioning data; separate repository storage or API use alone does not establish a GPL exemption. Apache-2.0 is [compatible with GPLv3](https://www.gnu.org/licenses/license-list.html#apache2). Distributing a combined ComfyUI/extension program must meet the applicable GPL source and license obligations as well as retaining these Apache notices; see the [FSF plug-in guidance](https://www.gnu.org/licenses/gpl-faq.html#GPLAndPlugins).

The verification checkouts are excluded from distribution and are not vendored. No model/LoRA weights, third-party package binaries, upstream example images or datasets are included. Workflow model filenames are references to user-supplied files, not a grant to redistribute those files. Obtain and use each model, LoRA, runtime and dataset under its own terms.

## Project identity and review scope

This is an independent adaptation. The name FreeFuse identifies the upstream method and provenance; it does not claim official affiliation, endorsement or trademark rights. Apache-2.0 section 6 does not grant general trademark permission.

The [2026-10-07 license audit](docs/license-audit.md) records the checked revisions, source scope, remediation and remaining limits. These notices describe the inspected source and do not certify every possible distribution or third-party right.
