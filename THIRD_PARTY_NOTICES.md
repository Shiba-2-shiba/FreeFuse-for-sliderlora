# Third-party notices

This project is distributed under Apache-2.0; the full license is in `LICENSE`.

## FreeFuse

- Source: https://github.com/yaoliliu/FreeFuse
- Local source revision: `f5570195e84d3bc8e7f6702fcb7b012b89372b8e`
- Source files: `freefuse_comfyui/freefuse_core/krea2_support.py`, `attention_replace.py`, `token_utils.py`.
- License: Apache-2.0.
- `slider_fuse/attention.py` adapts Krea2 observation and two-stage concept similarity. Changes include strict token/grid/sigma checks, target/protected groups independent of adapters, chunked similarity reduction, and no attention bias. The text template and prefix alignment follow the source/native encoder contract. No neighboring repository is imported at runtime.

## FreeFuse-for-anima

- Local source revision: `1b924b5dd1f7266fa6e5869331e67a2033e7ec2f`.
- Source files: `anima_freefuse/sampling.py`, `lora.py`, `conditioning.py`, `nodes.py`.
- License: Apache-2.0 (same license text in `LICENSE`).
- Its two-phase restart, reversible injection, exact-conditioning linkage and V3 layout informed this implementation. Krea2-specific image-only routing and subject registration replace the Anima T5/cross-attention contracts. Equal-area mask assignment and global text K/V LoRA are not used.

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

ComfyUI is an external runtime dependency. Its verification checkout is excluded from distribution and is not vendored. No model/LoRA weights or third-party package binaries are included.
