"""Base-only mask observation. The caller owns the shared-core guard."""
from __future__ import annotations

import torch

from .attention import AttentionCollector
from .lora import RoutingState
from .masks import generate_masks, patch_grid, postprocess_masks
from .sampling import tensor_hash, validate_settings


COLLECTION_KEY = "krea2_slider_mask_collection"


def collect_auto_mask(base, positive, negative, prompt_info, subjects, image, noise, sigmas, *,
                      grid, seed, steps, cfg, collect_step, collect_block, top_k_ratio, temperature,
                      fill_holes_max_area, mask_dilate_radius):
    """Return an owned CPU mask bank; never run the edited generation phase."""
    import comfy.sample
    import comfy.model_management
    import comfy.patcher_extension
    from .native_pair import _cleanup

    validate_settings(steps=steps, cfg=cfg, strength=0., mask_mode="auto", collect_step=collect_step,
        top_k_ratio=top_k_ratio, temperature=temperature, fill_holes_max_area=fill_holes_max_area,
        mask_dilate_radius=mask_dilate_radius)
    core = base.model.diffusion_model
    if isinstance(collect_block, bool) or not isinstance(collect_block, int) or not 0 <= collect_block < len(core.blocks):
        raise ValueError("collect_block is outside the Krea2 model")
    text_len = len(prompt_info.token_ids)
    probe = base.clone()
    state = RoutingState(phase="collect")
    collector = AttentionCollector({s.id: s.positions for s in subjects}, grid, collect_step,
                                  collect_block, top_k_ratio, temperature, sigmas)
    collector.cap_len = text_len
    collector.active = True
    used_sigmas = sigmas[:collect_step + 1]
    report = {"phase1_nfe": 0,
              "collection_attention_heads": {"query": core.blocks[collect_block].attn.heads,
                                             "key_value": core.blocks[collect_block].attn.kvheads},
              "collection_initial_noise_sha256": tensor_hash(noise),
              "collection_initial_latent_sha256": tensor_hash(image),
              "collection_full_sigmas_sha256": tensor_hash(sigmas),
              "collection_used_sigmas_sha256": tensor_hash(used_sigmas)}

    def inject(_patcher):
        if not collector._handles:
            collector.install(core, state)

    def eject(_patcher):
        collector.remove()

    def forward(apply_model, args):
        context = args["c"].get("c_crossattn")
        if (args.get("cond_or_uncond", [0]) != [0] or context is None
                or context.shape[0] != 1 or context.shape[1] != text_len):
            raise ValueError("Mask collection requires the matching positive batch1 conditioning")
        if patch_grid(args["input"], core.patch) != grid:
            raise ValueError("Mask collection input grid differs from the mask grid")
        collector.begin_forward(args["timestep"], text_len)
        report["phase1_nfe"] += 1
        return apply_model(args["input"], args["timestep"], **args["c"])

    try:
        probe.set_injections(COLLECTION_KEY, [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)])
        probe.model_options["model_function_wrapper"] = forward
        with torch.inference_mode():
            comfy.sample.sample(probe, noise.clone(), steps, cfg, "euler", "simple", positive, negative,
                                image.clone(), sigmas=used_sigmas.clone(), seed=seed)
        if report["phase1_nfe"] != collect_step or not collector.maps:
            raise RuntimeError("Incomplete automatic mask collection; check sigma/token/block alignment")
        # Snapshot before reset; postprocessing owns its own copies of the partition.
        maps = {name: value.detach().cpu().float().reshape(1, -1).clone()
                for name, value in collector.maps.items()}
        bank = postprocess_masks(generate_masks(maps, grid), max_hole_area=fill_holes_max_area,
                                 dilate_radius=mask_dilate_radius)
        report["observation"] = dict(collector.observation)
        return bank, report
    finally:
        _cleanup(lambda: comfy.model_management.unload_model_and_clones(probe, unload_additional_models=False),
                 collector.remove, collector.reset, state.clear)
