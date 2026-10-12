"""Isolated V3 surfaces for bounded observation-only attention sweeps."""
import json
from pathlib import Path

import folder_paths
from comfy_api.latest import io

from .nodes import CATEGORY, MasksType, PromptType, SubjectsType
from .slider_fuse.attention_sweep import (collect_attention_sweep, save_attention_sweep,
                                         select_attention_sweep, validate_sweep_config)

SweepType = io.Custom("KREA2_SLIDER_FUSE_ATTENTION_SWEEP")


class Krea2SliderFuseAttentionSweepCollect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Krea2SliderFuseAttentionSweepCollect",
            display_name="Krea2 Step / Block Attention Sweep", category=CATEGORY + "/Experimental",
            description="Observe steps 1/2/3 across selected Krea2 blocks with one three-forward prefix of the original eight-step schedule. Computes the existing k3 mask independently per tap. The input MODEL determines whether collection includes Slider weights.",
            inputs=[io.Model.Input("model"), io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
                    PromptType.Input("prompt_info"), SubjectsType.Input("subjects"), io.Latent.Input("latent"),
                    io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF),
                    io.Int.Input("steps", default=8, min=8, max=8),
                    io.Int.Input("trial_id", default=0, min=0, max=0x7FFFFFFF),
                    io.String.Input("config_json", multiline=True, default=json.dumps(validate_sweep_config({}), indent=2)),
                    io.String.Input("background_phrase", default=""),
                    io.Int.Input("background_occurrence", default=0, min=0, max=999),
                    io.Boolean.Input("audit_tensors", optional=True, default=False)],
            outputs=[SweepType.Output("sweep")])

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(collect_attention_sweep(**kwargs))


class Krea2SliderFuseAttentionSweepSelect(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Krea2SliderFuseAttentionSweepSelect",
            display_name="Select Krea2 Attention Sweep Tap", category=CATEGORY + "/Experimental",
            description="Select one successfully collected k3 mask. Failed or absent taps raise; no mask fallback.",
            inputs=[SweepType.Input("sweep"), io.Int.Input("step", default=2, min=1, max=3),
                    io.Int.Input("block", default=18, min=0, max=27)],
            outputs=[MasksType.Output("mask_bank"), io.String.Output("report")])

    @classmethod
    def execute(cls, sweep, step, block):
        bank, report = select_attention_sweep(sweep, step, block)
        return io.NodeOutput(bank, json.dumps(report, allow_nan=False))


class Krea2SliderFuseAttentionSweepSave(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Krea2SliderFuseAttentionSweepSave",
            display_name="Save Krea2 Attention Sweep", category=CATEGORY + "/Experimental", is_output_node=True,
            description="Retain every tap, raw role/head scores, candidate failures and shared collection provenance in a new bundle.",
            inputs=[SweepType.Input("sweep"), io.String.Input("filename_prefix", default="krea2_attention_sweep")], outputs=[])

    @classmethod
    def execute(cls, sweep, filename_prefix):
        saved = save_attention_sweep(Path(folder_paths.get_output_directory()).resolve(), filename_prefix, sweep)
        return io.NodeOutput(ui={"text": [json.dumps(saved)]})
