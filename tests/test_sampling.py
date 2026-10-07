import pytest
import torch

from slider_fuse.lora import RoutingState
from slider_fuse.sampling import run_phases, tensor_hash, validate_settings, validate_model_options


class Collector:
    active = False
    maps = None
    def reset(self): self.active = False


def test_restarts_from_identical_noise_latent_and_full_schedule():
    noise = torch.randn(1, 2, 3, 4); latent = torch.zeros_like(noise); sigmas = torch.tensor([1., .5, .2, 0.])
    state = RoutingState(); collector = Collector(); calls = []
    def sample(n, x, s):
        calls.append((n.clone(), x.clone(), s.clone(), state.phase))
        n.add_(99); x.add_(99)
        if state.phase == "collect": collector.maps = {"found": True}
        return x
    bank = {"masks": {"target": torch.ones(1, 2, 2)}}
    result, actual = run_phases(sample, noise, latent, sigmas, 2, collector, state, lambda maps: bank)
    assert actual is bank
    assert [c[3] for c in calls] == ["collect", "route"]
    assert torch.equal(calls[0][0], calls[1][0]) and torch.equal(calls[0][1], calls[1][1])
    assert torch.equal(calls[0][2], sigmas[:3]) and torch.equal(calls[1][2], sigmas)
    assert torch.equal(latent, torch.zeros_like(latent))
    assert state.phase == "off" and not collector.active


def test_phase_exception_clears_state_and_no_fallback():
    state = RoutingState(); collector = Collector()
    def fail(*args): raise RuntimeError("intentional")
    with pytest.raises(RuntimeError, match="intentional"):
        run_phases(fail, torch.ones(1), torch.zeros(1), torch.tensor([1., 0.]), 1, collector, state, lambda m: {})
    assert state.phase == "off" and not collector.active


def test_empty_collection_stops_before_route():
    state = RoutingState(); collector = Collector()
    with pytest.raises(RuntimeError, match="no concept"):
        run_phases(lambda *args: None, torch.ones(1), torch.zeros(1), torch.tensor([1., 0.]), 1, collector, state, lambda m: {})
    assert state.phase == "off"


def test_hash_supports_bfloat16():
    value = torch.tensor([1., 2.], dtype=torch.bfloat16)
    assert tensor_hash(value) == tensor_hash(value.clone())
    assert tensor_hash(value) != tensor_hash(value + 1)


@pytest.mark.parametrize("settings", [dict(cfg=2.), dict(mask_mode="bad"), dict(strength=float('nan')), dict(steps=0), dict(collect_step=9)])
def test_unsupported_sampling_settings_are_rejected(settings):
    values = dict(steps=8, cfg=1., strength=1., mask_mode="auto", collect_step=2, top_k_ratio=.3, temperature=4000.)
    values.update(settings)
    with pytest.raises(ValueError): validate_settings(**values)


@pytest.mark.parametrize("options", [{"model_function_wrapper": lambda *a: None},
    {"transformer_options": {"optimized_attention_override": object()}},
    {"transformer_options": {"patches": {"post_input": [object()]}}}])
def test_unverifiable_prior_runtime_patches_are_rejected(options):
    with pytest.raises(ValueError, match="unsupported"):
        validate_model_options(options)


def test_empty_options_and_global_weight_lora_are_not_rejected():
    validate_model_options({"transformer_options": {}})
