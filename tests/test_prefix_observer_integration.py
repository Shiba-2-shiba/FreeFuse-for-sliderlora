"""Real PyTorch passive-observer integration at a CPU ComfyUI boundary.

The patcher/sampler lifecycle is a double, not native ComfyUI or GPU evidence.
The observer tests compute attention inputs, Q/K/V, outputs and evidence with
real PyTorch modules and the unmodified passive collector. A separately labeled
synthetic test covers the pilot producer/artifact contract.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_native_pair import pair_environment, guider_environment
from test_prefix_pilot import EulerBoundary
from test_runtime import Block, TinyKrea, environment
from slider_fuse.diagnostics import DiagnosticRecorder
from slider_fuse.native_pair import make_prediction_mix_guider
from slider_fuse.paired_observation import PairedObserver
from slider_fuse.paired_prefix import PILOT_BLOCKS, PrefixEulerSampler
from slider_fuse.sampling import tensor_hash


class TwentySevenBlockCore(TinyKrea):
    """Tiny real attention network with every production pilot block present."""
    def __init__(self):
        super().__init__()
        self.patch = 1
        self.blocks = nn.ModuleList([Block() for _ in range(27)])

    def forward(self, x, sigma, context):
        text = self.txtfusion(context)
        value = torch.cat((text, x.flatten(2).transpose(1, 2)), dim=1)
        vec = sigma[:, None].expand(1, 4)
        for block in self.blocks:
            value = block(value, vec, None)
        return value[:, text.shape[1]:].transpose(1, 2).reshape_as(x)


@pytest.fixture
def observer_boundary(environment, guider_environment, monkeypatch):
    # environment supplies attention kernels; guider_environment supplies only
    # Comfy's outer lifecycle and patcher loading. Neither fabricates evidence.
    base, slider, events, samplers = guider_environment
    with torch.random.fork_rng():
        torch.manual_seed(1701)
        core = TwentySevenBlockCore().eval()
    base.model.diffusion_model = core
    context = torch.tensor([[[2., 0., 0., .2], [0., 2., 0., -.2],
                             [0., 0., 2., .4]]])
    calls = []

    def predict(model, x, sigma, uncond, cond, cfg, model_options, seed):
        assert model.current_patcher is not None
        assert model.current_patcher.factor == model.factor
        assert cfg == 1.
        calls.append((model.factor, x.clone(), sigma.clone()))
        # A branch-specific model boundary changes actual attention inputs.
        # No predictions or observer tensors are substituted.
        return core(x * (1 + model.factor * .02), sigma,
                    cond[0]['model_conds']['c_crossattn'].cond)

    monkeypatch.setattr(samplers, 'sampling_function', predict)
    schedule = torch.tensor([1., .94, .82, .69, .55, .4, .23, .1, 0.])
    noise = torch.linspace(-.5, .7, 16).reshape(1, 4, 2, 2)
    latent = torch.full_like(noise, .125)
    mask = torch.tensor([[[1., 0.], [1., 0.]]])
    return SimpleNamespace(base=base, slider=slider, core=core, context=context,
                           events=events, calls=calls, schedule=schedule,
                           noise=noise, latent=latent, mask=mask)


def make_run(boundary, monkeypatch, *, prefix, observe=True):
    observer = (PairedObserver({'target': [0], 'protected': [1], 'background': [2]},
                               blocks=PILOT_BLOCKS, keep_raw=True) if observe else None)
    if observer is not None:
        observer.bind(boundary.core, (2, 2), 3, boundary.schedule)
    report = {'sampler_nfe': 0, 'grid': [2, 2], 'text_token_count': 3}
    recorder = DiagnosticRecorder(set(), level='audit')
    guider = make_prediction_mix_guider(boundary.base, boundary.slider, boundary.mask,
        patch=1, recorder=recorder, report=report, mode='both', observer=observer)
    guider.set_conds([{'model_conds': {'c_crossattn': SimpleNamespace(cond=boundary.context)}}], [])
    underlying = EulerBoundary()

    def inner(noise, latent_image, device, sampler, sigmas, denoise_mask,
              callback, disable_pbar, seed, latent_shapes=None):
        # Mirror the native boundary: full model schedule precedes truncation.
        options = {'transformer_options': {'sample_sigmas': sigmas.clone()}}
        output = sampler.sample(guider, sigmas, {'model_options': options, 'seed': seed},
                                callback, noise, latent_image, denoise_mask, disable_pbar)
        return boundary.base.model.process_latent_out(output)

    monkeypatch.setattr(guider, 'inner_sample', inner)
    sampler = PrefixEulerSampler(underlying) if prefix else underlying
    return SimpleNamespace(guider=guider, observer=observer, report=report,
                           underlying=underlying, sampler=sampler)


def execute(boundary, run):
    with torch.inference_mode():
        value = run.guider.sample(boundary.noise, boundary.latent, run.sampler,
                                  boundary.schedule, seed=42)
    return value, run.guider.finish_report()


def source_snapshot(boundary):
    return {**{name: getattr(boundary, name).clone() for name in
               ('context', 'schedule', 'noise', 'latent', 'mask')},
            **{'weight.' + name: value.clone() for name, value in boundary.core.state_dict().items()}}


def assert_clean(boundary, originals):
    assert boundary.base.model.current_patcher is None
    assert ('cleanup_models', 1) in boundary.events
    assert all(not module._forward_hooks and not module._forward_pre_hooks
               for module in boundary.core.modules())
    assert all(module.forward == originals[name]
               for name, module in boundary.core.named_modules())


def test_real_observer_full8_and_prefix3_match_every_tap(observer_boundary, monkeypatch):
    boundary = observer_boundary
    originals = {name: module.forward for name, module in boundary.core.named_modules()}
    sources = source_snapshot(boundary)
    rng = torch.random.get_rng_state().clone()
    runs = []
    for prefix in (False, True):
        run = make_run(boundary, monkeypatch, prefix=prefix)
        value, traces = execute(boundary, run)
        count = 3 if prefix else 8
        assert run.report['sampler_nfe'] == count
        assert run.report['branch_nfe'] == {'base': count, 'slider': count}
        assert sum(run.report['branch_nfe'].values()) == (6 if prefix else 16)
        assert len(run.underlying.states) == count
        used, extra = run.underlying.calls[0]
        assert torch.equal(used, boundary.schedule[:4] if prefix else boundary.schedule)
        assert torch.equal(extra['model_options']['transformer_options']['sample_sigmas'], boundary.schedule)
        assert torch.equal(run.observer.sigmas, boundary.schedule)
        probe = run.observer.result(run.report)
        assert_clean(boundary, originals)
        runs.append((run, value, traces, probe))

    full, prefix = runs
    assert torch.equal(full[0].underlying.states[2], prefix[0].underlying.states[2])
    for name in ('trace_inputs', 'trace_sigmas', 'trace_base_predictions',
                 'trace_slider_predictions', 'trace_predictions'):
        assert torch.equal(full[2][name][:3], prefix[2][name]), name
    assert full[3]['frames'] == prefix[3]['frames']
    assert len(prefix[3]['frames']) == 6
    assert {(row['branch'], row['step']) for row in prefix[3]['frames']} == {
        (branch, step) for branch in ('base', 'slider') for step in (1, 2, 3)}
    for frame in prefix[3]['frames']:
        assert frame['input_sha256'] == tensor_hash(prefix[2]['trace_inputs'][frame['step'] - 1])
        assert [row['block'] for row in frame['taps']] == list(PILOT_BLOCKS)
        for tap in frame['taps']:
            assert tap['tensor_audit_enabled'] is True
            assert tap['block_input_before_sha256'] == tap['block_input_after_sha256']
            assert tap['tensor_sha256_before'] == tap['tensor_sha256_after']
    expected = {f'step_{step}.block_{block}' for step in (1, 2, 3) for block in PILOT_BLOCKS}
    for branch in ('base', 'slider'):
        assert set(full[3]['branches'][branch]) == set(prefix[3]['branches'][branch]) == expected
        for key in expected:
            left = full[3]['branches'][branch][key]['evidence_maps']
            right = prefix[3]['branches'][branch][key]['evidence_maps']
            assert set(left) == set(right)
            assert left  # Actual computed token/head score evidence at all taps.
            for name in left:
                assert torch.equal(left[name], right[name]), (branch, key, name)
                assert right[name].device.type == 'cpu' and not right[name].requires_grad
            raw = {name for name in right if name.startswith('capture.')}
            assert raw == ({'capture.attention_output', 'capture.img_q', 'capture.img_k'}
                           if key in ('step_3.block_18', 'step_3.block_23') else set())
    for block in (18, 23):
        key = f'step_3.block_{block}'
        for name in ('capture.attention_output', 'capture.img_q', 'capture.img_k'):
            a = prefix[3]['branches']['base'][key]['evidence_maps'][name]
            b = prefix[3]['branches']['slider'][key]['evidence_maps'][name]
            assert not torch.equal(a, b), (block, name)  # Meaningful branch coverage.
    assert not torch.equal(prefix[2]['trace_inputs'][0], prefix[2]['trace_inputs'][2])

    # Independent uninstrumented replay proves passive observation is neutral,
    # including the complete CPU eight-update trajectory and final output.
    plain = make_run(boundary, monkeypatch, prefix=False, observe=False)
    plain_value, plain_traces = execute(boundary, plain)
    assert torch.equal(plain_value, full[1])
    for name in ('trace_inputs', 'trace_predictions', 'trace_base_predictions', 'trace_slider_predictions'):
        assert torch.equal(plain_traces[name], full[2][name]), name
    assert_clean(boundary, originals)
    after = source_snapshot(boundary)
    assert sources.keys() == after.keys()
    assert all(torch.equal(sources[name], after[name]) for name in sources)
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize('failure', ['model', 'interrupt', 'install'])
def test_real_prefix_observer_releases_hooks_on_failure(observer_boundary, monkeypatch, failure):
    boundary = observer_boundary
    run = make_run(boundary, monkeypatch, prefix=True)
    sources = source_snapshot(boundary)
    error_type = KeyboardInterrupt if failure == 'interrupt' else RuntimeError
    if failure == 'install':
        from slider_fuse.experimental_collection import ExperimentalAttentionCollector
        original_install = ExperimentalAttentionCollector.install

        def fail_install(collector, core):
            original_install(collector, core)
            raise error_type('deliberate partial installation failure')

        monkeypatch.setattr(ExperimentalAttentionCollector, 'install', fail_install)
    else:
        original_forward = boundary.core.blocks[24].forward

        def fail_forward(*args, **kwargs):
            # Complete a base frame, then interrupt a slider frame after real
            # taps at 8, 10, 18 and 23 while collector hooks are still installed.
            if boundary.base.model.current_patcher is boundary.slider:
                raise error_type('deliberate branch interruption')
            return original_forward(*args, **kwargs)

        monkeypatch.setattr(boundary.core.blocks[24], 'forward', fail_forward)
    originals = {name: module.forward for name, module in boundary.core.named_modules()}
    with pytest.raises(error_type, match='deliberate'):
        execute(boundary, run)
    assert_clean(boundary, originals)
    assert run.guider.runner.active is None
    assert run.guider.runner.calls == ({'base': 0, 'slider': 0} if failure == 'install'
                                       else {'base': 1, 'slider': 0})
    assert len(run.observer.frames) == (0 if failure == 'install' else 1)
    after = source_snapshot(boundary)
    assert all(torch.equal(sources[name], after[name]) for name in sources)
    with pytest.raises(RuntimeError, match='Incomplete paired observation'):
        run.observer.result(run.report)


def test_pilot_producer_payload_round_trips_actual_artifact_contract(tmp_path, monkeypatch):
    """Synthetic producer seam, separate from real-attention tests above.

    The native-sampling boundary alone returns the artifact fixture's complete
    observations; the real pilot wrapper, R0 builder, saver, loader and comparator
    must agree on owned bootstrap masks and the native payload's extra fields.
    """
    import copy
    from test_prefix_artifacts import fixture as artifact_fixture
    from slider_fuse import native_pair
    from slider_fuse.conditioning import build_info, make_subjects
    from slider_fuse.diagnostics import DiagnosticPayload
    from slider_fuse.paired_prefix import sample_paired_pilot
    from slider_fuse.prefix_artifacts import (save_prefix_artifacts, load_prefix_artifacts,
                                              compare_prefix_artifacts)

    info = build_info('woman man room',
        [(151644, 1.), (90, 1.), (151644, 1.), (872, 1.), (198, 1.), (1, 1.), (2, 1.), (3, 1.)],
        lambda ids: ''.join({1: 'woman', 2: ' man', 3: ' room'}[i] for i in ids))
    subjects = make_subjects(info, 'woman', 'man')
    payloads = []

    def native_boundary(*args, _observer, _pilot_steps, **kwargs):
        fixture, _ = artifact_fixture('prefix' if _pilot_steps == 3 else 'full_reference')
        _observer.accumulators = {branch: SimpleNamespace(taps=copy.deepcopy(taps))
                                  for branch, taps in fixture['branches'].items()}
        _observer.frames = copy.deepcopy(fixture['frames'])
        bootstrap = {'grid': fixture['report']['grid'],
                     'masks': copy.deepcopy(fixture['bootstrap_masks'])}
        extra = copy.deepcopy(fixture['validation_tensors'])
        extra.update({'reference_' + role + '_mask': value.clone()
                      for role, value in bootstrap['masks'].items()})
        payload = DiagnosticPayload(copy.deepcopy(fixture['report']),
            extra['trace_predictions'][0].clone(), bootstrap['masks']['target'].clone(), extra)
        payloads.append((payload, bootstrap))
        return None, bootstrap, payload

    monkeypatch.setattr(native_pair, 'sample_krea2_prediction_mix', native_boundary)
    paths = []
    for mode in ('full_reference', 'prefix'):
        bank, probe = sample_paired_pilot(None, [], [], info, subjects, {}, 'unused.safetensors',
            background_phrase='room', pilot_mode=mode, strength=4., seed=42, steps=8)
        payload, bootstrap = payloads[-1]
        assert {'reference_target_mask', 'reference_protected_mask', 'reference_background_mask'} <= set(payload.extra_tensors)
        assert not any(name.startswith('reference_') for name in probe['validation_tensors'])
        assert set(probe['bootstrap_masks']) == {'target', 'protected', 'background'}
        saved = save_prefix_artifacts(tmp_path, mode, probe, bank)
        path = tmp_path / saved['manifest']
        manifest, restored = load_prefix_artifacts(path)
        assert manifest['complete'] is True
        assert len(probe['branches']['base']) == 15
        for role in ('target', 'protected', 'background'):
            key = 'bootstrap_masks/' + role
            assert torch.equal(restored[key], bootstrap['masks'][role])
            assert probe['bootstrap_masks'][role].data_ptr() != bootstrap['masks'][role].data_ptr()
        assert torch.equal(restored['validation_tensors/trace_inputs'], payload.extra_tensors['trace_inputs'])
        paths.append(path)
    result = compare_prefix_artifacts(*paths)
    assert result['pilot_parity'] is True
    assert result['compared_tap_count'] == 30
    assert result['reference_nfe'] == 16 and result['prefix_nfe'] == 6
    assert result['full_pipeline_parity'] is None
