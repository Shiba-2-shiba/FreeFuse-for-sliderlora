"""Focused next experiment contracts. Static graphs do not prove GPU/image quality."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_attention_validation_workflows import OUTPUTS, is_link

ROOT = Path(__file__).resolve().parents[1]
STEM = 'krea2_female_slider_attention_ablation'
COLLECT = 'Krea2SliderFuseExperimentalMaskCollect'
SELECT = 'Krea2SliderFuseExperimentalMaskSelect'
SAVE = 'Krea2SliderFuseExperimentalMaskSave'
SAMPLER = 'Krea2SliderFusePredictionMixSampler'
CASES = {'park': 444444, 'woman_front_strong_overlap': 42}


def generator():
    path = ROOT / 'tools/build_attention_ablation_workflows.py'
    assert path.is_file(), 'Missing focused ablation workflow generator'
    spec = importlib.util.spec_from_file_location('attention_ablation_generator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(folder, case, suffix):
    return json.loads((Path(folder) / (STEM + '_' + case + suffix)).read_text())


class AttentionAblationWorkflowTests(unittest.TestCase):
    def test_cold_only_changes_k_or_strength_and_reuses_one_k3_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            generator().write_workflows(output_dir=folder)
            for case, seed in CASES.items():
                api = read(folder, case, '_api.json')
                collectors = {sid: n['inputs'] for sid, n in api.items() if n['class_type'] == COLLECT}
                self.assertEqual(len(collectors), 2)
                configs = {json.loads(v['config_json'])['prototypes']: (sid, v) for sid, v in collectors.items()}
                self.assertEqual(set(configs), {1, 3})
                a, b = [v for _, v in configs.values()]
                self.assertEqual({k: v for k, v in a.items() if k != 'config_json'},
                                 {k: v for k, v in b.items() if k != 'config_json'})
                self.assertTrue(a['audit_tensors'])
                self.assertEqual(a['seed'], seed)
                self.assertEqual(a['steps'], 8)
                ca, cb = [json.loads(v['config_json']) for v in (a, b)]
                self.assertEqual({k: v for k, v in ca.items() if k != 'prototypes'},
                                 {k: v for k, v in cb.items() if k != 'prototypes'})
                self.assertEqual(ca['selected_steps'], [1, 2])
                self.assertEqual(ca['selected_blocks'], [16, 18])
                selects = {sid: n['inputs'] for sid, n in api.items() if n['class_type'] == SELECT}
                self.assertEqual(len(selects), 2)
                self.assertTrue(all(s['variant'] == 'multi_proto_bg' for s in selects.values()))
                samplers = {sid: n['inputs'] for sid, n in api.items() if n['class_type'] == SAMPLER}
                self.assertEqual(sorted(s['strength'] for s in samplers.values()), [2., 4., 4.])
                index = read(folder, case, '.index.json')
                rows = {row['label']: row for row in index['variants']}
                self.assertEqual(set(rows), {'bg_k1_s4', 'bg_k3_s4', 'bg_k3_s2'})
                k3a, k3b = [samplers[rows[label]['sampler_node']] for label in ('bg_k3_s4', 'bg_k3_s2')]
                self.assertEqual({k: v for k, v in k3a.items() if k != 'strength'},
                                 {k: v for k, v in k3b.items() if k != 'strength'})
                for label, row in rows.items():
                    s = samplers[row['sampler_node']]
                    self.assertEqual(s['strength'], row['strength'])
                    self.assertEqual(s['subjects'], [row['subjects_node'], 0])
                    self.assertEqual(json.loads(collectors[row['collector_node']]['config_json'])['prototypes'], row['requested_prototypes'])
                    subject = api[row['subjects_node']]['inputs']
                    preview = api[subject['target_mask'][0]]['inputs']
                    select = selects[preview['mask_bank'][0]]
                    self.assertEqual(select['suite'], [row['collector_node'], 0])
                    self.assertEqual(subject['protected_mask'], [subject['target_mask'][0], 1])
                    for key, value in dict(model=['2', 0], positive=['4', 0], negative=['5', 0],
                        latent=['7', 0], prompt_info=['4', 1], seed=seed, steps=8, cfg=1.,
                        mask_mode='manual', mix_scope='target_mask', diagnostic_level='audit',
                        fill_holes_max_area=0, mask_dilate_radius=0, selection_dilate_radius=0).items():
                        self.assertEqual(s[key], value, label + ':' + key)
                    self.assertIn(label, api[row['save_node']]['inputs']['filename_prefix'])
                classes = [n['class_type'] for n in api.values()]
                self.assertEqual(classes.count('LoraLoaderModelOnly'), 1)
                self.assertNotIn('KSampler', classes)
                self.assertNotIn('LoadImageMask', classes)
                self.assertEqual(classes.count('VAEDecode'), 3)
                for n in api.values():
                    if n['class_type'] == 'VAEDecode':
                        self.assertEqual(api[n['inputs']['samples'][0]]['class_type'], SAMPLER)

    def test_manifests_are_unexecuted_and_separate_quality_axes_and_seen_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            generator().write_workflows(output_dir=folder)
            for case in CASES:
                manifest = read(folder, case, '.index.json')
                cost = manifest['cost_accounting']
                self.assertEqual(cost['expected_cold_model_nfe'], 52)
                self.assertEqual(cost['expected_cold_two_cases_model_nfe'], 104)
                self.assertEqual(cost['conditional_reuse_model_nfe'], 34)
                self.assertEqual(cost['conditional_reuse_two_cases_model_nfe'], 68)
                self.assertIsNone(cost['actual_model_nfe'])
                self.assertEqual(manifest['validation']['gpu_execution'], 'not_run')
                self.assertEqual(manifest['holdout_policy']['previously_seen_seeds'], [42, 123, 777, 444444])
                self.assertEqual(manifest['holdout_policy']['confirmed_unseen_seeds'], [])
                record = manifest['evaluation_record_template']
                self.assertEqual(record['observation_parity']['status'], 'not_checked')
                for source in record['collection_evidence'].values():
                    for field in ('normalized_config', 'initial_noise_sha256', 'initial_latent_sha256',
                        'full_sigmas_sha256', 'used_sigmas_sha256', 'token_positions', 'conditioning',
                        'primary_tap', 'source_partition_sha256', 'implementation'):
                        self.assertIn(field, source)
                        self.assertIsNone(source[field])
                for row in record['variant_results']:
                    for key in ('duplicate_anatomy', 'target_edit_achievement', 'natural_body_proportions',
                        'male_face', 'male_eyeglasses', 'male_height', 'male_build', 'small_arm_hand_adjustment'):
                        self.assertIsNone(row[key]['value'])
                        self.assertTrue(row[key]['reason'])
                    self.assertIsNone(row['pose_gate']['passed'])
                    self.assertIsNone(row['preservation_vs_off']['value'])
                pose_text = ' '.join(manifest['pose_gate_criteria']).lower()
                self.assertNotIn('two adults', pose_text)
                self.assertIn('feet', pose_text)
                self.assertIn('younger', pose_text)
                if case == 'woman_front_strong_overlap':
                    self.assertIn('substantially occludes', pose_text)
                self.assertIn('residual', manifest['mask_semantics']['baseline'])
                self.assertIn('not equivalent', manifest['mask_semantics']['baseline'])

    def test_ui_api_schema_links_and_outputs_roundtrip(self):
        schemas = {}
        for cls in ast.parse((ROOT / 'nodes.py').read_text()).body:
            if isinstance(cls, ast.ClassDef):
                fn = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'define_schema'), None)
                if fn:
                    call = next(n.value for n in fn.body if isinstance(n, ast.Return))
                    schemas[cls.name] = next(k.value.elts for k in call.keywords if k.arg == 'inputs')
        with tempfile.TemporaryDirectory() as folder:
            generator().write_workflows(output_dir=folder)
            for case in CASES:
                api, ui = read(folder, case, '_api.json'), read(folder, case, '.json')
                nodes = {str(n['id']): n for n in ui['nodes']}
                links = {row[0]: row for row in ui['links']}
                self.assertEqual(set(api), {sid for sid, n in nodes.items() if n['type'] != 'Note'})
                self.assertEqual(ui['last_node_id'], max(int(sid) for sid in nodes))
                self.assertEqual(ui['last_link_id'], max(links))
                produced, consumed = [], []
                for sid, entry in api.items():
                    node = nodes[sid]
                    self.assertEqual(node['type'], entry['class_type'])
                    self.assertEqual([o['type'] for o in node['outputs']], OUTPUTS[node['type']])
                    named = node['widgets_values_named']
                    values = list(named.values())
                    if node['type'] in (COLLECT, SAMPLER):
                        values.insert(list(named).index('seed') + 1, 'fixed')
                    self.assertEqual(values, node['widgets_values'])
                    reconstructed = dict(named)
                    for slot, port in enumerate(node['inputs']):
                        if port['link'] is None:
                            continue
                        lid, src, out, dst, inp, kind = links[port['link']]
                        self.assertEqual((dst, inp), (int(sid), slot))
                        self.assertEqual(kind, port['type'])
                        self.assertEqual(kind, nodes[str(src)]['outputs'][out]['type'])
                        self.assertLess(nodes[str(src)]['order'], node['order'])
                        reconstructed[port['name']] = [str(src), out]
                        consumed.append(lid)
                    self.assertEqual(reconstructed, entry['inputs'])
                    for slot, port in enumerate(node['outputs']):
                        for lid in port['links']:
                            self.assertEqual(links[lid][1:3], [int(sid), slot])
                            produced.append(lid)
                    if entry['class_type'] in schemas:
                        fields = schemas[entry['class_type']]
                        names = [f.args[0].value for f in fields]
                        required = {f.args[0].value for f in fields if not any(k.arg == 'optional' and k.value.value is True for k in f.keywords)}
                        self.assertLessEqual(set(entry['inputs']), set(names))
                        self.assertLessEqual(required, set(entry['inputs']))
                        self.assertEqual(list(named), [name for name in names if name in entry['inputs'] and not is_link(entry['inputs'][name])])
                self.assertCountEqual(consumed, links)
                self.assertCountEqual(produced, links)

    def test_regeneration_is_exact_and_old_graphs_untouched(self):
        old = {p: p.read_bytes() for p in (ROOT / 'workflows').glob('*.json') if not p.name.startswith(STEM)}
        with tempfile.TemporaryDirectory() as folder:
            marker = Path(folder) / 'keep.txt'
            marker.write_text('preserved')
            paths = generator().write_workflows(output_dir=folder)
            self.assertEqual(len(paths), 6)
            for path in map(Path, paths):
                self.assertEqual(path.read_bytes(), (ROOT / 'workflows' / path.name).read_bytes())
                self.assertNotIn(b'\r\n', path.read_bytes())
            self.assertEqual(marker.read_text(), 'preserved')
        for path, data in old.items():
            self.assertEqual(path.read_bytes(), data)

    def test_case_seed_trial_cli_and_validation(self):
        g = generator()
        for kwargs in ({'case_id': 'bad'}, {'case_id': 'all', 'seed': 7}, {'seed': True},
                       {'seed': -1}, {'seed': 2**64}, {'trial_id': -1}, {'trial_id': True}, {'trial_id': 2**31}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    g.write_workflows(output_dir=folder, **kwargs)
                self.assertEqual(list(Path(folder).iterdir()), [])
        with tempfile.TemporaryDirectory() as folder:
            proc = subprocess.run([sys.executable, '-S', str(ROOT / 'tools/build_attention_ablation_workflows.py'),
                '--case', 'park', '--seed', '987654321', '--trial-id', '9', '--output-dir', folder], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            api = read(folder, 'park', '_api.json')
            for n in api.values():
                if n['class_type'] in (COLLECT, SAMPLER):
                    self.assertEqual(n['inputs']['seed'], 987654321)
                    self.assertEqual(n['inputs']['trial_id'], 9)


if __name__ == '__main__':
    unittest.main()


def reuse_history(tmp_path):
    """Persist a realistic synthetic park history using the existing artifact savers."""
    from test_attention_reuse import make_history
    prior = generator().load_tool('build_attention_validation_workflows.py')
    prompt = next(c['prompt'] for c in prior.build_cases() if c['case_id'] == 'park')
    return make_history(tmp_path, prompt=prompt, seed=444444, background_phrase='park path')


def reuse_kwargs(history):
    return dict(reuse_suite=history['suite_manifest'], reuse_edit=history['edit_report'],
        input_dir=history['input_dir'], reuse_target=history['target_image'],
        reuse_protected=history['protected_image'], asset_record=history['asset_record'])


def test_reuse_cli_runs_outside_checkout_and_has_only_two_new_edits(tmp_path):
    h = reuse_history(tmp_path)
    output = tmp_path / 'generated'
    args = [sys.executable, '-B', str(ROOT / 'tools/build_attention_ablation_workflows.py'),
        '--case', 'park', '--output-dir', str(output)]
    for name, value in reuse_kwargs(h).items():
        args.extend(['--' + name.replace('_', '-'), str(value)])
    proc = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    paths = json.loads(proc.stdout)['files']
    assert len(paths) == 3
    api = read(output, 'park_reuse', '_api.json')
    ui = read(output, 'park_reuse', '.json')
    manifest = read(output, 'park_reuse', '.index.json')
    collectors = [n['inputs'] for n in api.values() if n['class_type'] == COLLECT]
    assert len(collectors) == 1
    assert json.loads(collectors[0]['config_json'])['prototypes'] == 1
    samplers = [n['inputs'] for n in api.values() if n['class_type'] == SAMPLER]
    assert sorted(s['strength'] for s in samplers) == [2., 4.]
    loaders = {sid: n['inputs'] for sid, n in api.items() if n['class_type'] == 'LoadImageMask'}
    assert loaders == {'130': {'image': 'target.png', 'channel': 'red'},
                       '131': {'image': 'protected.png', 'channel': 'red'}}
    assert api['140']['inputs']['target_mask'] == ['130', 0]
    assert api['140']['inputs']['protected_mask'] == ['131', 0]
    assert api['171']['inputs']['subjects'] == ['140', 0]
    assert manifest['cost_accounting']['expected_this_graph_model_nfe'] == 34
    assert manifest['reuse_verification']['new_execution_hashes_verified'] is False
    assert {r['label'] for r in manifest['variants']} == {'bg_k1_s4', 'bg_k3_s2'}
    assert manifest['evaluation_record_template']['reused_k3_s4']['evaluation']['strength'] == 4.
    # Reconstruct all links/widgets, including core LoadImageMask, without Comfy.
    nodes = {str(n['id']): n for n in ui['nodes'] if n['type'] != 'Note'}
    links = {row[0]: row for row in ui['links']}
    assert set(nodes) == set(api)
    for sid, node in nodes.items():
        reconstructed = dict(node['widgets_values_named'])
        for i, port in enumerate(node['inputs']):
            if port['link'] is not None:
                _, origin, slot, destination, inslot, kind = links[port['link']]
                assert (destination, inslot) == (int(sid), i)
                assert nodes[str(origin)]['outputs'][slot]['type'] == kind == port['type']
                reconstructed[port['name']] = [str(origin), slot]
        assert reconstructed == api[sid]['inputs']
    for row in manifest['variants']:
        group = next(g for g in ui['groups'] if g['title'] == row['label'])
        sampler = nodes[row['sampler_node']]
        x, y, w, height = group['bounding']
        assert x <= sampler['pos'][0] <= x + w and y <= sampler['pos'][1] <= y + height


def test_reuse_rejects_missing_or_mismatched_controls_before_writing(tmp_path):
    import pytest
    h = reuse_history(tmp_path)
    output = tmp_path / 'output'
    g = generator()
    for kwargs in (dict(reuse_suite=h['suite_manifest']),
        dict(case_id='all', **reuse_kwargs(h)),
        dict(case_id='park', seed=42, **reuse_kwargs(h)),
        dict(case_id='woman_front_strong_overlap', **reuse_kwargs(h))):
        with pytest.raises(ValueError):
            g.write_workflows(output_dir=output, **kwargs)
        assert not output.exists()
    from test_attention_reuse import update_document
    update_document(h, 'suite_manifest', lambda d: d['report'].update(background_phrase='trees'))
    with pytest.raises(ValueError, match='background'):
        g.write_workflows(output_dir=output, case_id='park', **reuse_kwargs(h))
    assert not output.exists()
