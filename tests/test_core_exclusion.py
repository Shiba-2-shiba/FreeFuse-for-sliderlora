"""Offline final-core preparation contracts: real PNG files, no torch/inference."""
from copy import deepcopy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools/prepare_core_exclusion.py'
MANUAL = 'krea2_female_slider_hybrid_manual_d'


class CoreExclusionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.output = self.directory / 'prepared'

    def png(self, name, cells=(), size=(7, 6), mode='L'):
        image = Image.new('L', size, 0)
        for cell in cells:
            image.putpixel(cell, 255)
        if mode != 'L':
            image = image.convert(mode)
        path = self.directory / (name + '.png')
        image.save(path)
        return path

    def inputs(self):
        return {
            'effective': self.png('ambiguous_D_r4_filename', [(1, 1), (2, 1), (3, 1), (2, 2)]),
            'protected': self.png('P', [(5, 4)]),
            # Outside P and outside M are both allowed; both become protected.
            'core': self.png('core', [(2, 1), (2, 2), (4, 4)]),
        }

    def run_cli(self, paths=None, *options, provenance='actual_saved', reviewed='false', expected_grid=(7, 6)):
        self.assertTrue(TOOL.is_file(), 'The final-core preparation CLI has not been implemented')
        args = [sys.executable, str(TOOL)]
        for name, path in (paths or self.inputs()).items():
            args.extend(['--' + name, str(path)])
        args.extend(['--output-dir', str(self.output), '--effective-provenance', provenance,
                     '--core-human-verified', reviewed, '--expected-grid', *map(str, expected_grid), *map(str, options)])
        result = subprocess.run(args, capture_output=True, text=True)
        self.assertTrue(result.stdout.strip(), 'CLI must return JSON: ' + result.stderr)
        return result, json.loads(result.stdout)

    def pixels(self, name):
        with Image.open(self.output / name) as image:
            return image.size, {i for i, value in enumerate(image.tobytes()) if value == 255}

    def test_subtracts_core_after_expansion_and_unions_protection(self):
        paths = self.inputs()
        before = {name: path.read_bytes() for name, path in paths.items()}
        result, report = self.run_cli(paths)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.pixels('target_final.png'), ((7, 6), {8, 10}))
        self.assertEqual(self.pixels('protected_final.png'), ((7, 6), {9, 16, 32, 33}))
        self.assertEqual(self.pixels('core_removed.png'), ((7, 6), {9, 16}))
        self.assertEqual(before, {name: path.read_bytes() for name, path in paths.items()})
        self.assertEqual(report['counts'], {'source_effective': 4, 'source_protected': 1,
                         'core': 3, 'final_target': 2, 'final_protected': 4, 'removed': 2,
                         'core_outside_source_protected': 3})
        self.assertEqual(report['replay']['selection_dilate_radius'], 0)
        self.assertEqual(report['replay']['gpu_execution'], 'not_run')
        self.assertEqual(report, json.loads((self.output / 'manifest.json').read_text()))

    def test_manifest_separates_file_geometry_and_asserted_provenance(self):
        paths = self.inputs()
        result, report = self.run_cli(paths, reviewed='true')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(report['inputs']['effective']['file_sha256'], hashlib.sha256(paths['effective'].read_bytes()).hexdigest())
        self.assertEqual(report['inputs']['effective']['provenance'], 'actual_saved')
        self.assertEqual(report['inputs']['effective']['provenance_verified_by_tool'], False)
        self.assertEqual(report['core_review'], {'human_verified': True, 'source': 'explicit_caller_assertion', 'semantic_verification_by_tool': False})
        self.assertEqual(report['limitations']['protected_appearance'], 'not_established')
        self.assertEqual(report['limitations']['source_run_alignment'], 'not_verified')
        self.assertNotIn('case', report)
        self.assertEqual(report['outputs']['target_final']['file_sha256'], hashlib.sha256((self.output / 'target_final.png').read_bytes()).hexdigest())
        self.assertEqual(len(report['inputs']['effective']['geometry_sha256']), 64)

    def test_reconstructed_source_is_never_labeled_saved_actual_evidence(self):
        result, report = self.run_cli(provenance='reconstructed')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(report['inputs']['effective']['provenance'], 'reconstructed')
        self.assertFalse(report['replay']['actual_saved_source_asserted'])
        self.assertFalse(report['core_review']['human_verified'])

    def test_binary_geometry_hash_ignores_png_encoding_but_includes_grid(self):
        paths = self.inputs()
        _, first = self.run_cli(paths)
        self.output = self.directory / 'rgb_prepared'
        paths['effective'] = self.png('rgb', [(1, 1), (2, 1), (3, 1), (2, 2)], mode='RGB')
        _, second = self.run_cli(paths)
        self.assertEqual(first['inputs']['effective']['geometry_sha256'], second['inputs']['effective']['geometry_sha256'])
        self.assertNotEqual(first['inputs']['effective']['file_sha256'], second['inputs']['effective']['file_sha256'])
        self.output = self.directory / 'reshaped'
        paths = {'effective': self.png('wideM', [(8, 0), (9, 0), (10, 0), (2, 1)], size=(14, 3)),
                 'protected': self.png('wideP', [(5, 2)], size=(14, 3)),
                 'core': self.png('wideC', [(9, 0)], size=(14, 3))}
        _, third = self.run_cli(paths, expected_grid=(14, 3))
        self.assertNotEqual(first['inputs']['effective']['geometry_sha256'], third['inputs']['effective']['geometry_sha256'])

    def test_source_protected_overlap_is_rejected_without_silent_repair(self):
        paths = self.inputs()
        paths['protected'] = self.png('overlap', [(2, 1)])
        result, report = self.run_cli(paths)
        self.assertEqual(result.returncode, 2)
        self.assertIn('source_effective_protected_overlap', report['error'])
        self.assertFalse(self.output.exists())

    def test_empty_inputs_and_empty_final_target_are_rejected(self):
        for name in ('effective', 'protected', 'core', 'final_target'):
            with self.subTest(name=name):
                paths = self.inputs()
                if name == 'final_target':
                    paths['core'] = paths['effective']
                else:
                    paths[name] = self.png('empty')
                result, report = self.run_cli(paths)
                self.assertEqual(result.returncode, 2)
                self.assertIn('empty_' + name, report['error'])
                self.assertFalse(self.output.exists())

    def test_grid_mismatch_is_rejected_without_resizing(self):
        paths = self.inputs()
        paths['core'] = self.png('wrong_grid', [(2, 1)], size=(14, 12))
        result, report = self.run_cli(paths)
        self.assertEqual(result.returncode, 2)
        self.assertIn('grid_mismatch', report['error'])
        self.assertFalse(self.output.exists())

    def test_strict_loader_rejects_soft_alpha_palette_and_unequal_rgb(self):
        for variant in ('soft', 'alpha', 'palette', 'color', 'transparency'):
            with self.subTest(variant=variant):
                paths = self.inputs()
                with Image.open(paths['core']) as source:
                    image = source.copy()
                if variant == 'soft':
                    image.putpixel((0, 0), 127)
                elif variant == 'alpha':
                    image = image.convert('RGBA')
                elif variant == 'palette':
                    image = image.convert('P')
                elif variant == 'color':
                    image = image.convert('RGB'); image.putpixel((0, 0), (255, 0, 0))
                else:
                    image.info['transparency'] = 0
                image.save(paths['core'])
                result, _ = self.run_cli(paths)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.output.exists())

    def test_nonintersecting_core_still_becomes_immutable_protected_support(self):
        paths = self.inputs()
        paths['core'] = self.png('outside', [(4, 4)])
        result, report = self.run_cli(paths)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(report['counts']['removed'], 0)
        self.assertEqual(report['counts']['final_target'], 4)
        self.assertEqual(self.pixels('protected_final.png')[1], {32, 33})

    def test_existing_output_directory_and_symlink_are_never_overwritten(self):
        self.output.mkdir()
        marker = self.output / 'target_final.png'
        marker.write_bytes(b'keep')
        result, _ = self.run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(marker.read_bytes(), b'keep')
        linked = self.directory / 'linked'
        linked.symlink_to(self.output, target_is_directory=True)
        self.output = linked
        result, _ = self.run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(marker.read_bytes(), b'keep')

    def test_expected_sampling_grid_is_required_and_must_match(self):
        result, report = self.run_cli(expected_grid=(64, 64))
        self.assertEqual(result.returncode, 2)
        self.assertIn('expected_grid_mismatch', report['error'])
        self.assertFalse(self.output.exists())

    def module(self):
        self.assertTrue(TOOL.is_file())
        spec = importlib.util.spec_from_file_location('tools.prepare_core_exclusion', TOOL)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_late_write_failure_rolls_back_new_bundle(self):
        module = self.module()
        self.assertIn('expected_grid', inspect.signature(module.prepare).parameters)
        real_open = Path.open
        def fail_manifest(path, *args, **kwargs):
            if path == self.output / 'manifest.json':
                raise OSError('simulated full filesystem')
            return real_open(path, *args, **kwargs)
        with patch.object(Path, 'open', fail_manifest):
            with self.assertRaisesRegex(OSError, 'simulated full filesystem'):
                module.prepare(**self.inputs(), output_dir=self.output,
                               effective_provenance='actual_saved', core_human_verified=False,
                               expected_grid=(7, 6))
        self.assertFalse(self.output.exists())

    def source_pair(self, change=None):
        api = json.loads((ROOT / 'workflows' / (MANUAL + '_api.json')).read_text())
        ui = json.loads((ROOT / 'workflows' / (MANUAL + '.json')).read_text())
        # Distinct source settings catch regeneration from default case/seed/model.
        api['4']['inputs']['prompt'] = 'Explicit caller source prompt, no filename inference.'
        api['21']['inputs'].update(seed=123456, trial_id=17)
        api['1']['inputs']['unet_name'] = 'caller_model.safetensors'
        api['2']['inputs']['strength_model'] = 0.65
        if change:
            change(api)
        for node in ui['nodes']:
            if node['type'] == 'Note' or str(node['id']) not in api:
                continue
            inputs = api[str(node['id'])]['inputs']
            named = {key: inputs[key] for key in node['widgets_values_named']}
            values = list(named.values())
            if node['id'] == 21:
                values.insert(list(named).index('seed') + 1, 'fixed')
            elif node['type'] == 'LoadImageMask':
                values.append('image')
            node['widgets_values_named'], node['widgets_values'] = named, values
        api_path, ui_path = self.directory / 'source_api.json', self.directory / 'source_ui.json'
        api_path.write_text(json.dumps(api)); ui_path.write_text(json.dumps(ui))
        return api, ui, api_path, ui_path

    def replay_inputs(self):
        return {'effective': self.png('M64', [(1, 1), (2, 1), (3, 1)], size=(64, 64)),
                'protected': self.png('P64', [(5, 4)], size=(64, 64)),
                'core': self.png('C64', [(2, 1)], size=(64, 64))}

    def replay(self, api_path, ui_path, *options):
        return self.run_cli(self.replay_inputs(), '--workflow-api', api_path,
                            '--workflow-ui', ui_path, '--output-prefix', 'E_core_clipped/test',
                            *options, expected_grid=(64, 64))

    def test_replay_changes_only_masks_radius_and_output_prefixes(self):
        source, source_ui, api_path, ui_path = self.source_pair()
        before = api_path.read_bytes(), ui_path.read_bytes()
        result, report = self.replay(api_path, ui_path)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        replay = json.loads((self.output / 'replay_api.json').read_text())
        ui = json.loads((self.output / 'replay.json').read_text())
        self.assertEqual(set(replay), set(source))
        for sid, original in source.items():
            expected = deepcopy(original['inputs'])
            if sid == '200': expected['image'] = 'target_final.png'
            if sid == '201': expected['image'] = 'protected_final.png'
            if sid == '21': expected['selection_dilate_radius'] = 0
            if 'filename_prefix' in expected:
                self.assertTrue(replay[sid]['inputs']['filename_prefix'].startswith('E_core_clipped/test/'))
                expected['filename_prefix'] = replay[sid]['inputs']['filename_prefix']
            self.assertEqual(replay[sid]['class_type'], original['class_type'])
            self.assertEqual(replay[sid]['inputs'], expected, sid)
        self.assertEqual(before, (api_path.read_bytes(), ui_path.read_bytes()))
        self.assertEqual(report['replay']['workflow'], {'api': 'replay_api.json', 'ui': 'replay.json'})
        self.assertEqual(report['replay']['grid_mapping_assumption'], 'pinned_Krea2_latent_image_16px_per_mask_cell')
        self.assertEqual(report['replay']['source_workflow_api']['file_sha256'], hashlib.sha256(before[0]).hexdigest())
        self.assertIn('E_core_clipped', ' '.join(str(n) for n in ui['nodes'] if n['type'] == 'Note'))
        self.assertEqual({n['widgets_values'][3] for n in ui['nodes'] if n['id'] == 21}, {'fixed'})
        self.module().validate_manual_pair(replay, ui)

    def test_replay_input_subdir_is_optional_and_safe(self):
        _, _, api_path, ui_path = self.source_pair()
        result, _ = self.replay(api_path, ui_path, '--input-subdir', 'core_run')
        self.assertEqual(result.returncode, 0)
        replay = json.loads((self.output / 'replay_api.json').read_text())
        self.assertEqual(replay['200']['inputs']['image'], 'core_run/target_final.png')
        self.output = self.directory / 'invalid'
        result, report = self.replay(api_path, ui_path, '--input-subdir', '../escape')
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.output.exists())

    def test_replay_rejects_source_mask_transform_or_unknown_graph(self):
        mutations = [lambda a: a['21']['inputs'].update(mask_mode='auto'),
                     lambda a: a['21']['inputs'].update(fill_holes_max_area=2),
                     lambda a: a['21']['inputs'].update(mask_dilate_radius=1),
                     lambda a: a['200']['inputs'].update(channel='alpha'),
                     lambda a: a['202']['inputs'].update(operation='add'),
                     lambda a: a['21']['inputs'].update(model=['1', 0]),
                     lambda a: a.update({'999': deepcopy(a['1'])})]
        for change in mutations:
            with self.subTest(change=change):
                _, _, api_path, ui_path = self.source_pair(change)
                result, report = self.replay(api_path, ui_path)
                self.assertEqual(result.returncode, 2, report)
                self.assertFalse(self.output.exists())

    def test_replay_rejects_ui_widget_link_or_seed_control_drift(self):
        for variant in ('named', 'positional', 'link', 'seed_control', 'disabled'):
            with self.subTest(variant=variant):
                _, ui, api_path, ui_path = self.source_pair()
                sampler = next(n for n in ui['nodes'] if n['id'] == 21)
                if variant == 'named': sampler['widgets_values_named']['seed'] += 1
                elif variant == 'positional': sampler['widgets_values'][2] += 1
                elif variant == 'link': sampler['inputs'][0]['link'] = None
                elif variant == 'seed_control': sampler['widgets_values'][3] = 'randomize'
                else: sampler['mode'] = 2
                ui_path.write_text(json.dumps(ui))
                result, report = self.replay(api_path, ui_path)
                self.assertEqual(result.returncode, 2, report)
                self.assertFalse(self.output.exists())

    def test_replay_rejects_grid_incompatible_with_source_latent(self):
        _, _, api_path, ui_path = self.source_pair(lambda a: a['7']['inputs'].update(width=2048))
        result, report = self.replay(api_path, ui_path)
        self.assertEqual(result.returncode, 2)
        self.assertIn('replay_grid_mismatch', report['error'])
        self.assertFalse(self.output.exists())

    def test_replay_requires_both_explicit_source_formats(self):
        _, _, api_path, _ = self.source_pair()
        result, report = self.run_cli(None, '--workflow-api', api_path)
        self.assertEqual(result.returncode, 2)
        self.assertIn('workflow_api_and_ui_required_together', report['error'])
        self.assertFalse(self.output.exists())

    def test_replay_rejects_unsupported_scalar_domains_before_writing(self):
        invalid = [('7', 'batch_size', 2), ('21', 'cfg', 2), ('21', 'steps', -4),
                   ('21', 'steps', 101), ('21', 'strength', 11), ('21', 'trial_id', -1),
                   ('21', 'trial_id', 2**31), ('21', 'collect_step', 0),
                   ('21', 'collect_block', 101), ('21', 'top_k_ratio', 0),
                   ('21', 'temperature', 100001), ('21', 'diagnostic_level', 'full'),
                   ('21', 'selection_dilate_radius', 17), ('20', 'target_occurrence', -1),
                   ('20', 'protected_occurrence', 1000), ('3', 'type', 'flux')]
        for index, (sid, key, value) in enumerate(invalid):
            with self.subTest(sid=sid, key=key, value=value):
                self.output = self.directory / ('invalid_scalar_' + str(index))
                _, _, api_path, ui_path = self.source_pair(lambda a: a[sid]['inputs'].update({key: value}))
                result, report = self.replay(api_path, ui_path)
                self.assertEqual(result.returncode, 2, report)
                self.assertFalse(self.output.exists())

    def test_replay_accepts_every_supported_source_radius_and_forces_zero(self):
        for radius in (1, 8, 16):
            with self.subTest(radius=radius):
                self.output = self.directory / ('source_radius_' + str(radius))
                _, _, api_path, ui_path = self.source_pair(lambda a: a['21']['inputs'].update(selection_dilate_radius=radius))
                result, report = self.replay(api_path, ui_path)
                self.assertEqual(result.returncode, 0, report)
                self.assertEqual(report['replay']['source_selection_dilate_radius'], radius)
                replay = json.loads((self.output / 'replay_api.json').read_text())
                self.assertEqual(replay['21']['inputs']['selection_dilate_radius'], 0)

    def test_replay_rejects_malformed_metadata_and_typed_graph_ambiguities(self):
        variants = ('metadata_string', 'metadata_list', 'title_list', 'api_bool_slot',
                    'ui_string_note_id', 'ui_bool_link_slot', 'ui_bool_port_link', 'ui_bool_widget')
        for variant in variants:
            with self.subTest(variant=variant):
                self.output = self.directory / variant
                api, ui, api_path, ui_path = self.source_pair()
                if variant == 'metadata_string': api['1']['_meta'] = 'bad'
                elif variant == 'metadata_list': api['1']['_meta'] = []
                elif variant == 'title_list': api['1']['_meta']['title'] = []
                elif variant == 'api_bool_slot': api['4']['inputs']['clip'][1] = False
                elif variant == 'ui_string_note_id':
                    next(n for n in ui['nodes'] if n['type'] == 'Note')['id'] = '1'
                elif variant == 'ui_bool_link_slot': ui['links'][0][2] = False
                elif variant == 'ui_bool_port_link':
                    next(n for n in ui['nodes'] if n['id'] == 2)['inputs'][0]['link'] = True
                else:
                    sampler = next(n for n in ui['nodes'] if n['id'] == 21)
                    sampler['widgets_values_named']['cfg'] = True
                    sampler['widgets_values'][5] = True
                api_path.write_text(json.dumps(api)); ui_path.write_text(json.dumps(ui))
                result, report = self.replay(api_path, ui_path)
                self.assertEqual(result.returncode, 2, report)
                self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
