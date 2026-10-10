"""Prepare a final-mask core exclusion, without model inference or resizing.

Supply M=post-expansion effective support, P=matching saved protection and a
human-authored core on the identical binary mask grid. Output T=M & ~core & ~P,
P_final=P | core and removed=M & core. M/P overlap is an error, never repaired.
Every input and T must be nonempty. The core may extend beyond both M and P.
This is validation infrastructure, not an anatomy estimator or a no-OFF-image
automation pipeline; the caller must supply the source masks and core.

--effective-provenance and --core-human-verified are explicit caller assertions,
not semantic or run-alignment certifications. Reconstructed masks are labeled
as reconstructed; filenames never establish their case or source. Outputs are
created in a NEW directory; existing directories/symlinks are refused. On a
write failure, files created by this call are removed. JSON goes to stdout;
exit 0 means prepared, exit 2 means invalid inputs or an I/O error.

Replay must use selection_dilate_radius=0, fill_holes_max_area=0 and
mask_dilate_radius=0. Excluding direct support does not establish preservation
of the protected person's appearance, background identity or image quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import struct
import sys

from PIL import Image

try:
    from .audit_mask_support import load_binary_png
    from . import build_hybrid_validation_workflows as generator
except ImportError:
    from audit_mask_support import load_binary_png
    import build_hybrid_validation_workflows as generator


def geometry_sha256(size, mask):
    """Hash grid dimensions and row-major 0/1 cells independently of encoding."""
    return hashlib.sha256(b'binary_mask_geometry_v1\0' + struct.pack('>QQ', *size) + bytes(mask)).hexdigest()


def png_bytes(size, mask):
    stream = BytesIO()
    Image.frombytes('L', size, bytes(value * 255 for value in mask)).save(stream, format='PNG')
    return stream.getvalue()


ROOT = Path(__file__).resolve().parents[1]
MANUAL = 'krea2_female_slider_hybrid_manual_d'


def validate_manual_pair(api, ui):
    """Reject any graph/schema drift or UI/API execution mismatch in manual D.

    This is deliberately not a generic ComfyUI editor. The pinned manual-D
    topology and widget/port schemas are the only supported template.
    """
    reference = generator.read_json(ROOT / 'workflows' / (MANUAL + '_api.json'))
    reference_ui = generator.read_json(ROOT / 'workflows' / (MANUAL + '.json'))
    if not isinstance(api, dict) or set(api) != set(reference):
        raise ValueError('unsupported_manual_graph: node IDs differ from pinned manual D')
    for sid, expected in reference.items():
        actual = api[sid]
        if (not isinstance(actual, dict) or actual.get('class_type') != expected['class_type']
                or not isinstance(actual.get('inputs'), dict)
                or set(actual['inputs']) != set(expected['inputs'])):
            raise ValueError('unsupported_manual_schema: ' + sid)
        metadata = actual.get('_meta', {})
        if not isinstance(metadata, dict) or ('title' in metadata and not isinstance(metadata['title'], str)):
            raise ValueError('invalid_api_metadata: ' + sid)
        for key, value in actual['inputs'].items():
            original = expected['inputs'][key]
            if isinstance(original, list):
                valid = (isinstance(value, list) and len(value) == 2
                         and type(value[0]) is str and type(value[1]) is int
                         and value[1] >= 0 and value == original)
            elif type(original) is int:
                valid = type(value) is int
            elif type(original) is float:
                valid = type(value) in (int, float) and math.isfinite(value)
            else:
                valid = isinstance(value, str) and bool(value.strip())
            if not valid:
                raise ValueError('unsupported_manual_input: ' + sid + '.' + key)
    sampler = api['21']['inputs']
    for key, expected in {'mask_mode': 'manual', 'mix_scope': 'target_mask',
                          'fill_holes_max_area': 0, 'mask_dilate_radius': 0}.items():
        if sampler[key] != expected:
            raise ValueError('unsupported_source_mask_transform: ' + key)
    # Bounds are the pinned custom node's define_schema contract, checked
    # without importing nodes.py, torch or the inference runtime.
    ranges = {'strength': (-10, 10), 'steps': (1, 100), 'cfg': (1, 1),
              'trial_id': (0, 0x7fffffff), 'collect_step': (1, 100),
              'collect_block': (0, 100), 'top_k_ratio': (.001, 1),
              'temperature': (.001, 100000), 'selection_dilate_radius': (0, 16)}
    for key, (minimum, maximum) in ranges.items():
        if not minimum <= sampler[key] <= maximum:
            raise ValueError('unsupported_source_setting: ' + key)
    if sampler['diagnostic_level'] not in ('audit', 'summary'):
        raise ValueError('unsupported_source_setting: diagnostic_level')
    if api['7']['inputs']['batch_size'] != 1 or api['3']['inputs']['type'] != 'krea2':
        raise ValueError('pinned_manual_replay_requires_batch1_and_krea2_clip')
    for key in ('target_occurrence', 'protected_occurrence'):
        if not 0 <= api['20']['inputs'][key] <= 999:
            raise ValueError('unsupported_source_setting: ' + key)
    if not 0 <= sampler['seed'] <= 0xffffffffffffffff:
        raise ValueError('invalid_seed')
    if any(api[sid]['inputs']['channel'] != 'red' for sid in ('200', '201')):
        raise ValueError('manual_masks_require_red_channel')
    for sid in ('200', '201'):
        generator.input_filename(api[sid]['inputs']['image'])
    if any(api['202']['inputs'][key] != reference['202']['inputs'][key] for key in ('x', 'y', 'operation')):
        raise ValueError('manual_subtraction_must_be_unchanged')
    if not isinstance(ui, dict) or ui.get('version') != 0.4:
        raise ValueError('unsupported_ui_version')
    all_nodes = ui.get('nodes', [])
    if (not isinstance(all_nodes, list) or any(not isinstance(n, dict)
            or type(n.get('id')) is not int or n['id'] <= 0 for n in all_nodes)):
        raise ValueError('invalid_ui_node_id')
    if len({n['id'] for n in all_nodes}) != len(all_nodes):
        raise ValueError('duplicate_ui_node')
    nodes = {str(n['id']): n for n in all_nodes if n['type'] != 'Note'}
    if set(nodes) != set(api):
        raise ValueError('ui_api_node_mismatch')
    templates = {str(n['id']): n for n in reference_ui['nodes'] if n['type'] != 'Note'}
    links_list = ui.get('links', [])
    if (not isinstance(links_list, list) or any(not isinstance(link, list) or len(link) != 6
            or any(type(value) is not int or value < 0 for value in link[:5])
            or any(link[index] <= 0 for index in (0, 1, 3))
            or not isinstance(link[5], str) for link in links_list)):
        raise ValueError('invalid_ui_link')
    links = {link[0]: link for link in links_list}
    if len(links) != len(links_list):
        raise ValueError('duplicate_ui_link')
    consumed, produced = [], []
    for sid, node in nodes.items():
        template, entry = templates[sid], api[sid]
        if node.get('type') != entry['class_type'] or type(node.get('mode')) is not int or node['mode'] != 0:
            raise ValueError('ui_api_type_or_mode_mismatch: ' + sid)
        names = list(template['widgets_values_named'])
        expected_named = {key: entry['inputs'][key] for key in names}
        expected_values = list(expected_named.values())
        if sid == '21':
            expected_values.insert(names.index('seed') + 1, 'fixed')
        elif node['type'] == 'LoadImageMask':
            expected_values.append('image')
        if (node.get('widgets_values_named') != expected_named or node.get('widgets_values') != expected_values
                or any(type(value) is bool for value in node['widgets_values_named'].values())
                or any(type(value) is bool for value in node['widgets_values'])):
            raise ValueError('ui_api_widget_mismatch: ' + sid)
        for direction in ('inputs', 'outputs'):
            if [(p['name'], p['type']) for p in node.get(direction, [])] != [(p['name'], p['type']) for p in template[direction]]:
                raise ValueError('unsupported_ui_ports: ' + sid)
        for slot, port in enumerate(node['inputs']):
            if type(port.get('link')) is not int or port['link'] <= 0:
                raise ValueError('invalid_ui_input_link: ' + sid)
            link = links.get(port.get('link'))
            source = entry['inputs'][port['name']]
            if link != [port.get('link'), int(source[0]), source[1], int(sid), slot, port['type']]:
                raise ValueError('ui_api_link_mismatch: ' + sid)
            consumed.append(link[0])
        for slot, port in enumerate(node['outputs']):
            for lid in port.get('links') or []:
                if type(lid) is not int or lid not in links or links[lid][1:3] != [int(sid), slot]:
                    raise ValueError('ui_api_output_link_mismatch: ' + sid)
                produced.append(lid)
    if Counter(consumed) != Counter(links.keys()) or Counter(produced) != Counter(links.keys()):
        raise ValueError('ui_link_coverage_mismatch')


def build_replay(api_path, ui_path, expected_grid, input_subdir, output_prefix):
    """Read an explicit source pair and make only a bounded radius-zero replay."""
    api_bytes, ui_bytes = Path(api_path).read_bytes(), Path(ui_path).read_bytes()
    api, ui = json.loads(api_bytes), json.loads(ui_bytes)
    validate_manual_pair(api, ui)
    latent = api['7']['inputs']
    # The pinned Krea2 graph uses a 16-image-pixel token/mask cell. This is an
    # explicit template assumption, not proof of a live runtime's tensor grid.
    if (latent['width'], latent['height']) != (expected_grid[0] * 16, expected_grid[1] * 16):
        raise ValueError('replay_grid_mismatch: pinned Krea2 template requires latent image size = expected mask grid x 16')
    if not isinstance(output_prefix, str) or not output_prefix:
        raise ValueError('output_prefix_required_for_replay')
    generator.input_filename(output_prefix + '/output.png')
    if input_subdir:
        generator.input_filename(input_subdir + '/target_final.png')
    target_name = (input_subdir + '/' if input_subdir else '') + 'target_final.png'
    protected_name = (input_subdir + '/' if input_subdir else '') + 'protected_final.png'
    replay = deepcopy(api)
    replay['200']['inputs']['image'] = target_name
    replay['201']['inputs']['image'] = protected_name
    replay['21']['inputs']['selection_dilate_radius'] = 0
    export_names = {'23': 'E_core_clipped_final', '102': 'E_core_clipped_effective_prediction_mask',
                    '104': 'E_core_clipped_selection_added_mask', '106': 'E_core_clipped_reference_target',
                    '108': 'E_core_clipped_reference_protected', '204': 'E_core_clipped_target_input',
                    '206': 'E_core_clipped_protected_input', '208': 'E_core_clipped_target_minus_protected'}
    for sid, name in export_names.items():
        replay[sid]['inputs']['filename_prefix'] = output_prefix + '/' + name
    for sid, title in {'20': 'E_core_clipped: prepared final target and core-unioned protection',
                        '21': 'E_core_clipped: unchanged source settings, radius 0',
                        '200': 'Prepared final target M & ~core & ~P',
                        '201': 'Prepared final protection P | core',
                        '202': 'Idempotent safety subtraction of prepared protection'}.items():
        replay[sid].setdefault('_meta', {})['title'] = title
    replay_ui = generator.make_ui(replay, ui, manual=True, radius=0)
    notes = [
        'E_core_clipped: load target_final.png and protected_final.png from this preparation bundle. White=1, black=0; channel=red. Preserve the exact same grid. This is post-expansion core clipping, not a second dilation.',
        'Source model, prompt, seed, sampler settings and trial_id are unchanged; only loaded masks, selection radius=0 and output prefixes change. Core review and source provenance in manifest.json are explicit caller assertions, not automatic certifications.',
        'The pinned Krea2 template assumes one mask cell per 16 image pixels. Verify the live sampler grid and source-run alignment. Audit the actually saved effective mask against target_final.png, protected_final.png and the original core. Direct-mask exclusion does not prove image or protected-person preservation.',
    ]
    for node, note in zip((n for n in replay_ui['nodes'] if n['type'] == 'Note'), notes):
        node['widgets_values'] = [note]
        node['title'] = 'Core replay instructions'
    validate_manual_pair(replay, replay_ui)
    payloads = {'replay_api.json': (json.dumps(replay, indent=2, allow_nan=False) + '\n').encode(),
                'replay.json': (json.dumps(replay_ui, indent=2, allow_nan=False) + '\n').encode()}
    details = {'workflow': {'api': 'replay_api.json', 'ui': 'replay.json'},
               'experiment_label': 'E_core_clipped', 'output_prefix': output_prefix,
               'input_filenames': {'target': target_name, 'protected': protected_name},
               'grid_mapping_assumption': 'pinned_Krea2_latent_image_16px_per_mask_cell',
               'source_workflow_api': {'path': str(api_path), 'file_sha256': hashlib.sha256(api_bytes).hexdigest()},
               'source_workflow_ui': {'path': str(ui_path), 'file_sha256': hashlib.sha256(ui_bytes).hexdigest()},
               'source_selection_dilate_radius': api['21']['inputs']['selection_dilate_radius'],
               'trial_id': api['21']['inputs']['trial_id'], 'ui_api_static_validation': 'passed',
               'workflow_file_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}
    return payloads, details


def prepare(effective, protected, core, output_dir, effective_provenance,
            core_human_verified, expected_grid, workflow_api=None, workflow_ui=None,
            input_subdir=None, output_prefix=None):
    """Validate, prepare and exclusively save one mask bundle; return its manifest."""
    if (workflow_api is None) != (workflow_ui is None):
        raise ValueError('workflow_api_and_ui_required_together')
    if workflow_api is None and (input_subdir is not None or output_prefix is not None):
        raise ValueError('replay_options_require_explicit_workflow_pair')
    if effective_provenance not in ('actual_saved', 'reconstructed'):
        raise ValueError('effective_provenance must be actual_saved or reconstructed')
    if type(core_human_verified) is not bool:
        raise ValueError('core_human_verified must be an explicit boolean')
    if (not isinstance(expected_grid, (tuple, list)) or len(expected_grid) != 2
            or any(type(value) is not int or value <= 0 for value in expected_grid)):
        raise ValueError('expected_grid must contain two positive integers')
    expected_grid = tuple(expected_grid)
    masks, inputs = {}, {}
    size = None
    for name, path in (('effective', effective), ('protected', protected), ('core', core)):
        # Hash exactly the byte snapshot which the trusted strict loader decodes.
        data = Path(path).read_bytes()
        grid, mask = load_binary_png(BytesIO(data))
        if size is not None and grid != size:
            raise ValueError('grid_mismatch: all masks must share one grid; no resizing')
        size = grid
        if size != expected_grid:
            raise ValueError('expected_grid_mismatch: input is not on the explicitly expected sampler grid')
        if not any(mask):
            raise ValueError('empty_' + name)
        masks[name] = mask
        inputs[name] = {'path': str(path), 'file_sha256': hashlib.sha256(data).hexdigest(),
                        'geometry_sha256': geometry_sha256(size, mask), 'cells': sum(mask)}
    inputs['effective'].update(provenance=effective_provenance, provenance_verified_by_tool=False)
    source, protection, exclusion = (masks[name] for name in ('effective', 'protected', 'core'))
    if any(m and p for m, p in zip(source, protection)):
        raise ValueError('source_effective_protected_overlap: use a matching disjoint saved source pair')
    target = bytearray(m and not c and not p for m, p, c in zip(source, protection, exclusion))
    final_protected = bytearray(p or c for p, c in zip(protection, exclusion))
    removed = bytearray(m and c for m, c in zip(source, exclusion))
    if not any(target):
        raise ValueError('empty_final_target')
    output_masks = {'target_final': target, 'protected_final': final_protected, 'core_removed': removed}
    payloads = {name + '.png': png_bytes(size, mask) for name, mask in output_masks.items()}
    report = {
        'schema_version': 1, 'status': 'prepared_not_gpu_validated',
        'scope': 'binary_saved_grid_preparation_only',
        'expected_sampler_grid': {'width': expected_grid[0], 'height': expected_grid[1],
                                  'source': 'explicit_caller_assertion', 'runtime_alignment_verified': False},
        'grid': {'width': size[0], 'height': size[1], 'unit': 'mask_grid_cell'},
        'formula': {'target_final': 'effective & ~core & ~protected',
                    'protected_final': 'protected | core', 'core_removed': 'effective & core'},
        'inputs': inputs,
        'core_review': {'human_verified': core_human_verified,
                        'source': 'explicit_caller_assertion', 'semantic_verification_by_tool': False},
        'counts': {'source_effective': sum(source), 'source_protected': sum(protection),
                   'core': sum(exclusion), 'final_target': sum(target),
                   'final_protected': sum(final_protected), 'removed': sum(removed),
                   'core_outside_source_protected': sum(c and not p for c, p in zip(exclusion, protection))},
        'outputs': {name: {'filename': name + '.png',
                           'file_sha256': hashlib.sha256(payloads[name + '.png']).hexdigest(),
                           'geometry_sha256': geometry_sha256(size, mask), 'cells': sum(mask)}
                    for name, mask in output_masks.items()},
        'replay': {'selection_dilate_radius': 0, 'mask_dilate_radius': 0,
                   'fill_holes_max_area': 0, 'gpu_execution': 'not_run',
                   'workflow': 'not_requested',
                   'actual_saved_source_asserted': effective_provenance == 'actual_saved',
                   'human_review_required_before_use': not core_human_verified},
        'limitations': {'protected_appearance': 'not_established', 'image_quality': 'not_assessed',
                        'background_identity': 'not_established', 'source_run_alignment': 'not_verified',
                        'direct_support_exclusion_only': True,
                        'source_provenance': 'caller_asserted_not_independently_verified'},
    }
    if workflow_api is not None:
        replay_payloads, details = build_replay(workflow_api, workflow_ui, expected_grid, input_subdir, output_prefix)
        payloads.update(replay_payloads)
        report['replay'].update(details)
    payloads['manifest.json'] = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode()
    destination = Path(output_dir)
    destination.mkdir()  # Exclusive ownership: refuse existing directories and symlinks.
    created = []
    try:
        for name, data in payloads.items():
            path = destination / name
            with path.open('xb') as stream:
                created.append(path)
                stream.write(data)
    except BaseException:
        for path in reversed(created):
            path.unlink()
        destination.rmdir()
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('effective', 'protected', 'core'):
        parser.add_argument('--' + name, type=Path, required=True, help='Strict binary PNG on the shared saved mask grid')
    parser.add_argument('--expected-grid', type=int, nargs=2, required=True, metavar=('WIDTH', 'HEIGHT'),
                        help='Observed or independently verified sampler mask grid, not the output image size')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; parent must already exist')
    parser.add_argument('--effective-provenance', choices=('actual_saved', 'reconstructed'), required=True)
    parser.add_argument('--core-human-verified', choices=('true', 'false'), required=True,
                        help='Explicit caller review assertion; never certified automatically')
    parser.add_argument('--workflow-api', type=Path, help='Explicit matching source manual-D API workflow')
    parser.add_argument('--workflow-ui', type=Path, help='Explicit matching source manual-D UI workflow')
    parser.add_argument('--input-subdir', help='Optional relative ComfyUI input folder; default is flat filenames')
    parser.add_argument('--output-prefix', help='Required with replay; use a distinct experiment prefix such as E_core_clipped/run1')
    args = vars(parser.parse_args(argv))
    args['core_human_verified'] = args['core_human_verified'] == 'true'
    try:
        report = prepare(**args)
    except (OSError, ValueError, KeyError, TypeError, Image.DecompressionBombError) as error:
        print(json.dumps({'status': 'failed', 'error': str(error)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
