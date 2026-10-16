"""Regression checks for configuration, array lineage and published evidence."""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training'))
import config
import numpy as np
import h5py
import pandas as pd
from sklearn.metrics import f1_score
from split_groups import group_rows
from patch_pool import cache_local_slice


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EncoderGeometryTests(unittest.TestCase):
    def test_resampling_preserves_footprint(self):
        from encoder_geometry import ENCODER_GSD_M, resized_gsd
        self.assertAlmostEqual(ENCODER_GSD_M * 224, 1280)
        self.assertEqual(resized_gsd(10, 128, 128), 10)
        for value in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                resized_gsd(10, 128, value)

    def test_geometry_guards_saved_artifacts(self):
        from encoder_geometry import ENCODER_GSD_M, validate_geometry
        protocol = json.loads((ROOT / 'training/full_nuisance_balance/clay_full_sweep/protocol.json').read_text())
        self.assertEqual(validate_geometry(protocol), ENCODER_GSD_M)
        checkpoint = {'gsd': ENCODER_GSD_M, 'resize': [224, 224]}
        self.assertEqual(validate_geometry(checkpoint, checkpoint=True), ENCODER_GSD_M)
        for metadata in [{}, dict(protocol, gsd_m=1), dict(protocol, native_size=64)]:
            with self.assertRaises(ValueError):
                validate_geometry(metadata)
        with self.assertRaises(ValueError):
            validate_geometry(dict(checkpoint, resize=[128, 128]), checkpoint=True)


class ConfigurationTests(unittest.TestCase):
    def test_defaults_and_repository_relative_paths(self):
        with patch.dict(os.environ, {'KELP_CONFIG': str(ROOT / 'example_config.json')}):
            self.assertEqual(config.get_path('CLAY_META'), ROOT / 'configs/metadata.yaml')
            self.assertEqual(config.get_path('FULL_NUISANCE_EMBEDDINGS'), ROOT / 'training/full_nuisance_balance/clay_full_sweep/embeddings.npz')
            with self.assertRaisesRegex(ValueError, 'FIELD_MEASUREMENTS'):
                config.get_path('FIELD_MEASUREMENTS')

    def test_overrides_references_and_cycles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            with patch.dict(os.environ, {'KELP_CONFIG': str(path)}):
                path.write_text(json.dumps({'paths': {'DATA_DIR': 'runs/example'}}))
                self.assertEqual(config.get_path('CANONICAL_CHIPS_H5'), ROOT / 'runs/example/chips/chips.h5')
                path.write_text(json.dumps({'paths': {'DATA_DIR': '{CHIPS_DIR}'}}))
                with self.assertRaisesRegex(ValueError, 'Circular'):
                    config.get_path('CANONICAL_CHIPS_H5')
                path.write_text(json.dumps({'paths': {'TYPO': 'bad'}}))
                with self.assertRaisesRegex(ValueError, 'Unknown'):
                    config.get_path('CLAY_META')

    def test_default_template_has_no_dangling_keys(self):
        defaults = json.loads((ROOT / 'example_config.json').read_text())['paths']
        with patch.dict(os.environ, {'KELP_CONFIG': str(ROOT / 'example_config.json')}):
            for key, value in defaults.items():
                if value and key not in {'PNEO_SEPTEMBER_DIR'}:
                    config.get_path(key)


class ConsolidationTests(unittest.TestCase):
    def test_duplicate_arrays_preserve_associations_and_flag_conflicts(self):
        module = load_module('consolidation', ROOT / 'chip_generation/consolidate_chips.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            array = np.ones((8, 128, 128), dtype=np.uint16)
            for tag in ['present', 'absent']:
                source = root / tag
                source.mkdir()
                inputs.append(source)
                with h5py.File(source / 'chips.h5', 'w') as f:
                    f.create_dataset('chips', data=array[None])
                    f.attrs['n_chips'] = 1
                with (source / 'chips_metadata.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['status', 'chipIndex', 'class_tag', 'observation_id'])
                    writer.writeheader()
                    writer.writerow(dict(status='ok', chipIndex=0, class_tag=tag, observation_id=tag+'-observation'))
            output = root / 'out'
            result = module.consolidate(inputs, output)
            self.assertEqual((result['arrays'], result['associations'], result['conflicting_arrays']), (1, 2, 1))
            with h5py.File(output / 'chips.h5') as f:
                np.testing.assert_array_equal(f['chips'][0], array)
            with (output / 'observation_chips.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({r['observation_id'] for r in rows}, {'present-observation', 'absent-observation'})
            self.assertEqual({r['sha256'] for r in rows}, {hashlib.sha256(array.tobytes()).hexdigest()})
            with self.assertRaises(FileExistsError):
                module.consolidate(inputs, output)


class EvidenceTests(unittest.TestCase):
    def test_published_partitions_and_initialization(self):
        root = ROOT / 'training/full_nuisance_balance'
        pool = pd.read_csv(ROOT / 'data_collection/eligible_chips.csv')
        self.assertEqual((len(pool), int(pool.label.sum())), (1198, 984))
        initial = pd.read_csv(ROOT / 'data_collection/split_initialization.csv')
        union = set()
        for seed in [42, 123, 456]:
            frame = pd.read_csv(root / f'seed_{seed}/selected.csv')
            self.assertEqual((len(frame), int(frame.label.sum())), (636, 424))
            self.assertTrue(frame.chip_id.is_unique)
            groups = group_rows(frame)
            self.assertEqual(pd.DataFrame({'group': groups, 'split': frame.split}).groupby('group').split.nunique().max(), 1)
            split = json.loads((root / f'seed_{seed}/split_seed{seed}.json').read_text())
            for name in ['train', 'val', 'test']:
                subset = frame[frame.split.eq(name)]
                self.assertEqual(int(subset.label.sum()), 2 * int(subset.label.eq(0).sum()))
                self.assertEqual(set(subset.chip_id), set(split[name+'_chip_ids']))
            self.assertTrue(set(frame.loc[frame.label.eq(0), 'chip_id']) <= set(initial.loc[initial.seed.eq(seed), 'chip_id']))
            union.update(frame.chip_id)
        self.assertEqual(union, set(pd.read_csv(ROOT / 'training/final_models/selected.csv').chip_id))
        self.assertTrue(union <= set(pool.chip_id))

    def test_saved_mlp_metrics_recompute_from_predictions(self):
        root = ROOT / 'training/full_nuisance_balance/clay_full_sweep'
        predictions = pd.read_csv(root / 'predictions.csv')
        metrics = pd.read_csv(root / 'metrics.csv')
        for row in metrics.itertuples():
            q = predictions[predictions.pool_size.eq(row.pool_size) & predictions.seed.eq(row.seed) & predictions.split.eq(row.split)]
            actual = f1_score(q.label, q.probability.ge(row.threshold), average='macro')
            self.assertAlmostEqual(actual, row.macro_f1, places=12)

    def test_final_and_field_metrics_match_saved_predictions(self):
        final = ROOT / 'training/final_models'
        predictions = pd.read_csv(final / 'predictions.csv')
        for row in pd.read_csv(final / 'metrics.csv').itertuples():
            q = predictions[predictions.model.eq(row.model) & predictions.split.eq(row.split)]
            actual = f1_score(q.label, q.probability.ge(row.threshold), average='macro')
            self.assertAlmostEqual(actual, row.macro_f1, places=12)
        field = ROOT / 'training/sesimbra_evaluation'
        predictions = pd.read_csv(field / 'predictions.csv')
        for row in pd.read_csv(field / 'metrics.csv').itertuples():
            q = predictions[predictions.model.eq(row.model) & predictions.method.eq(row.method)]
            if row.campaign != 'all':
                q = q[q.campaign.eq(row.campaign)]
            self.assertTrue(q.threshold.eq(row.threshold).all())
            self.assertEqual(len(q), row.N)
            if q.label.nunique() == 2:
                actual = f1_score(q.label, q.probability.ge(row.threshold), average='macro')
                self.assertAlmostEqual(actual, row.macro_f1, places=12)
            else:
                self.assertTrue(pd.isna(row.macro_f1))

    def test_pool_slices_match_published_convention(self):
        self.assertEqual([cache_local_slice(p) for p in [1, 2, 3, 5]], [(3, 4), (3, 5), (2, 5), (1, 6)])

    def test_custom_output_report_links_stay_with_current_evidence(self):
        from report_links import relocate_links
        canonical = ROOT / 'training/full_nuisance_balance'
        output = ROOT / 'runs/example'
        text = '[code](../train_simple_clay_heads.py) [metrics](clay_full_sweep/metrics.csv)'
        result = relocate_links(text, output, canonical)
        self.assertIn('(../../training/train_simple_clay_heads.py)', result)
        self.assertIn('(clay_full_sweep/metrics.csv)', result)

    def test_fresh_run_contains_only_inputs(self):
        module = load_module('prepare_run', ROOT / 'scripts/prepare_run.py')
        with tempfile.TemporaryDirectory() as directory:
            out = module.prepare(Path(directory) / 'run')
            self.assertTrue((out / 'seed_42/selected.csv').is_file())
            self.assertFalse((out / 'clay_full_sweep').exists())
            with self.assertRaises(FileExistsError):
                module.prepare(out)


if __name__ == '__main__':
    unittest.main()
