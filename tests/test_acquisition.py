"""Regression tests for crop geometry and external-cache preflight."""
import csv
import datetime as dt
import runpy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chip_generation.chip_common import BANDS, IncompleteChipError, read_chip


class CropGeometryTests(unittest.TestCase):
    def test_edge_rejected_and_interior_centre_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gradient.tif'
            array = np.tile(np.arange(200, dtype=np.uint16), (200, 1))
            with rasterio.open(path, 'w', driver='GTiff', width=200, height=200,
                               count=1, dtype='uint16', crs='EPSG:32629',
                               transform=from_origin(500000, 5000000, 10, 10)) as dst:
                dst.write(array, 1)
            item = SimpleNamespace(assets={band: SimpleNamespace(href=str(path)) for band in BANDS})
            lon, lat = transform('EPSG:32629', 'EPSG:4326', [500100], [4999000])
            with self.assertRaises(IncompleteChipError):
                read_chip(item, lon[0], lat[0])
            lon, lat = transform('EPSG:32629', 'EPSG:4326', [501000], [4999000])
            chip = read_chip(item, lon[0], lat[0])
            self.assertEqual(chip.shape, (8, 128, 128))
            self.assertEqual(int(chip[0, 64, 64]), 100)
            np.testing.assert_array_equal(chip[0, 64], np.arange(36, 164))


class AdjacentTileTests(unittest.TestCase):
    def test_adjacent_projection_preserves_target_grid(self):
        spec = importlib.util.spec_from_file_location('coverage', ROOT / 'chip_generation/tile_coverage.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        lon, lat = -5.99, 54.0
        x, y = transform('EPSG:4326', 'EPSG:32629', [lon], [lat])
        tf = from_origin(x[0]-1500, y[0]+1500, 10, 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'adjacent.tif'
            array = np.tile(np.arange(300, dtype=np.uint16), (300, 1))
            with rasterio.open(path, 'w', driver='GTiff', width=300, height=300,
                               count=1, dtype='uint16', crs='EPSG:32629', transform=tf) as dst:
                dst.write(array, 1)
            item = {'properties': {'proj:epsg': 32629}, 'assets': {
                band: {'href': str(path), 'proj:shape': [300, 300],
                       'proj:transform': list(tf)[:6]} for band in BANDS}}
            self.assertTrue(module.complete(item, lon, lat, 32630))
            result = module.read_on_grid(item, lon, lat, 32630)
            self.assertEqual(result.shape, (8, 128, 128))
            self.assertLessEqual(abs(int(result[0, 64, 64])-150), 1)
            self.assertTrue(np.isfinite(result).all())
            item['assets']['blue']['proj:shape'] = [20, 20]
            self.assertFalse(module.complete(item, lon, lat, 32630))


class AcquisitionWorkflowTests(unittest.TestCase):
    def test_overlap_requires_same_acquisition_and_preserves_requested_crs(self):
        from chip_generation import tile_coverage as module
        def item(name, datatake, epsg):
            props = {'platform': 'sentinel-2a', 's2:datatake_id': datatake,
                     's2:processing_baseline': '05.00', 'proj:code': f'EPSG:{epsg}'}
            data = {'properties': props, 'assets': {band: {} for band in BANDS}}
            return SimpleNamespace(id=name, properties=props, to_dict=lambda: data)
        source = item('edge', 'take1', 32630)
        wrong = item('different-acquisition', 'take2', 32629)
        adjacent = item('adjacent', 'take1', 32629)
        array = np.ones((8, 128, 128), dtype=np.uint16)
        with patch.object(module, 'read_chip', side_effect=IncompleteChipError), \
             patch.object(module, 'complete', return_value=True) as coverage, \
             patch.object(module, 'read_on_grid', return_value=array) as read:
            selected, result, crs = module.read_acquisition_chip(
                source, [(0, wrong), (0, adjacent)], -5.99, 54.)
            self.assertIs(selected, adjacent)
            self.assertIs(result, array)
            self.assertEqual(crs, 32630)
            coverage.assert_called_once_with(adjacent.to_dict(), -5.99, 54., 32630)
            read.assert_called_once_with(adjacent.to_dict(), -5.99, 54., 32630)
            with self.assertRaises(IncompleteChipError):
                module.read_acquisition_chip(source, [(0, wrong)], -5.99, 54.)

    def test_absent_cli_outputs_consolidate_and_keep_one_scene_per_day(self):
        from chip_generation import chip_common as module
        from chip_generation.consolidate_chips import consolidate
        from chip_generation import tile_coverage
        items = [SimpleNamespace(id=name, datetime=dt.datetime(2020, 1, day))
                 for name, day in [('tile-a', 1), ('tile-b', 1), ('tile-c', 2)]]
        array = np.full((8, 128, 128), 100, np.uint16)
        array[1] = np.tile(np.arange(128, dtype=np.uint16)*10, (128, 1))
        array[7] = 6
        def read(item, candidates, lon, lat):
            return item, array.copy(), 32629
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root/'candidates.csv'
            candidates.write_text('gbifID,decimalLatitude,decimalLongitude,eventDate,species\nobs1,41,-8,2020-01-01,Laminaria hyperborea\n')
            argv = ['fetch_absent.py', str(candidates), str(root/'fetched'), '--workers', '1', '--disable-snap']
            with patch.object(sys, 'argv', argv), patch.object(module.Client, 'open'), \
                 patch.object(module, 'search_scenes', return_value=[(0, item) for item in items]), \
                 patch.object(tile_coverage, 'read_acquisition_chip', side_effect=read):
                runpy.run_path(str(ROOT/'chip_generation/fetch_absent.py'), run_name='__main__')
            with (root/'fetched/chips_metadata.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['sceneID'] for r in rows], ['tile-a', 'tile-c'])
            self.assertTrue(all(r['class_tag'] == 'absent' for r in rows))
            self.assertTrue(all(r['output_crs'] == '32629' for r in rows))
            summary = consolidate([root/'fetched'], root/'collection')
            self.assertEqual(summary['associations'], 2)
            self.assertEqual(summary['arrays'], 1)


class PreflightTests(unittest.TestCase):
    def test_protocol_follows_overridden_cache(self):
        spec = importlib.util.spec_from_file_location('check_inputs', ROOT / 'scripts/check_inputs.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache' / 'embeddings.npz'
            cache.parent.mkdir()
            cache.touch()
            experiment = root / 'experiment'
            for seed in [42, 123, 456]:
                target = experiment / f'seed_{seed}' / 'selected.csv'
                target.parent.mkdir(parents=True)
                target.touch()
            stale = experiment / 'clay_full_sweep' / 'protocol.json'
            stale.parent.mkdir()
            stale.write_text('{}')
            override = root / 'override.json'
            override.write_text(json.dumps({'paths': {
                'FULL_NUISANCE_EMBEDDINGS': str(cache),
                'FULL_NUISANCE_BALANCE_OUTPUT': str(experiment)}}))
            with patch.dict(os.environ, {'KELP_CONFIG': str(override)}):
                self.assertEqual(len(module.check('final')), 1)
                cache.with_name('protocol.json').write_text('{}')
                stale.unlink()
                self.assertEqual(module.check('final'), [])


if __name__ == '__main__':
    unittest.main()
