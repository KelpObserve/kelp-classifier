"""Audit selected chip footprints against public STAC raster extents, without imagery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

from rasterio.transform import Affine
from rasterio.warp import transform, transform_bounds
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from chip_generation.chip_common import BANDS, require_complete_window, IncompleteChipError
from chip_generation.tile_coverage import source_crs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, help='Chip inventory; defaults to ELIGIBLE_CHIPS')
    args = parser.parse_args()
    manifest = args.manifest or config.get_path('ELIGIBLE_CHIPS')
    with manifest.open() as stream:
        rows = {r['chip_id']: r for r in csv.DictReader(stream)}
    base = config.get_service('earth_search').rstrip('/')

    def fetch(scene):
        try:
            with urlopen(f'{base}/collections/sentinel-2-l2a/items/{scene}', timeout=30) as response:
                return scene, json.load(response), None
        except Exception as exc:
            return scene, None, str(exc)

    scenes = sorted({r['sceneID'] for r in rows.values()})
    items = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for scene, item, error in pool.map(fetch, scenes):
            items[scene] = (item, error)
    results = []
    for chip, row in sorted(rows.items()):
        item, error = items[row['sceneID']]
        record = {'chip_id': chip, 'scene_id': row['sceneID']}
        try:
            if error:
                raise ValueError(error)
            lon, lat = float(row['chip_center_lon']), float(row['chip_center_lat'])
            margins = []
            for band in BANDS:
                asset = item['assets'][band]
                props = item['properties']
                epsg = source_crs(props, asset).to_epsg()
                height, width = asset['proj:shape']
                tf = Affine(*asset['proj:transform'][:6])
                output_epsg = int(float(row['output_crs'])) if row.get('output_crs') else epsg
                x, y = transform('EPSG:4326', f'EPSG:{output_epsg}', [lon], [lat])
                bounds = transform_bounds(f'EPSG:{output_epsg}', f'EPSG:{epsg}', x[0]-640, y[0]-640, x[0]+640, y[0]+640, densify_pts=21)
                window = from_bounds(*bounds, tf)
                require_complete_window(window, width, height)
                margins.append(min(window.col_off, width-window.col_off-window.width,
                                   window.row_off, height-window.row_off-window.height))
            record.update(status='inside', minimum_margin_source_pixels=min(margins))
        except IncompleteChipError:
            record.update(status='intersects_boundary', band=band)
        except Exception as exc:
            record.update(status='unverified', reason=str(exc))
        results.append(record)
    summary = {status: sum(r['status'] == status for r in results)
               for status in ['inside', 'intersects_boundary', 'unverified']}
    evidence = {
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'method': '1280 m square about recorded chip centre, checked against all eight asset proj:shape/transform extents from current public STAC metadata.',
        'boundary': 'Checks raster extents, not internal nodata, asset immutability or pixel values. No source arrays or models are modified.',
        'scene_count': len(scenes), 'chip_count': len(rows), 'summary': summary,
        'scenes': {scene: {'error': error} if error else {
            'metadata_sha256': hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest(),
            'epsg': source_crs(item['properties']).to_epsg(),
            'assets': {band: {k: item['assets'][band][k] for k in ['proj:shape', 'proj:transform']}
                       for band in BANDS}} for scene, (item, error) in items.items()},
        'chips': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    raise SystemExit(bool(summary['intersects_boundary'] or summary['unverified']))


if __name__ == '__main__':
    main()
