"""Download the Sesimbra crop of the selected campaign S2 scene (rasterio, numpy).

Retains source digital numbers and records scale/offset in scene_item.json.
Uses the existing benchmark's extent, bands and resampling convention.
"""
import argparse
import csv
from datetime import date
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from chip_generation.chip_common import require_complete_window

os.environ.setdefault('AWS_NO_SIGN_REQUEST', 'YES')
os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, rowcol
from rasterio.warp import transform
from rasterio.windows import from_bounds

ROOT = config.get_path('S2_CAMPAIGNS_DIR')
BANDS = ['blue', 'green', 'red', 'rededge1', 'nir', 'swir16', 'scl']
BBOX = [-9.235, 38.395, -9.170, 38.425]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign', choices=['june', 'july', 'september'])
    args = parser.parse_args()
    root = ROOT / args.campaign
    item = json.loads((root / 'scene_item.json').read_text())
    with rasterio.open(item['assets']['blue']['href']) as src:
        crs = src.crs
    xs, ys = transform('EPSG:4326', crs, BBOX[::2], BBOX[1::2])
    x0, x1 = np.floor(min(xs)/10)*10, np.ceil(max(xs)/10)*10
    y0, y1 = np.floor(min(ys)/10)*10, np.ceil(max(ys)/10)*10
    width, height = int((x1-x0)/10), int((y1-y0)/10)
    tf = Affine(10, 0, x0, 0, -10, y1)
    cube = np.zeros((len(BANDS), height, width), dtype=np.uint16)
    for i, band in enumerate(BANDS):
        with rasterio.open(item['assets'][band]['href']) as src:
            if src.crs != crs:
                raise ValueError('Band CRS mismatch')
            window = from_bounds(x0, y0, x1, y1, src.transform)
            require_complete_window(window, src.width, src.height)
            cube[i] = src.read(1, window=window,
                               out_shape=(height,width),
                               resampling=Resampling.nearest if band=='scl' else Resampling.bilinear)
        print(f'Downloaded {band}: {height} x {width}', flush=True)
    out = root / 'scene.npz'
    np.savez_compressed(out, cube=cube, transform=np.array(tuple(tf)), crs=str(crs),
                        scene_id=item['id'], bands=np.array(BANDS))
    with (ROOT / 'field_observations.csv').open() as stream:
        events = [r for r in csv.DictReader(stream) if r['date'][:7]==item['properties']['datetime'][:7]]
    xx, yy = transform('EPSG:4326',crs,[float(r['lon']) for r in events],[float(r['lat']) for r in events])
    rr, cc = rowcol(tf,xx,yy)
    scl = [int(cube[-1,r,c]) for r,c in zip(rr,cc)]
    report = dict(scene_id=item['id'], acquisition_datetime=item['properties']['datetime'],
                  observation_dates=sorted({r['date'] for r in events}),
                  day_differences={d:(date.fromisoformat(item['properties']['datetime'][:10])-date.fromisoformat(d)).days for d in sorted({r['date'] for r in events})},
                  selection='Campaign-specific scene selected by date proximity and SCL checks at observation locations; see README.md.',
                  catalogue=config.get_service('earth_search'),
                  tile_cloud_percent=item['properties']['eo:cloud_cover'], bbox_wgs84=BBOX,
                  bands=BANDS, shape=list(cube.shape), crs=str(crs), transform=list(tf),
                  radiometry='Unmodified source digital numbers; scale and offset are in scene_item.json.',
                  observation_scl_counts=dict(Counter(scl)),
                  cloud_or_shadow_events=sum(v in (3,8,9,10) for v in scl),
                  nodata_events=sum(v==0 for v in scl),
                  sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                  status='Downloaded input imagery; classifier inference has not been run.')
    (root / 'acquisition.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
