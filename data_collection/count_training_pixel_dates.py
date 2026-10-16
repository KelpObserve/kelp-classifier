"""Count selected training observations and unique native S2 pixel-date labels.

Uses the current seed-aware experiment manifest, not all source observations
linked to an image. Requires rasterio. Paths are supplied by root config.json.
"""
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from rasterio.warp import transform


def count_rows(rows):
    pixels = defaultdict(list)
    observations = defaultdict(set)
    for row in rows:
        status = row['occurrenceStatus']
        observations[status].add((round(float(row['source_decimalLatitude']), 4),
                                  round(float(row['source_decimalLongitude']), 4), row['eventDate']))
        tile = row['tile']
        epsg = int(float(row['output_crs'])) if row.get('output_crs') else (32600 if tile[2] >= "N" else 32700) + int(tile[:2])
        crs = f'EPSG:{epsg}'
        xs, ys = transform('EPSG:4326', crs, [float(row['chip_center_lon'])], [float(row['chip_center_lat'])])
        key = (crs, math.floor(xs[0]/10)*10, math.floor(ys[0]/10)*10, row['sceneDate'])
        pixels[key].append(row)
    labels=[]
    for key,members in pixels.items():
        values={r['occurrenceStatus'] for r in members}
        if len(values)!=1:raise ValueError(f'Contradictory labels at {key}')
        labels.append(next(iter(values)))
    return dict(location_date_observations={k:len(v) for k,v in observations.items()},
                selected_chips=dict(Counter(r['occurrenceStatus'] for r in rows)),
                pixel_date_labels=dict(Counter(labels)),
                duplicate_pixel_date_groups=sum(len(v)>1 for v in pixels.values()))


def main():
    source=config.get_path('TRAINING_SELECTION')
    with source.open() as f:rows=list(csv.DictReader(f))
    seeds=sorted({r['seed'] for r in rows}) if 'seed' in rows[0] else []
    unique={r['chip_id']:r for r in rows}
    report=dict(scope='Current experiment: each seed includes train, validation and test. Union is a deduplicated inventory, not a combined final-fit model.',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),union=count_rows(list(unique.values())),per_seed={seed:count_rows([r for r in rows if r['seed']==seed]) for seed in seeds},observation_key='Latitude/longitude rounded to four decimals and observation date, within class.',pixel_key='UTM CRS, native 10 m cell and acquisition calendar date.')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
