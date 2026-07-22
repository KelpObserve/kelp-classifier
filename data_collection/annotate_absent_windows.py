"""For every raw ABSENT record from GBIF/OBIS/JNCC, compute allowed_max_days
= 30 or 60 by looking up nearby raw PRESENT records.

Rule:
  If any PRESENT record within 1 km AND ±90 days of this ABSENT observation
  exists -> allowed_max_days = 30 (strict - kelp was near here recently).
  Otherwise -> 60 (extended - no nearby PRESENT known).

Uses ALL raw source annotations as ecological truth. Does NOT consult chip
metadata.

Bug fixes (reviewer round 3):
  1. PRESENT records don't need an ID - only lat/lon/date. Splitting parse
     into parse_present (no id) and parse_absent (needs id) avoids dropping
     OBIS + JNCC PRESENT records from the BallTree ground truth.
  2. Per-source ID normalization - JSON keys must match what merged CSV
     writes. GBIF -> gbifID plain; OBIS -> 'obis:' + obisID; JNCC ->
     'jncc:' + sample_ref. Confirmed by inspecting merged CSV samples.

Output: data/absent_windows.json - {base_gbifID: 30 or 60}.
"""
import csv, json
import datetime as dt
from pathlib import Path
import numpy as np
from sklearn.neighbors import BallTree
# Locate repo root by walking up until config.py is found
import sys as _sys
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

REPO = config.REPO_ROOT
DATA = config.DATA_DIR

SOURCES = [
    ('gbif', REPO / 'data_collection/gbif/kelp_gbif_europe_absent.csv',
             REPO / 'data_collection/gbif/kelp_gbif_europe_present.csv'),
    ('obis', REPO / 'data_collection/obis/kelp_obis_europe_absent.csv',
             REPO / 'data_collection/obis/kelp_obis_europe_present.csv'),
    ('jncc', REPO / 'data_collection/jncc/kelp_jncc_absent.csv',
             REPO / 'data_collection/jncc/kelp_jncc_present.csv'),
]

RADIUS_KM       = 1.0
TIME_GAP_DAYS   = 90
DEFAULT_DAYS    = 30
EXTENDED_DAYS   = 60
EARTH_RADIUS_KM = 6371.0


def parse_present(r):
    try:
        lat = float(r.get('decimalLatitude') or r.get('latitude'))
        lon = float(r.get('decimalLongitude') or r.get('longitude'))
        date = dt.date.fromisoformat((r.get('eventDate') or r.get('event_date') or '')[:10])
    except (ValueError, TypeError, KeyError):
        return None
    return lat, lon, date


def parse_absent(r, source):
    try:
        lat = float(r.get('decimalLatitude') or r.get('latitude'))
        lon = float(r.get('decimalLongitude') or r.get('longitude'))
        date = dt.date.fromisoformat((r.get('eventDate') or r.get('event_date') or '')[:10])
    except (ValueError, TypeError, KeyError):
        return None
    if source == 'gbif':
        gid = str(r.get('gbifID') or '')
    elif source == 'obis':
        oid = str(r.get('obisID') or '')
        gid = 'obis:' + oid if oid else ''
    elif source == 'jncc':
        sr = str(r.get('sample_ref') or '')
        gid = 'jncc:' + sr if sr else ''
    else:
        gid = ''
    if not gid:
        return None
    return lat, lon, date, gid


present = []
for source, _, path in SOURCES:
    if not path.exists():
        print(f'skip PRESENT (missing): {path.name}'); continue
    n = 0
    for r in csv.DictReader(path.open()):
        parsed = parse_present(r)
        if parsed is None: continue
        present.append(parsed); n += 1
    print(f'loaded {n} PRESENT rows from {source}')
print(f'total PRESENT for tree: {len(present)}')

p_coords = np.radians([[lat, lon] for lat, lon, _ in present])
p_days   = np.array([(d - dt.date(1970, 1, 1)).days for _, _, d in present])
tree = BallTree(p_coords, metric='haversine')
radius_rad = RADIUS_KM / EARTH_RADIUS_KM
print(f'built BallTree ({len(present)} PRESENT records)')

result = {}
counts = {'total': 0, 'strict_30': 0, 'extended_60': 0, 'unparseable': 0}
by_source = {}
for source, path, _ in SOURCES:
    if not path.exists():
        print(f'skip ABSENT (missing): {path.name}'); continue
    strict_n, ext_n = 0, 0
    for r in csv.DictReader(path.open()):
        parsed = parse_absent(r, source)
        if parsed is None:
            counts['unparseable'] += 1; continue
        lat, lon, ev_date, gid = parsed
        counts['total'] += 1
        pt = np.radians([[lat, lon]])
        nn = tree.query_radius(pt, r=radius_rad)[0]
        ev_days = (ev_date - dt.date(1970, 1, 1)).days
        strict = False
        if len(nn) > 0:
            if np.abs(p_days[nn] - ev_days).min() <= TIME_GAP_DAYS:
                strict = True
        result[gid] = DEFAULT_DAYS if strict else EXTENDED_DAYS
        counts['strict_30' if strict else 'extended_60'] += 1
        strict_n += int(strict); ext_n += int(not strict)
    by_source[source] = {'strict_30': strict_n, 'extended_60': ext_n}
    print(f'{source} ABSENT: strict={strict_n}, extended={ext_n}')

print()
print(f'totals: {counts}')
print(f'per source: {by_source}')

out = DATA / 'absent_windows.json'
out.write_text(json.dumps(result))
print(f'wrote {out} - {len(result)} ABSENT records annotated')
