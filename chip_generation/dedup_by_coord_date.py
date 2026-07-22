"""Deduplicate the merged dataset so each (occurrenceStatus, lat, lon, eventDate)
key has only ONE base_gbifID.

For BOTH PRESENT and ABSENT rows:
  - Group by (status, lat, lon, eventDate)
  - Pick a winner base (most chips, then alphabetically-first base_gbifID)
  - Drop all rows belonging to non-winner bases at that key
  - In the winner rows, replace the species field with a comma-separated list
    of ALL species reported at that (status, lat, lon, eventDate) key.
    (For PRESENT this documents every kelp species observed at the site+day;
    for ABSENT it documents every species surveyed as not observed.)

Runs in-place on data/s2_chips_merged/. h5 compaction relies on the dataset
being created with maxshape=(None, ...); if it isn't, this script cannot
resize. Rewrites chips_metadata.csv with renumbered chipIndex/merged_index.

Idempotent: re-running does nothing after the first pass.
"""
import csv
from collections import defaultdict
from pathlib import Path
import h5py
# Locate repo root by walking up until config.py is found
import sys as _sys
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

DATA = config.CHIPS_DIR
CSV_P = DATA / 'chips_metadata.csv'
H5_P  = DATA / 'chips.h5'

def coord_key(r):
    try:
        lat = round(float(r['decimalLatitude']), 5)
        lon = round(float(r['decimalLongitude']), 5)
    except (ValueError, KeyError, TypeError):
        return None
    d = (r.get('eventDate') or '')[:10]
    if not d: return None
    return (r.get('occurrenceStatus'), lat, lon, d)

def base_of(r): return r['gbifID'].rsplit('_s', 1)[0]

def dedup(csv_p=CSV_P, h5_p=H5_P):
    rows = list(csv.DictReader(open(csv_p)))
    n_in = len(rows)
    print(f'input rows: {n_in}')

    # Group per (status, lat, lon, date) -> {base: [row_i, ...]}
    groups = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        k = coord_key(r)
        if k is None: continue
        groups[k][base_of(r)].append(i)

    n_multi = sum(1 for bases in groups.values() if len(bases) > 1)
    print(f'(status, lat, lon, date) keys with >1 base: {n_multi}')
    if n_multi == 0:
        print('nothing to dedup')
        return

    # Decide winner + combined species per key
    winners = {}           # key -> winner_base
    combined_species = {}  # key -> "sp1, sp2, sp3"
    for k, bases in groups.items():
        # winner: most chips, then alphabetical
        winner = sorted(bases.items(), key=lambda x: (-len(x[1]), x[0]))[0][0]
        winners[k] = winner
        # combined species list across ALL bases at this key
        species = set()
        for base, idxs in bases.items():
            for i in idxs:
                sp = (rows[i].get('species') or '').strip()
                if sp: species.add(sp)
        combined_species[k] = ', '.join(sorted(species))

    # Identify drop indices + update winner rows' species field
    drop_indices = set()
    updated = 0
    for i, r in enumerate(rows):
        k = coord_key(r)
        if k is None: continue
        winner = winners[k]
        b = base_of(r)
        if b != winner:
            drop_indices.add(i)
        else:
            # Update species field to the combined list
            if r.get('species') != combined_species[k]:
                r['species'] = combined_species[k]
                updated += 1

    n_drop = len(drop_indices)
    n_keep = n_in - n_drop
    print(f'dropping {n_drop} rows; updating species on {updated} kept winner rows')
    print(f'keeping {n_keep} rows')

    keep_rows = [(i, r) for i, r in enumerate(rows) if i not in drop_indices]

    # In-place compact h5 (needs maxshape=None)
    with h5py.File(h5_p, 'a') as f:
        ds = f['chips']
        assert ds.shape[0] == n_in, f'h5/csv mismatch: {ds.shape[0]} vs {n_in}'
        print(f'h5 chips before: {ds.shape}')
        tail = ds.shape[1:]
        for new_i, (old_i, _) in enumerate(keep_rows):
            if new_i != old_i:
                ds[new_i] = ds[old_i]
        ds.resize((n_keep,) + tail)
        f.attrs['n_chips'] = n_keep
        print(f'h5 chips after:  {ds.shape}')

    # Rewrite CSV with union of fields + renumbered indices
    all_fields = []
    seen = set()
    for _, r in keep_rows:
        for k in r.keys():
            if k not in seen: seen.add(k); all_fields.append(k)
    if 'merged_index' not in seen: all_fields.append('merged_index')

    with csv_p.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=all_fields, extrasaction='ignore')
        w.writeheader()
        for new_i, (_, r) in enumerate(keep_rows):
            r['chipIndex'] = new_i
            r['merged_index'] = new_i
            w.writerow({k: r.get(k, '') for k in all_fields})
    print(f'wrote {csv_p}  ({n_keep} rows)')


if __name__ == '__main__':
    dedup()
