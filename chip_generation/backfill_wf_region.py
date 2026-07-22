'''Backfill wf_region column for rows with empty value by recomputing from SCL band.
Same formula as qc_chip: wf_region = fraction of SCL==6 in the 23x23 centre block [50:73, 50:73].
Also fills fill_region and cloud_region if empty (same source, computed in the same pass).
Writes to a new CSV then atomically replaces the original after backup.'''
import csv, shutil, sys
from pathlib import Path
import numpy as np
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

MERGED = config.CHIPS_DIR
CSV_IN = MERGED / 'chips_metadata.csv'
H5     = MERGED / 'chips.h5'

# Backup
BACKUP = CSV_IN.with_suffix('.csv.backup_before_wfbackfill')
if not BACKUP.exists():
    shutil.copy2(CSV_IN, BACKUP)
    print(f'backed up CSV -> {BACKUP}')
else:
    print(f'backup exists at {BACKUP} (skipping)')

# Same as chip_common.py
SCL_BAND_IDX = 7
REGION_R0, REGION_R1 = 50, 73
WATER_SCL = 6
CLOUD_SCL = (3, 8, 9, 10)
NODATA_SCL = 0

rows = list(csv.DictReader(open(CSV_IN)))
fields = list(rows[0].keys())
print(f'rows: {len(rows)}  fields: {len(fields)}')

# Which rows need backfill on wf_region?
need_ci = []
for i, r in enumerate(rows):
    if r.get('wf_region','').strip() == '':
        try: ci = int(r['chipIndex'])
        except (ValueError, KeyError): continue
        need_ci.append((i, ci))
print(f'rows missing wf_region: {len(need_ci)}')

if not need_ci:
    print('nothing to backfill'); sys.exit(0)

n_updated_wf = 0
n_updated_fill = 0
n_updated_cloud = 0
n_failed = 0

with h5py.File(H5, 'r') as f:
    chips = f['chips']
    print(f'chips.h5 shape: {chips.shape}')
    for i, ci in need_ci:
        if not (0 <= ci < chips.shape[0]):
            n_failed += 1; continue
        try:
            scl = chips[ci, SCL_BAND_IDX]
        except Exception:
            n_failed += 1; continue
        reg = scl[REGION_R0:REGION_R1, REGION_R0:REGION_R1]
        wf   = float((reg == WATER_SCL).mean())
        fillr = float((reg == NODATA_SCL).mean())
        cldr  = float(np.isin(reg, CLOUD_SCL).mean())
        r = rows[i]
        if r.get('wf_region','').strip() == '':
            r['wf_region'] = f'{round(wf,3)}'; n_updated_wf += 1
        if r.get('fill_region','').strip() == '':
            r['fill_region'] = f'{round(fillr,3)}'; n_updated_fill += 1
        if r.get('cloud_region','').strip() == '':
            r['cloud_region'] = f'{round(cldr,3)}'; n_updated_cloud += 1

print(f'backfilled: wf_region={n_updated_wf}  fill_region={n_updated_fill}  cloud_region={n_updated_cloud}  failed(bad chipIndex)={n_failed}')

# Write atomically
tmp = CSV_IN.with_suffix('.csv.tmp')
with tmp.open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k,'') for k in fields})
tmp.replace(CSV_IN)
print(f'wrote {CSV_IN} ({sum(1 for _ in open(CSV_IN))-1} data rows)')
