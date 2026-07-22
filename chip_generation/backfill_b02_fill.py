'''Backfill b02_range and fill_region for empty rows.
Same formulas as chip_common.qc_chip:
  fill_region = fraction of SCL == 0 (NODATA) in the 23x23 centre block [50:73, 50:73]
  b02_range   = 95th - 5th percentile of B02 pixels > 0 in that block (if >20 nonzero pixels)
Writes atomically after backup.'''
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

BACKUP = CSV_IN.with_suffix('.csv.backup_before_b02fill_backfill')
if not BACKUP.exists():
    shutil.copy2(CSV_IN, BACKUP); print(f'backup -> {BACKUP}')
else: print(f'backup exists at {BACKUP}')

SCL_BAND_IDX = 7
B02_BAND_IDX = 1
REGION_R0, REGION_R1 = 50, 73
NODATA_SCL = 0

rows = list(csv.DictReader(open(CSV_IN)))
fields = list(rows[0].keys())
print(f'rows: {len(rows)}')

need = []
for i, r in enumerate(rows):
    empty_b02  = r.get('b02_range','').strip() == ''
    empty_fill = r.get('fill_region','').strip() == ''
    if not (empty_b02 or empty_fill): continue
    try: ci = int(r['chipIndex'])
    except (ValueError, KeyError): continue
    need.append((i, ci, empty_b02, empty_fill))

print(f'rows needing backfill: {len(need)}')
if not need: sys.exit(0)

n_b02 = 0; n_fill = 0; n_fail = 0
with h5py.File(H5, 'r') as f:
    chips = f['chips']
    print(f'chips.h5 shape: {chips.shape}')
    for i, ci, empty_b02, empty_fill in need:
        if not (0 <= ci < chips.shape[0]):
            n_fail += 1; continue
        try:
            chip = chips[ci]
        except Exception:
            n_fail += 1; continue
        scl = chip[SCL_BAND_IDX]
        b02 = chip[B02_BAND_IDX]
        reg_scl = scl[REGION_R0:REGION_R1, REGION_R0:REGION_R1]
        reg_b02 = b02[REGION_R0:REGION_R1, REGION_R0:REGION_R1]
        if empty_fill:
            rows[i]['fill_region'] = f'{round(float((reg_scl == NODATA_SCL).mean()), 3)}'
            n_fill += 1
        if empty_b02:
            reg_b02_nz = reg_b02[reg_b02 > 0]
            if len(reg_b02_nz) > 20:
                b02_range = float(np.percentile(reg_b02_nz, 95) - np.percentile(reg_b02_nz, 5))
            else:
                b02_range = 0.0
            rows[i]['b02_range'] = f'{round(b02_range, 1)}'
            n_b02 += 1

print(f'backfilled: b02_range={n_b02}  fill_region={n_fill}  failed={n_fail}')

tmp = CSV_IN.with_suffix('.csv.tmp')
with tmp.open('w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=fields)
    w.writeheader()
    for r in rows: w.writerow({k: r.get(k,'') for k in fields})
tmp.replace(CSV_IN)
print(f'wrote {CSV_IN}')
