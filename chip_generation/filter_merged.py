"""Stage 1c (new rule): apply three deletions to the merged dataset:

  Rule 1  Multi-species same-day ABSENT chips (12 coords):
          Same (lat, lon, eventDate) has PRESENT of some other species
          → the ABSENT rows are wrong for our any-kelp binary task.
  Rule 2  Mixed-fetch bases (all 405):
          Bases with both a single-scene "original" chip and one or more
          "_sN" multi-scene chips are the pattern that produced the OBIS
          misalignment. Delete all chips (original + _sN); re-fetch in Stage 2.
  Rule 3  Beyond-window ABSENT chips:
          |sceneDate − eventDate| > allowed_max_days (30 or 60 per record).
          Read the flag list from absent_beyond_window.csv (produced by 02).

Backs up first. After the rewrite, chipIndex/merged_index are renumbered.
Do NOT run any consumer that indexes by chipIndex/merged_index or CSV row
position between here and Stage 4 (splits regen + Clay cache rebuild).
"""
import argparse
import csv, os, shutil, datetime as dt
from collections import defaultdict
from pathlib import Path
import numpy as np, h5py
# Locate repo root by walking up until config.py is found
import sys as _sys
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

ap = argparse.ArgumentParser()
ap.add_argument("--skip-backup", action="store_true",
                help="Skip creating a merged-dir backup before rewrite (for low-disk situations)")
args = ap.parse_args()

MERGED = config.CHIPS_DIR
META_IN = MERGED / "chips_metadata.csv"
H5_IN   = MERGED / "chips.h5"
BEYOND_WINDOW = config.TMP_DIR / "absent_beyond_window.csv"

rows = list(csv.DictReader(META_IN.open()))
print(f"input rows: {len(rows)}")

# --- rule 1: multi-species same-day ABSENT ---
by_coord_date = defaultdict(list)
for i, r in enumerate(rows):
    try:
        lat = round(float(r["decimalLatitude"]), 4)
        lon = round(float(r["decimalLongitude"]), 4)
    except (ValueError, KeyError): continue
    date = (r.get("eventDate") or "")[:10]
    if not date: continue
    by_coord_date[(lat, lon, date)].append((i, r))
drop_multi_species = set()
for _, group in by_coord_date.items():
    statuses = {r.get("occurrenceStatus","?") for _, r in group}
    if statuses == {"PRESENT", "ABSENT"}:
        for i, r in group:
            if r.get("occurrenceStatus") == "ABSENT":
                drop_multi_species.add(i)
print(f"rule 1 (multi-species same-day ABSENT): {len(drop_multi_species)} rows")

# --- rule 2: mixed-fetch bases ---
by_base = defaultdict(list)
for i, r in enumerate(rows):
    if r.get("occurrenceStatus") == "PRESENT": continue
    by_base[r["gbifID"].rsplit("_s", 1)[0]].append((i, r))
drop_mixed_fetch = set()
for base, group in by_base.items():
    has_orig = any(r["gbifID"] == base for _, r in group)
    has_multi = any(r["gbifID"].startswith(base + "_s") for _, r in group)
    if has_orig and has_multi:
        for i, _ in group: drop_mixed_fetch.add(i)
n_bases = len({rows[i]['gbifID'].rsplit('_s',1)[0] for i in drop_mixed_fetch})
print(f"rule 2 (mixed-fetch bases): {len(drop_mixed_fetch)} rows ({n_bases} bases)")

# --- rule 3: beyond-window ABSENT (from 02 output) ---
drop_beyond_window = set()
if BEYOND_WINDOW.exists():
    for r in csv.DictReader(BEYOND_WINDOW.open()):
        drop_beyond_window.add(int(r["row"]))
    print(f"rule 3 (beyond allowed_max_days): {len(drop_beyond_window)} rows "
          f"(loaded from {BEYOND_WINDOW.name})")
else:
    print(f"rule 3: no {BEYOND_WINDOW.name} found — skipping (run 02 first)")

to_drop = drop_multi_species | drop_mixed_fetch | drop_beyond_window
print(f"total to drop: {len(to_drop)} rows")

# Backup
if not args.skip_backup:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = MERGED.parent / f"s2_chips_merged.backup_stage1_{ts}"
    print(f"backup → {backup}")
    shutil.copytree(MERGED, backup)
else:
    print("--skip-backup: no backup dir created")

# Write new h5 and CSV
keep_idx = [i for i in range(len(rows)) if i not in to_drop]
with h5py.File(H5_IN, "r") as f_in:
    chips_ds = f_in["chips"]
    assert chips_ds.shape[0] == len(rows)
    src_tail = chips_ds.shape[1:]
    src_dtype = chips_ds.dtype
    src_chunks = chips_ds.chunks
    src_compress = chips_ds.compression
    tmp_h5 = H5_IN.with_suffix(".h5.tmp")
    with h5py.File(tmp_h5, "w") as f_out:
        for k in f_in.attrs.keys():
            if k == "n_chips": continue
            f_out.attrs[k] = f_in.attrs[k]
        f_out.attrs["n_chips"] = len(keep_idx)
        ds_out = f_out.create_dataset("chips",
            shape=(len(keep_idx),) + src_tail,
            dtype=src_dtype, chunks=src_chunks, compression=src_compress)
        for new_i, src_i in enumerate(keep_idx):
            ds_out[new_i] = chips_ds[src_i]
os.replace(tmp_h5, H5_IN)

kept_rows = [rows[i] for i in keep_idx]
seen = []; seen_set = set()
for r in kept_rows:
    for k in r.keys():
        if k not in seen_set: seen_set.add(k); seen.append(k)
if "merged_index" not in seen_set: seen.append("merged_index")

tmp_csv = META_IN.with_suffix(".csv.tmp")
with tmp_csv.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=seen, extrasaction="ignore")
    w.writeheader()
    for new_i, r in enumerate(kept_rows):
        r["chipIndex"] = new_i
        r["merged_index"] = new_i
        w.writerow({k: r.get(k, "") for k in seen})
os.replace(tmp_csv, META_IN)
print(f"wrote {H5_IN} ({len(kept_rows)} chips) and {META_IN}")
