"""Merge all successful chips from every fetch pass into a single dataset.

Reads from all `~/kelp-detection/data/gbif_s2_chips_*` fetch dirs, keeps only
status='ok' rows. Applies two dedup passes:
  1. by gbifID (collapses same-source refetches of the same scene)
  2. by (species, lat, lon, eventDate) via common.dedup_key (collapses
     cross-source duplicates — same observation reported in both GBIF and OBIS)
Writes merged_chips.h5 + merged_metadata.csv.
"""
import csv
import sys
from pathlib import Path
import h5py
import numpy as np
# Locate repo root by walking up until config.py is found
import sys as _sys
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_collection"))
from common import dedup_key  # (species, lat, lon, date) → collapses cross-source duplicates

DATA_ROOT = config.DATA_DIR
# #7 (reviewer): widen from gbif_s2_chips_* to *_s2_chips_* so future OBIS /
# JNCC / snap fetch directories are picked up automatically. The merged output
# dir itself lives at s2_chips_merged/ (no underscore prefix) and is excluded.
SOURCES = sorted([
    p for p in DATA_ROOT.iterdir()
    if p.is_dir()
    and p.name != "s2_chips_merged"
    and ".backup" not in p.name  # any *.backup* dir (#7 reviewer)
    and ("s2_chips_" in p.name)
    and (p / "chips.h5").exists()
])
OUT_DIR = config.CHIPS_DIR
OUT_DIR.mkdir(exist_ok=True)

print(f"sources found: {len(SOURCES)}")
for s in SOURCES:
    print(f"  {s.name}")

# #11 (reviewer): stream chip data through the merge instead of holding
# every 128x128x8x uint16 in RAM. Pass 1 collects only per-chip REFERENCES
# (source dir + row index inside its h5) plus metadata. Pass 2 opens each
# source h5 once, reads and writes chips one at a time. Memory drops from
# O(N * 128 * 128 * 8 * 2 bytes) ≈ 2.6 GB to O(one chip).
seen_ids = set()
refs = []  # list of (src_dir, chip_idx_in_src, meta_row)
all_meta = []
for src in SOURCES:
    h5p = src / "chips.h5"
    mp = src / "chips_metadata.csv"
    if not h5p.exists():
        print(f"  ! {src.name}: SKIPPED (no chips.h5)")
        continue
    try:
        # Just probe attrs — don't materialise the data
        with h5py.File(h5p, "r") as f:
            _ = int(f.attrs.get("n_chips", f["chips"].shape[0]))
    except OSError as e:
        print(f"  ! {src.name}: SKIPPED ({e})")
        continue
    rows = [r for r in csv.DictReader(mp.open())
            if r["status"] == "ok"
            and r.get("class_tag") not in ("absent_water", "absent_land")]
    rows.sort(key=lambda r: int(r["chipIndex"]))
    kept = 0
    for r in rows:
        gid = r["gbifID"]
        if gid in seen_ids:
            continue
        seen_ids.add(gid)
        idx = int(r["chipIndex"])
        refs.append((src, idx, r))
        all_meta.append(r)
        kept += 1
    print(f"  {src.name}: {len(rows)} ok, kept {kept} new")

print(f"after gbifID dedup: {len(refs)} chips")

# Cross-source dedup by (species, lat, lon, date). Sources are iterated in
# sorted order (alphabetical), so first-seen wins.
seen_keys = set()
kept_refs, kept_meta = [], []
dropped_xsrc = 0
for (src, idx, r) in refs:
    k = dedup_key(r)
    if k is not None and k in seen_keys:
        dropped_xsrc += 1
        continue
    if k is not None:
        seen_keys.add(k)
    kept_refs.append((src, idx, r))
    kept_meta.append(r)
refs, all_meta = kept_refs, kept_meta
print(f"cross-source dedup dropped {dropped_xsrc} chips; kept {len(refs)}")

from collections import Counter
print(f"class balance: {Counter(r['occurrenceStatus'] for r in all_meta)}")
print(f"by class_tag: {Counter(r['class_tag'] for r in all_meta)}")

# Stream: open each source h5 once (cached), read + write chip by chip.
n = len(refs)
h5_cache = {}
def _open(src):
    if src not in h5_cache:
        h5_cache[src] = h5py.File(src / "chips.h5", "r")
    return h5_cache[src]

try:
    with h5py.File(OUT_DIR / "chips.h5", "w") as f:
        f.attrs["n_chips"] = n
        f.attrs["bands"] = np.array(["B01", "B02", "B03", "B04", "B05", "B08", "B11", "SCL"], dtype="S")
        f.attrs["chip_size"] = 128
        f.attrs["pixel_m"] = 10.0
        # maxshape=(None, ...) makes the dataset resizable so downstream tools
        # (e.g. dedup_by_coord_date.py) can shrink or grow chips.h5 in place
        # without a full rewrite. Older merges omitted this and produced
        # fixed-size datasets — those crash the dedup script on rebuild.
        ds = f.create_dataset("chips", shape=(n, 8, 128, 128), maxshape=(None, 8, 128, 128), dtype=np.uint16,
                               chunks=(1, 8, 128, 128), compression="lzf")
        for i, (src, idx, _r) in enumerate(refs):
            ds[i] = _open(src)["chips"][idx]
finally:
    for h in h5_cache.values():
        h.close()

# Write metadata with a merged_index column (0-indexed row in merged h5).
# B1 (reviewer): union field names across ALL rows so a column present in
# some source CSVs but not source[0] (e.g. snap_lat/snap_lon added when
# snap-to-water re-anchors a chip) isn't silently dropped by DictWriter.
with (OUT_DIR / "chips_metadata.csv").open("w", newline="") as fh:
    all_fields = []
    seen = set()
    for r in all_meta:
        for k in r.keys():
            if k not in seen:
                seen.add(k); all_fields.append(k)
    if "merged_index" not in seen:
        all_fields.append("merged_index")
    w = csv.DictWriter(fh, fieldnames=all_fields, extrasaction="ignore")
    w.writeheader()
    for i, r in enumerate(all_meta):
        # Renumber chipIndex to the merged h5 row (was the source-h5 row for
        # each source, which caused every downstream script that used
        # chipIndex to pull the WRONG chip from the merged h5). Post-merge,
        # chipIndex == merged_index == CSV row position.
        r["chipIndex"] = i
        r["merged_index"] = i
        w.writerow({k: r.get(k, "") for k in all_fields})

print(f"wrote {OUT_DIR / 'chips.h5'} and metadata csv")
print(f"final: {n} chips × {len(all_fields)} metadata fields")
