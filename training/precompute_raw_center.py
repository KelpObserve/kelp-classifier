"""Precompute raw S2 reflectance at chip centre (pixel 64, 64) for every
chip that lives in the Clay embedding cache.

Feeds `train_pixel_head.py`, which concatenates the raw centre pixel onto
Clay's CLS + centre patch features. Kept in a separate small npz so we
never touch the existing identity cache.

Cache format:
    raw_center.npz
      gbif_ids   (N,)           — chip IDs (same order as identity.npz)
      raw_center (N, 6)  float32 — Clay-normalised reflectance for
                                    [blue, green, red, rededge1, nir, swir16]
                                    at pixel (64, 64) of the 128×128 chip

Env vars honoured (same as precompute_clay_embeddings.py):
  KELP_CHIPS_SUBSET  which chip pool to read (merged | submerged)
  KELP_DATA_DIR      data root

Usage:
  KELP_CHIPS_SUBSET=submerged python training/precompute_raw_center.py
"""
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml
from box import Box

_here = Path(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

CHIPS_H5  = config.CHIPS_H5
META_CSV  = config.CHIPS_META
CLAY_META = config.CLAY_META
IDENTITY_NPZ = config.CLAY_CACHE_NPZ
OUT = config.CLAY_CACHE_NPZ.with_name(
    f"raw_center{'' if config.CHIPS_SUBSET == 'merged' else '_' + config.CHIPS_SUBSET}.npz"
)

CLAY_BAND_IDX = [1, 2, 3, 4, 5, 6]   # blue, green, red, rededge1, nir, swir16
CLAY_KEYS = ["blue", "green", "red", "rededge1", "nir", "swir16"]

# Labelled pixel = chip centre. See the design note in train_pixel_head.py.
CENTER_R, CENTER_C = 64, 64


def main():
    if not IDENTITY_NPZ.exists():
        raise SystemExit(f"identity cache not found: {IDENTITY_NPZ}\n"
                         f"Run precompute_clay_embeddings.py first.")
    cache = np.load(IDENTITY_NPZ)
    gbif_ids = list(map(str, cache["gbif_ids"]))
    print(f"identity cache: {len(gbif_ids)} chips  ({IDENTITY_NPZ.name})")

    # Reuse previously-cached raw-centre values if present (additive, same
    # protocol as precompute_clay_embeddings.py). Never wipes on delta runs.
    prev = {}
    if OUT.exists():
        p = np.load(OUT)
        for gid, v in zip(map(str, p["gbif_ids"]), p["raw_center"]):
            prev[gid] = v.astype(np.float32)
        print(f"existing raw_center cache: {len(prev)} chips  ({OUT.name})")

    meta_all = list(csv.DictReader(META_CSV.open()))
    row_by_id = {r["gbifID"]: i for i, r in enumerate(meta_all)}
    todo = [g for g in gbif_ids if g not in prev]
    missing = [g for g in todo if g not in row_by_id]
    if missing:
        print(f"WARN: {len(missing)} identity-cache IDs are not in chips_metadata.csv "
              f"(first 3: {missing[:3]}) — will be skipped.")
    todo = [g for g in todo if g in row_by_id]
    print(f"extracting raw centre for {len(todo)} new chips  "
          f"({len(prev)} carried over)")

    # Clay per-band mean/std for consistent normalisation (matches
    # precompute_clay_embeddings.py). Normalising here means the pixel
    # feature is on the same scale as the Clay-encoded features.
    band_stats = Box(yaml.safe_load(CLAY_META.open()))["sentinel-2-l2a"]["bands"]
    means = np.array([float(band_stats["mean"][k]) for k in CLAY_KEYS], dtype=np.float32)
    stds  = np.array([float(band_stats["std"][k])  for k in CLAY_KEYS], dtype=np.float32)

    t0 = time.time()
    with h5py.File(CHIPS_H5, "r") as f:
        chips_ds = f["chips"]
        for i, gid in enumerate(todo):
            hdf_row = row_by_id[gid]
            raw = chips_ds[hdf_row, CLAY_BAND_IDX, CENTER_R, CENTER_C].astype(np.float32)
            prev[gid] = (raw - means) / stds
            if (i + 1) % 500 == 0 or i + 1 == len(todo):
                print(f"  {i+1}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)

    final_ids = [g for g in gbif_ids if g in prev]
    final_arr = np.stack([prev[g] for g in final_ids]).astype(np.float32)
    print(f"writing raw_center cache: {len(final_ids)} chips  shape={final_arr.shape}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.stem + ".tmp" + OUT.suffix)
    np.savez(tmp, gbif_ids=np.array(final_ids), raw_center=final_arr)
    import os as _os
    _os.replace(tmp, OUT)
    print(f"wrote {OUT}  (atomic)")


if __name__ == "__main__":
    main()
