"""Precompute Clay v1.5 identity embeddings for the chips referenced by the splits.

Input: text file with one chip gbifID per line (defaults to splits/used_chip_ids.txt)
Reads chips from the merged HDF5 by matching gbifID → row index in chips_metadata.csv.
Writes ONE embedding per chip to data/clay_cache/identity.npz.

Now that ABSENT chips come from real multi-scene observations, we no longer
need the 8-augmentation cache.

Usage:
  python precompute_clay_embeddings.py [chip_ids.txt]
"""
import csv
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from box import Box
from claymodel.module import ClayMAEModule
# Locate repo root by walking up until config.py is found
import sys as _sys
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

CHIPS_H5  = config.CHIPS_H5
META_CSV  = config.CHIPS_META
CLAY_CKPT = config.CLAY_CKPT
CLAY_META = config.CLAY_META
DEFAULT_IDS = config.SPLITS_DIR / "used_chip_ids.txt"
OUT = config.CLAY_CACHE_NPZ

CLAY_BAND_IDX = [1, 2, 3, 4, 5, 6]
CLAY_KEYS = ["blue", "green", "red", "rededge1", "nir", "swir16"]
CLAY_WAVES = [0.493, 0.560, 0.665, 0.704, 0.842, 1.610]


def latlon_time(meta):
    """Return (ll, tv): Clay's positional and temporal encodings.

    N3: uses snapped coords when the chip was re-anchored via snap-to-water,
    because Clay's positional encoder must be told where the chip actually is
    — not the observer's reported position, which can be up to 300 m off after
    a snap.
    """
    import datetime as dt
    # Prefer snap coords when populated
    try:
        slat = float(meta.get("snap_lat","") or 0); slon = float(meta.get("snap_lon","") or 0)
    except (ValueError, TypeError):
        slat = slon = 0.0
    try:
        if slat and slon:
            lat, lon = slat, slon
        else:
            lat = float(meta["decimalLatitude"]); lon = float(meta["decimalLongitude"])
    except (KeyError, ValueError, TypeError):
        lat = lon = 0.0
    ll = np.array([np.sin(np.radians(lat)), np.cos(np.radians(lat)),
                   np.sin(np.radians(lon)), np.cos(np.radians(lon))], dtype=np.float32)
    tv = np.zeros(4, dtype=np.float32)
    try:
        d = dt.date.fromisoformat((meta.get("sceneDate") or "")[:10])
        wk = d.isocalendar()[1]
        tv[0] = np.sin(2 * np.pi * wk / 52); tv[1] = np.cos(2 * np.pi * wk / 52)
    except (ValueError, TypeError):
        pass
    return ll, tv


def main():
    ids_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IDS
    wanted = [line.strip() for line in ids_path.open() if line.strip()]
    wanted_set = set(wanted)
    print(f"loaded {len(wanted)} chip IDs from {ids_path}")

    # Locate wanted chips in the merged HDF5 via chips_metadata.csv
    meta_all = list(csv.DictReader(META_CSV.open()))
    row_by_id = {r["gbifID"]: (i, r) for i, r in enumerate(meta_all)}

    # Incremental: skip chips already in cache — but require the sceneID to
    # still match what's in metadata. A re-fetch can re-order scenes at a base
    # gbifID and hand a different sceneID to `gbifID_s1` than last time; without
    # this check the stale embedding is silently reused (reviewer finding C1).
    existing_emb = {}
    scene_of = {}  # cached gbifID → sceneID (for logging invalidation)
    if OUT.exists():
        cached = np.load(OUT)
        # Materialize arrays ONCE — otherwise cached["emb"][i] re-reads the
        # entire 37 MB array from the npz each iteration and each dict-value
        # view keeps its parent array alive → RAM blows up on large caches.
        cached_ids    = list(cached["gbif_ids"])
        cached_scenes = list(cached["scene_ids"]) if "scene_ids" in cached.files else [""] * len(cached_ids)
        cached_emb    = cached["emb"]  # single materialization
        invalidated = 0
        for i, gid in enumerate(cached_ids):
            cs = cached_scenes[i] if i < len(cached_scenes) else ""
            row = row_by_id.get(gid)
            current_scene = row[1].get("sceneID", "") if row else ""
            if not cs or cs != current_scene:
                invalidated += 1
                continue
            existing_emb[gid] = cached_emb[i].copy()  # detach from parent
            scene_of[gid] = cs
        del cached, cached_emb  # free npz + parent array before Clay load
        print(f"existing cache: {len(existing_emb)} valid, {invalidated} invalidated (missing/stale sceneID)")
    missing = [g for g in wanted if g not in row_by_id]
    if missing:
        print(f"WARN: {len(missing)} requested IDs missing from chips.h5 (first 5: {missing[:5]})")

    # New chips to compute; keep already-cached ones as-is
    todo = [(g, *row_by_id[g]) for g in wanted if g in row_by_id and g not in existing_emb]
    print(f"embedding {len(todo)} new chips (skipping {len(wanted) - len(todo) - len(missing)} cached)")

    with h5py.File(CHIPS_H5, "r") as f:
        chips_ds = f["chips"]
        d = yaml.safe_load(CLAY_META.open())
        band_stats = Box(d)["sentinel-2-l2a"]["bands"]
        means = {k: float(band_stats["mean"][k]) for k in CLAY_KEYS}
        stds  = {k: float(band_stats["std"][k])  for k in CLAY_KEYS}
        waves = torch.tensor(CLAY_WAVES, device="cuda", dtype=torch.float32)
        gsd   = torch.tensor(10.0, device="cuda", dtype=torch.float32)
        model = ClayMAEModule.load_from_checkpoint(
            str(CLAY_CKPT), model_size="large", shuffle=False, mask_ratio=0.0,
            map_location="cuda",
        ).eval()
        print("Clay loaded", flush=True)

        # E1 fix: batch Clay encoder to amortise H2D + kernel launch cost
        BATCH = 32
        N = len(todo)
        new_emb = np.zeros((N, 2048), dtype=np.float32)
        new_ids = [gid for gid, _, _ in todo]
        t0 = time.time()
        with torch.no_grad():
            for i0 in range(0, N, BATCH):
                batch = todo[i0:i0 + BATCH]
                B = len(batch)
                # Assemble batched inputs
                pix_np = np.zeros((B, len(CLAY_KEYS), 128, 128), dtype=np.float32)
                ll_np = np.zeros((B, 4), dtype=np.float32)
                tv_np = np.zeros((B, 4), dtype=np.float32)
                for bi, (_, hdf_row, meta) in enumerate(batch):
                    ll, tv = latlon_time(meta)
                    pix_np[bi] = chips_ds[hdf_row, CLAY_BAND_IDX].astype(np.float32)
                    ll_np[bi] = ll; tv_np[bi] = tv
                x = torch.from_numpy(pix_np)
                x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
                for j, kk in enumerate(CLAY_KEYS):
                    x[:, j] = (x[:, j] - means[kk]) / stds[kk]
                pix = x.to("cuda", dtype=torch.float32)
                datacube = {"pixels": pix, "gsd": gsd, "waves": waves,
                            "platform": "sentinel-2-l2a",
                            "time":   torch.tensor(tv_np, device="cuda", dtype=torch.float32),
                            "latlon": torch.tensor(ll_np, device="cuda", dtype=torch.float32)}
                out = model.model.encoder(datacube)
                if isinstance(out, tuple): out = out[0]
                cls = out[:, 0, :]
                patches = out[:, 1:, :].reshape(B, 28, 28, 1024)
                centre = patches[:, 11:16, 11:16, :].mean(dim=(1, 2))
                new_emb[i0:i0 + B] = torch.cat([cls, centre], dim=-1).cpu().numpy()
                if (i0 + B) % 500 == 0 or i0 + B == N:
                    print(f"  {i0 + B}/{N}  {time.time()-t0:.0f}s", flush=True)

    # Merge: existing cache + new embeddings, restricted to `wanted` set, in wanted order
    for gid, e in zip(new_ids, new_emb):
        existing_emb[gid] = e
    final_ids = [g for g in wanted if g in existing_emb]
    final_emb = np.stack([existing_emb[g] for g in final_ids])
    final_scenes = np.array([row_by_id[g][1].get("sceneID","") for g in final_ids])
    OUT.parent.mkdir(exist_ok=True, parents=True)
    # #4 (reviewer): np.savez isn't atomic. A crash mid-write can leave the
    # file half-baked — either a BadZipFile (recoverable) or, worse, a
    # loadable file with fewer entries than expected which then trips
    # train_head's `assert not missing` and blows away hours of GPU work.
    #
    # Write to a sibling then atomically rename. Note np.savez appends `.npz`
    # if the path doesn't already end in `.npz`, so the tmp name has to end in
    # `.npz` (identity.tmp.npz) — otherwise savez writes identity.npz.tmp.npz
    # and the os.replace(identity.npz.tmp, ...) below then fails.
    tmp = OUT.with_name(OUT.stem + ".tmp" + OUT.suffix)  # identity.tmp.npz
    np.savez(tmp, gbif_ids=np.array(final_ids), scene_ids=final_scenes, emb=final_emb)
    import os as _os
    _os.replace(tmp, OUT)
    print(f"wrote {OUT}  shape={final_emb.shape}  (atomic; with scene_ids for cache validation)")


if __name__ == "__main__":
    main()
