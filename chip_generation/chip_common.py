"""Shared functions for fetching Sentinel-2 chips at kelp record locations.

Two QC modes:
  - STRICT  : for PRESENT records — cloud<10, requires water at chip centre
  - RELAXED : for ABSENT records — cloud<60, no water requirement, wider time window

Two fetch modes:
  - single-scene : one chip per record (used for PRESENT and default for ABSENT)
  - multi-scene  : up to K chips per record (used for ABSENT, K=8 default)
"""
import csv
import datetime as dt
import os
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import rasterio
from pystac_client import Client
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = ["coastal", "blue", "green", "red", "rededge1", "nir", "swir16", "scl"]
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B08", "B11", "SCL"]
CHIP_SIZE = 128
PIXEL_M = 10.0

# QC profiles keyed by chip class.
#
# All chip-level QC operates on the "prediction region": a 23x23 central pixel
# block (chip rows 50:73) that maps to Clay's centre-pool patches[11:16, 11:16]
# on the 28x28 patch grid after 128->224 resize. QC matches the region the
# model actually emphasises in addition to the CLS token.
QC_STRICT_PRESENT = dict(
    scene_cloud_max=60,       # STAC pre-screen: skip scenes with >60% global cloud (efficiency)
    region_cloud_max=0.40,    # ≤40% cloud/shadow/cirrus pixels in region
    region_water_min=0.30,    # ≥30% water pixels in region
    region_fill_max=0.70,     # ≤70% no-data pixels in region
    region_b02_range_min=30,  # ≥30 uint16 dynamic range in B02 (rejects atmospheric distortion)
    time_window_days=30,      # search ±30 d around eventDate
    snap_cap_m=300,           # if reported coord fails water QC, snap up to 300 m to nearest water
)

QC_RELAXED_ABSENT = dict(
    scene_cloud_max=60,       # STAC pre-screen: skip scenes with >60% global cloud (efficiency)
    region_cloud_max=0.40,    # same cloud cap in region
    region_water_min=0.30,    # SYMMETRIC with PRESENT (fix for reviewer L2)
    region_fill_max=0.70,     # still reject mostly-fill chips (edge-of-tile black blocks)
    region_b02_range_min=30,  # same distortion filter
    time_window_days=60,      # wider window because many absent sites are cloudy
    snap_cap_m=300,           # snap to nearest water within 300 m if reported coord is on land
)

# Bands are numbered 0..7 in the HDF5. Clay uses 1..6 (blue..swir16); SCL is 7.
SCL_BAND_IDX = 7
CLOUD_SCL = (3, 8, 9, 10)   # cloud_shadow, cloud_med, cloud_high, cirrus
WATER_SCL = (6,)
LAND_SCL = (4, 5, 7, 11)    # vegetation, bare, unclassified, snow
NODATA_SCL = (0,)           # S2 fill / no_data — appears at tile edges & masked pixels
# Prediction region: 23x23 pixel block at chip centre, maps to Clay's centre-pool
# patches[11:16, 11:16] on the 28x28 patch grid after 128->224 resize.
REGION_R0, REGION_R1 = 50, 73  # inclusive:exclusive on the 128 chip

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


# ---------------- chip reading + QC ----------------

def read_chip(item, lon, lat) -> np.ndarray:
    """Read a 128x128, 8-band chip centred at (lon, lat) from an S2 STAC item.

    E2: 8 COG reads are parallelised across a small thread pool. Each read is
    network I/O bound so threading gives near-linear speedup up to ~8 workers.
    """
    from concurrent.futures import ThreadPoolExecutor
    chip = np.zeros((len(BANDS), CHIP_SIZE, CHIP_SIZE), dtype=np.uint16)
    def read_one(idx_key):
        i, key = idx_key
        # SCL is a categorical mask; anything but nearest-neighbour interpolates
        # class codes into fractional values that no longer equal WATER_SCL[0]==6
        # after uint16 rounding, undercounting water at coastal boundaries and
        # over-triggering snap-to-water (reviewer finding N1).
        rsmpl = (rasterio.enums.Resampling.nearest if key == "scl"
                 else rasterio.enums.Resampling.bilinear)
        with rasterio.open(item.assets[key].href) as src:
            xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
            cx, cy = xs[0], ys[0]
            half = (CHIP_SIZE * PIXEL_M) / 2
            window = from_bounds(cx - half, cy - half, cx + half, cy + half, src.transform)
            data = src.read(1, window=window, out_shape=(CHIP_SIZE, CHIP_SIZE),
                            resampling=rsmpl)
        chip[i] = data.astype(np.uint16)
    with ThreadPoolExecutor(max_workers=len(BANDS)) as pool:
        list(pool.map(read_one, list(enumerate(BANDS))))
    return chip


def snap_to_water(item, lon, lat, cap_m=300):
    """Snap (lon, lat) to the nearest water pixel in item's SCL within cap_m.

    Reads only the SCL band over a widened window centered on the input point.
    Returns (new_lon, new_lat, snap_dist_m) if a water pixel is found inside the
    cap, or None if the location is truly inland (no water within cap_m).

    Used by _worker when the reported observer location fails the water-region
    QC — a shore observer's "no kelp" report legitimately applies to the water
    they were looking at, not the rock they stood on. Symmetrically, a rocky-
    coast PRESENT observation can be re-anchored on the nearshore kelp bed.
    """
    cap_px = int(round(cap_m / PIXEL_M))
    wide = CHIP_SIZE + 2 * cap_px
    with rasterio.open(item.assets["scl"].href) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        cx, cy = xs[0], ys[0]
        half = (wide * PIXEL_M) / 2
        window = from_bounds(cx - half, cy - half, cx + half, cy + half, src.transform)
        scl = src.read(1, window=window, out_shape=(wide, wide),
                       resampling=rasterio.enums.Resampling.nearest).astype(np.int16)
        src_crs = src.crs
    ccx, ccy = wide // 2, wide // 2
    # Already on water? no snap needed.
    if 0 <= ccy < wide and 0 <= ccx < wide and scl[ccy, ccx] == WATER_SCL[0]:
        return lon, lat, 0.0
    water = (scl == WATER_SCL[0])
    if not water.any():
        return None
    ys_w, xs_w = np.where(water)
    dx = xs_w - ccx; dy = ys_w - ccy
    d2 = dx * dx + dy * dy
    cap2 = cap_px * cap_px
    inside = d2 <= cap2
    if not inside.any():
        return None
    d2_masked = np.where(inside, d2, d2.max() + 1)
    idx = int(np.argmin(d2_masked))
    wy, wx = int(ys_w[idx]), int(xs_w[idx])
    snap_dist_m = float(np.sqrt(d2[idx])) * PIXEL_M
    # Pixel (wx, wy) → CRS coords in src_crs. The widened window we requested
    # spanned CRS x in [cx-half, cx+half] mapped to output cols 0..wide-1, and
    # CRS y in [cy-half, cy+half] mapped to output rows 0..wide-1 (top-down).
    crs_x = (cx - half) + (wx + 0.5) * PIXEL_M
    crs_y = (cy + half) - (wy + 0.5) * PIXEL_M
    lons, lats = warp_transform(src_crs, "EPSG:4326", [crs_x], [crs_y])
    return float(lons[0]), float(lats[0]), snap_dist_m



def qc_chip(chip: np.ndarray, qc: dict, lon=None, lat=None, item=None):
    """Return (status, metrics). status='ok' if the chip passes the QC config.

    All checks operate on the 23x23-pixel *prediction region* at chip centre
    (rows/cols 50:73) — the area Clay's centre-pool sees (patches[11:16, 11:16]).
    Aligning QC with what the model actually looks at prevents:
      - ABSENT chips passing on flat-zero centres (finding #2 from review)
      - PRESENT chips landing on land in the region the model emphasises
      - Chips with atmospheric distortion (very low B02 dynamic range) that
        look like flat unnatural surfaces
    """
    scl = chip[SCL_BAND_IDX]
    b02 = chip[1]  # blue band
    reg_scl = scl[REGION_R0:REGION_R1, REGION_R0:REGION_R1]
    reg_b02 = b02[REGION_R0:REGION_R1, REGION_R0:REGION_R1]
    wf_region    = float((reg_scl == WATER_SCL[0]).mean())
    fill_region  = float(np.isin(reg_scl, NODATA_SCL).mean())
    cloud_region = float(np.isin(reg_scl, CLOUD_SCL).mean())
    fill_whole   = float(np.isin(scl, NODATA_SCL).mean())  # kept for record only
    reg_b02_nz = reg_b02[reg_b02 > 0]
    if len(reg_b02_nz) > 20:
        b02_range = float(np.percentile(reg_b02_nz, 95) - np.percentile(reg_b02_nz, 5))
    else:
        b02_range = 0.0
    m = dict(wf_region=round(wf_region, 3),
             fill_region=round(fill_region, 3),
             cloud_region=round(cloud_region, 3),
             fill_whole=round(fill_whole, 3),
             b02_range=round(b02_range, 1))
    if fill_region > qc.get("region_fill_max", 0.70):
        return "bad_fill", m
    if cloud_region > qc.get("region_cloud_max", 0.40):
        return "bad_cloud", m
    if wf_region < qc.get("region_water_min", 0.0):
        return "bad_water", m
    if b02_range < qc.get("region_b02_range_min", 0):
        return "bad_distortion", m
    return "ok", m


# ---------------- STAC search helper ----------------

def search_scenes(client, lon, lat, date, qc, time_window_days=None):
    """Return (cloud_cover, item) tuples sorted cleanest-first, one per day.

    Scene-level cloud_cover serves two purposes:
      1. Efficiency pre-screen: skip scenes with global cloud > scene_cloud_max
         (typically 60%). Without this we'd STAC-search + read + reject a huge
         number of pathologically cloudy scenes.
      2. Sort key: within accepted scenes, try cleanest first.
    The correctness check is still chip-level (cloud_region in qc_chip).

    time_window_days: optional int, overrides qc["time_window_days"] when
    provided. Used by ABSENT fetch — each candidate carries a per-record
    allowed_max_days (30 or 60) computed from spatial+temporal proximity to
    raw PRESENT annotations. PRESENT calls with time_window_days=None to use
    QC_STRICT_PRESENT's 30-day default.
    """
    # Clamp to qc["time_window_days"] so a rogue JSON value (e.g. 600) can't drive an unbounded STAC search (reviewer round 3, minor).
    twd = int(time_window_days) if time_window_days is not None else qc["time_window_days"]
    twd = min(twd, qc["time_window_days"])
    start = date - dt.timedelta(days=twd)
    end   = date + dt.timedelta(days=twd)
    box = 0.005
    items = list(client.search(
        collections=[COLLECTION],
        bbox=(lon-box, lat-box, lon+box, lat+box),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": qc["scene_cloud_max"]}},
        limit=100,
    ).items())
    seen = set(); out = []
    for item in items:
        if item.datetime is None: continue
        d = item.datetime.date().isoformat()
        if d in seen: continue
        seen.add(d)
        cc = item.properties.get("eo:cloud_cover", 100)
        out.append((cc, item))
    out.sort(key=lambda x: x[0])
    return out


# ---------------- driver: fetch chips for a set of records ----------------

def _worker(subset, qc, max_scenes, exclude_scene_ids, state, log):
    client = Client.open(STAC_URL)
    targets = state.get("species_targets") or {}
    disable_snap = state.get("disable_snap", False)
    snap_cap = 0 if disable_snap else qc.get("snap_cap_m", 0)
    class_tag = state.get("class_tag", "")
    for record in subset:
        if state.get("stop"): return  # graceful global stop
        sp = record.get("species") or ""
        if targets and sp in targets:
            # Hold h5_lock so the check-and-set with the increment inside the
            # write path (further below) is atomic. Without this two threads
            # can each see count < target for the same species and both proceed
            # to fetch a chip, overshooting the target.
            with state["h5_lock"]:
                if state["species_counts"].get(sp, 0) >= targets[sp]:
                    continue
        gid = record["gbifID"]
        try:
            lat = float(record["decimalLatitude"]); lon = float(record["decimalLongitude"])
            date = dt.date.fromisoformat(record["eventDate"][:10])
        except Exception:
            continue
        # Per-record time window (ABSENT only). allowed_max_days is stamped on
        # the candidate row upstream (fetch_absent.py) based on
        # nearby-PRESENT proximity — 30 by default, 60 when no PRESENT record
        # exists within 1 km + ±90 d of this observation. Missing/blank falls
        # back to the QC default. PRESENT candidates never carry the field.
        allowed = record.get("allowed_max_days")
        try:
            allowed = int(allowed) if allowed not in (None, "") else None
        except (ValueError, TypeError):
            allowed = None
        candidates = None
        for attempt in range(3):
            try:
                candidates = search_scenes(client, lon, lat, date, qc,
                                           time_window_days=allowed)
                break
            except Exception:
                time.sleep(2 ** attempt)
        if candidates is None:
            state["counts"]["stac_error"] = state["counts"].get("stac_error", 0) + 1
            continue
        if not candidates:
            # STAC returned zero scenes in the search window. This happens
            # when the eventDate is pre-Sentinel-2 (< 2015-06) or when every
            # scene in the ±window exceeded scene_cloud_max. Silently skipping
            # made this look like every record was land/water-QC-failing;
            # instead surface it so pre-S2 records are obvious.
            state["counts"]["no_stac_scenes"] = state["counts"].get("no_stac_scenes", 0) + 1
            continue
        existing = exclude_scene_ids.get(gid, set())

        # ---- Pass 1: try all candidates without snap; collect passes ----
        passes = []           # (item, chip, m, snap_dist_m, snap_lon, snap_lat)
        failed_items = []     # (days_off_abs, cc, item) for snap fallback pick
        any_bad_water = False
        for cc, item in candidates:
            if len(passes) >= max_scenes: break
            if item.id in existing: continue
            chip = None
            for attempt in range(3):
                try:
                    chip = read_chip(item, lon, lat); break
                except Exception:
                    time.sleep(2 ** attempt)
            if chip is None:
                state["counts"]["read_error"] = state["counts"].get("read_error", 0) + 1
                continue
            status, m = qc_chip(chip, qc, lon=lon, lat=lat, item=item)
            if status == "ok":
                passes.append((item, chip, m, 0.0, "", ""))
            else:
                state["counts"][status] = state["counts"].get(status, 0) + 1
                if status == "bad_water": any_bad_water = True
                days_off = abs((item.datetime.date() - date).days) if item.datetime else 999
                failed_items.append((days_off, cc, item, status))

        # ---- Pass 2: snap fallback — only if NO passing chip, snap enabled,
        # AND at least one Pass 1 failure was bad_water (reviewer fix #2).
        # Snap can only recover chips that failed for water reasons; cloud /
        # fill / distortion failures don't benefit from moving 300 m.
        if not passes and snap_cap > 0 and any_bad_water and failed_items:
            # Reviewer round 2: filter to bad_water candidates before sorting.
            # cloud/fill/dynamic-range failures can't be fixed by moving 300 m.
            # any_bad_water=True already guarantees at least one bad_water candidate.
            bad_water_candidates = [f for f in failed_items if f[3] == "bad_water"]
            bad_water_candidates.sort(key=lambda x: (x[0], x[1]))  # closest date, then cleanest
            _, _, item, _ = bad_water_candidates[0]
            snapped = None
            try:
                snapped = snap_to_water(item, lon, lat, cap_m=snap_cap)
            except Exception:
                pass
            if snapped is None:
                state["counts"]["snap_no_water"] = state["counts"].get("snap_no_water", 0) + 1
            else:
                s_lon, s_lat, sd = snapped
                new_chip = None
                for attempt in range(3):
                    try:
                        new_chip = read_chip(item, s_lon, s_lat); break
                    except Exception:
                        time.sleep(2 ** attempt)
                if new_chip is not None:
                    n_status, n_m = qc_chip(new_chip, qc, lon=s_lon, lat=s_lat, item=item)
                    if n_status == "ok":
                        passes.append((item, new_chip, n_m, sd, s_lon, s_lat))
                        state["counts"]["snap_success"] = state["counts"].get("snap_success", 0) + 1
                    else:
                        state["counts"][f"snap_{n_status}"] = state["counts"].get(f"snap_{n_status}", 0) + 1

        # ---- Write out all passes ----
        for pi, (item, chip, m, snap_dist_m, snap_lon, snap_lat) in enumerate(passes, 1):
            with state["h5_lock"]:
                ci = state["idx"]
                state["h5"]["chips"][ci] = chip
                state["idx"] += 1
            out_gid = f"{gid}_s{pi}" if max_scenes > 1 else gid
            row = dict(record)
            row["gbifID"] = out_gid
            row["sceneID"] = item.id
            row["sceneDate"] = item.datetime.date().isoformat() if item.datetime else ""
            row["daysOff"] = (item.datetime.date() - date).days if item.datetime else ""
            row["cloudCover"] = m["cloud_region"]
            row["wf_region"] = m["wf_region"]
            row["fill_region"] = m["fill_region"]
            row["cloud_region"] = m["cloud_region"]
            row["fill_whole"] = m["fill_whole"]
            row["b02_range"] = m["b02_range"]
            row["snap_dist_m"] = round(snap_dist_m, 1)
            row["snap_lon"] = snap_lon
            row["snap_lat"] = snap_lat
            row["status"] = "ok"
            row["chipIndex"] = ci
            with state["csv_lock"]:
                state["csv"].writerow({k: row.get(k, "") for k in state["fields"]})
            state["counts"]["ok"] += 1
            if targets:
                with state["h5_lock"]:
                    sp_key = record.get("species") or ""
                    state["species_counts"][sp_key] = state["species_counts"].get(sp_key, 0) + 1
                    if all(state["species_counts"].get(s, 0) >= tgt
                           for s, tgt in targets.items()):
                        state["stop"] = True
        if passes:
            state["counts"]["records_with_chip"] = state["counts"].get("records_with_chip", 0) + 1


def fetch_chips(candidates, qc, out_dir, class_tag,
                max_scenes_per_record=1, exclude_scene_ids=None, workers=6,
                species_targets=None, disable_snap=False):
    """Fetch S2 chips for a list of candidate records.

    Arguments
      candidates    : list of dicts with gbifID, decimalLatitude, decimalLongitude, eventDate
      qc            : QC config dict (see QC_STRICT_PRESENT / QC_RELAXED_ABSENT)
      out_dir       : Path to write chips.h5 and chips_metadata.csv
      class_tag     : label written into the CSV ('present' or 'absent_real')
      max_scenes_per_record : cap per gbifID (1 = classic, 8 = multi-scene)
      exclude_scene_ids     : dict {gbifID: {sceneID, ...}} — skip these scenes
                              (use when re-fetching to complement existing chips)
      workers       : concurrent worker threads
    """
    exclude_scene_ids = exclude_scene_ids or {}
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True, parents=True)
    log_path = out_dir / "fetch.log"
    log_lock = threading.Lock()

    def log(msg):
        with log_lock:
            ts = dt.datetime.utcnow().isoformat(timespec="seconds") + "+00:00"
            with log_path.open("a") as fh:
                fh.write(f"{ts} {msg}\n")
            print(f"{ts} {msg}", flush=True)

    # Add extra fields to CSV
    if not candidates:
        log("no candidates"); return
    fields = list(candidates[0].keys()) + [
        "sceneID","sceneDate","daysOff","cloudCover",
        "wf_region","fill_region","cloud_region","fill_whole","b02_range",
        "snap_dist_m","snap_lon","snap_lat","allowed_max_days",
        "status","chipIndex","class_tag"]
    for r in candidates:
        r["class_tag"] = class_tag

    max_chips = len(candidates) * max_scenes_per_record
    csv_path = out_dir / "chips_metadata.csv"
    csv_fh = csv_path.open("w", newline="")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=list(dict.fromkeys(fields)))
    csv_writer.writeheader()

    h5_path = out_dir / "chips.h5"
    h5f = h5py.File(h5_path, "w")
    h5f.attrs["bands"] = np.array(BAND_NAMES, dtype="S")
    h5f.attrs["chip_size"] = CHIP_SIZE
    h5f.attrs["pixel_m"] = PIXEL_M
    h5f.create_dataset("chips", shape=(max_chips, len(BANDS), CHIP_SIZE, CHIP_SIZE),
                       dtype=np.uint16, chunks=(1, 8, CHIP_SIZE, CHIP_SIZE), compression="lzf")

    state = {"h5": h5f, "csv": csv_writer, "idx": 0,
             "fields": csv_writer.fieldnames,
             "csv_lock": threading.Lock(), "h5_lock": threading.Lock(),
             "counts": {"ok": 0},
             "species_targets": species_targets,
             "species_counts": {},
             "disable_snap": bool(disable_snap),
             "class_tag": class_tag,
             "stop": False}

    per_worker = (len(candidates) + workers - 1) // workers
    threads = []
    for w in range(workers):
        sub = candidates[w*per_worker:(w+1)*per_worker]
        if not sub: continue
        t = threading.Thread(target=_worker,
                             args=(sub, qc, max_scenes_per_record,
                                    exclude_scene_ids, state, log))
        t.start(); threads.append(t)
        log(f"worker {w} started on {len(sub)} records")

    t0 = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        with state["csv_lock"]: csv_fh.flush()
        extra = ""
        if state.get("species_targets"):
            extra = f"  species_counts={dict(state['species_counts'])}"
        log(f"progress: {state['counts']}  idx={state['idx']}  elapsed={time.time()-t0:.0f}s{extra}")
        if state.get("stop"):
            log("all species targets met — signalling graceful stop")
            break
    for t in threads: t.join()

    h5f.attrs["n_chips"] = state["idx"]
    csv_fh.close(); h5f.close()
    log(f"finished: {state['counts']}")
    return state["counts"]
