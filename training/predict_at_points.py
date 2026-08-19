"""training/predict_at_points.py — chip-oracle P(kelp) per CCMar quadrat.

For each unique CCMar quadrat coordinate, fetch a 128×128 S2 chip centred
exactly on that point (so the labelled bed sits where the pool=1 head was
trained to look — the chip centre patch), run Clay + head, and compare to
the raster's P(kelp) at the same pixel + the CCMar aggregate cover.

This isolates "did the model see this bed" from "did the tile-rasteriser
smear the signal across a 46 m patch that overlaps mostly open water" — if
the chip-oracle probs are much higher than the raster probs at the same
coords, the rasteriser is diluting a signal the model actually has.

Efficiency: all CCMar points sit within a ~120 m radius, so one bbox read
that spans (all points + 128 px margin) gets us the raw S2 pixels for every
chip. Then chips are sliced in-memory and pushed through Clay in one batch.

Usage:
    python training/predict_at_points.py \\
        --ccmar   /path/to/kelp_observations_with_coordinates.csv \\
        --scene   S2A_29SMC_20260609_0_L2A \\
        --raster  training/results/sesimbra_arrabida_prob.tif \\
        --out-csv training/results/sesimbra_arrabida_pointpreds.csv \\
        --out-png training/results/sesimbra_arrabida_pointpreds.png
"""
import argparse
import csv
import datetime as dt
import os
import sys as _sys
from collections import defaultdict
from pathlib import Path
from pathlib import Path as _P

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from box import Box
from claymodel.module import ClayMAEModule
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from rasterio.windows import from_bounds

_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent

import config
from patch_pool import centre_pool_slice

CHIP_SIZE = 128
PIXEL_M = 10.0
CLAY_BANDS = ["blue", "green", "red", "rededge1", "nir", "swir16"]
STAC_KEYS  = CLAY_BANDS + ["scl"]
CLAY_WAVES = [0.493, 0.560, 0.665, 0.704, 0.842, 1.610]

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def load_ccmar(csv_path):
    """Aggregate CCMar rows per unique (lat, lon). Returns list of dicts
    {lat, lon, cover, has_kelp, n_rows}."""
    by_ll = defaultdict(list)
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                ll = (round(float(r["latitude"]), 6), round(float(r["longitude"]), 6))
            except (ValueError, KeyError):
                continue
            by_ll[ll].append(r)
    quads = []
    for (lat, lon), rows in by_ll.items():
        kelp_rows = [r for r in rows if (r.get("species") or "").strip().lower() != "empty"]
        cover_vals = []
        for r in kelp_rows:
            try: cover_vals.append(float(r["cover_pct"]))
            except (ValueError, KeyError, TypeError): pass
        quads.append({
            "lat": lat, "lon": lon,
            "cover": sum(cover_vals) if cover_vals else 0.0,
            "has_kelp": bool(kelp_rows), "n_rows": len(rows),
        })
    return quads


def stac_item(scene_id):
    client = Client.open("https://earth-search.aws.element84.com/v1")
    for item in client.search(collections=["sentinel-2-l2a"], ids=[scene_id]).items():
        return item
    raise ValueError(f"scene {scene_id} not found")


def read_bbox_covering(item, quads, chip_margin_px=CHIP_SIZE):
    """Read one bbox from all bands that fully contains a CHIP_SIZE chip
    centred on every quadrat. Returns (arr, transform, crs), where arr is
    uint16 (B, H, W) on the scene's UTM 10-m grid."""
    lats = [q["lat"] for q in quads]
    lons = [q["lon"] for q in quads]
    with rasterio.open(item.assets["blue"].href) as ref:
        crs = ref.crs
    xs, ys = warp_transform("EPSG:4326", crs, lons, lats)
    x0, x1 = min(xs) - chip_margin_px * PIXEL_M, max(xs) + chip_margin_px * PIXEL_M
    y0, y1 = min(ys) - chip_margin_px * PIXEL_M, max(ys) + chip_margin_px * PIXEL_M
    x0 = np.floor(x0 / PIXEL_M) * PIXEL_M
    x1 = np.ceil (x1 / PIXEL_M) * PIXEL_M
    y0 = np.floor(y0 / PIXEL_M) * PIXEL_M
    y1 = np.ceil (y1 / PIXEL_M) * PIXEL_M
    W = int((x1 - x0) / PIXEL_M)
    H = int((y1 - y0) / PIXEL_M)
    transform = Affine(PIXEL_M, 0, x0, 0, -PIXEL_M, y1)
    arr = np.zeros((len(STAC_KEYS), H, W), dtype=np.uint16)
    for i, key in enumerate(STAC_KEYS):
        rsmpl = Resampling.nearest if key == "scl" else Resampling.bilinear
        with rasterio.open(item.assets[key].href) as src:
            win = from_bounds(x0, y0, x1, y1, src.transform)
            arr[i] = src.read(1, window=win, out_shape=(H, W), resampling=rsmpl)
    return arr, transform, crs


def chip_from_bbox(arr, transform, crs, lat, lon):
    """Extract a CHIP_SIZE×CHIP_SIZE chip centred at (lat, lon).
    Returns (chip[6, 128, 128] float32, chip_row_col_of_center)."""
    xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
    cx, cy = float(xs[0]), float(ys[0])
    col_c = (cx - transform.c) / PIXEL_M                     # native col (float)
    row_c = (transform.f - cy) / PIXEL_M                     # native row (float, top-down)
    col0 = int(round(col_c - CHIP_SIZE / 2))
    row0 = int(round(row_c - CHIP_SIZE / 2))
    if row0 < 0 or col0 < 0 or row0 + CHIP_SIZE > arr.shape[1] or col0 + CHIP_SIZE > arr.shape[2]:
        raise RuntimeError(f"chip at (lat={lat}, lon={lon}) falls off bbox — grow chip_margin_px")
    chip = arr[0:6, row0:row0+CHIP_SIZE, col0:col0+CHIP_SIZE].astype(np.float32)
    return chip, (row_c - row0, col_c - col0)


def raster_prob_at(raster_path, lat, lon):
    """Nearest-pixel P(kelp) from the pre-computed GeoTIFF at (lat, lon)."""
    with rasterio.open(raster_path) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        try:
            r, c = src.index(xs[0], ys[0])
        except (rasterio.errors.RasterioError, IndexError):
            return float("nan")
        prob = src.read(1)
        if 0 <= r < prob.shape[0] and 0 <= c < prob.shape[1]:
            return float(prob[r, c])
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ccmar", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--raster", default=None,
                    help="Optional GeoTIFF from rasterise.py; if set, records the "
                         "raster's P(kelp) at each point for comparison.")
    ap.add_argument("--head", default=str(config.RESULTS_DIR / "head_single_p1" / "model.pt"))
    ap.add_argument("--pool-size", type=int, default=1,
                    help="Centre-pool size for head input (default 1, matches deployment head)")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-png", required=True)
    args = ap.parse_args()

    quads = load_ccmar(args.ccmar)
    print(f"CCMar: {len(quads)} unique quadrats  "
          f"({sum(q['has_kelp'] for q in quads)} with kelp)")

    print(f"scene: {args.scene}")
    item = stac_item(args.scene)
    print(f"  datetime: {item.properties['datetime']}   "
          f"cloud%={item.properties.get('eo:cloud_cover', '?')}")

    print("reading bbox covering all CCMar points + 128 px margin ...")
    arr, transform, crs = read_bbox_covering(item, quads)
    print(f"  arr shape={arr.shape}  crs={crs}")

    # ------ Clay + head setup ------
    d = yaml.safe_load(config.CLAY_META.open())
    band_stats = Box(d)["sentinel-2-l2a"]["bands"]
    means = np.array([band_stats["mean"][k] for k in CLAY_BANDS], dtype=np.float32).reshape(1, -1, 1, 1)
    stds  = np.array([band_stats["std" ][k] for k in CLAY_BANDS], dtype=np.float32).reshape(1, -1, 1, 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    waves = torch.tensor(CLAY_WAVES, device=device, dtype=torch.float32)
    gsd   = torch.tensor(10.0, device=device, dtype=torch.float32)
    date = dt.date.fromisoformat(item.properties["datetime"][:10])
    wk = date.isocalendar()[1]
    tv_row = np.array([np.sin(2*np.pi*wk/52), np.cos(2*np.pi*wk/52), 0, 0], dtype=np.float32)

    print("loading Clay v1.5...")
    clay = ClayMAEModule.load_from_checkpoint(
        str(config.CLAY_CKPT), model_size="large", shuffle=False, mask_ratio=0.0,
        map_location=device,
    ).eval()
    head = MLPHead().to(device)
    head.load_state_dict(torch.load(args.head, map_location="cpu", weights_only=True))
    head.eval()

    pool_lo, pool_hi = centre_pool_slice(args.pool_size)
    print(f"  head pool: patches[{pool_lo}:{pool_hi}, {pool_lo}:{pool_hi}]")

    # ------ Build a batch of chips ------
    chips = np.zeros((len(quads), 6, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
    ll    = np.zeros((len(quads), 4), dtype=np.float32)
    for i, q in enumerate(quads):
        chip, _ = chip_from_bbox(arr, transform, crs, q["lat"], q["lon"])
        chips[i] = chip
        lat, lon = q["lat"], q["lon"]
        ll[i] = [np.sin(np.radians(lat)), np.cos(np.radians(lat)),
                 np.sin(np.radians(lon)), np.cos(np.radians(lon))]
    chips = (chips - means) / stds

    with torch.no_grad():
        x = torch.from_numpy(chips).to(device)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        ll_t = torch.from_numpy(ll).to(device)
        tv_t = torch.from_numpy(np.tile(tv_row, (len(quads), 1))).to(device)
        datacube = {"pixels": x, "gsd": gsd, "waves": waves,
                    "platform": "sentinel-2-l2a",
                    "time": tv_t, "latlon": ll_t}
        out = clay.model.encoder(datacube)
        if isinstance(out, tuple): out = out[0]
        cls = out[:, 0, :]                                       # (N, 1024)
        patches = out[:, 1:, :].reshape(len(quads), 28, 28, 1024)
        pool = patches[:, pool_lo:pool_hi, pool_lo:pool_hi, :].mean(dim=(1, 2))  # (N, 1024)
        emb = torch.cat([cls, pool], dim=-1)                     # (N, 2048)
        probs = torch.sigmoid(head(emb)).cpu().numpy()           # (N,)

    print(f"predictions: min={probs.min():.3f}  max={probs.max():.3f}  mean={probs.mean():.3f}")

    # ------ Optional raster comparison ------
    raster_probs = None
    if args.raster:
        raster_probs = np.array([raster_prob_at(args.raster, q["lat"], q["lon"]) for q in quads])
        print(f"raster @ points: min={np.nanmin(raster_probs):.3f}  max={np.nanmax(raster_probs):.3f}  "
              f"mean={np.nanmean(raster_probs):.3f}  NaN={int(np.isnan(raster_probs).sum())}")

    # ------ Write CSV ------
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w") as f:
        w = csv.writer(f)
        cols = ["lat", "lon", "cover_pct_sum", "has_kelp", "n_rows",
                "prob_chip_oracle"]
        if raster_probs is not None:
            cols.append("prob_raster_at_pixel")
        w.writerow(cols)
        for i, q in enumerate(quads):
            row = [q["lat"], q["lon"], q["cover"], int(q["has_kelp"]), q["n_rows"],
                   round(float(probs[i]), 4)]
            if raster_probs is not None:
                row.append(round(float(raster_probs[i]), 4) if not np.isnan(raster_probs[i]) else "")
            w.writerow(row)
    print(f"wrote {args.out_csv}")

    # ------ Scatter plot ------
    covers = np.array([q["cover"] for q in quads])
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    if raster_probs is not None:
        ax.scatter(covers, raster_probs, c="#dd8452", edgecolors="#8b3d17",
                   s=70, alpha=0.85, label="from raster (pixel value)")
    ax.scatter(covers, probs, c="#3aa53a", edgecolors="#003c00",
               s=90, alpha=0.9, label="chip-oracle (centred chip)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, zorder=0,
               label="deployment threshold t=0.5")
    ax.set_xlabel("CCMar cover_pct (sum across kelp species per quadrat)")
    ax.set_ylabel("P(kelp) — pool=1 deployment head")
    ax.set_title(f"Arrábida CCMar quadrats — model vs field cover\n"
                 f"Scene {args.scene} ({date.isoformat()}),   {len(quads)} quadrats")
    ax.set_ylim(0, 1.02); ax.set_xlim(-2, max(covers.max(), 100)*1.05)
    ax.grid(alpha=0.3); ax.legend()
    Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_png, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
