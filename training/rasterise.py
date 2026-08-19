"""training/rasterise.py — dense P(kelp) raster over a bbox from an S2 scene.

Tiles the bbox with overlapping 128×128 chips, runs Clay per chip, applies the
pool=1 head to every 28×28 patch position (not just the centre pool), mosaics
with overlap-averaging, bilinearly upsamples to 10 m, applies an SCL land /
cloud / cirrus mask post-hoc, writes a Float32 GeoTIFF in the scene's UTM CRS.

The pool=1 head takes `CLS ‖ one patch embedding → scalar`. Feeding it every
patch's embedding in turn yields a 28×28 prediction grid per chip; each patch
covers ~46 m of ground, so the 10 m raster is a bilinear upsample of a ~46 m
grid — spatially addressable at 10 m, effectively resolved at ~46 m.

Usage:
    python training/rasterise.py \\
        --bbox -9.235 38.395 -9.170 38.425 \\
        --scene S2B_29SMD_20260721_0_L2A \\
        --out training/results/sesimbra_arrabida_prob.tif
"""
import argparse
import datetime as dt
import os
import sys as _sys
import time
from pathlib import Path
from pathlib import Path as _P

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

# Locate repo root by walking up until config.py is found
_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config
from patch_pool import PATCH_GRID

CHIP_SIZE = 128
PIXEL_M = 10.0
# STAC band keys in the order Clay expects for its 6 spectral inputs; SCL
# tacked on for post-hoc masking.
CLAY_BANDS = ["blue", "green", "red", "rededge1", "nir", "swir16"]
STAC_KEYS  = CLAY_BANDS + ["scl"]
CLAY_WAVES = [0.493, 0.560, 0.665, 0.704, 0.842, 1.610]
CLOUD_SCL  = (3, 8, 9, 10)   # shadow, cloud_med, cloud_high, cirrus
LAND_SCL   = (4, 5, 7, 11)   # vegetation, bare, unclassified, snow
NODATA_SCL = (0,)

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


# Same architecture as training/train_single.py, so state_dict loads cleanly.
class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def stac_item(scene_id):
    client = Client.open("https://earth-search.aws.element84.com/v1")
    for item in client.search(collections=["sentinel-2-l2a"], ids=[scene_id]).items():
        return item
    raise ValueError(f"scene {scene_id} not found on Earth-Search STAC")


def read_bbox(item, bbox_wgs84):
    """Read all needed bands over bbox_wgs84 into (B, H, W) uint16, aligned to
    the scene's UTM 10-m grid. SCL nearest-resampled; others bilinear."""
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    with rasterio.open(item.assets["blue"].href) as ref:
        crs = ref.crs
        xs, ys = warp_transform("EPSG:4326", crs, [lon_min, lon_max], [lat_min, lat_max])
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        # Snap to 10 m grid so pixel boundaries land on Sentinel-2's native grid.
        x0 = np.floor(x0 / PIXEL_M) * PIXEL_M
        x1 = np.ceil (x1 / PIXEL_M) * PIXEL_M
        y0 = np.floor(y0 / PIXEL_M) * PIXEL_M
        y1 = np.ceil (y1 / PIXEL_M) * PIXEL_M
    W = int((x1 - x0) / PIXEL_M)
    H = int((y1 - y0) / PIXEL_M)
    # north-up transform: (col_size, 0, x_origin, 0, -row_size, y_top)
    transform = Affine(PIXEL_M, 0, x0, 0, -PIXEL_M, y1)
    arr = np.zeros((len(STAC_KEYS), H, W), dtype=np.uint16)
    for i, key in enumerate(STAC_KEYS):
        rsmpl = Resampling.nearest if key == "scl" else Resampling.bilinear
        with rasterio.open(item.assets[key].href) as src:
            win = from_bounds(x0, y0, x1, y1, src.transform)
            arr[i] = src.read(1, window=win, out_shape=(H, W), resampling=rsmpl)
        print(f"  read {key:<9} shape={arr[i].shape}", flush=True)
    return arr, transform, crs, (x0, y0, x1, y1)


def rasterise(arr, transform, crs, item, head_state, batch=32, stride_px=64):
    B, H, W = arr.shape
    ys = list(range(0, max(1, H - CHIP_SIZE + 1), stride_px))
    xs = list(range(0, max(1, W - CHIP_SIZE + 1), stride_px))
    if not ys or ys[-1] + CHIP_SIZE < H: ys.append(max(0, H - CHIP_SIZE))
    if not xs or xs[-1] + CHIP_SIZE < W: xs.append(max(0, W - CHIP_SIZE))
    anchors = [(y, x) for y in ys for x in xs]
    print(f"  bbox pixel grid: {H}×{W} at {PIXEL_M:.0f} m")
    print(f"  chip anchors:    {len(anchors)} (stride {stride_px} px = {stride_px*PIXEL_M:.0f} m)")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    d = yaml.safe_load(config.CLAY_META.open())
    band_stats = Box(d)["sentinel-2-l2a"]["bands"]
    means = np.array([band_stats["mean"][k] for k in CLAY_BANDS], dtype=np.float32).reshape(1, -1, 1, 1)
    stds  = np.array([band_stats["std" ][k] for k in CLAY_BANDS], dtype=np.float32).reshape(1, -1, 1, 1)
    waves = torch.tensor(CLAY_WAVES, device=device, dtype=torch.float32)
    gsd   = torch.tensor(10.0, device=device, dtype=torch.float32)

    print("  loading Clay v1.5...", flush=True)
    clay = ClayMAEModule.load_from_checkpoint(
        str(config.CLAY_CKPT), model_size="large", shuffle=False, mask_ratio=0.0,
        map_location=device,
    ).eval()
    head = MLPHead().to(device)
    head.load_state_dict(head_state)
    head.eval()

    # Time embedding: one date per scene.
    date = dt.date.fromisoformat(item.properties["datetime"][:10])
    wk = date.isocalendar()[1]
    tv_row = np.array([np.sin(2*np.pi*wk/52), np.cos(2*np.pi*wk/52), 0, 0], dtype=np.float32)

    prob_sum = np.zeros((H, W), dtype=np.float32)
    prob_cnt = np.zeros((H, W), dtype=np.uint16)

    t0 = time.time()
    with torch.no_grad():
        for i0 in range(0, len(anchors), batch):
            batch_anchors = anchors[i0:i0 + batch]
            Bb = len(batch_anchors)
            pix = np.zeros((Bb, 6, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
            ll  = np.zeros((Bb, 4), dtype=np.float32)
            for bi, (y, x) in enumerate(batch_anchors):
                pix[bi] = arr[0:6, y:y+CHIP_SIZE, x:x+CHIP_SIZE].astype(np.float32)
                cx = transform.c + (x + CHIP_SIZE/2) * PIXEL_M
                cy = transform.f + (y + CHIP_SIZE/2) * (-PIXEL_M)
                lons, lats = warp_transform(crs, "EPSG:4326", [cx], [cy])
                lat, lon = lats[0], lons[0]
                ll[bi] = [np.sin(np.radians(lat)), np.cos(np.radians(lat)),
                          np.sin(np.radians(lon)), np.cos(np.radians(lon))]
            pix = (pix - means) / stds
            x_t = torch.from_numpy(pix).to(device)
            x_t = F.interpolate(x_t, size=(224, 224), mode="bilinear", align_corners=False)
            ll_t = torch.from_numpy(ll).to(device)
            tv_t = torch.from_numpy(np.tile(tv_row, (Bb, 1))).to(device)
            datacube = {"pixels": x_t, "gsd": gsd, "waves": waves,
                        "platform": "sentinel-2-l2a",
                        "time": tv_t, "latlon": ll_t}
            out = clay.model.encoder(datacube)
            if isinstance(out, tuple): out = out[0]
            cls_tok = out[:, 0, :]                                  # (Bb, 1024)
            patches = out[:, 1:, :].reshape(Bb, PATCH_GRID, PATCH_GRID, 1024)
            # Head input = CLS ‖ patch[i,j] for every (i, j).
            cls_exp = cls_tok.unsqueeze(1).unsqueeze(1).expand(-1, PATCH_GRID, PATCH_GRID, -1)
            head_in = torch.cat([cls_exp, patches], dim=-1)          # (Bb, 28, 28, 2048)
            logits  = head.net(head_in.view(-1, 2048)).view(Bb, PATCH_GRID, PATCH_GRID)
            probs   = torch.sigmoid(logits).cpu().numpy()            # (Bb, 28, 28)
            # Upsample 28×28 → 128×128 (bilinear) and splat into the mosaic.
            probs_t = torch.from_numpy(probs).unsqueeze(1)           # (Bb, 1, 28, 28)
            probs_128 = F.interpolate(probs_t, size=(CHIP_SIZE, CHIP_SIZE),
                                      mode="bilinear", align_corners=False).squeeze(1).numpy()
            for bi, (y, x) in enumerate(batch_anchors):
                prob_sum[y:y+CHIP_SIZE, x:x+CHIP_SIZE] += probs_128[bi]
                prob_cnt[y:y+CHIP_SIZE, x:x+CHIP_SIZE] += 1
            if (i0 + Bb) % 100 == 0 or i0 + Bb == len(anchors):
                print(f"  {i0 + Bb}/{len(anchors)}  {time.time()-t0:.0f}s", flush=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        prob = np.where(prob_cnt > 0, prob_sum / prob_cnt, np.nan)

    # SCL land/cloud/cirrus mask — applied POST-HOC (does not influence Clay
    # or the head). Model output over land is meaningless anyway; setting
    # those pixels to NaN keeps the raster honest without introducing any
    # supervisory leakage.
    scl = arr[6]
    mask = np.isin(scl, list(CLOUD_SCL) + list(LAND_SCL) + list(NODATA_SCL))
    prob[mask] = np.nan
    print(f"  masked {int(mask.sum())} pixels ({100*mask.mean():.1f}% of raster)")
    return prob, transform, crs


def write_gtiff(prob, transform, crs, out_path):
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": prob.shape[0], "width": prob.shape[1],
        "crs": crs, "transform": transform,
        "compress": "deflate", "tiled": True,
        "nodata": float("nan"),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(prob.astype(np.float32), 1)
    n_valid = int((~np.isnan(prob)).sum())
    print(f"wrote {out_path}  shape={prob.shape}  valid={n_valid}  "
          f"P(kelp) range=[{np.nanmin(prob):.3f}, {np.nanmax(prob):.3f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    ap.add_argument("--scene", required=True, help="Sentinel-2 L2A scene ID (Element84 STAC)")
    ap.add_argument("--head", default=str(config.RESULTS_DIR / "head_single_p1" / "model.pt"),
                    help="Path to pool=1 deployment head state_dict")
    ap.add_argument("--stride-px", type=int, default=64,
                    help="Chip stride in pixels (default 64 = half-chip overlap)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", required=True, help="Output GeoTIFF path")
    args = ap.parse_args()

    print(f"scene: {args.scene}")
    item = stac_item(args.scene)
    print(f"  datetime: {item.properties['datetime']}   "
          f"cloud%={item.properties.get('eo:cloud_cover', '?')}")

    print("reading bbox from S2 COGs...")
    arr, transform, crs, bbox_utm = read_bbox(item, args.bbox)
    print(f"  bbox UTM: {bbox_utm}  crs: {crs}")

    print("loading head...")
    head_state = torch.load(args.head, map_location="cpu", weights_only=True)

    prob, transform, crs = rasterise(arr, transform, crs, item, head_state,
                                     batch=args.batch, stride_px=args.stride_px)

    write_gtiff(prob, transform, crs, args.out)


if __name__ == "__main__":
    main()
