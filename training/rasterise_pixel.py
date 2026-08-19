"""training/rasterise_pixel.py — per-pixel P(kelp) raster using the pixel head
(CLS + interpolated Clay patch + raw S2 at every pixel).

Same tiling / bbox / SCL masking as rasterise.py, but the head is now evaluated
at EVERY 10 m pixel of every chip (128×128 = 16 384 head calls per chip)
instead of at the 28×28 Clay patch positions.

Per-pixel feature (2054-D):
  CLS (1024)                          — chip-global Clay context, broadcast
  interpolated Clay patch (1024)      — bilinear upsample of 28×28 → 128×128
  raw S2 (6)                          — Clay-normalised 6-band reflectance

Head is run per-chip (rather than batched across chips) to bound VRAM: the
biggest tensor per chip is (16 384, 2054) → ~134 MB fp32.

Usage:
  python training/rasterise_pixel.py \\
      --bbox -9.235 38.395 -9.170 38.425 \\
      --scene S2B_29SMD_20260721_0_L2A \\
      --head training/results_submerged/head_single_pixel/model.pt \\
      --out  training/results_submerged/sesimbra_arrabida_prob_pixel.tif
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

_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config
from patch_pool import PATCH_GRID

CHIP_SIZE = 128
PIXEL_M = 10.0
EMBED_DIM = 2048 + 6

CLAY_BANDS = ["blue", "green", "red", "rededge1", "nir", "swir16"]
STAC_KEYS  = CLAY_BANDS + ["scl"]
CLAY_WAVES = [0.493, 0.560, 0.665, 0.704, 0.842, 1.610]
CLOUD_SCL  = (3, 8, 9, 10)
LAND_SCL   = (4, 5, 7, 11)
NODATA_SCL = (0,)

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")


class MLPHead(nn.Module):
    """2054-D pixel head, matches train_single_pixel.py."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 512), nn.ReLU(), nn.Dropout(0.3),
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
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    with rasterio.open(item.assets["blue"].href) as ref:
        crs = ref.crs
        xs, ys = warp_transform("EPSG:4326", crs, [lon_min, lon_max], [lat_min, lat_max])
        x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
        x0 = np.floor(x0 / PIXEL_M) * PIXEL_M; x1 = np.ceil(x1 / PIXEL_M) * PIXEL_M
        y0 = np.floor(y0 / PIXEL_M) * PIXEL_M; y1 = np.ceil(y1 / PIXEL_M) * PIXEL_M
    W = int((x1 - x0) / PIXEL_M); H = int((y1 - y0) / PIXEL_M)
    transform = Affine(PIXEL_M, 0, x0, 0, -PIXEL_M, y1)
    arr = np.zeros((len(STAC_KEYS), H, W), dtype=np.uint16)
    for i, key in enumerate(STAC_KEYS):
        rsmpl = Resampling.nearest if key == "scl" else Resampling.bilinear
        with rasterio.open(item.assets[key].href) as src:
            win = from_bounds(x0, y0, x1, y1, src.transform)
            arr[i] = src.read(1, window=win, out_shape=(H, W), resampling=rsmpl)
        print(f"  read {key:<9} shape={arr[i].shape}", flush=True)
    return arr, transform, crs, (x0, y0, x1, y1)


def rasterise(arr, transform, crs, item, head_state, batch=16, stride_px=64):
    B, H, W = arr.shape
    ys = list(range(0, max(1, H - CHIP_SIZE + 1), stride_px))
    xs = list(range(0, max(1, W - CHIP_SIZE + 1), stride_px))
    if not ys or ys[-1] + CHIP_SIZE < H: ys.append(max(0, H - CHIP_SIZE))
    if not xs or xs[-1] + CHIP_SIZE < W: xs.append(max(0, W - CHIP_SIZE))
    anchors = [(y, x) for y in ys for x in xs]
    print(f"  bbox pixel grid: {H}×{W} at {PIXEL_M:.0f} m")
    print(f"  chip anchors:    {len(anchors)} (stride {stride_px} px = {stride_px*PIXEL_M:.0f} m)")
    print(f"  head calls per chip: {CHIP_SIZE*CHIP_SIZE}  (pixel head)")

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
            pix_norm = (pix - means) / stds                                  # (Bb, 6, 128, 128) normalised
            x_224 = F.interpolate(torch.from_numpy(pix_norm).to(device),
                                    size=(224, 224), mode="bilinear", align_corners=False)
            ll_t = torch.from_numpy(ll).to(device)
            tv_t = torch.from_numpy(np.tile(tv_row, (Bb, 1))).to(device)
            datacube = {"pixels": x_224, "gsd": gsd, "waves": waves,
                        "platform": "sentinel-2-l2a", "time": tv_t, "latlon": ll_t}
            out = clay.model.encoder(datacube)
            if isinstance(out, tuple): out = out[0]
            cls_tok = out[:, 0, :]                                            # (Bb, 1024)
            patches = out[:, 1:, :].reshape(Bb, PATCH_GRID, PATCH_GRID, 1024)  # (Bb, 28, 28, 1024)

            # Bilinear upsample of the 28×28 patch grid → 128×128 (per pixel context)
            patches_up = F.interpolate(
                patches.permute(0, 3, 1, 2),                                  # (Bb, 1024, 28, 28)
                size=(CHIP_SIZE, CHIP_SIZE), mode="bilinear", align_corners=False,
            ).permute(0, 2, 3, 1)                                             # (Bb, 128, 128, 1024)

            # Raw S2 per pixel (already Clay-normalised, same scale as training)
            raw_s2 = torch.from_numpy(pix_norm).to(device).permute(0, 2, 3, 1)  # (Bb, 128, 128, 6)

            # Head per chip to bound VRAM
            probs_np = np.zeros((Bb, CHIP_SIZE, CHIP_SIZE), dtype=np.float32)
            for bi in range(Bb):
                cls_chip = cls_tok[bi].view(1, 1, 1024).expand(CHIP_SIZE, CHIP_SIZE, 1024)
                head_in = torch.cat([cls_chip, patches_up[bi], raw_s2[bi]], dim=-1)  # (128, 128, 2054)
                logits = head.net(head_in.view(-1, EMBED_DIM)).view(CHIP_SIZE, CHIP_SIZE)
                probs_np[bi] = torch.sigmoid(logits).cpu().numpy()

            for bi, (y, x) in enumerate(batch_anchors):
                prob_sum[y:y+CHIP_SIZE, x:x+CHIP_SIZE] += probs_np[bi]
                prob_cnt[y:y+CHIP_SIZE, x:x+CHIP_SIZE] += 1
            if (i0 + Bb) % 50 == 0 or i0 + Bb == len(anchors):
                print(f"  {i0 + Bb}/{len(anchors)}  {time.time()-t0:.0f}s", flush=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        prob = np.where(prob_cnt > 0, prob_sum / prob_cnt, np.nan)

    scl = arr[6]
    mask = np.isin(scl, list(CLOUD_SCL) + list(LAND_SCL) + list(NODATA_SCL))
    prob[mask] = np.nan
    print(f"  masked {int(mask.sum())} pixels ({100*mask.mean():.1f}% of raster)")
    return prob, transform, crs


def write_gtiff(prob, transform, crs, out_path):
    profile = {"driver": "GTiff", "dtype": "float32", "count": 1,
               "height": prob.shape[0], "width": prob.shape[1],
               "crs": crs, "transform": transform,
               "compress": "deflate", "tiled": True, "nodata": float("nan")}
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
    ap.add_argument("--scene", required=True)
    ap.add_argument("--head", default=str(config.RESULTS_DIR / "head_single_pixel" / "model.pt"))
    ap.add_argument("--stride-px", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16,
                    help="Clay batch size; head runs per chip regardless")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"scene: {args.scene}")
    item = stac_item(args.scene)
    print(f"  datetime: {item.properties['datetime']}   "
          f"cloud%={item.properties.get('eo:cloud_cover', '?')}")

    print("reading bbox from S2 COGs...")
    arr, transform, crs, bbox_utm = read_bbox(item, args.bbox)
    print(f"  bbox UTM: {bbox_utm}  crs: {crs}")

    print(f"loading head: {args.head}")
    head_state = torch.load(args.head, map_location="cpu", weights_only=True)

    prob, transform, crs = rasterise(arr, transform, crs, item, head_state,
                                     batch=args.batch, stride_px=args.stride_px)
    write_gtiff(prob, transform, crs, args.out)


if __name__ == "__main__":
    main()
