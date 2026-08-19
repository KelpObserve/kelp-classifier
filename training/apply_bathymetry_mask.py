"""apply_bathymetry_mask.py — download EMODnet bathymetry over a bbox and
mask any raster pixel where depth > --max-depth-m to NaN.

EMODnet Bathymetry is free, covers European seas at ~115 m native resolution
via a public WCS endpoint. Depth values are negative below sea level; we
convert to positive depths internally (positive = deeper).

Usage:
  python training/apply_bathymetry_mask.py \\
      --raster training/results_submerged/sesimbra_arrabida_prob.tif \\
      --out    training/results_submerged/sesimbra_arrabida_prob_depth30.tif \\
      --max-depth-m 30 \\
      --bathy-out training/results_submerged/emodnet_bathy_arrabida.tif
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds


# EMODnet WCS endpoint (free, no auth)
EMODNET_WCS = "https://ows.emodnet-bathymetry.eu/wcs"
COVERAGE_ID = "emodnet__mean"


def fetch_bathymetry(bbox_ll, out_path):
    """Download EMODnet bathymetry (mean layer) for a WGS84 lat/lon bbox.
    Returns the path to the downloaded GeoTIFF."""
    lon_min, lat_min, lon_max, lat_max = bbox_ll
    # WCS 2.0.1 GetCoverage
    params = {
        "service": "WCS",
        "version": "2.0.1",
        "request": "GetCoverage",
        "coverageid": COVERAGE_ID,
        "format": "image/tiff",
        "subset": [f"Lat({lat_min},{lat_max})", f"Long({lon_min},{lon_max})"],
    }
    print(f"fetching EMODnet bathymetry for lat {lat_min:.3f}..{lat_max:.3f}, "
          f"lon {lon_min:.3f}..{lon_max:.3f}")
    r = requests.get(EMODNET_WCS, params=params, timeout=120)
    if r.status_code != 200 or not r.headers.get("Content-Type", "").startswith("image/tiff"):
        raise RuntimeError(f"WCS error {r.status_code}: {r.text[:400]}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"  wrote {out_path}  ({len(r.content)/1024:.1f} KiB)")
    return out_path


def load_bathymetry_on_raster_grid(bathy_path, target_raster_path):
    """Reproject EMODnet bathymetry onto the target raster's grid.
    Returns (depth_m array of same shape as target, positive = deeper).

    Expects the source file to use the raw EMODnet convention (negative =
    below sea level). Fresh WCS fetches and files written by --bathy-out
    both use this convention — see main().
    """
    with rasterio.open(target_raster_path) as tgt:
        H, W = tgt.height, tgt.width
        dst_transform = tgt.transform
        dst_crs = tgt.crs
    with rasterio.open(bathy_path) as src:
        dst = np.full((H, W), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata, dst_nodata=np.nan,
        )
    # EMODnet convention: negative below sea level. Flip so positive = depth below sea.
    v = dst[np.isfinite(dst)]
    if v.size and np.median(v) > 0:
        # Coastal bboxes span some land (positive elevation) but the median
        # over a mostly-water raster should be negative under the raw
        # convention. A positive median almost always means the source file
        # was written with an already-flipped depth (bug from an older
        # --bathy-out). Refusing here would break re-runs; warn loudly so
        # the operator can regenerate the cache if needed.
        print(f"  WARN: bathymetry median={np.median(v):.1f} > 0 — source "
              f"file may already be depth-positive (pre-flip). Assuming raw "
              f"EMODnet (negative-below) and flipping anyway; if masking "
              f"looks inverted, refetch by dropping --bathy-cache.")
    depth = -dst
    return depth


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raster", required=True, help="Input prediction GeoTIFF")
    ap.add_argument("--out", required=True, help="Output masked GeoTIFF")
    ap.add_argument("--max-depth-m", type=float, default=30.0,
                    help="Mask pixels deeper than this (m below sea level)")
    ap.add_argument("--bathy-out", default=None,
                    help="Optional: save the downloaded+reprojected bathymetry as GeoTIFF here. "
                         "Stored in raw EMODnet convention (negative = below sea level) so it can "
                         "be re-used losslessly via --bathy-cache.")
    ap.add_argument("--bathy-cache", default=None,
                    help="If given AND exists, load bathymetry from this path instead of fetching")
    args = ap.parse_args()

    # 1. Get raster bbox in WGS84
    with rasterio.open(args.raster) as ds:
        raster_bounds = ds.bounds; raster_crs = ds.crs
        prob = ds.read(1); profile = ds.profile
    bbox_ll = transform_bounds(raster_crs, "EPSG:4326",
                                raster_bounds.left, raster_bounds.bottom,
                                raster_bounds.right, raster_bounds.top,
                                densify_pts=21)
    print(f"raster: {args.raster}  shape={prob.shape}  crs={raster_crs}")
    print(f"  WGS84 bbox for bathymetry request: {bbox_ll}")

    # 2. Fetch or load cached bathymetry
    if args.bathy_cache and Path(args.bathy_cache).exists():
        print(f"using cached bathymetry: {args.bathy_cache}")
        raw_path = args.bathy_cache
    else:
        raw_path = args.bathy_cache or "/tmp/emodnet_raw.tif"
        fetch_bathymetry(bbox_ll, raw_path)

    # 3. Reproject onto raster grid
    print("reprojecting bathymetry onto raster grid...")
    depth = load_bathymetry_on_raster_grid(raw_path, args.raster)
    valid = np.isfinite(depth)
    if valid.sum():
        print(f"  depth stats over raster (m below sea level, positive = deeper):")
        v = depth[valid]
        print(f"    min={v.min():.1f}  p25={np.percentile(v,25):.1f}  "
              f"med={np.percentile(v,50):.1f}  p75={np.percentile(v,75):.1f}  "
              f"p90={np.percentile(v,90):.1f}  max={v.max():.1f}")

    # 4. Save the reprojected bathymetry if requested.
    # Store the RAW EMODnet convention (negative = below sea level) so the
    # file is a drop-in replacement for a fresh WCS fetch on subsequent
    # --bathy-cache runs. Storing depth-positive here would double-flip on
    # reload because load_bathymetry_on_raster_grid unconditionally negates.
    if args.bathy_out:
        with rasterio.open(args.raster) as ds:
            out_profile = ds.profile.copy()
        out_profile.update(dtype="float32", count=1, nodata=float("nan"),
                            compress="deflate", tiled=True)
        Path(args.bathy_out).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(args.bathy_out, "w", **out_profile) as dst:
            dst.write((-depth).astype(np.float32), 1)
        print(f"  wrote reprojected bathymetry (raw EMODnet convention, "
              f"negative = below sea level): {args.bathy_out}")

    # 5. Mask prediction where depth > threshold
    too_deep = np.isfinite(depth) & (depth > args.max_depth_m)
    n_before = int((~np.isnan(prob)).sum())
    prob_masked = prob.copy()
    prob_masked[too_deep] = np.nan
    n_after = int((~np.isnan(prob_masked)).sum())
    print(f"\nmasking pixels where depth > {args.max_depth_m} m:")
    print(f"  valid pixels before: {n_before}  after: {n_after}  "
          f"masked: {n_before - n_after} ({100*(n_before-n_after)/max(1,n_before):.1f}%)")

    # 6. Write masked raster
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, nodata=float("nan"),
                        compress="deflate", tiled=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.out, "w", **out_profile) as dst:
        dst.write(prob_masked.astype(np.float32), 1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
