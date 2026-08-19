"""plot_pneo_overlay.py — two Sesimbra visualisation figures for a given
S2 P(kelp) raster:

  A. `<out_prefix>_over_pneo.png`
        PNEO RGB base + semi-transparent S2 P(kelp) heatmap on top.
        Shows what the model claims about the water surface, over the
        actual PNEO imagery of the seabed.

  B. `<out_prefix>_with_ccmar.png`
        S2 heatmap base + CCMar quadrat observations overlaid (green =
        PRESENT, red hollow = ABSENT; marker area ∝ mean cover_pct per
        quadrat). Ground-truth check against the model's prediction.

Both figures share the same bbox — the intersection of the S2 raster and
PNEO scene footprints — so PNEO detail and S2 prediction are directly
comparable.

Usage:
  python training/plot_pneo_overlay.py \
      --raster training/results_submerged/sesimbra_arrabida_prob.tif \
      --pneo   /path/to/000598128_1_2_STD_A \
      --ccmar  /path/to/kelp_observations_with_coordinates.csv \
      --out-prefix training/results_submerged/sesimbra_pool1
"""
import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as warp_transform


# ---------------- I/O ----------------

def load_pneo_rgb(scene_dir, bbox_utm, out_shape, out_transform, out_crs):
    """Load PNEO RGB (bands R, G, B) resampled onto out_transform/out_shape.

    Returns (rgb (H, W, 3) float in [0, 1], and a mask of nodata).
    """
    rgb_path = glob.glob(str(Path(scene_dir) / "IMG_01_PNEO*_MS-FS" / "*RGB*R1C1.JP2"))[0]
    x0, y0, x1, y1 = bbox_utm
    H, W = out_shape
    window = None
    with rasterio.open(rgb_path) as ds:
        assert ds.crs == out_crs, f"PNEO CRS {ds.crs} != raster CRS {out_crs}"
        window = rasterio.windows.from_bounds(x0, y0, x1, y1, ds.transform)
        arr = ds.read(out_shape=(3, H, W), window=window,
                       resampling=Resampling.bilinear)
    # Simple percentile stretch on non-zero pixels for a readable base image
    rgb = arr.astype(np.float32)
    nodata = (rgb.sum(axis=0) == 0)
    for i in range(3):
        band = rgb[i]
        valid = band[~nodata]
        if valid.size == 0: continue
        p2, p98 = np.percentile(valid, [2, 98])
        band = np.clip((band - p2) / (p98 - p2 + 1e-9), 0, 1)
        rgb[i] = band
    rgb = np.moveaxis(rgb, 0, -1)   # (H, W, 3)
    rgb[nodata] = 0
    return rgb, nodata


def load_ccmar_quadrats(csv_path):
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
        covers = []
        for r in kelp_rows:
            try: covers.append(float(r["cover_pct"]))
            except (ValueError, KeyError, TypeError): pass
        quads.append({"lat": lat, "lon": lon,
                      "cover": sum(covers) if covers else 0.0,
                      "has_kelp": bool(kelp_rows)})
    return quads


def add_scale_bar(ax, x0, y0, length_m, height_m):
    ax.add_patch(plt.Rectangle((x0, y0), length_m/2, height_m,
                                facecolor="black", edgecolor="black", lw=0.5, zorder=10))
    ax.add_patch(plt.Rectangle((x0 + length_m/2, y0), length_m/2, height_m,
                                facecolor="white", edgecolor="black", lw=0.5, zorder=10))
    def fmt(m): return f"{m:.0f} m" if m < 1000 else f"{m/1000:g} km"
    ax.text(x0, y0 + height_m*3, "0", ha="center", va="bottom", fontsize=8, zorder=10)
    ax.text(x0 + length_m/2, y0 + height_m*3, fmt(length_m/2), ha="center", va="bottom", fontsize=8, zorder=10)
    ax.text(x0 + length_m, y0 + height_m*3, fmt(length_m), ha="center", va="bottom", fontsize=8, zorder=10)


def add_north(ax, extent):
    x_min, x_max, y_min, y_max = extent
    span_x = x_max - x_min; span_y = y_max - y_min
    ax.annotate("N", xy=(x_max - span_x*0.04, y_max - span_y*0.05),
                xytext=(x_max - span_x*0.04, y_max - span_y*0.13),
                arrowprops=dict(facecolor="white", edgecolor="black", width=3, headwidth=10, headlength=10),
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="white", zorder=10,
                path_effects=[])


# ---------------- Main plotting ----------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raster", required=True)
    ap.add_argument("--pneo", required=True, help="Path to PNEO scene STD_A directory")
    ap.add_argument("--ccmar", required=True)
    ap.add_argument("--out-prefix", required=True,
                    help="Prefix for output PNGs; suffix _over_pneo.png / _with_ccmar.png added")
    ap.add_argument("--heatmap-alpha", type=float, default=0.55,
                    help="Base alpha for the heatmap; also multiplied by P(kelp) "
                         "so low-probability areas fade into the base image.")
    ap.add_argument("--pred-colormap", default="Reds",
                    help="Matplotlib colormap for P(kelp). Default Reds — high "
                         "P(kelp) shows as deep red, low as near-transparent white. "
                         "Alternatives: 'hot', 'viridis', 'YlOrRd', 'magma'.")
    ap.add_argument("--pool-label", default="pool=1",
                    help="Short label of the model (goes in titles)")
    args = ap.parse_args()

    # ---- 1. Load raster ----
    print(f"reading raster: {args.raster}")
    with rasterio.open(args.raster) as src:
        prob = src.read(1)
        crs = src.crs
        tf = src.transform
        H, W = prob.shape
        left, top = tf.c, tf.f
        right = left + W * tf.a
        bot = top + H * tf.e
        extent = (left, right, bot, top)
        bbox_utm = (left, bot, right, top)
    print(f"  shape={prob.shape}  crs={crs}")

    # ---- 2. Load PNEO RGB clipped/resampled onto raster grid ----
    print(f"loading PNEO RGB from: {args.pneo}")
    pneo_rgb, pneo_nodata = load_pneo_rgb(args.pneo, bbox_utm, prob.shape, tf, crs)
    print(f"  PNEO RGB resampled to raster shape {pneo_rgb.shape}, "
          f"nodata fraction {pneo_nodata.mean():.2f}")

    # ---- 3. Load CCMar quadrats ----
    print(f"reading CCMar: {args.ccmar}")
    quads = load_ccmar_quadrats(args.ccmar)
    print(f"  {len(quads)} unique quadrats; {sum(q['has_kelp'] for q in quads)} with kelp")
    lons = [q["lon"] for q in quads]; lats = [q["lat"] for q in quads]
    xs_q, ys_q = warp_transform("EPSG:4326", crs, lons, lats)

    # ---- FIGURE A — PNEO base + S2 heatmap overlay ----
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.imshow(pneo_rgb, extent=extent, origin="upper", zorder=1)
    # Per-pixel alpha proportional to probability so low-P areas fade to reveal
    # PNEO underneath; high-P areas render as opaque colour.
    alpha_arr = np.where(np.isfinite(prob),
                         np.clip(prob, 0, 1) * args.heatmap_alpha, 0)
    im = ax.imshow(prob, extent=extent, origin="upper",
                   cmap=args.pred_colormap, vmin=0, vmax=1,
                   alpha=alpha_arr, interpolation="bilinear",
                   zorder=2)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel(f"Easting (m, {crs.to_string()})")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Arrábida — S2 submerged head {args.pool_label} P(kelp)\n"
                 f"over PNEO RGB (2026-07-08, ~1.2 m)")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
    cbar.set_label(f"P(kelp) — S2 submerged head {args.pool_label}")
    span_x = extent[1] - extent[0]; span_y = extent[3] - extent[2]
    add_scale_bar(ax, extent[0] + span_x*0.03, extent[2] + span_y*0.04,
                  length_m=1000, height_m=span_y*0.007)
    out_a = f"{args.out_prefix}_over_pneo.png"
    Path(out_a).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_a, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_a}")

    # ---- FIGURE B — S2 heatmap + CCMar overlay ----
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    ax.set_facecolor("#f7f7f7")
    # Land (NaN in raster) shown as light grey; heatmap on top of water.
    land = np.isnan(prob).astype(np.float32)
    ax.imshow(np.where(land > 0, 0.7, np.nan), extent=extent, origin="upper",
              cmap=mcolors.ListedColormap(["#c8c8c8"]), vmin=0, vmax=1,
              interpolation="none", zorder=1)
    im = ax.imshow(prob, extent=extent, origin="upper",
                   cmap=args.pred_colormap, vmin=0, vmax=1, interpolation="bilinear",
                   zorder=2)
    # CCMar quadrats
    covers = np.array([q["cover"] for q in quads], dtype=np.float32)
    sizes = 20 + (np.clip(covers, 0, 100) / 100.0) * 280
    face = ["#3aa53a" if q["has_kelp"] else "#ffffff" for q in quads]
    edge = ["#003c00" if q["has_kelp"] else "#8b0000" for q in quads]
    ax.scatter(xs_q, ys_q, s=sizes, c=face, edgecolors=edge, linewidths=1.2,
               zorder=4, alpha=0.95)
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel(f"Easting (m, {crs.to_string()})")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"Arrábida — S2 submerged head {args.pool_label} P(kelp) with CCMar quadrats\n"
                 f"green = kelp observed; hollow red = empty; marker area ∝ mean cover_pct")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
    cbar.set_label(f"P(kelp) — S2 submerged head {args.pool_label}")
    add_scale_bar(ax, extent[0] + span_x*0.03, extent[2] + span_y*0.04,
                  length_m=1000, height_m=span_y*0.007)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#3aa53a",
               markeredgecolor="#003c00", markersize=10, label="CCMar quadrat, kelp observed"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ffffff",
               markeredgecolor="#8b0000", markersize=8, label="CCMar quadrat, empty"),
        Line2D([0], [0], color="#c8c8c8", lw=8, label="Land / cloud (masked)"),
    ], loc="lower right", framealpha=0.92, fontsize=9)
    out_b = f"{args.out_prefix}_with_ccmar.png"
    fig.savefig(out_b, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_b}")


if __name__ == "__main__":
    main()
