"""training/plot_sesimbra_map.py — kelp probability heatmap + CCMar overlays.

Reads the P(kelp) GeoTIFF written by rasterise.py, overlays CCMar quadrat
observations (aggregated per unique lat/lon), writes a clean PNG. No basemap
tiles — the raster's NaN-over-land mask acts as the coastline (filled grey);
water pixels get the viridis heatmap.

Two modes:
  Wide      : shows the full raster extent.
  Zoom-in   : --zoom-around lat lon radius_m crops the display and rescales
              the scale bar; useful for the tight CCMar quadrat cluster.

Usage:
    python training/plot_sesimbra_map.py \\
        --raster training/results/sesimbra_arrabida_prob.tif \\
        --ccmar  /path/to/kelp_observations_with_coordinates.csv \\
        --out    training/results/sesimbra_arrabida_map.png
    python training/plot_sesimbra_map.py --raster ... --ccmar ... \\
        --zoom-around 38.4090 -9.2020 250 \\
        --out training/results/sesimbra_arrabida_map_zoom.png
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform


def load_ccmar(csv_path):
    """Return list of quadrat dicts with keys {lon, lat, cover, n_rows, species,
    has_kelp}. `cover` = sum of cover_pct across all kelp-species rows for the
    quadrat (kelp species = anything other than the literal "EMPTY")."""
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
            try:
                cover_vals.append(float(r["cover_pct"]))
            except (ValueError, KeyError, TypeError):
                pass
        cover = sum(cover_vals) if cover_vals else 0.0
        species = sorted({(r.get("species") or "").strip() for r in kelp_rows if r.get("species")})
        quads.append({
            "lat": lat, "lon": lon,
            "cover": cover, "n_rows": len(rows), "n_kelp_rows": len(kelp_rows),
            "species": species,
            "has_kelp": bool(kelp_rows),
        })
    return quads


def add_scale_bar(ax, x0, y0, length_m=1000, height_m=60, label=None):
    """Simple two-tone scale bar in map units (metres). `label` overrides
    the auto-generated km label; useful for sub-km scales (e.g. "100 m")."""
    ax.add_patch(plt.Rectangle((x0, y0), length_m/2, height_m,
                                facecolor="black", edgecolor="black", lw=0.5, zorder=10))
    ax.add_patch(plt.Rectangle((x0 + length_m/2, y0), length_m/2, height_m,
                                facecolor="white", edgecolor="black", lw=0.5, zorder=10))
    def fmt(m):
        return f"{m:.0f} m" if m < 1000 else f"{m/1000:g} km"
    ax.text(x0, y0 + height_m*3, "0", ha="center", va="bottom", fontsize=8, zorder=10)
    ax.text(x0 + length_m/2, y0 + height_m*3, fmt(length_m/2),
            ha="center", va="bottom", fontsize=8, zorder=10)
    ax.text(x0 + length_m, y0 + height_m*3, label or fmt(length_m),
            ha="center", va="bottom", fontsize=8, zorder=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raster", required=True)
    ap.add_argument("--ccmar", required=True)
    ap.add_argument("--zoom-around", nargs=3, type=float, default=None,
                    metavar=("LAT", "LON", "RADIUS_M"),
                    help="If set, crop the display to a square of side 2*RADIUS_M "
                         "centred on (LAT, LON). Otherwise show the whole raster.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Arrábida — P(kelp) at 10 m + CCMar June 2026 quadrats")
    args = ap.parse_args()

    print(f"reading raster: {args.raster}")
    with rasterio.open(args.raster) as src:
        prob = src.read(1)
        crs = src.crs
        tr = src.transform
        # extent for imshow: (left, right, bottom, top) in map units
        h, w = prob.shape
        left, top = tr.c, tr.f
        right = left + w * tr.a
        bot   = top  + h * tr.e   # tr.e is negative (north-up)
        extent = (left, right, bot, top)
    print(f"  shape={prob.shape}  crs={crs}  extent={extent}")

    print(f"reading CCMar: {args.ccmar}")
    quads = load_ccmar(args.ccmar)
    print(f"  {len(quads)} unique quadrats; {sum(q['has_kelp'] for q in quads)} with kelp")

    # Reproject CCMar (WGS84) points into the raster CRS.
    lons = [q["lon"] for q in quads]
    lats = [q["lat"] for q in quads]
    xs, ys = warp_transform("EPSG:4326", crs, lons, lats)

    # Resolve zoom window in the raster CRS if requested.
    zoom_bounds = None
    if args.zoom_around:
        z_lat, z_lon, z_r = args.zoom_around
        z_x, z_y = warp_transform("EPSG:4326", crs, [z_lon], [z_lat])
        cx, cy = float(z_x[0]), float(z_y[0])
        zoom_bounds = (cx - z_r, cx + z_r, cy - z_r, cy + z_r)
        print(f"  zoom window: centre=({cx:.0f}, {cy:.0f})  radius={z_r:.0f} m")

    # -------- figure --------
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    ax.set_facecolor("#f7f7f7")   # water background where prob happens to be near 0

    # Land + cloud pixels are NaN in prob; render them as a light-grey layer.
    land = np.isnan(prob).astype(np.float32)
    ax.imshow(np.where(land > 0, 0.7, np.nan), extent=extent, origin="upper",
              cmap=mcolors.ListedColormap(["#c8c8c8"]), vmin=0, vmax=1,
              interpolation="none", zorder=1)

    # Kelp probability heatmap on top of water.
    im = ax.imshow(prob, extent=extent, origin="upper",
                   cmap="viridis", vmin=0, vmax=1, interpolation="bilinear",
                   zorder=2)

    # CCMar quadrats: size ∝ mean cover_pct across kelp-species rows (max 1).
    # Zoom mode: smaller markers + partial alpha so individual dots read; the
    # 30 quadrats span ~120 m and would otherwise fuse into one blob.
    covers = np.array([q["cover"] for q in quads], dtype=np.float32)
    if zoom_bounds is not None:
        sizes = 40 + (np.clip(covers, 0, 100) / 100.0) * 200
        alpha = 0.75
        edge_lw = 0.9
    else:
        sizes = 20 + (np.clip(covers, 0, 100) / 100.0) * 280
        alpha = 0.95
        edge_lw = 1.2
    face   = ["#3aa53a" if q["has_kelp"] else "#ffffff" for q in quads]
    edge   = ["#003c00" if q["has_kelp"] else "#8b0000" for q in quads]

    ax.scatter(xs, ys, s=sizes, c=face, edgecolors=edge, linewidths=edge_lw,
               zorder=4, alpha=alpha, label="CCMar quadrats (size ∝ cover_pct)")

    # Axis limits: full raster or zoom window.
    if zoom_bounds is not None:
        x_min, x_max, y_min, y_max = zoom_bounds
    else:
        x_min, x_max, y_min, y_max = left, right, bot, top
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    span_x = x_max - x_min
    span_y = y_max - y_min

    # Scale bar sized to ~20 % of the horizontal span, rounded to a friendly number.
    def _round_scale(m):
        for cand in (10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10):
            if cand <= m: return cand
        return 10
    bar_len = _round_scale(span_x * 0.2)
    x0 = x_min + span_x * 0.03
    y0 = y_min + span_y * 0.05
    add_scale_bar(ax, x0, y0, length_m=bar_len, height_m=span_y * 0.007)

    # north arrow at top-right
    ax.annotate("N", xy=(x_max - span_x*0.04, y_max - span_y*0.05),
                xytext=(x_max - span_x*0.04, y_max - span_y*0.13),
                arrowprops=dict(facecolor="black", width=3, headwidth=10, headlength=10),
                ha="center", va="center", fontsize=12, fontweight="bold", zorder=10)

    ax.set_aspect("equal")
    ax.set_xlabel(f"Easting (m, {crs.to_string()})")
    ax.set_ylabel("Northing (m)")
    ax.set_title(args.title)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
    cbar.set_label("P(kelp) — pool=1 deployment head")

    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#3aa53a",
               markeredgecolor="#003c00", markersize=10, label="CCMar quadrat, kelp observed"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ffffff",
               markeredgecolor="#8b0000", markersize=8, label="CCMar quadrat, empty"),
        Line2D([0], [0], color="#c8c8c8", lw=8, label="Land / cloud (masked)"),
    ]
    ax.legend(handles=legend_els, loc="lower right", framealpha=0.92, fontsize=9)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
