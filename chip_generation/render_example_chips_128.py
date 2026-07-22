"""Render 128×128 chips as inspection PNGs.

Reads from ~/kelp-detection/data/s2_chips_merged/, picks a few
PRESENT + ABSENT with diverse country/genus, renders as multi-panel
figures at real size (128×128 → 640×640 on-screen).
"""
import csv
from pathlib import Path

import h5py
import numpy as np
import matplotlib
# Locate repo root by walking up until config.py is found
import sys as _sys
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHIP_DIR = config.CHIPS_DIR
OUT_DIR = config.REPO_ROOT / "chip_generation" / "tools_output_examples" / "chips_128"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B08", "B11", "SCL"]
SCL_LABELS = {
    0: "no data", 1: "saturated", 2: "dark", 3: "cloud shadow",
    4: "vegetation", 5: "bare/sand", 6: "water", 7: "unclassified",
    8: "cloud med", 9: "cloud high", 10: "thin cirrus", 11: "snow/ice",
}


def load_partial():
    h5p = CHIP_DIR / "chips.h5"
    mp = CHIP_DIR / "chips_metadata.csv"
    with h5py.File(h5p, "r") as f:
        n = int(f.attrs.get("n_chips", f["chips"].shape[0]))
        chips = f["chips"][:n]
    rows = [r for r in csv.DictReader(mp.open()) if r["status"] == "ok"]
    rows.sort(key=lambda r: int(r["chipIndex"]))
    return chips, rows


def stretch(x, lo_p=1, hi_p=99):
    lo = np.percentile(x, lo_p)
    hi = np.percentile(x, hi_p)
    if hi <= lo:
        hi = lo + 1
    return np.clip((x - lo) / (hi - lo), 0, 1)


def render_panel(chip, meta_row, out_path):
    """Six-panel figure: RGB, NIR-R-G false, NDWI, blue only, SCL, per-band curves."""
    refl = chip.astype(np.float32) / 10000.0
    b01, b02, b03, b04, b05, b08, b11, scl = refl
    scl_int = chip[7].astype(int)

    fig = plt.figure(figsize=(15, 10))
    grid = plt.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.2)

    # RGB (B04, B03, B02) with pixel stretch
    ax = fig.add_subplot(grid[0, 0])
    rgb = np.stack([stretch(b04), stretch(b03), stretch(b02)], axis=-1)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title("RGB (B04/B03/B02)  stretched 1-99%")
    ax.scatter([64], [64], marker="+", c="yellow", s=200, linewidth=2)
    ax.set_xticks([]); ax.set_yticks([])

    # NIR false colour (B08, B04, B03)
    ax = fig.add_subplot(grid[0, 1])
    fc = np.stack([stretch(b08), stretch(b04), stretch(b03)], axis=-1)
    ax.imshow(fc, interpolation="nearest")
    ax.set_title("NIR false color (B08/B04/B03)")
    ax.scatter([64], [64], marker="+", c="yellow", s=200, linewidth=2)
    ax.set_xticks([]); ax.set_yticks([])

    # NDWI (green - NIR) / (green + NIR)
    ax = fig.add_subplot(grid[0, 2])
    ndwi = (b03 - b08) / np.clip(b03 + b08, 1e-6, None)
    im = ax.imshow(ndwi, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"NDWI  mean={ndwi.mean():.2f}")
    ax.scatter([64], [64], marker="+", c="yellow", s=200, linewidth=2)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Blue band (B02) — water bottom visibility
    ax = fig.add_subplot(grid[1, 0])
    im = ax.imshow(b02, cmap="Blues_r", interpolation="nearest",
                    vmin=np.percentile(b02, 1), vmax=np.percentile(b02, 99))
    ax.set_title(f"B02 blue reflectance  mean={b02.mean():.3f}")
    ax.scatter([64], [64], marker="+", c="red", s=200, linewidth=2)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)

    # SCL
    ax = fig.add_subplot(grid[1, 1])
    ax.imshow(scl_int, cmap="tab20", vmin=0, vmax=11, interpolation="nearest")
    counts = np.bincount(scl_int.ravel(), minlength=12)
    dominant = counts.argmax()
    ax.set_title(f"SCL  dominant={SCL_LABELS.get(dominant)}  "
                 f"water={100*counts[6]/scl_int.size:.0f}%")
    ax.scatter([64], [64], marker="+", c="red", s=200, linewidth=2)
    ax.set_xticks([]); ax.set_yticks([])

    # Per-band centre 3×3 mean spectrum
    ax = fig.add_subplot(grid[1, 2])
    centre = chip[:, 63:66, 63:66].astype(np.float32).mean(axis=(1, 2)) / 10000.0
    wl = [0.443, 0.493, 0.560, 0.665, 0.704, 0.842, 1.610, 0]
    ax.bar(range(7), centre[:7], color="tab:blue")
    ax.set_xticks(range(7))
    ax.set_xticklabels(BAND_NAMES[:7], rotation=45)
    ax.set_ylabel("Reflectance at chip centre")
    ax.set_title("Per-band reflectance (centre 3x3)")
    ax.grid(True, alpha=0.3)

    txt = (f"{meta_row['occurrenceStatus']}  {meta_row['genus']}  "
           f"{meta_row['country']}\n"
           f"lat={float(meta_row['decimalLatitude']):.3f}  "
           f"lon={float(meta_row['decimalLongitude']):.3f}   "
           f"coord ±{meta_row.get('coordinateUncertaintyInMeters', '?')} m\n"
           f"event {meta_row['eventDate']}   scene {meta_row['sceneDate']} "
           f"({meta_row['daysOff']} d off)   cloud {meta_row.get('cloudCover', '?')}%\n"
           f"scene: {meta_row['sceneID']}")
    fig.suptitle(txt, fontsize=11, y=0.98)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    chips, meta = load_partial()
    print(f"loaded {len(chips)} chips from partial fetch")
    idx_p = [i for i, r in enumerate(meta) if r["occurrenceStatus"] == "PRESENT"]
    idx_a = [i for i, r in enumerate(meta) if r["occurrenceStatus"] == "ABSENT"]
    print(f"PRESENT={len(idx_p)}  ABSENT={len(idx_a)}")

    rng = np.random.default_rng(42)

    def pick_diverse(idx_list, n, key):
        by = {}
        for i in idx_list:
            by.setdefault(meta[i].get(key) or "UNKNOWN", []).append(i)
        keys = sorted(by, key=lambda k: -len(by[k]))
        picked = []
        for k in keys:
            picked.append(int(rng.choice(by[k])))
            if len(picked) >= n:
                break
        return picked

    p_pick = pick_diverse(idx_p, 4, "genus")
    a_pick = pick_diverse(idx_a, 4, "country") if idx_a else []
    print(f"picked  PRESENT={p_pick}  ABSENT={a_pick}")

    for idx in p_pick:
        m = meta[idx]
        out_path = OUT_DIR / f"PRESENT_{m['genus']}_{m['country'].split()[0]}_{m['eventDate'][:10]}.png"
        render_panel(chips[idx], m, out_path)
        print(f"  wrote {out_path.name}")
    for idx in a_pick:
        m = meta[idx]
        out_path = OUT_DIR / f"ABSENT_{m['genus']}_{m['country'].split()[0]}_{m['eventDate'][:10]}.png"
        render_panel(chips[idx], m, out_path)
        print(f"  wrote {out_path.name}")


if __name__ == "__main__":
    main()
