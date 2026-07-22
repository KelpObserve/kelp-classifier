"""Render QC-gated example chips. Shows all successful (ok + bad_scl) chips.

Includes water-fraction stats + QC status prominently so the user can see
what the filter is doing.
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
OUT_DIR = config.REPO_ROOT / "chip_generation" / "tools_output_examples" / "qc"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B08", "B11", "SCL"]
SCL_LABELS = {
    0: "nodata", 1: "sat", 2: "dark", 3: "cldshd",
    4: "veg", 5: "sand", 6: "water", 7: "unclass",
    8: "cldmed", 9: "cldhi", 10: "cirrus", 11: "snow",
}


def load_all():
    with h5py.File(CHIP_DIR / "chips.h5", "r") as f:
        n = int(f.attrs.get("n_chips", f["chips"].shape[0]))
        chips = f["chips"][:n]
    rows = [r for r in csv.DictReader((CHIP_DIR / "chips_metadata.csv").open())
            if r["status"] in ("ok", "bad_scl")]
    rows.sort(key=lambda r: int(r["chipIndex"]))
    return chips, rows


def stretch(x, lo_p=1, hi_p=99):
    lo = np.percentile(x, lo_p)
    hi = np.percentile(x, hi_p)
    if hi <= lo: hi = lo + 1
    return np.clip((x - lo) / (hi - lo), 0, 1)


def render_panel(chip, m, out_path):
    refl = chip.astype(np.float32) / 10000.0
    b01, b02, b03, b04, b05, b08, b11, scl = refl
    scl_int = chip[7].astype(int)

    fig = plt.figure(figsize=(15, 10))
    grid = plt.GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.20)

    # RGB
    ax = fig.add_subplot(grid[0, 0])
    rgb = np.stack([stretch(b04), stretch(b03), stretch(b02)], axis=-1)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title("RGB (B04/B03/B02) 1-99% stretched")
    ax.scatter([64], [64], marker="+", c="yellow", s=200, linewidth=2)
    ax.set_xticks([]); ax.set_yticks([])

    # NIR false colour
    ax = fig.add_subplot(grid[0, 1])
    fc = np.stack([stretch(b08), stretch(b04), stretch(b03)], axis=-1)
    ax.imshow(fc, interpolation="nearest")
    ax.set_title("NIR false color (B08/B04/B03)")
    ax.scatter([64], [64], marker="+", c="yellow", s=200, linewidth=2)
    ax.set_xticks([]); ax.set_yticks([])

    # NDWI
    ax = fig.add_subplot(grid[0, 2])
    ndwi = (b03 - b08) / np.clip(b03 + b08, 1e-6, None)
    im = ax.imshow(ndwi, cmap="RdBu", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"NDWI mean={ndwi.mean():.2f}")
    ax.scatter([64], [64], marker="+", c="yellow", s=200, linewidth=2)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Blue band
    ax = fig.add_subplot(grid[1, 0])
    im = ax.imshow(b02, cmap="Blues_r", interpolation="nearest",
                    vmin=np.percentile(b02, 1), vmax=np.percentile(b02, 99))
    ax.set_title(f"B02 blue reflectance mean={b02.mean():.3f}")
    ax.scatter([64], [64], marker="+", c="red", s=200, linewidth=2)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)

    # SCL with legend
    ax = fig.add_subplot(grid[1, 1])
    ax.imshow(scl_int, cmap="tab20", vmin=0, vmax=11, interpolation="nearest")
    counts = np.bincount(scl_int.ravel(), minlength=12)
    top = counts.argsort()[::-1][:3]
    label = "  ".join(f"{SCL_LABELS[c]}={100*counts[c]/scl_int.size:.0f}%"
                       for c in top if counts[c] > 0)
    ax.set_title(f"SCL top: {label}")
    ax.scatter([64], [64], marker="+", c="red", s=200, linewidth=2)
    ax.set_xticks([]); ax.set_yticks([])

    # Per-band spectrum at chip centre
    ax = fig.add_subplot(grid[1, 2])
    centre = chip[:, 63:66, 63:66].astype(np.float32).mean(axis=(1, 2)) / 10000.0
    ax.bar(range(7), centre[:7], color="tab:blue")
    ax.set_xticks(range(7))
    ax.set_xticklabels(BAND_NAMES[:7], rotation=45)
    ax.set_ylabel("Reflectance at chip centre 3x3")
    ax.set_title("Per-band reflectance")
    ax.grid(True, alpha=0.3)

    qc_colour = "tab:green" if m["status"] == "ok" else "tab:red"
    txt = (f"[{m['status'].upper()}]  {m['occurrenceStatus']}  "
           f"{m['genus']}  {m['country']}\n"
           f"water fractions:  point 5x5={m['waterFracPoint']}   "
           f"chip centre 3x3={m['waterFracCentre']}   whole chip={m['waterFracChip']}\n"
           f"lat={float(m['decimalLatitude']):.3f}  "
           f"lon={float(m['decimalLongitude']):.3f}   "
           f"coord ±{m.get('coordinateUncertaintyInMeters', '?')} m\n"
           f"event {m['eventDate']}   scene {m['sceneDate']} "
           f"({m['daysOff']} d off, cloud {m['cloudCover']}%)")
    fig.suptitle(txt, fontsize=11, y=0.99, color=qc_colour, weight="bold")
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    chips, meta = load_all()
    print(f"loaded {len(chips)} chips ({sum(1 for m in meta if m['status']=='ok')} ok, "
          f"{sum(1 for m in meta if m['status']=='bad_scl')} bad_scl)")
    for i, m in enumerate(meta):
        out_path = OUT_DIR / f"{m['status']}_{m['occurrenceStatus']}_{m['genus']}_{m['country'].split()[0]}_{m['eventDate'][:10]}.png"
        render_panel(chips[i], m, out_path)
        print(f"  wrote {out_path.name}")


if __name__ == "__main__":
    main()
