"""plot_pixel_vs_pool.py — compare the pixel head (2054-D, CLS + centre patch
+ raw S2 at (64,64)) against pool={1,2,3,5} chip heads across the 3 seeds
used for the submerged pool sweep.

Reads (all optional, warns on missing):
  results_submerged/head_seed{S}_p{P}/results.json   for P in {1,2,3,5}, S in seeds
  results_submerged/pixel/head_seed{S}/results.json  for S in seeds

Writes:
  results_submerged/pixel/pixel_vs_pool.png
  results_submerged/pixel/pixel_vs_pool.csv

Usage:
  KELP_CHIPS_SUBSET=submerged python training/plot_pixel_vs_pool.py \
      --seeds 42 123 456
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_here = Path(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

POOL_SIZES = [1, 2, 3, 5]
METRICS = [
    ("roc_auc",  "ROC-AUC"),
    ("pr_auc",   "PR-AUC"),
    ("macro_f1", "macro F1 @ t=0.5"),
]


def load_one(path):
    if not path.exists():
        return None
    r = json.load(path.open())
    d = r.get("test_default_t0.5", {})
    return {
        "roc_auc": d.get("roc_auc"),
        "pr_auc":  d.get("pr_auc"),
        "macro_f1": d.get("macro_f1"),
        "f1_absent": d.get("f1_absent"),
        "f1_present": d.get("f1_present"),
    }


def collect(seeds):
    """Return {config_name: {metric: [values across seeds]}}."""
    root = config.RESULTS_DIR
    out = {}
    for p in POOL_SIZES:
        name = f"pool={p}"
        out[name] = {m: [] for m, _ in METRICS}
        for s in seeds:
            path = root / f"head_seed{s}_p{p}" / "results.json"
            r = load_one(path)
            if r is None:
                print(f"  WARN missing: {path}")
                continue
            for m, _ in METRICS:
                if r[m] is not None:
                    out[name][m].append(r[m])
    # pixel head
    name = "pixel head\n(CLS+patch+raw S2)"
    out[name] = {m: [] for m, _ in METRICS}
    for s in seeds:
        path = root / "pixel" / f"head_seed{s}" / "results.json"
        r = load_one(path)
        if r is None:
            print(f"  WARN missing: {path}")
            continue
        for m, _ in METRICS:
            if r[m] is not None:
                out[name][m].append(r[m])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    args = ap.parse_args()

    print(f"[config] subset={config.CHIPS_SUBSET}  seeds={args.seeds}  root={config.RESULTS_DIR.name}")
    data = collect(args.seeds)

    # Save CSV
    out_dir = config.RESULTS_DIR / "pixel"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pixel_vs_pool.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "metric", "n_seeds", "mean", "std", "values"])
        for name, per_m in data.items():
            for m, _ in METRICS:
                vals = per_m[m]
                w.writerow([name, m, len(vals),
                            f"{np.mean(vals):.4f}" if vals else "",
                            f"{np.std(vals):.4f}" if vals else "",
                            ";".join(f"{v:.4f}" for v in vals)])
    print(f"wrote {csv_path}")

    # Plot: grouped bar chart, one panel per metric, error bars = ±1 std
    names = list(data.keys())
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, len(METRICS),
                              figsize=(4.2 * len(METRICS), 5),
                              constrained_layout=True)
    if len(METRICS) == 1: axes = [axes]

    colors = ["#4c72b0", "#55a868", "#dd8452", "#c44e52", "#8172b3"]
    for ax, (m, title) in zip(axes, METRICS):
        means = [np.mean(data[n][m]) if data[n][m] else np.nan for n in names]
        stds  = [np.std(data[n][m])  if data[n][m] else 0      for n in names]
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=colors[:len(names)], alpha=0.9, edgecolor="black", linewidth=0.6)
        for b, mv, sv, nvals in zip(bars, means, stds, [len(data[n][m]) for n in names]):
            if not np.isnan(mv):
                ax.text(b.get_x() + b.get_width()/2, mv + sv + 0.005,
                        f"{mv:.3f}\n(n={nvals})",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
        ax.set_title(title, fontsize=11)

    fig.suptitle(f"Pixel head vs pool sweep — submerged pool, "
                  f"{len(args.seeds)} seeds ({', '.join(map(str, args.seeds))})",
                  fontsize=12, y=1.02)
    png_path = out_dir / "pixel_vs_pool.png"
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
