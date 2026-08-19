"""plot_model_comparison.py — two comparison plots across
{merged pool=1, submerged pool=1, submerged pool=5}:

  1. Per-species F1 (grouped bar chart) from the deployment val evals.
  2. Sesimbra pixel-aggregated evaluation (AUC + F1_P + F1_A).

Usage:
  python plot_model_comparison.py \
      --species-csv merged_p1=results/eval_by_species_merged_p1.csv \
      --species-csv submerged_p1=results_submerged/eval_by_species_submerged_p1.csv \
      --species-csv submerged_p5=results_submerged/eval_by_species_submerged_p5.csv \
      --sesimbra-csv results_submerged/sesimbra_model_comparison.csv \
      --out-species-png <path> --out-sesimbra-png <path>
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ["merged_p1", "submerged_p1", "submerged_p5"]
MODEL_LABELS = {
    "merged_p1":    "merged, pool=1\n(all species)",
    "submerged_p1": "submerged, pool=1\n(4 deep species)",
    "submerged_p5": "submerged, pool=5\n(4 deep species)",
}
MODEL_COLORS = {
    "merged_p1":    "#4c72b0",
    "submerged_p1": "#dd8452",
    "submerged_p5": "#55a868",
}


def read_species_csv(path):
    """Return {group_name: {roc_auc, f1_present, f1_absent, n_pos, n_neg}}."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["group"]] = {
                "roc_auc": float(r["roc_auc"]) if r["roc_auc"] else float("nan"),
                "f1_present": float(r["f1_present"]) if r["f1_present"] else float("nan"),
                "f1_absent": float(r["f1_absent"]) if r["f1_absent"] else float("nan"),
                "n_pos": int(r["n_pos"]) if r["n_pos"] else 0,
                "n_neg": int(r["n_neg"]) if r["n_neg"] else 0,
            }
    return out


def plot_species(species_csvs, out_png):
    """Grouped bar chart: for each species (+ OVERALL + ABSENT), bars per model."""
    per_model = {k: read_species_csv(v) for k, v in species_csvs.items()}
    # Ordered species list: union across models, sorted by max n_pos across models
    all_groups = set()
    for m in per_model.values(): all_groups.update(m.keys())
    # Sort: OVERALL first, then PRESENT:* sorted by size, then ABSENT last
    species_groups = sorted(
        [g for g in all_groups if g.startswith("PRESENT:")],
        key=lambda g: -max(per_model[m].get(g, {}).get("n_pos", 0) for m in per_model),
    )
    ordered = ["OVERALL"] + species_groups + ["ABSENT"]

    n_groups = len(ordered)
    n_models = len(species_csvs)
    x = np.arange(n_groups)
    bar_w = 0.8 / n_models

    fig, axes = plt.subplots(2, 1, figsize=(max(11, n_groups * 1.4), 9),
                              sharex=True, constrained_layout=True)

    # Top panel: F1_present per species per model
    ax = axes[0]
    for i, (mname, _) in enumerate([(m, species_csvs[m]) for m in MODEL_ORDER if m in species_csvs]):
        vals = [per_model[mname].get(g, {}).get("f1_present", np.nan) for g in ordered]
        bars = ax.bar(x + (i - (n_models-1)/2) * bar_w, vals, bar_w,
                      label=MODEL_LABELS[mname], color=MODEL_COLORS[mname], alpha=0.9)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("F1_PRESENT (threshold=0.5)")
    ax.set_title("Per-species F1 (positive = species-s PRESENT chips vs all ABSENT chips)")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3); ax.legend(loc="lower left", fontsize=9)

    # Bottom panel: n_pos per species per model (sample sizes)
    ax = axes[1]
    for i, mname in enumerate([m for m in MODEL_ORDER if m in species_csvs]):
        vals = [per_model[mname].get(g, {}).get("n_pos", 0) for g in ordered]
        ax.bar(x + (i - (n_models-1)/2) * bar_w, vals, bar_w,
               color=MODEL_COLORS[mname], alpha=0.9, label=MODEL_LABELS[mname])
    ax.set_ylabel("n positive chips")
    ax.set_title("Sample size per species per model (val set)")
    ax.grid(axis="y", alpha=0.3)
    labels = [g.replace("PRESENT:", "") for g in ordered]
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)

    fig.suptitle("Model comparison — per-species F1 on deployment val set",
                 fontsize=13, y=1.005)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


def plot_sesimbra(csv_path, out_png):
    """AUC + F1_P + F1_A bar chart per model, from Sesimbra pixel-aggregated eval."""
    rows = list(csv.DictReader(open(csv_path)))
    by_name = {r["name"]: r for r in rows}
    names = [m for m in MODEL_ORDER if m in by_name]
    metrics = [
        ("auc",             "ROC-AUC (higher = better)"),
        ("f1_present_best", "F1_PRESENT (best-threshold)"),
        ("f1_absent_best",  "F1_ABSENT (best-threshold)"),
    ]
    x = np.arange(len(metrics))
    bar_w = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for i, mname in enumerate(names):
        vals = [float(by_name[mname][k]) for k, _ in metrics]
        bars = ax.bar(x + (i - (len(names)-1)/2) * bar_w, vals, bar_w,
                      label=MODEL_LABELS[mname], color=MODEL_COLORS[mname], alpha=0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in metrics], fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_ylabel("score")
    ax.grid(axis="y", alpha=0.3); ax.legend(loc="lower left", fontsize=9)
    # Note the tiny negative-sample size honestly in the title
    npos = int(rows[0]["n_positive"]); nneg = int(rows[0]["n_negative"])
    ax.set_title(f"Sesimbra pixel-aggregated eval (OR-based)\n"
                 f"CCMar quadrats → {npos} PRESENT pixels + {nneg} ABSENT pixels at 10 m",
                 fontsize=12)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--species-csv", action="append", required=True,
                    help="name=path, repeat per model")
    ap.add_argument("--sesimbra-csv", required=True)
    ap.add_argument("--out-species-png", required=True)
    ap.add_argument("--out-sesimbra-png", required=True)
    args = ap.parse_args()
    csvs = dict(spec.split("=", 1) for spec in args.species_csv)
    plot_species(csvs, args.out_species_png)
    plot_sesimbra(args.sesimbra_csv, args.out_sesimbra_png)


if __name__ == "__main__":
    main()
