"""Plot all test metrics for a pool-size sweep (works for any results dir).

Usage:
  python plot_pool_sweep_generic.py <results_dir> <out_png>
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

results_dir = Path(sys.argv[1])
out_png     = sys.argv[2]

POOLS = [1, 2, 3, 5]
SEEDS = [42, 123, 456]
METRICS = [
    ("roc_auc",       "ROC-AUC"),
    ("pr_auc",        "PR-AUC"),
    ("balanced_acc",  "balanced_acc @ 0.5"),
    ("macro_f1",      "macro_F1 @ 0.5"),
    ("rec_absent",    "ABSENT recall"),
    ("prec_absent",   "ABSENT precision"),
    ("rec_present",   "PRESENT recall"),
    ("prec_present",  "PRESENT precision"),
]
SEED_COLORS = ["#4c72b0", "#dd8452", "#55a868"]
MEAN_COLOR  = "#000000"

data = {}
for pool in POOLS:
    for seed in SEEDS:
        p = results_dir / f"head_seed{seed}_p{pool}" / "results.json"
        if not p.exists():
            print(f"MISSING: {p}", file=sys.stderr); continue
        d = json.load(open(p))["test_default_t0.5"]
        for m, _ in METRICS:
            data.setdefault(m, {}).setdefault(pool, []).append(d[m])

fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
for ax, (m, label) in zip(axes.flat, METRICS):
    for si, s in enumerate(SEEDS):
        y = [data[m][p][si] for p in POOLS if p in data[m]]
        px = [p for p in POOLS if p in data[m]]
        ax.plot(px, y, marker="o", color=SEED_COLORS[si], alpha=0.55,
                linewidth=1.2, markersize=6, label=f"seed {s}")
    means = [np.mean(data[m][p]) for p in POOLS if p in data[m]]
    stds  = [np.std(data[m][p], ddof=1) for p in POOLS if p in data[m]]
    px = [p for p in POOLS if p in data[m]]
    ax.errorbar(px, means, yerr=stds, marker="s", color=MEAN_COLOR,
                linewidth=2, markersize=7, capsize=4, label="mean ± σ", zorder=5)
    ax.set_title(label, fontsize=11)
    ax.set_xticks(POOLS); ax.grid(alpha=0.3)
    yall = [v for p in POOLS if p in data[m] for v in data[m][p]]
    ymin, ymax = min(yall), max(yall); rng = ymax - ymin
    ax.set_ylim(ymin - 0.15 * rng - 0.005, ymax + 0.15 * rng + 0.005)

for ax in axes[-1]:
    ax.set_xlabel("POOL_SIZE  (patches per side)")

axes[0, 0].legend(fontsize=8, loc="best")
subset = "submerged" if "submerged" in str(results_dir) else "merged"
fig.suptitle(f"Kelp classifier: test-set metrics vs centre-pool size — {subset.upper()} pool\n"
             f"3 seeds (42/123/456) × 4 pools (1, 2, 3, 5).  Effective footprint ≈ POOL_SIZE × 46 m native.",
             fontsize=12, y=1.00)
fig.tight_layout()
Path(out_png).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, dpi=140, bbox_inches="tight")
print(f"wrote {out_png}")
