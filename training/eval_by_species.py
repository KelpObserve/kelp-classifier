"""eval_by_species.py — per-species + ABSENT F1 for a trained head, evaluated
on its own deployment val split.

For each species s that appears as PRESENT in the val set, restrict the
eval to (chips of species s marked PRESENT) ∪ (all ABSENT chips), then
compute F1 at threshold t=0.5 with the PRESENT class as positive. This
tells us how well the model separates species-s kelp from any ABSENT
signal, independent of other species' performance.

Also reports the overall F1 across all species and F1_ABSENT.

Usage:
  # Merged deployment head (pool=1, all species)
  KELP_CHIPS_SUBSET=merged   KELP_POOL_SIZE=1  python training/eval_by_species.py \
      --head training/results/head_single_p1/model.pt \
      --out  training/results/eval_by_species_merged_p1.csv

  # Submerged deployment head (pool=1)
  KELP_CHIPS_SUBSET=submerged KELP_POOL_SIZE=1  python training/eval_by_species.py \
      --head training/results_submerged/head_single_p1/model.pt \
      --out  training/results_submerged/eval_by_species_submerged_p1.csv

  # Submerged deployment head (pool=5)
  KELP_CHIPS_SUBSET=submerged KELP_POOL_SIZE=5  python training/eval_by_species.py \
      --head training/results_submerged/head_single_p5/model.pt \
      --out  training/results_submerged/eval_by_species_submerged_p5.csv
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                             roc_auc_score)

_here = Path(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config
from patch_pool import cache_local_slice


class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def load_val_chips_and_meta():
    """Load the deployment val split and join to chips_metadata."""
    split = json.load((config.SPLITS_DIR / "split_single.json").open())
    val_ids = set(split["val_chip_ids"])
    meta_by_id = {}
    with config.CHIPS_META.open() as f:
        for r in csv.DictReader(f):
            if r["gbifID"] in val_ids:
                meta_by_id[r["gbifID"]] = r
    return val_ids, meta_by_id


def load_embeddings_for_ids(ids):
    """Return (X (N, 2048), kept_ids) using cache + POOL_SIZE aggregation."""
    npz = np.load(config.CLAY_CACHE_NPZ, allow_pickle=True)
    assert "centre_patches" in npz.files, "cache is old format — re-run precompute"
    cache_ids = list(map(str, npz["gbif_ids"]))
    idx_of = {g: i for i, g in enumerate(cache_ids)}
    cls_all = npz["cls"]                   # (M, 1024)
    patches_all = npz["centre_patches"]    # (M, 7, 7, 1024)
    lo, hi = cache_local_slice(config.POOL_SIZE)
    pool_all = patches_all[:, lo:hi, lo:hi, :].mean(axis=(1, 2))  # (M, 1024)
    emb_all = np.concatenate([cls_all, pool_all], axis=1).astype(np.float32)

    kept, rows = [], []
    for gid in ids:
        if gid in idx_of:
            kept.append(gid); rows.append(idx_of[gid])
    X = emb_all[np.array(rows)] if rows else np.zeros((0, 2048), dtype=np.float32)
    return X, kept


def species_set(row):
    """Return set of lowercase species (splits comma-list)."""
    s = (row.get("species") or "").strip().lower()
    return {x.strip() for x in s.split(",") if x.strip()}


def compute_f1(y_true, y_pred_binary):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred_binary, labels=[0, 1],
                                                  zero_division=0)
    return {
        "prec_absent": float(p[0]), "rec_absent": float(r[0]), "f1_absent": float(f[0]),
        "prec_present": float(p[1]), "rec_present": float(r[1]), "f1_present": float(f[1]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--head", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    print(f"[cfg] subset={config.CHIPS_SUBSET}  pool={config.POOL_SIZE}  "
          f"splits={config.SPLITS_DIR.name}  cache={config.CLAY_CACHE_NPZ.name}")

    val_ids, meta_by_id = load_val_chips_and_meta()
    print(f"[val] {len(val_ids)} chip IDs; {len(meta_by_id)} matched in metadata")

    # Order chips by val_ids that are also in meta
    ids_ordered = [g for g in val_ids if g in meta_by_id]
    X, kept = load_embeddings_for_ids(ids_ordered)
    print(f"[emb] loaded embeddings for {len(kept)} / {len(ids_ordered)} chips")

    # Predict
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = MLPHead().to(device)
    head.load_state_dict(torch.load(args.head, map_location="cpu", weights_only=True))
    head.eval()
    with torch.no_grad():
        probs = torch.sigmoid(head(torch.from_numpy(X).to(device))).cpu().numpy()

    # Labels + species per chip
    labels = np.array([1 if meta_by_id[g].get("occurrenceStatus") == "PRESENT" else 0
                       for g in kept], dtype=int)
    sps = [species_set(meta_by_id[g]) for g in kept]

    pred_binary = (probs >= args.threshold).astype(int)
    t = args.threshold

    # ---- overall stats ----
    overall = compute_f1(labels, pred_binary)
    overall["roc_auc"] = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
    overall["pr_auc"] = float(average_precision_score(labels, probs))
    overall["n_pos"] = int(labels.sum()); overall["n_neg"] = int(len(labels) - labels.sum())
    print(f"[overall] ROC={overall['roc_auc']:.4f}  F1_P={overall['f1_present']:.4f}  "
          f"F1_A={overall['f1_absent']:.4f}  n_pos={overall['n_pos']} n_neg={overall['n_neg']}")

    # ---- per-species: species-s-present vs ABSENT ----
    absent_mask = labels == 0
    all_species = Counter()
    for i, sp_set in enumerate(sps):
        if labels[i] == 1:
            for s in sp_set: all_species[s] += 1
    print(f"[species] {len(all_species)} species observed in val PRESENT set")

    rows_out = []
    # header row: overall
    rows_out.append({
        "group": "OVERALL", "n_positive": int(labels.sum()), "n_negative": int(absent_mask.sum()),
        "threshold": t,
        **overall,
    })
    for sp, n in sorted(all_species.items(), key=lambda x: -x[1]):
        sp_mask = np.array([labels[i] == 1 and sp in sps[i] for i in range(len(labels))])
        pos_i = np.where(sp_mask)[0]
        eval_i = np.concatenate([pos_i, np.where(absent_mask)[0]])
        y = labels[eval_i]; p = probs[eval_i]; yhat = (p >= t).astype(int)
        stats = compute_f1(y, yhat)
        try: auc = float(roc_auc_score(y, p))
        except ValueError: auc = float("nan")
        try: pr_auc = float(average_precision_score(y, p))
        except: pr_auc = float("nan")
        stats["roc_auc"] = auc; stats["pr_auc"] = pr_auc
        stats["n_pos"] = int(len(pos_i)); stats["n_neg"] = int(absent_mask.sum())
        rows_out.append({
            "group": f"PRESENT:{sp}", "n_positive": len(pos_i),
            "n_negative": int(absent_mask.sum()), "threshold": t,
            **stats,
        })

    # Also: ABSENT-only F1 = F1_A from overall (already reported); include as its own row
    rows_out.append({
        "group": "ABSENT", "n_positive": int(absent_mask.sum()),
        "n_negative": int(labels.sum()), "threshold": t,
        "prec_present": overall["prec_absent"], "rec_present": overall["rec_absent"],
        "f1_present": overall["f1_absent"], "prec_absent": overall["prec_present"],
        "rec_absent": overall["rec_present"], "f1_absent": overall["f1_present"],
        "roc_auc": overall["roc_auc"], "pr_auc": overall["pr_auc"],
        "n_pos": int(absent_mask.sum()), "n_neg": int(labels.sum()),
    })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fields = ["group", "n_positive", "n_negative", "threshold",
              "roc_auc", "pr_auc",
              "f1_present", "prec_present", "rec_present",
              "f1_absent", "prec_absent", "rec_absent",
              "n_pos", "n_neg"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows_out:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
