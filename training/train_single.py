"""train_single.py — train one MLP head on a train/val split (no test).

Reads:
  splits/split_single.json    (built by make_single_split.py)
  clay_cache/identity.npz     (CLS + centre 7×7 patches, built by
                               precompute_clay_embeddings.py; pool aggregation
                               done here at load time per KELP_POOL_SIZE)
Writes:
  results/head_single_p{POOL_SIZE}/results.json   (history + val metrics)
  results/head_single_p{POOL_SIZE}/model.pt       (state_dict of best-val_macro_f1)

CLI: python train_single.py [--seed 42] [--epochs 80]
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

# Locate repo root by walking up until config.py is found
import sys as _sys
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             precision_recall_fscore_support, average_precision_score,
                             confusion_matrix)

from patch_pool import cache_local_slice

EMBED_DIM = 2048
BATCH_TRAIN = 64
LR = 1e-3
FOCAL_GAMMA = 2.0
PATIENCE = 15
MAX_EPOCHS_DEFAULT = 80

class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBED_DIM, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def sigmoid_focal_loss(logits, targets, alpha=0.5, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    return (ce * ((1 - p_t) ** gamma) * (alpha * targets + (1 - alpha) * (1 - targets))).mean()


def load_data():
    """Load metadata + split + embeddings. Filter chips to those present in cache."""
    rows = list(csv.DictReader(config.CHIPS_META.open()))
    by_gid = {r["gbifID"]: r for r in rows}

    split = json.load((config.SPLITS_DIR / "split_single.json").open())
    train_ids = set(split["train_chip_ids"])
    val_ids   = set(split["val_chip_ids"])
    print(f"[split] train={len(train_ids)} val={len(val_ids)} chips (before embed-cache filter)")

    npz = np.load(config.CLAY_CACHE_NPZ, allow_pickle=True)
    assert "centre_patches" in npz.files, (
        f"{config.CLAY_CACHE_NPZ} is old-format (has {sorted(npz.files)}); "
        f"expected 'cls' + 'centre_patches'. Re-run precompute_clay_embeddings.py."
    )
    ids  = list(map(str, npz["gbif_ids"]))
    cls_all     = npz["cls"]              # (N, 1024)
    patches_all = npz["centre_patches"]   # (N, CACHE_K, CACHE_K, 1024)
    lo, hi = cache_local_slice(config.POOL_SIZE)
    pool_all = patches_all[:, lo:hi, lo:hi, :].mean(axis=(1, 2))
    embs = np.concatenate([cls_all, pool_all], axis=1).astype(np.float32)
    print(f"[cache] {config.CLAY_CACHE_NPZ.name}  N={len(ids)}  pool={config.POOL_SIZE} "
          f"(cache slice [{lo}:{hi}])")
    idx_of = {g: i for i, g in enumerate(ids)}

    def build(id_set):
        keep, y, X = [], [], []
        for cid in id_set:
            if cid not in idx_of or cid not in by_gid: continue
            keep.append(cid)
            X.append(embs[idx_of[cid]])
            y.append(1 if by_gid[cid].get("occurrenceStatus") == "PRESENT" else 0)
        return np.stack(X).astype(np.float32), np.array(y, dtype=np.float32), keep

    X_tr, y_tr, tr_ids = build(train_ids)
    X_va, y_va, va_ids = build(val_ids)
    n_missing_tr = len(train_ids) - len(tr_ids)
    n_missing_va = len(val_ids) - len(va_ids)
    if n_missing_tr or n_missing_va:
        print(f"[cache] dropped {n_missing_tr} train + {n_missing_va} val chips (not in embedding cache)")
    print(f"[after cache filter] train={len(tr_ids)} ({int(y_tr.sum())} P / {int(len(y_tr) - y_tr.sum())} A)"
          f"   val={len(va_ids)} ({int(y_va.sum())} P / {int(len(y_va) - y_va.sum())} A)")
    return X_tr, y_tr, X_va, y_va


def eval_at_threshold(y, p, t):
    yhat = (p >= t).astype(int)
    bal = balanced_accuracy_score(y, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, labels=[0,1], zero_division=0)
    try: roc = roc_auc_score(y, p)
    except: roc = float("nan")
    try: pr_auc = average_precision_score(y, p)
    except: pr_auc = float("nan")
    return {
        "t": float(t),
        "balanced_acc": float(bal),
        "macro_f1": float((f1[0]+f1[1])/2),
        "prec_absent": float(prec[0]), "rec_absent": float(rec[0]), "f1_absent": float(f1[0]),
        "prec_present": float(prec[1]), "rec_present": float(rec[1]), "f1_present": float(f1[1]),
        "confusion": confusion_matrix(y, yhat).tolist(),
        "roc_auc": float(roc), "pr_auc": float(pr_auc),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS_DEFAULT)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.use_deterministic_algorithms(True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}   seed={args.seed}   max_epochs={args.epochs}")

    X_tr_np, y_tr_np, X_va_np, y_va_np = load_data()

    # Balance train by class subsampling (same as train_head.py)
    pos = np.where(y_tr_np == 1)[0]; neg = np.where(y_tr_np == 0)[0]
    n = min(len(pos), len(neg))
    rng = np.random.default_rng(args.seed)
    keep_pos = rng.choice(pos, n, replace=False)
    keep_neg = rng.choice(neg, n, replace=False)
    keep = np.concatenate([keep_pos, keep_neg])
    rng.shuffle(keep)
    X_tr_np = X_tr_np[keep]; y_tr_np = y_tr_np[keep]
    print(f"[balanced train] N={len(y_tr_np)} (P={int(y_tr_np.sum())}, A={int(len(y_tr_np)-y_tr_np.sum())})")

    X_tr_t = torch.from_numpy(X_tr_np).to(device); y_tr_t = torch.from_numpy(y_tr_np).to(device)
    X_va_t = torch.from_numpy(X_va_np).to(device); y_va_t = torch.from_numpy(y_va_np).to(device)

    model = MLPHead().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    N_tr = X_tr_t.shape[0]
    best_val_mf1 = -1.0; best_state = None; no_improve = 0
    history = []

    print("\nepoch  train_loss  val_loss  val_bal  val_mF1  f1_absent  f1_present")
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(N_tr, device=device)
        loss_sum, nb = 0.0, 0
        for i in range(0, N_tr, BATCH_TRAIN):
            b = perm[i:i+BATCH_TRAIN]
            logits = model(X_tr_t[b])
            loss = sigmoid_focal_loss(logits, y_tr_t[b], alpha=0.5, gamma=FOCAL_GAMMA)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss.item()); nb += 1
        train_loss = loss_sum / nb

        model.eval()
        with torch.no_grad():
            va_logits = model(X_va_t)
            val_loss = float(sigmoid_focal_loss(va_logits, y_va_t, alpha=0.5, gamma=FOCAL_GAMMA))
            va_prob = torch.sigmoid(va_logits).cpu().numpy()
        y_va_int = y_va_np.astype(int)
        va_pred = (va_prob >= 0.5).astype(int)
        val_bal = balanced_accuracy_score(y_va_int, va_pred)
        _, _, f, _ = precision_recall_fscore_support(y_va_int, va_pred, labels=[0,1], zero_division=0)
        mF1 = float((f[0]+f[1])/2)
        print(f"{epoch:>4}   {train_loss:>8.4f}   {val_loss:>7.4f}  {val_bal:>6.3f}  "
              f"{mF1:>6.3f}  {f[0]:>7.3f}  {f[1]:>7.3f}", flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_bal_acc": val_bal, "val_macro_f1": mF1,
                        "f1_absent": float(f[0]), "f1_present": float(f[1])})

        if mF1 > best_val_mf1 + 1e-5:
            best_val_mf1 = mF1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nearly stop epoch {epoch} (best val_macro_f1={best_val_mf1:.4f})")
                break

    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_prob = torch.sigmoid(model(X_va_t)).cpu().numpy()

    y_va_int = y_va_np.astype(int)
    default = eval_at_threshold(y_va_int, va_prob, 0.5)

    # Val-tuned threshold: maximise val macro_f1 over t in [0.30, 0.70]
    best_t = 0.5; best_mf1 = default["macro_f1"]
    for t in np.arange(0.30, 0.7001, 0.01):
        yhat = (va_prob >= t).astype(int)
        _, _, f, _ = precision_recall_fscore_support(y_va_int, yhat, labels=[0,1], zero_division=0)
        m = float((f[0]+f[1])/2)
        if m > best_mf1: best_mf1 = m; best_t = float(t)
    tuned = eval_at_threshold(y_va_int, va_prob, best_t)

    print("\n=== VAL RESULTS ===")
    print(f"  @ t=0.5:   bal_acc={default['balanced_acc']:.4f}  mF1={default['macro_f1']:.4f}  "
          f"ROC={default['roc_auc']:.4f}  ABSENT R/P={default['rec_absent']:.3f}/{default['prec_absent']:.3f}  "
          f"PRESENT R/P={default['rec_present']:.3f}/{default['prec_present']:.3f}")
    print(f"  @ t={best_t:.2f}:  bal_acc={tuned['balanced_acc']:.4f}  mF1={tuned['macro_f1']:.4f}  "
          f"ABSENT R/P={tuned['rec_absent']:.3f}/{tuned['prec_absent']:.3f}  "
          f"PRESENT R/P={tuned['rec_present']:.3f}/{tuned['prec_present']:.3f}")

    out_dir = config.RESULTS_DIR / f"head_single_p{config.POOL_SIZE}"
    out_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with (out_dir / "results.json").open("w") as f:
        json.dump({
            "seed": args.seed,
            "pool_size": config.POOL_SIZE,
            "history": history,
            "val_default_t0.5": default,
            "val_tuned": tuned,
            "counts": {"train_N": len(y_tr_np), "val_N": len(y_va_np),
                       "val_P": int(y_va_np.sum()), "val_A": int(len(y_va_np) - y_va_np.sum())},
        }, f, indent=2)
    print(f"\nwrote {out_dir}/results.json and model.pt")

if __name__ == "__main__":
    main()
