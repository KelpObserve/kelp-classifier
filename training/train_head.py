"""Train MLP head on cached Clay embeddings, using a fixed split.

Reads:
  splits/split_seed{SEED}.json     → tile lists per split
  splits/used_chip_ids.txt         → whitelist of chip IDs (enforces MAX_SCENES cap)
  data/clay_cache/identity.npz     → precomputed embeddings

Env vars:
  HEAD_SEED    seed for the fixed split (default 42; must be one of 42/123/456/789/1000)

Writes:
  results/head_seed{SEED}/results.json
  results/head_seed{SEED}/model.pt
"""
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
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

# #5 (reviewer): must be set BEFORE `import torch` for cuBLAS to see it when
# use_deterministic_algorithms(True) is enabled inside main(). Setting it
# afterwards is unreliable — cuBLAS may have already been initialised via
# a torch autograd path.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)

from patch_pool import cache_local_slice

REPO = Path(__file__).resolve().parent
META_CSV = config.CHIPS_META
CACHE = config.CLAY_CACHE_NPZ
SPLITS = REPO / "splits"
OUT_ROOT = REPO / "results"

SEED = int(os.environ.get("HEAD_SEED", 42))
# EMBED_DIM stays 2048 (CLS 1024 + mean-pooled centre 1024) regardless of
# POOL_SIZE — the pool is a spatial mean, not a concat.
EMBED_DIM = 2048
BATCH_TRAIN = 64
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 25
FOCAL_GAMMA = 2.0
device = "cuda" if torch.cuda.is_available() else "cpu"


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
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t).pow(gamma) * ce).mean()


def load_chip_data(seed):
    """Load metadata + fixed split; return (train, val, test) as lists of
    dicts each with keys {gbifID, tile, is_absent, embedding_row, ...}."""
    split = json.load((SPLITS / f"split_seed{seed}.json").open())
    train_ids = set(split["train_chip_ids"])
    val_ids   = set(split["val_chip_ids"])
    test_ids  = set(split["test_chip_ids"])
    all_ids   = train_ids | val_ids | test_ids
    # Sanity: splits are disjoint
    assert not (train_ids & val_ids), f"train n val overlap: {len(train_ids & val_ids)} chips"
    assert not (train_ids & test_ids), f"train n test overlap: {len(train_ids & test_ids)} chips"
    assert not (val_ids & test_ids), f"val n test overlap: {len(val_ids & test_ids)} chips"
    print(f"[split seed={seed}] train={len(train_ids)} chips, val={len(val_ids)} chips, test={len(test_ids)} chips")

    meta = list(csv.DictReader(META_CSV.open()))
    for r in meta:
        m = re.match(r"S2[ABC]_([0-9]{2}[A-Z]{3})_", r.get("sceneID", ""))
        r["tile"] = m.group(1) if m else "UNK"
        # Explicit equality on both sides — a row with a blank or unrecognized
        # occurrenceStatus is treated as neither absent nor present here, matching
        # make_splits.py. (In practice train_head only sees chips whose IDs are
        # in the split, so blanks never land here — this just keeps the label
        # rule symmetric across the two scripts.)
        r["is_absent"]  = (r.get("occurrenceStatus") == "ABSENT")
        r["is_present"] = (r.get("occurrenceStatus") == "PRESENT")

    keep = [r for r in meta if r["gbifID"] in all_ids]
    assert len(keep) == len(all_ids), (
        f"metadata match {len(keep)} != split chip_ids {len(all_ids)}")

    # Load embedding cache. New format stores CLS + centre patches; we build
    # the 2048-D input on the fly by mean-pooling the POOL_SIZE×POOL_SIZE
    # window inside the cached centre patches.
    cache = np.load(CACHE)
    assert "centre_patches" in cache.files, (
        f"{CACHE} is old-format (has {sorted(cache.files)}); expected "
        f"'cls' + 'centre_patches'. Re-run precompute_clay_embeddings.py."
    )
    row_by_id = {gid: i for i, gid in enumerate(cache["gbif_ids"])}
    cls_all     = cache["cls"]              # (N, 1024)
    patches_all = cache["centre_patches"]   # (N, CACHE_K, CACHE_K, 1024)
    lo, hi = cache_local_slice(config.POOL_SIZE)
    pool_all = patches_all[:, lo:hi, lo:hi, :].mean(axis=(1, 2))  # (N, 1024)
    emb = np.concatenate([cls_all, pool_all], axis=1).astype(np.float32)  # (N, 2048)
    print(f"[cache] {CACHE.name}  N={emb.shape[0]}  pool={config.POOL_SIZE} "
          f"(cache slice [{lo}:{hi}])")
    missing = [r["gbifID"] for r in keep if r["gbifID"] not in row_by_id]
    # C3 (reviewer): missing embeddings must not silently drop. If a split
    # chip is missing from the cache the training set is not what was planned;
    # regenerate the cache instead of running on a smaller-than-designed set.
    assert not missing, (
        f"{len(missing)} split chips missing from embedding cache "
        f"(first 3: {missing[:3]}). Re-run precompute_clay_embeddings.py.")
    for r in keep:
        r["emb_row"] = row_by_id[r["gbifID"]]

    train = [r for r in keep if r["gbifID"] in train_ids]
    val   = [r for r in keep if r["gbifID"] in val_ids]
    test  = [r for r in keep if r["gbifID"] in test_ids]
    assert len(train) + len(val) + len(test) == len(keep), (
        f"count mismatch: {len(train)} + {len(val)} + {len(test)} != {len(keep)}")
    return train, val, test, emb


def build_xy(rows, emb, balance_class=False, rng=None):
    if balance_class:
        pos = [r for r in rows if not r["is_absent"]]
        neg = [r for r in rows if r["is_absent"]]
        target = min(len(pos), len(neg))
        if len(pos) > target:
            keep = rng.choice(len(pos), target, replace=False)
            pos = [pos[i] for i in keep]
        if len(neg) > target:
            keep = rng.choice(len(neg), target, replace=False)
            neg = [neg[i] for i in keep]
        rows = pos + neg
    idx = np.array([r["emb_row"] for r in rows])
    X = emb[idx]
    # Explicit is_present branch: any row that is neither present nor absent
    # never makes it here (make_splits skips those), but this defends against a
    # future ingest that adds a third status value from silently flipping y.
    y = np.array([1 if r["is_present"] else 0 for r in rows], dtype=np.float32)
    return X, y, rows


def main():
    # #5 (reviewer): strict reproducibility on GPU. CUBLAS_WORKSPACE_CONFIG is
    # set at module top before torch imports; here we pin the RNGs and force
    # deterministic cuDNN + cuBLAS ops. warn_only=False → any non-deterministic
    # op raises loudly instead of silently drifting the reported metric.
    print(f"[config] POOL_SIZE={config.POOL_SIZE}  cache={config.CLAY_CACHE_NPZ.name} (shared across pools)")
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception:
        pass  # older torch versions don't have this
    rng = np.random.default_rng(SEED)

    train, val, test, emb = load_chip_data(SEED)
    print(f"  train: {len(train)} chips ({sum(r['is_absent'] for r in train)} A / {sum(not r['is_absent'] for r in train)} P)")
    print(f"  val:   {len(val)}  ({sum(r['is_absent'] for r in val)} A / {sum(not r['is_absent'] for r in val)} P)")
    print(f"  test:  {len(test)} ({sum(r['is_absent'] for r in test)} A / {sum(not r['is_absent'] for r in test)} P)")

    X_tr, y_tr, tr_meta = build_xy(train, emb, balance_class=True, rng=rng)
    X_va, y_va, _ = build_xy(val, emb)
    X_te, y_te, te_meta = build_xy(test, emb)
    print(f"  balanced train: {len(y_tr)} ({int(y_tr.sum())} P, {int(len(y_tr)-y_tr.sum())} A)")

    X_tr_t = torch.from_numpy(X_tr).to(device); y_tr_t = torch.from_numpy(y_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device); y_va_t = torch.from_numpy(y_va).to(device)
    X_te_t = torch.from_numpy(X_te).to(device); y_te_t = torch.from_numpy(y_te).to(device)

    model = MLPHead().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    N_tr = X_tr_t.shape[0]
    best_val_macro_f1 = -1.0
    best_state = None
    no_improve = 0
    history = []

    print("\nepoch  train_loss  val_loss  val_bal  val_mF1  f1_absent  f1_present")
    for epoch in range(1, MAX_EPOCHS + 1):
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
        va_pred = (va_prob >= 0.5).astype(int)
        val_bal = balanced_accuracy_score(y_va.astype(int), va_pred)
        _, _, f, _ = precision_recall_fscore_support(y_va.astype(int), va_pred,
                                                       labels=[0, 1], zero_division=0)
        mF1 = float(f.mean())
        print(f"{epoch:>4}   {train_loss:>8.4f}   {val_loss:>7.4f}  {val_bal:>6.3f}  "
              f"{mF1:>6.3f}  {f[0]:>7.3f}  {f[1]:>7.3f}", flush=True)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_bal_acc": val_bal, "val_macro_f1": mF1,
                        "f1_absent": float(f[0]), "f1_present": float(f[1])})
        if mF1 > best_val_macro_f1 + 1e-5:
            best_val_macro_f1 = mF1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nearly stop epoch {epoch} (best val_macro_f1={best_val_macro_f1:.4f})")
                break

    if best_state: model.load_state_dict(best_state)

    # Test — default t=0.5 and threshold-tuned
    model.eval()
    with torch.no_grad():
        va_prob = torch.sigmoid(model(X_va_t)).cpu().numpy()
        te_prob = torch.sigmoid(model(X_te_t)).cpu().numpy()

    def metrics_at(y, prob, t):
        pred = (prob >= t).astype(int)
        bal = balanced_accuracy_score(y, pred)
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
        cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
        return {"t": float(t), "balanced_acc": float(bal), "macro_f1": float(f.mean()),
                "prec_absent": float(p[0]), "rec_absent": float(r[0]), "f1_absent": float(f[0]),
                "prec_present": float(p[1]), "rec_present": float(r[1]), "f1_present": float(f[1]),
                "confusion": cm}

    yi = y_te.astype(int)
    default = metrics_at(yi, te_prob, 0.5)
    grid = np.linspace(0.05, 0.95, 91)
    yi_v = y_va.astype(int)
    def val_macro_f1_at(t):
        pred = (va_prob >= t).astype(int)
        _, _, f, _ = precision_recall_fscore_support(yi_v, pred, labels=[0, 1], zero_division=0)
        return float(f.mean())
    # Threshold selected on val macro_F1 — same objective as model selection
    # (best_val_macro_f1 during training).
    best_t = max(grid, key=val_macro_f1_at)
    tuned = metrics_at(yi, te_prob, float(best_t))
    default["roc_auc"] = float(roc_auc_score(yi, te_prob))
    default["pr_auc"] = float(average_precision_score(yi, te_prob))
    tuned["roc_auc"] = default["roc_auc"]  # same model, threshold-independent
    tuned["pr_auc"] = default["pr_auc"]

    print("\n=== TEST RESULTS ===")
    print(f"  @ t=0.5:  bal_acc={default['balanced_acc']:.4f}  ABSENT R/P={default['rec_absent']:.3f}/{default['prec_absent']:.3f}  ROC={default['roc_auc']:.3f}")
    print(f"  @ t={best_t:.2f}:  bal_acc={tuned['balanced_acc']:.4f}  ABSENT R/P={tuned['rec_absent']:.3f}/{tuned['prec_absent']:.3f}")

    out_dir = OUT_ROOT / f"head_seed{SEED}_p{config.POOL_SIZE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "results.json").write_text(json.dumps({
        "seed": SEED,
        "pool_size": config.POOL_SIZE,
        "history": history,
        "test_default_t0.5": default,
        "test_tuned": tuned,
        "counts": {"train_P": int(y_tr.sum()), "train_A": int(len(y_tr) - y_tr.sum()),
                    "val_P": int(y_va.sum()), "val_A": int(len(y_va) - y_va.sum()),
                    "test_P": int(y_te.sum()), "test_A": int(len(y_te) - y_te.sum())},
    }, indent=2))
    print(f"wrote {out_dir}/results.json")


if __name__ == "__main__":
    main()
