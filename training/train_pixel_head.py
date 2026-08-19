"""Train MLP head on per-pixel features at the labelled chip-centre pixel.

Design notes
------------
Every chip is centred on its observation coord (PRESENT: reported ≤10 m unc;
ABSENT: snap-to-water). The labelled pixel is therefore always the chip-centre
S2 pixel (64, 64) in the 128×128 native grid — pixel (14, 14) in Clay's 28×28
patch grid (~46 m per patch).

Input feature per chip (2054-D):
  CLS token                    (1024,)  — chip-global Clay context
  centre patch at (14, 14)     (1024,)  — Clay's ~46 m centre patch feature
  raw S2 reflectance at (64,64)(  6,)   — Clay-normalised 6-band raw pixel

This is pool=1 (2048-D) + 6 raw S2 features at the labelled pixel. Head
architecture stays MLP 2054→512→128→1 with dropout 0.3.

Same 3-seed protocol as train_head.py: reads splits_submerged/split_seed*.json
and reuses the identity_submerged.npz Clay cache. NEVER overwrites any
existing head / results — outputs go to results_submerged/pixel/head_seed*/.

Env vars:
  HEAD_SEED    seed for the fixed split (default 42)
  KELP_CHIPS_SUBSET  (submerged)

Writes:
  results_submerged/pixel/head_seed{SEED}/results.json
  results_submerged/pixel/head_seed{SEED}/model.pt
"""
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)

META_CSV = config.CHIPS_META
CACHE    = config.CLAY_CACHE_NPZ
RAW_NPZ  = config.CLAY_CACHE_NPZ.with_name(
    f"raw_center{'' if config.CHIPS_SUBSET == 'merged' else '_' + config.CHIPS_SUBSET}.npz"
)
SPLITS   = config.SPLITS_DIR
OUT_ROOT = config.RESULTS_DIR / "pixel"

SEED = int(os.environ.get("HEAD_SEED", 42))
# CLS(1024) + centre patch (14,14) (1024) + raw S2 at (64,64) (6) = 2054
EMBED_DIM = 2048 + 6
BATCH_TRAIN = 64
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 25
FOCAL_GAMMA = 2.0
device = "cuda" if torch.cuda.is_available() else "cpu"

# Chip centre native pixel (64, 64) → Clay 28×28 patch (14, 14).
# 64/128 * 224 = 112; 112 // 8 = 14. Cached CACHE_K=7 window is patches
# [10..16], so (14, 14) is at cache-local (4, 4).
CENTRE_PATCH_LOCAL_R = 4
CENTRE_PATCH_LOCAL_C = 4


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
    """Load metadata + split; join to Clay cache AND raw-centre cache.
    Returns train/val/test lists and the (N, 2054) feature matrix."""
    split = json.load((SPLITS / f"split_seed{seed}.json").open())
    train_ids = set(split["train_chip_ids"])
    val_ids   = set(split["val_chip_ids"])
    test_ids  = set(split["test_chip_ids"])
    all_ids   = train_ids | val_ids | test_ids
    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    print(f"[split seed={seed}] train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    meta = list(csv.DictReader(META_CSV.open()))
    for r in meta:
        m = re.match(r"S2[ABC]_([0-9]{2}[A-Z]{3})_", r.get("sceneID", ""))
        r["tile"] = m.group(1) if m else "UNK"
        r["is_absent"]  = (r.get("occurrenceStatus") == "ABSENT")
        r["is_present"] = (r.get("occurrenceStatus") == "PRESENT")
    keep = [r for r in meta if r["gbifID"] in all_ids]
    assert len(keep) == len(all_ids), f"metadata {len(keep)} != split {len(all_ids)}"

    # Clay identity cache: CLS + centre 7×7 patches
    cache = np.load(CACHE)
    assert "centre_patches" in cache.files, (
        f"{CACHE} is old-format — re-run precompute_clay_embeddings.py.")
    cache_ids = list(map(str, cache["gbif_ids"]))
    row_by_id = {g: i for i, g in enumerate(cache_ids)}
    cls_all     = cache["cls"]              # (N, 1024)
    patches_all = cache["centre_patches"]   # (N, 7, 7, 1024)
    centre_all  = patches_all[:, CENTRE_PATCH_LOCAL_R, CENTRE_PATCH_LOCAL_C, :]  # (N, 1024)

    # Raw centre-pixel cache
    if not RAW_NPZ.exists():
        raise SystemExit(f"raw_center cache not found: {RAW_NPZ}\n"
                         f"Run precompute_raw_center.py first.")
    raw = np.load(RAW_NPZ)
    raw_by_id = {g: i for i, g in enumerate(map(str, raw["gbif_ids"]))}
    raw_all = raw["raw_center"]             # (M, 6)
    print(f"[cache] identity: N={len(cache_ids)}  raw_center: N={len(raw_by_id)}")

    missing_c = [r["gbifID"] for r in keep if r["gbifID"] not in row_by_id]
    missing_r = [r["gbifID"] for r in keep if r["gbifID"] not in raw_by_id]
    assert not missing_c, f"{len(missing_c)} chips missing from identity cache (first 3: {missing_c[:3]})"
    assert not missing_r, f"{len(missing_r)} chips missing from raw_center cache (first 3: {missing_r[:3]})"

    # Assemble 2054-D features aligned to `keep`
    N = len(keep)
    emb = np.zeros((N, EMBED_DIM), dtype=np.float32)
    for i, r in enumerate(keep):
        gid = r["gbifID"]
        ci = row_by_id[gid]
        ri = raw_by_id[gid]
        emb[i, :1024]      = cls_all[ci]
        emb[i, 1024:2048]  = centre_all[ci]
        emb[i, 2048:]      = raw_all[ri]
        r["emb_row"] = i

    train = [r for r in keep if r["gbifID"] in train_ids]
    val   = [r for r in keep if r["gbifID"] in val_ids]
    test  = [r for r in keep if r["gbifID"] in test_ids]
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
    y = np.array([1 if r["is_present"] else 0 for r in rows], dtype=np.float32)
    return X, y, rows


def main():
    print(f"[config] subset={config.CHIPS_SUBSET}  seed={SEED}  "
          f"identity={CACHE.name}  raw_center={RAW_NPZ.name}  "
          f"splits={SPLITS.name}  out={OUT_ROOT.name}")
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception:
        pass
    rng = np.random.default_rng(SEED)

    train, val, test, emb = load_chip_data(SEED)
    print(f"  train: {len(train)} chips ({sum(r['is_absent'] for r in train)} A / {sum(not r['is_absent'] for r in train)} P)")
    print(f"  val:   {len(val)}  ({sum(r['is_absent'] for r in val)} A / {sum(not r['is_absent'] for r in val)} P)")
    print(f"  test:  {len(test)} ({sum(r['is_absent'] for r in test)} A / {sum(not r['is_absent'] for r in test)} P)")

    X_tr, y_tr, _ = build_xy(train, emb, balance_class=True, rng=rng)
    X_va, y_va, _ = build_xy(val, emb)
    X_te, y_te, _ = build_xy(test, emb)
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
    best_t = max(grid, key=val_macro_f1_at)
    tuned = metrics_at(yi, te_prob, float(best_t))
    default["roc_auc"] = float(roc_auc_score(yi, te_prob))
    default["pr_auc"]  = float(average_precision_score(yi, te_prob))
    tuned["roc_auc"]   = default["roc_auc"]
    tuned["pr_auc"]    = default["pr_auc"]

    print("\n=== TEST RESULTS ===")
    print(f"  @ t=0.5:  bal_acc={default['balanced_acc']:.4f}  ROC={default['roc_auc']:.3f}  "
          f"F1_A={default['f1_absent']:.3f}  F1_P={default['f1_present']:.3f}")
    print(f"  @ t={best_t:.2f}: bal_acc={tuned['balanced_acc']:.4f}  "
          f"F1_A={tuned['f1_absent']:.3f}  F1_P={tuned['f1_present']:.3f}")

    out_dir = OUT_ROOT / f"head_seed{SEED}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "results.json").write_text(json.dumps({
        "seed": SEED,
        "arch": "pixel_head_2054d",
        "embed_dim": EMBED_DIM,
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
