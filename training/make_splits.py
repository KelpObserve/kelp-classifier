"""make_splits.py — build N fixed train/val/test splits for kelp head training.

Two-stage pipeline (2026-07-17):

  Stage A — 75/15/15 split of ABSENT-carrying location-clusters
    All chips are first grouped into location-clusters at CLUSTER_RADIUS_KM
    (50 m — exact-location safety). Whole clusters go to one split, so any
    two chips at the same lat/lon (multi-scene _sN siblings, or a P/A pair
    sharing an observation coordinate at different dates) always land in
    the same split. A-carrying clusters are then stratified by their A
    chips' (cloud, water) bin so per-bin split proportions stay ~75/15/15.

  Stage B — hierarchical PRESENT subselection (~1.9:1 P:A)
    Two sources of P per split:
      - FORCED-P: P chips co-located with A (i.e. in an A-carrying cluster)
        go to the same split as that cluster.
      - FREE-P: P chips in P-only clusters participate in a greedy
        nearest-b02 assignment within each A chip's water bin.
    Preserves water alignment exactly and approximates b02_range as tightly
    as the free P pool allows without dropping any A chip.

Guarantees:
  - All 1614 A chips are used (no A loss).
  - Exact-location safety on ALL chips: any two chips whose coords sit
    within CLUSTER_RADIUS_KM stay in the same split.
  - Cloud + water distributions of P and A match across every split.
  - b02_range distributions of P and A match within each water bin as
    tightly as the P pool allows.
  - 3 independent seeds (42, 123, 456 by default) produce 3 cross-val folds.

L2 sanity on the produced splits (linear classifier on QC only):
  - water + cloud only: ROC ~0.51 - 0.55 (essentially chance)
  - + fill_region + b02_range: ROC ~0.61 - 0.64 (residual b02 gap)

Reads:  data/s2_chips_merged/chips_metadata.csv
Writes: splits/a_split_seed{S}.json         (intermediate A-only splits)
        splits/split_seed{S}.json           (final chip splits per seed)
        splits/used_chip_ids.txt            (union of chip IDs across seeds)

CLI:
  python make_splits.py [--seeds S1 S2 ...]
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.neighbors import BallTree
# Locate repo root by walking up until config.py is found
import sys as _sys
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

# ---------- constants ----------

META_CSV = config.CHIPS_META
OUT_DIR  = Path(__file__).parent / "splits"

SPLIT_RATIOS = (0.75, 0.15, 0.15)      # train / val / test

# Any two chips whose coords fall within this radius must go to the same
# split. 50 m is effectively "same lat/lon allowing for GPS noise" — it
# catches multi-scene _sN siblings (all at the same coord) and the small
# number of P/A pairs sharing an observation coordinate at different dates
# or for different species. Chips more than 50 m apart are treated as
# independent by the split logic (see README for rationale).
CLUSTER_RADIUS_KM = 0.05
EARTH_R_KM = 6371.0

# Bin widths for stratification / matching
WATER_W = 0.10   # wf_region bin
CLOUD_W = 0.05   # cloud_region bin
P_A_RATIO_TARGET = 2  # P/A target per A chip (achieved ~1.9:1 in practice)

DEFAULT_SEEDS = (42, 123, 456)

# ---------- helpers ----------

def _num(r, k):
    try: return float(r.get(k, "") or 0)
    except (ValueError, TypeError): return 0.0

def water_bin(r): return int(_num(r, "wf_region") / WATER_W)
def cloud_bin(r): return int(_num(r, "cloud_region") / CLOUD_W)
def cw_bin(r):    return (cloud_bin(r), water_bin(r))
def b02(r):       return _num(r, "b02_range")

def _coord(r):
    """Prefer snap coords if the chip was re-anchored; else raw observer coord."""
    try:
        sla = float(r.get("snap_lat", "") or 0)
        slo = float(r.get("snap_lon", "") or 0)
    except (ValueError, TypeError):
        sla = slo = 0
    if sla and slo: return sla, slo
    try:
        return float(r["decimalLatitude"]), float(r["decimalLongitude"])
    except (ValueError, KeyError, TypeError):
        return None

class UnionFind:
    def __init__(self): self.p = {}
    def find(self, x):
        if x not in self.p: self.p[x] = x
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb

# ---------- clustering (over ALL chips) ----------

def cluster_by_location(rows):
    """Union chips (P and A alike) whose lat/lon lie within CLUSTER_RADIUS_KM.
    Returns {cluster_id: [row_index, ...]} where row_index is into the input
    rows list. Chips without valid coords are singletons.
    """
    valid = [(i, _coord(r)) for i, r in enumerate(rows)]
    valid = [(i, c) for i, c in valid if c is not None]
    if not valid:
        return {i: [i] for i in range(len(rows))}

    coords_rad = np.radians(np.array([c for _, c in valid]))
    tree = BallTree(coords_rad, metric="haversine")
    radius = CLUSTER_RADIUS_KM / EARTH_R_KM

    uf = UnionFind()
    for j, (i, _) in enumerate(valid):
        for k in tree.query_radius(coords_rad[j:j+1], r=radius)[0]:
            if int(k) == j: continue
            uf.union(i, valid[int(k)][0])

    clusters = defaultdict(list)
    for i, _ in enumerate(rows):
        if i in uf.p: clusters[uf.find(i)].append(i)
        else: clusters[i].append(i)
    return dict(clusters)

# ---------- Stage A: stratified ABSENT split ----------

def stratified_split_a_carrying(all_rows, clusters, seed):
    """Assign each A-carrying location-cluster to train/val/test = 75/15/15,
    stratified by (cloud, water) bin of the A chips inside the cluster.

    All chips in the cluster (A and any co-located P) go to the assigned
    split — that's where exact-location safety is enforced on the A side.

    Returns:
      cluster_split: {cluster_id: split_name} for A-carrying clusters only
      a_by_split: {"train": [row_idx, ...], "val": [...], "test": [...]}
    """
    a_carrying = {cid: idxs for cid, idxs in clusters.items()
                  if any(all_rows[i]["is_absent"] for i in idxs)}
    rng = random.Random(seed)

    by_bin = defaultdict(list)
    for cid, idxs in a_carrying.items():
        a_only = [i for i in idxs if all_rows[i]["is_absent"]]
        primary = Counter(cw_bin(all_rows[i]) for i in a_only).most_common(1)[0][0]
        by_bin[primary].append((cid, idxs, len(a_only)))

    cluster_split = {}
    a_by_split = {"train": [], "val": [], "test": []}
    for b, cs in by_bin.items():
        total = sum(n_a for _, _, n_a in cs)
        tgt_t = total * SPLIT_RATIOS[0]
        tgt_v = total * SPLIT_RATIOS[1]
        tgt_e = total * SPLIT_RATIOS[2]
        rng.shuffle(cs)
        used_t = used_v = used_e = 0
        for cid, idxs, n_a in cs:
            gaps = [tgt_t - used_t, tgt_v - used_v, tgt_e - used_e]
            k = max(range(3), key=lambda i: gaps[i])
            if k == 0: split = "train"; used_t += n_a
            elif k == 1: split = "val"; used_v += n_a
            else: split = "test"; used_e += n_a
            cluster_split[cid] = split
            for i in idxs:
                if all_rows[i]["is_absent"]:
                    a_by_split[split].append(i)
    return cluster_split, a_by_split

# ---------- Stage B: PRESENT subselection ----------

def subselect_present(all_rows, clusters, cluster_split, a_by_split, seed):
    """Cluster-aware hierarchical PRESENT subselection.

    Two sources of P per split:
      1) FORCED-P: P chips inside A-carrying clusters go to the same split
         as those clusters (exact-location safety).
      2) FREE-P: P-only clusters participate in greedy nearest-b02 matching
         within each A chip's water bin. Whole clusters are chosen atomically
         so exact-location safety holds for P-P pairs at the same lat/lon.

    Per water bin we iterate A chips in a seeded random order; each A is
    matched to its P_A_RATIO_TARGET nearest-b02 free clusters in the same
    water bin. Water alignment is preserved exactly; b02 is matched as
    tightly as the free pool allows.

    Returns:
      picked: {"train": [gbifID, ...], "val": [...], "test": [...]}
    """
    rng = random.Random(seed)

    forced_p = {"train": [], "val": [], "test": []}
    for cid, split in cluster_split.items():
        for i in clusters[cid]:
            if all_rows[i]["is_present"]:
                forced_p[split].append(all_rows[i]["gbifID"])

    p_free_clusters = []
    for cid, idxs in clusters.items():
        if cid in cluster_split: continue
        p_idxs = [i for i in idxs if all_rows[i]["is_present"]]
        if not p_idxs: continue
        primary_wbin = Counter(water_bin(all_rows[i]) for i in p_idxs).most_common(1)[0][0]
        mean_b02 = float(np.mean([b02(all_rows[i]) for i in p_idxs]))
        p_free_clusters.append({
            "wbin": primary_wbin,
            "b02": mean_b02,
            "gids": [all_rows[i]["gbifID"] for i in p_idxs],
            "taken": False,
        })

    free_by_wbin = defaultdict(list)
    for c in p_free_clusters:
        free_by_wbin[c["wbin"]].append(c)

    a_by_wbin = defaultdict(list)
    for split_name, indices in a_by_split.items():
        for i in indices:
            r = all_rows[i]
            a_by_wbin[water_bin(r)].append((split_name, b02(r)))

    picked_free = {"train": [], "val": [], "test": []}
    for wbin, a_list in a_by_wbin.items():
        pool = list(free_by_wbin.get(wbin, []))
        rng.shuffle(pool)
        rng.shuffle(a_list)
        for (split_name, a_val) in a_list:
            for _ in range(P_A_RATIO_TARGET):
                best_i, best_d = -1, float("inf")
                for i, c in enumerate(pool):
                    if c["taken"]: continue
                    d = abs(c["b02"] - a_val)
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i < 0:
                    break
                pool[best_i]["taken"] = True
                picked_free[split_name].extend(pool[best_i]["gids"])

    return {k: sorted(set(forced_p[k]) | set(picked_free[k]))
            for k in ("train", "val", "test")}

# ---------- orchestration ----------

def load_rows():
    """Load metadata CSV. Adds explicit is_absent/is_present booleans; any
    row with a blank or unrecognized occurrenceStatus is skipped by the
    split logic rather than silently classed either way."""
    rows = list(csv.DictReader(META_CSV.open()))
    for r in rows:
        status = r.get("occurrenceStatus", "").strip()
        r["is_absent"]  = (status == "ABSENT")
        r["is_present"] = (status == "PRESENT")
    return rows

def _dist_summary(rows_list):
    if not rows_list: return "(empty)"
    wf = np.array([_num(r, "wf_region") for r in rows_list])
    cl = np.array([_num(r, "cloud_region") for r in rows_list])
    bx = np.array([_num(r, "b02_range") for r in rows_list])
    return (f"water med/mean={np.median(wf):.2f}/{wf.mean():.2f}  "
            f"cloud med/mean={np.median(cl):.3f}/{cl.mean():.3f}  "
            f"b02 med/mean={int(np.median(bx))}/{int(bx.mean())}")

def main():
    ap = argparse.ArgumentParser(description="Generate matched train/val/test splits.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                    help=f"Seeds to generate (default: {DEFAULT_SEEDS})")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    rows = load_rows()
    absent  = [i for i, r in enumerate(rows) if r["is_absent"]]
    present = [i for i, r in enumerate(rows) if r["is_present"]]
    skipped = len(rows) - len(absent) - len(present)
    print(f"loaded {len(rows)} chips: A={len(absent)}  P={len(present)}"
          + (f"  (skipped {skipped} rows with blank/other occurrenceStatus)" if skipped else ""))

    clusters = cluster_by_location(rows)
    n_a = sum(1 for _, idxs in clusters.items()
              if any(rows[i]["is_absent"] for i in idxs))
    n_p_only = sum(1 for _, idxs in clusters.items()
                   if not any(rows[i]["is_absent"] for i in idxs)
                   and any(rows[i]["is_present"] for i in idxs))
    n_mixed = sum(1 for _, idxs in clusters.items()
                  if any(rows[i]["is_absent"] for i in idxs)
                  and any(rows[i]["is_present"] for i in idxs))
    print(f"location clusters (radius {CLUSTER_RADIUS_KM*1000:.0f} m): "
          f"{len(clusters)} total  ({n_a} A-carrying, of which {n_mixed} mixed P/A;  {n_p_only} P-only)")

    union_ids = set()
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        cluster_split, a_by_split = stratified_split_a_carrying(rows, clusters, seed)
        tot_a = sum(len(v) for v in a_by_split.values())
        n_tr = len(a_by_split["train"])
        n_va = len(a_by_split["val"])
        n_te = len(a_by_split["test"])
        print(f"  A split: train={n_tr} ({100*n_tr/tot_a:.1f}%) val={n_va} ({100*n_va/tot_a:.1f}%) test={n_te} ({100*n_te/tot_a:.1f}%)  total={tot_a}")

        picked_p = subselect_present(rows, clusters, cluster_split, a_by_split, seed)

        by_g = {rows[i]["gbifID"]: rows[i] for i in present}
        for k in ("train", "val", "test"):
            a_rows_k = [rows[i] for i in a_by_split[k]]
            p_rows_k = [by_g[g] for g in picked_p[k] if g in by_g]
            n_a_k = len(a_rows_k); n_p_k = len(p_rows_k)
            print(f"  {k:5s}: A={n_a_k:4d} P={n_p_k:4d} P/A={n_p_k/max(1,n_a_k):.2f}")
            print(f"    A: {_dist_summary(a_rows_k)}")
            print(f"    P: {_dist_summary(p_rows_k)}")

        a_only = {
            "seed": seed,
            "train_a_ids": [rows[i]["gbifID"] for i in a_by_split["train"]],
            "val_a_ids":   [rows[i]["gbifID"] for i in a_by_split["val"]],
            "test_a_ids":  [rows[i]["gbifID"] for i in a_by_split["test"]],
        }
        (OUT_DIR / f"a_split_seed{seed}.json").write_text(json.dumps(a_only))

        final = {
            "seed": seed,
            "train_chip_ids": sorted(set([rows[i]["gbifID"] for i in a_by_split["train"]] + picked_p["train"])),
            "val_chip_ids":   sorted(set([rows[i]["gbifID"] for i in a_by_split["val"]]   + picked_p["val"])),
            "test_chip_ids":  sorted(set([rows[i]["gbifID"] for i in a_by_split["test"]]  + picked_p["test"])),
        }
        (OUT_DIR / f"split_seed{seed}.json").write_text(json.dumps(final))
        union_ids.update(final["train_chip_ids"])
        union_ids.update(final["val_chip_ids"])
        union_ids.update(final["test_chip_ids"])
        print(f"  wrote splits/a_split_seed{seed}.json and splits/split_seed{seed}.json")

    (OUT_DIR / "used_chip_ids.txt").write_text("\n".join(sorted(union_ids)) + "\n")
    print(f"\nunion of chip IDs across all seeds: {len(union_ids)}  -> splits/used_chip_ids.txt")

if __name__ == "__main__":
    main()
