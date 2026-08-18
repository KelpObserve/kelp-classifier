"""make_single_split.py — build a single train/val split (default 85 / 15).

Same guarantees as the 3-seed splitter in make_splits.py:

  Stage A — 85/15 split of A-carrying location-clusters
    All chips are grouped into 50 m location clusters. A-carrying clusters
    (i.e. clusters containing at least one ABSENT chip) are stratified by
    (cloud_region / 0.05, wf_region / 0.10) bin so per-bin proportions stay
    near the target. Whole clusters go to one split — no cross-split leakage
    on any pair of chips within 50 m of each other. All ABSENT chips are
    used.

  Stage B — hierarchical PRESENT subselection (~1.9-2.2 : 1 P:A)
    Forced-P: P chips inside A-carrying clusters go to the same split as
      those clusters.
    Free-P: for each split's A chips, greedy nearest-b02 matching within
      the same water bin.

Cluster granularity + coarse bins mean any single seed leaves some residual
train/val divergence on water/cloud. `--n-candidates N` runs the splitter N
times with consecutive seeds and keeps the split whose worst per-class KS D
(over {PRESENT, ABSENT} × {water, cloud}) is smallest — cheap way to trade
CPU for tighter distribution matching without touching the location-safety
or bin-stratification guarantees.

Reads:  chips_metadata.csv (via config.CHIPS_META)
Writes: splits/split_single.json  ({"train_chip_ids": [...], "val_chip_ids": [...]})
        splits/a_split_single.json (intermediate A-only view)

CLI:  python make_single_split.py [--seed 42] [--train-frac 0.85] [--n-candidates 1]
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

# Locate repo root by walking up until config.py is found
import sys as _sys
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

# Reuse the primitives from the 3-way splitter (unchanged logic)
from make_splits import (
    cluster_by_location, load_rows,
    cw_bin, water_bin, b02,
    _num, _dist_summary,
    CLUSTER_RADIUS_KM, P_A_RATIO_TARGET,
)

import random

DEFAULT_SEED = 42
DEFAULT_TRAIN_FRAC = 0.85

# ---------- Stage A: A-carrying cluster split, but two-way ----------

def stratified_a_split_two_way(all_rows, clusters, seed, train_frac):
    """Two-way (train / val) analogue of stratified_split_a_carrying."""
    val_frac = 1.0 - train_frac
    a_carrying = {cid: idxs for cid, idxs in clusters.items()
                  if any(all_rows[i]["is_absent"] for i in idxs)}
    rng = random.Random(seed)

    by_bin = defaultdict(list)
    for cid, idxs in a_carrying.items():
        a_only = [i for i in idxs if all_rows[i]["is_absent"]]
        primary = Counter(cw_bin(all_rows[i]) for i in a_only).most_common(1)[0][0]
        by_bin[primary].append((cid, idxs, len(a_only)))

    cluster_split = {}
    a_by_split = {"train": [], "val": []}
    for b, cs in by_bin.items():
        total = sum(n_a for _, _, n_a in cs)
        tgt_t = total * train_frac
        tgt_v = total * val_frac
        rng.shuffle(cs)
        used_t = used_v = 0
        for cid, idxs, n_a in cs:
            # Assign to whichever split is most under its per-bin target.
            gaps = [tgt_t - used_t, tgt_v - used_v]
            k = max(range(2), key=lambda i: gaps[i])
            if k == 0: split = "train"; used_t += n_a
            else:      split = "val";   used_v += n_a
            cluster_split[cid] = split
            for i in idxs:
                if all_rows[i]["is_absent"]:
                    a_by_split[split].append(i)
    return cluster_split, a_by_split


# ---------- Stage B: hierarchical PRESENT subselection (two-way) ----------

def subselect_present_two_way(all_rows, clusters, cluster_split, a_by_split, seed):
    """Same greedy nearest-b02 matching as the 3-way version but keyed on
    two splits (train / val) instead of three."""
    rng = random.Random(seed)

    forced_p = {"train": [], "val": []}
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
            "wbin": primary_wbin, "b02": mean_b02,
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

    picked_free = {"train": [], "val": []}
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
            for k in ("train", "val")}


# ---------- Distribution-divergence scoring (for --n-candidates) ----------

def _feat(row, key):
    """Return float(row[key]) or None if missing/unparseable."""
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def score_split(rows, a_by_split, picked_p):
    """Return dict of KS D over the four (class, feature) pairs plus the max.

    Score is the max KS statistic; lower is better. Uses scipy.stats.ks_2samp;
    imported lazily so single-candidate runs don't need scipy installed.
    """
    from scipy.stats import ks_2samp  # lazy
    by_g = {r["gbifID"]: r for r in rows}

    def bucket(k):
        # Return dict split → list[chip rows] restricted to occurrenceStatus == k.
        out = {"train": [], "val": []}
        # ABSENT chips are indexed by row index directly.
        for split_name, indices in a_by_split.items():
            for i in indices:
                if rows[i].get("occurrenceStatus") == k:
                    out[split_name].append(rows[i])
        # PRESENT chips are indexed by gbifID via picked_p.
        for split_name, gids in picked_p.items():
            for g in gids:
                r = by_g.get(g)
                if r is not None and r.get("occurrenceStatus") == k:
                    out[split_name].append(r)
        return out

    scores = {}
    for label in ("PRESENT", "ABSENT"):
        b = bucket(label)
        for feat in ("wf_region", "cloud_region"):
            tr = [x for x in (_feat(r, feat) for r in b["train"]) if x is not None]
            va = [x for x in (_feat(r, feat) for r in b["val"])   if x is not None]
            if len(tr) < 2 or len(va) < 2:
                scores[f"{label}.{feat}"] = float("nan")
                continue
            scores[f"{label}.{feat}"] = float(ks_2samp(tr, va).statistic)
        # ABSENT of a single-cluster-only bin can have zero val chips; guard.
    scores["max"] = max(v for v in scores.values() if v == v)  # skip NaN
    return scores


def run_candidate(rows, clusters, seed, train_frac):
    """One splitter run for a given seed. Returns (cluster_split, a_by_split, picked_p)."""
    cluster_split, a_by_split = stratified_a_split_two_way(rows, clusters, seed, train_frac)
    picked_p = subselect_present_two_way(rows, clusters, cluster_split, a_by_split, seed)
    return cluster_split, a_by_split, picked_p


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"Base RNG seed (default: {DEFAULT_SEED}). With --n-candidates>1, "
                         "consecutive seeds {seed, seed+1, ...} are tried.")
    ap.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC,
                    help=f"Fraction of A chips going to train (default: {DEFAULT_TRAIN_FRAC})")
    ap.add_argument("--n-candidates", type=int, default=1,
                    help="If >1, run the splitter this many times with consecutive seeds "
                         "and keep the one whose worst per-class KS D over water+cloud is "
                         "smallest. Requires scipy.")
    args = ap.parse_args()
    if not 0.5 <= args.train_frac <= 0.95:
        raise SystemExit("--train-frac must be between 0.5 and 0.95")
    if args.n_candidates < 1:
        raise SystemExit("--n-candidates must be >= 1")

    config.SPLITS_DIR.mkdir(exist_ok=True, parents=True)

    rows = load_rows()
    absent  = [i for i, r in enumerate(rows) if r["is_absent"]]
    present = [i for i, r in enumerate(rows) if r["is_present"]]
    skipped = len(rows) - len(absent) - len(present)
    print(f"loaded {len(rows)} chips: A={len(absent)}  P={len(present)}"
          + (f"  (skipped {skipped} rows)" if skipped else ""))

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

    if args.n_candidates == 1:
        chosen_seed = args.seed
        print(f"\nseed={chosen_seed}  train_frac={args.train_frac}")
        cluster_split, a_by_split, picked_p = run_candidate(rows, clusters, chosen_seed, args.train_frac)
    else:
        print(f"\nsearching {args.n_candidates} candidates (seeds {args.seed}..{args.seed + args.n_candidates - 1})  "
              f"train_frac={args.train_frac}")
        candidates = []  # (max_ks, seed, cluster_split, a_by_split, picked_p, scores)
        for cand_seed in range(args.seed, args.seed + args.n_candidates):
            cs, abs_, pp = run_candidate(rows, clusters, cand_seed, args.train_frac)
            scores = score_split(rows, abs_, pp)
            candidates.append((scores["max"], cand_seed, cs, abs_, pp, scores))
        candidates.sort(key=lambda t: t[0])
        print("  top 5 candidates by max KS D (lower = tighter distribution match):")
        for max_d, sd, _, _, _, sc in candidates[:5]:
            print(f"    seed={sd:>4}  max_D={max_d:.3f}  "
                  f"P.wf={sc['PRESENT.wf_region']:.3f} P.cl={sc['PRESENT.cloud_region']:.3f} "
                  f"A.wf={sc['ABSENT.wf_region']:.3f} A.cl={sc['ABSENT.cloud_region']:.3f}")
        worst = candidates[-1]
        print(f"  worst candidate: seed={worst[1]}  max_D={worst[0]:.3f}")
        max_d, chosen_seed, cluster_split, a_by_split, picked_p, chosen_scores = candidates[0]
        print(f"\nchose seed={chosen_seed}  max_D={max_d:.3f}")

    tot_a = sum(len(v) for v in a_by_split.values())
    n_tr = len(a_by_split["train"]); n_va = len(a_by_split["val"])
    print(f"  A split: train={n_tr} ({100*n_tr/tot_a:.1f}%)  val={n_va} ({100*n_va/tot_a:.1f}%)  total={tot_a}")
    by_g = {rows[i]["gbifID"]: rows[i] for i in present}
    for k in ("train", "val"):
        a_rows_k = [rows[i] for i in a_by_split[k]]
        p_rows_k = [by_g[g] for g in picked_p[k] if g in by_g]
        n_a_k = len(a_rows_k); n_p_k = len(p_rows_k)
        print(f"  {k:5s}: A={n_a_k:4d} P={n_p_k:4d} P/A={n_p_k/max(1,n_a_k):.2f}")
        print(f"    A: {_dist_summary(a_rows_k)}")
        print(f"    P: {_dist_summary(p_rows_k)}")

    a_only = {
        "seed": chosen_seed,
        "train_a_ids": [rows[i]["gbifID"] for i in a_by_split["train"]],
        "val_a_ids":   [rows[i]["gbifID"] for i in a_by_split["val"]],
    }
    (config.SPLITS_DIR / "a_split_single.json").write_text(json.dumps(a_only))

    final = {
        "seed": chosen_seed,
        "train_frac": args.train_frac,
        "n_candidates": args.n_candidates,
        "train_chip_ids": sorted(set([rows[i]["gbifID"] for i in a_by_split["train"]] + picked_p["train"])),
        "val_chip_ids":   sorted(set([rows[i]["gbifID"] for i in a_by_split["val"]]   + picked_p["val"])),
    }
    (config.SPLITS_DIR / "split_single.json").write_text(json.dumps(final))
    print(f"  wrote splits/a_split_single.json and splits/split_single.json")
    print(f"  chip_ids: train={len(final['train_chip_ids'])} val={len(final['val_chip_ids'])}")

if __name__ == "__main__":
    main()
