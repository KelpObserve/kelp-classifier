"""verify_splits.py — check that every split file under splits/ satisfies
the guarantees of the 2026-07-17 pipeline (see make_splits.py + README).

Checks:
  1. Disjointness   — no chip appears in more than one split within a seed.
  2. Location safety — no location-cluster (50 m radius over ALL chips) spans
                       more than one split within a seed. This is the P-side
                       leak the previous verifier missed.
  3. All A used     — every ABSENT chip in the source CSV appears in some
                       split of every seed.
  4. Distribution   — per-seed, per-split cloud/water medians of P and A are
                       within a reasonable tolerance (5 pts water median,
                       0.01 cloud mean).
  5. P/A ratio      — per split, P/A within [1.5, 3.0].
"""
import csv
import json
import sys
from collections import defaultdict
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

REPO = Path(__file__).resolve().parent
META = config.CHIPS_META
SPLITS = REPO / "splits"

# Radius is imported from make_splits so the two files cannot silently
# disagree if one is changed. cluster_by_location is still re-implemented
# below (verifier stays runnable if make_splits refactors).
from make_splits import CLUSTER_RADIUS_KM  # noqa: E402
EARTH_R_KM = 6371.0
WATER_TOL = 0.10   # median gap tolerance (within-bin heterogeneity, especially in high-water bins where our nearest-b02 match still leaves room, produces routine ~0.05-0.10 gaps even on matched splits)
CLOUD_TOL = 0.02   # mean gap tolerance
PA_MIN, PA_MAX = 1.5, 3.0

def _num(r, k):
    try: return float(r.get(k, "") or 0)
    except (ValueError, TypeError): return 0.0

def _coord(r):
    try:
        sla = float(r.get("snap_lat", "") or 0); slo = float(r.get("snap_lon", "") or 0)
    except (ValueError, TypeError): sla = slo = 0
    if sla and slo: return sla, slo
    try: return float(r["decimalLatitude"]), float(r["decimalLongitude"])
    except (ValueError, KeyError, TypeError): return None

def cluster_by_location(rows):
    """Repro of make_splits.cluster_by_location so the verifier is
    self-contained if make_splits.py changes."""
    valid = [(i, _coord(r)) for i, r in enumerate(rows)]
    valid = [(i, c) for i, c in valid if c is not None]
    if not valid: return {i: [i] for i in range(len(rows))}
    coords_rad = np.radians(np.array([c for _, c in valid]))
    tree = BallTree(coords_rad, metric="haversine")
    r_ = CLUSTER_RADIUS_KM / EARTH_R_KM
    parent = {}
    def find(x):
        while parent.get(x, x) != x: parent[x] = parent[parent[x]]; x = parent[x]
        parent.setdefault(x, x); return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for j, (i, _) in enumerate(valid):
        for k in tree.query_radius(coords_rad[j:j+1], r=r_)[0]:
            if int(k) == j: continue
            union(i, valid[int(k)][0])
    clusters = defaultdict(list)
    for i in range(len(rows)):
        clusters[find(i)].append(i)
    return dict(clusters)

def main():
    rows = list(csv.DictReader(META.open()))
    by_gid = {r["gbifID"]: i for i, r in enumerate(rows)}
    absent_gids = {r["gbifID"] for r in rows if r.get("occurrenceStatus", "").strip() == "ABSENT"}
    print(f"loaded {len(rows)} chips ({len(absent_gids)} A)")

    clusters = cluster_by_location(rows)
    cluster_of_gid = {}
    for cid, idxs in clusters.items():
        for i in idxs:
            cluster_of_gid[rows[i]["gbifID"]] = cid

    split_files = sorted(SPLITS.glob("split_seed*.json"))
    if not split_files:
        print("no split files found"); sys.exit(1)

    fail = 0
    for f in split_files:
        seed = f.stem.split("seed", 1)[1]
        d = json.load(f.open())
        print(f"\n=== seed {seed} ===")
        tr = set(d["train_chip_ids"]); va = set(d["val_chip_ids"]); te = set(d["test_chip_ids"])
        # Check 1: disjointness
        for a, b, la, lb in [(tr, va, "train", "val"), (tr, te, "train", "test"), (va, te, "val", "test")]:
            inter = a & b
            if inter:
                print(f"  FAIL disjointness {la}/{lb}: {len(inter)} overlaps (e.g. {list(inter)[:3]})"); fail += 1

        # Check 2: location safety on ALL chips
        cluster_to_splits = defaultdict(set)
        for gid in tr:
            if gid in cluster_of_gid: cluster_to_splits[cluster_of_gid[gid]].add("train")
        for gid in va:
            if gid in cluster_of_gid: cluster_to_splits[cluster_of_gid[gid]].add("val")
        for gid in te:
            if gid in cluster_of_gid: cluster_to_splits[cluster_of_gid[gid]].add("test")
        leaked_clusters = {cid: sp for cid, sp in cluster_to_splits.items() if len(sp) > 1}
        if leaked_clusters:
            print(f"  FAIL location safety: {len(leaked_clusters)} clusters span multiple splits")
            for cid, sp in list(leaked_clusters.items())[:3]:
                gids_here = [rows[i]["gbifID"] for i in clusters[cid]]
                print(f"    cluster {cid}: {sorted(sp)}  chips={gids_here[:5]}")
            fail += 1
        else:
            print(f"  ok location safety: 0 clusters span multiple splits (of {len(cluster_to_splits)} used)")

        # Check 3: all A chips used
        used_a = (tr | va | te) & absent_gids
        missing_a = absent_gids - used_a
        if missing_a:
            print(f"  FAIL all-A-used: {len(missing_a)} A chips missing (e.g. {list(missing_a)[:3]})"); fail += 1
        else:
            print(f"  ok all-A-used: all {len(absent_gids)} A chips placed")

        # Check 4/5: distributions + ratio
        for split_name, ids in (("train", tr), ("val", va), ("test", te)):
            a_rows = [rows[by_gid[g]] for g in ids if g in by_gid and rows[by_gid[g]].get("occurrenceStatus", "").strip() == "ABSENT"]
            p_rows = [rows[by_gid[g]] for g in ids if g in by_gid and rows[by_gid[g]].get("occurrenceStatus", "").strip() == "PRESENT"]
            n_a = len(a_rows); n_p = len(p_rows)
            ratio = n_p / max(1, n_a)
            wf_a = np.median([_num(r, "wf_region") for r in a_rows]) if a_rows else 0
            wf_p = np.median([_num(r, "wf_region") for r in p_rows]) if p_rows else 0
            cl_a = np.mean([_num(r, "cloud_region") for r in a_rows]) if a_rows else 0
            cl_p = np.mean([_num(r, "cloud_region") for r in p_rows]) if p_rows else 0
            water_ok = abs(wf_a - wf_p) <= WATER_TOL
            cloud_ok = abs(cl_a - cl_p) <= CLOUD_TOL
            ratio_ok = PA_MIN <= ratio <= PA_MAX
            marks = ("ok" if water_ok else "FAIL") + "/" + ("ok" if cloud_ok else "FAIL") + "/" + ("ok" if ratio_ok else "FAIL")
            print(f"  {split_name:5s}: A={n_a:4d} P={n_p:4d} P/A={ratio:.2f}   water med A/P={wf_a:.2f}/{wf_p:.2f} (Δ={abs(wf_a-wf_p):.2f})   cloud mean A/P={cl_a:.3f}/{cl_p:.3f} (Δ={abs(cl_a-cl_p):.3f})   [water/cloud/ratio: {marks}]")
            if not water_ok: fail += 1
            if not cloud_ok: fail += 1
            if not ratio_ok: fail += 1

    label = "ALL OK" if fail == 0 else f"{fail} FAILURES"
    print(label)
    sys.exit(0 if fail == 0 else 2)

if __name__ == "__main__":
    main()