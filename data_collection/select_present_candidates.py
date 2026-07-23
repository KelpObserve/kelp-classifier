"""select_present_candidates.py — narrow the 32,773 deduped PRESENT candidates
down to the ones geographically closest to ABSENT candidates.

Two categories:
  - forced_p: PRESENT candidates within 50 m of any ABSENT candidate.
    These will co-cluster with the ABSENT chip under our 50 m location safety,
    so they're needed regardless.
  - free_p: the N closest PRESENT candidates outside 50 m, ranked by
    haversine distance to nearest ABSENT candidate.

Rationale:
  Stage B of the split algorithm picks P chips within the same water bin as
  each A chip. P chips physically close to A chips are much more likely to be
  in the same water bin (same coastal / depth zone), so ranking by proximity
  maximizes the chance of good b02/water matching.

Also applies PRESENT's coordinateUncertaintyInMeters <= 115 pre-filter (which
fetch_present.py would enforce anyway) so we don't waste "distance-nearest"
slots on candidates that would be dropped later.

Reads:
  data_collection/deduped_present_candidates.csv
  data_collection/deduped_absent_candidates.csv

Writes:
  data_collection/fetch_present_candidates.csv (forced_p + top-N free_p)

CLI:
  python select_present_candidates.py [--n-free 6000]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.neighbors import BallTree

REPO = Path(__file__).resolve().parent.parent
DC = REPO / "data_collection"
EARTH_R_KM = 6371.0
CLUSTER_RADIUS_M = 50
UNC_MAX_M = 115  # matches fetch_present.py's pre-filter


def parse_float(s):
    try: return float(s or 0)
    except (ValueError, TypeError): return 0.0


def load(csv_path):
    rows = list(csv.DictReader(csv_path.open()))
    print(f"  {csv_path.name}: {len(rows)} rows")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-free", type=int, default=6000,
                    help="number of non-forced P candidates to keep, sorted by proximity")
    args = ap.parse_args()

    print("=== load ===")
    p_rows = load(DC / "deduped_present_candidates.csv")
    a_rows = load(DC / "deduped_absent_candidates.csv")

    # Pre-filter P by coordinate uncertainty (fetch_present.py drops these anyway)
    def unc_ok(r):
        u = parse_float(r.get("coordinateUncertaintyInMeters"))
        return u == 0 or u <= UNC_MAX_M  # 0 = missing, treated as pass
    n_before = len(p_rows)
    p_rows = [r for r in p_rows if unc_ok(r)]
    print(f"  P after unc<={UNC_MAX_M} filter: {len(p_rows)} (dropped {n_before-len(p_rows)})")

    # BallTree over A locations (radians for haversine)
    a_coords = np.radians(np.array([[parse_float(r["decimalLatitude"]),
                                     parse_float(r["decimalLongitude"])] for r in a_rows]))
    tree = BallTree(a_coords, metric="haversine")

    # Distance from each P candidate to its nearest A candidate
    p_coords = np.radians(np.array([[parse_float(r["decimalLatitude"]),
                                     parse_float(r["decimalLongitude"])] for r in p_rows]))
    d_rad, _ = tree.query(p_coords, k=1)
    d_m = d_rad[:, 0] * EARTH_R_KM * 1000.0

    print()
    print("=== distance histogram (P -> nearest A) ===")
    for lo, hi in [(0, 50), (50, 500), (500, 2000), (2000, 10000), (10000, 100000), (100000, 1e9)]:
        n = int(((d_m >= lo) & (d_m < hi)).sum())
        label = f">= {hi}" if hi == 1e9 else f"{lo:>6d} - {hi:>6d} m"
        print(f"  {label:>17s}: {n:>6d}")

    forced_mask = d_m < CLUSTER_RADIUS_M
    free_mask = ~forced_mask
    n_forced = int(forced_mask.sum())
    print()
    print(f"forced_p (within {CLUSTER_RADIUS_M} m of an A): {n_forced}")

    # Rank free_p by distance and take top n_free
    free_indices = np.argsort(d_m[free_mask])[: args.n_free]
    free_p_rows = np.array(p_rows, dtype=object)[free_mask][free_indices]
    forced_p_rows = np.array(p_rows, dtype=object)[forced_mask]
    keep_rows = list(forced_p_rows) + list(free_p_rows)
    print(f"free_p (top {args.n_free} nearest, farthest kept dist = "
          f"{d_m[free_mask][free_indices[-1]] / 1000:.2f} km): {len(free_p_rows)}")
    print(f"total to fetch: {len(keep_rows)}")

    # Write
    out = DC / "fetch_present_candidates.csv"
    fields = list(p_rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in keep_rows: w.writerow(r)
    print(f"wrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
