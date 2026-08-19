"""build_submerged_candidates.py — candidate CSVs for the four deep-submerged
kelp species (blade sub-surface at 5-30 m), EXCLUDING any observation already
represented in the merged chip pool.

  Saccorhiza polyschides
  Laminaria hyperborea
  Laminaria ochroleuca
  Phyllariopsis brevipes

PRESENT filter matches build_high_precision_candidates.py: unc <= P_MAX_UNC
(default 10 m); blank unc is dropped unless --include-blank-unc.
ABSENT: no unc filter (a "no kelp seen" claim over an area is robust to a
noisy anchor point — the fetch pipeline snaps to water anyway).

Cross-reference: any candidate gbifID that already appears (with or without
_sN scene suffix) in $KELP_DATA_DIR/s2_chips_merged/chips_metadata.csv is
skipped — the fetch pipeline would otherwise redo work it already did.

Reads:
  data_collection/deduped_present_candidates.csv
  data_collection/deduped_absent_candidates.csv
  $KELP_DATA_DIR/s2_chips_merged/chips_metadata.csv

Writes (default):
  data_collection/submerged_present_candidates.csv  (PRESENT, new only)
  data_collection/submerged_absent_candidates.csv   (ABSENT, new only)
"""
import argparse
import csv
import re
import sys as _sys
from collections import Counter
from pathlib import Path
from pathlib import Path as _P

_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

DC = Path(__file__).resolve().parent

TARGET = {
    "saccorhiza polyschides",
    "laminaria hyperborea",
    "laminaria ochroleuca",
    "phyllariopsis brevipes",
}

DEFAULT_P_MAX_UNC = 10.0


def base_id(gid):
    """Strip _sN scene-index suffix (the merged pool stores multiple chips
    per raw record as gbifID_s1, gbifID_s2, …)."""
    return re.sub(r"_s\d+$", "", gid)


def unc_val(r):
    u = (r.get("coordinateUncertaintyInMeters") or "").strip()
    if not u: return None
    try: return float(u)
    except (ValueError, TypeError): return None


def load_already_fetched():
    """Return set of base gbifIDs already present in the merged chip pool."""
    path = config.CHIPS_META
    if not path.exists():
        print(f"WARN: {path} not found — treating all candidates as new")
        return set()
    ids = set()
    with path.open() as f:
        for row in csv.DictReader(f):
            ids.add(base_id(row["gbifID"]))
    return ids


def filter_present(rows, max_unc, include_blank):
    kept, n_species, n_blank, n_over = [], 0, 0, 0
    for r in rows:
        if (r.get("species") or "").strip().lower() not in TARGET: continue
        n_species += 1
        v = unc_val(r)
        if v is None:
            if include_blank: kept.append(r)
            else: n_blank += 1
        elif v <= max_unc:
            kept.append(r)
        else:
            n_over += 1
    return kept, n_species, n_blank, n_over


def filter_absent(rows):
    return [r for r in rows if (r.get("species") or "").strip().lower() in TARGET]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--p-max-unc-m", type=float, default=DEFAULT_P_MAX_UNC)
    ap.add_argument("--include-blank-unc", action="store_true",
                    help="Keep PRESENT records with blank uncertainty (default: drop).")
    ap.add_argument("--present-out", default=str(DC / "submerged_present_candidates.csv"))
    ap.add_argument("--absent-out",  default=str(DC / "submerged_absent_candidates.csv"))
    args = ap.parse_args()

    print("loading deduped candidate CSVs...")
    p_all = list(csv.DictReader((DC / "deduped_present_candidates.csv").open()))
    a_all = list(csv.DictReader((DC / "deduped_absent_candidates.csv").open()))
    print(f"  deduped_present: {len(p_all)}   deduped_absent: {len(a_all)}")

    print("\nspecies filter (PRESENT with unc filter, ABSENT no unc filter)...")
    p_kept, n_sp_p, n_blank, n_over = filter_present(p_all, args.p_max_unc_m, args.include_blank_unc)
    a_kept = filter_absent(a_all)
    print(f"  PRESENT: {n_sp_p} match species → {len(p_kept)} after unc<=({args.p_max_unc_m})  "
          f"(dropped {n_blank} blank + {n_over} over)")
    print(f"  ABSENT:  {len(a_kept)} match species (no unc filter)")

    print("\nloading already-fetched gbifIDs from merged chip pool...")
    already = load_already_fetched()
    print(f"  {len(already)} unique base gbifIDs already fetched")

    p_new = [r for r in p_kept if r["gbifID"] not in already]
    a_new = [r for r in a_kept if r["gbifID"] not in already]
    print(f"\nafter cross-reference:")
    print(f"  PRESENT new = {len(p_new)}  (skipping {len(p_kept)-len(p_new)} already-fetched)")
    print(f"  ABSENT  new = {len(a_new)}  (skipping {len(a_kept)-len(a_new)} already-fetched)")

    def write(rows, path, label):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fields = list(p_all[0].keys())
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows: w.writerow({k: r.get(k, "") for k in fields})
        print(f"wrote {path}  ({len(rows)} rows, {label})")
        if rows:
            print(f"  by species: {dict(Counter((r.get('species') or '').strip().lower() for r in rows))}")
            print(f"  by source:  {dict(Counter(r.get('source','?') for r in rows))}")

    print()
    write(p_new, args.present_out, "PRESENT, new only")
    print()
    write(a_new, args.absent_out,  "ABSENT, new only")


if __name__ == "__main__":
    main()
