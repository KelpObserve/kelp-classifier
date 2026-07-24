"""build_high_precision_candidates.py — combine the deduped PRESENT+ABSENT
candidates into a single CSV where PRESENT records are restricted to
coordinateUncertaintyInMeters <= P_MAX_UNC_M (default 10 m) and ABSENT records
are kept in full.

Rationale: high-precision PRESENT anchor points (diver GPS) give clean
training signal, while ABSENT observations tend to have blank/loose
uncertainty (marine biology transects log ranges, not points) and we want
to keep them all.

Reads:
  data_collection/deduped_present_candidates.csv
  data_collection/deduped_absent_candidates.csv

Writes:
  data_collection/high_precision_candidates.csv  (combined)
"""
import argparse
import csv
from pathlib import Path

DC = Path(__file__).resolve().parent
DEFAULT_P_MAX_UNC = 10.0


def unc_val(r):
    u = (r.get("coordinateUncertaintyInMeters") or "").strip()
    if not u: return None
    try: return float(u)
    except (ValueError, TypeError): return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--p-max-unc-m", type=float, default=DEFAULT_P_MAX_UNC,
                    help=f"Max coordinateUncertaintyInMeters for PRESENT (default {DEFAULT_P_MAX_UNC}). "
                         "PRESENT records with blank/unparseable uncertainty are excluded.")
    ap.add_argument("--include-blank-unc", action="store_true",
                    help="Include PRESENT records with blank uncertainty (default: exclude).")
    args = ap.parse_args()

    p_all = list(csv.DictReader((DC / "deduped_present_candidates.csv").open()))
    a_all = list(csv.DictReader((DC / "deduped_absent_candidates.csv").open()))
    print(f"deduped_present_candidates.csv: {len(p_all)} rows")
    print(f"deduped_absent_candidates.csv:  {len(a_all)} rows")

    p_kept = []
    n_blank = n_over = 0
    for r in p_all:
        v = unc_val(r)
        if v is None:
            if args.include_blank_unc:
                p_kept.append(r)
            else:
                n_blank += 1
        elif v <= args.p_max_unc_m:
            p_kept.append(r)
        else:
            n_over += 1
    print(f"PRESENT filter (unc <= {args.p_max_unc_m} m, "
          f"blank={'kept' if args.include_blank_unc else 'excluded'}): "
          f"kept {len(p_kept)} (dropped {n_blank} blank + {n_over} over-threshold)")

    combined = p_kept + a_all
    fields = list(p_all[0].keys())  # both files share the same schema
    out = DC / "high_precision_candidates.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in combined: w.writerow({k: r.get(k, "") for k in fields})
    from collections import Counter
    print(f"\nwrote {out.relative_to(DC.parent)}: {len(combined)} rows")
    print(f"  by class:  {dict(Counter(r['occurrenceStatus'] for r in combined))}")
    print(f"  by source: {dict(Counter(r.get('source','?') for r in combined).most_common())}")


if __name__ == "__main__":
    main()
