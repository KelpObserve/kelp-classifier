"""build_high_precision_candidates.py — build a PRESENT-only high-precision
candidate list, restricted to coordinateUncertaintyInMeters <= P_MAX_UNC_M
(default 10 m).

Rationale: high-precision PRESENT anchor points (diver GPS) give clean
training signal. ABSENT is handled by a separate pipeline
(deduped_absent_candidates.csv → fetch_absent.py) and must NOT be mixed
in here: fetch_present.py stamps class_tag=present on every row it
processes, so any ABSENT smuggled into this file would end up
double-fetched and mislabeled.

Reads:
  data_collection/deduped_present_candidates.csv

Writes:
  data_collection/high_precision_candidates.csv  (PRESENT only)
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
    print(f"deduped_present_candidates.csv: {len(p_all)} rows")

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

    fields = list(p_all[0].keys())

    from collections import Counter
    def write(rows, path, label):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows: w.writerow({k: r.get(k, "") for k in fields})
        print(f"wrote {path.relative_to(DC.parent)}: {len(rows)} rows  ({label})")
        print(f"  by class:  {dict(Counter(r['occurrenceStatus'] for r in rows))}")
        print(f"  by source: {dict(Counter(r.get('source','?') for r in rows).most_common())}")

    # (1) Full high-precision set (all valid eventDates, PRESENT only)
    write(p_kept,
          DC / "high_precision_candidates.csv",
          f"PRESENT ≤ {args.p_max_unc_m} m, ≥ 2016")

    # (2) PNEO-era subset: eventDate ≥ 2021-04-01 (PNEO 3 first light)
    import re
    def year_month(r):
        m = re.match(r"(\d{4})-(\d{2})", (r.get("eventDate") or "").strip())
        if not m: return (0, 0)
        return (int(m.group(1)), int(m.group(2)))
    pneo_cutoff = (2021, 4)
    pneo_rows = [r for r in p_kept if year_month(r) >= pneo_cutoff]
    print()
    write(pneo_rows,
          DC / "high_precision_pneo_era_candidates.csv",
          "same, restricted to eventDate ≥ 2021-04 (PNEO 3+ era)")


if __name__ == "__main__":
    main()
