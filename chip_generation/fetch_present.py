"""Thin wrapper: fetch PRESENT chips (strict QC, 1 scene per record).

Pre-filters candidates by coord uncertainty (records with declared
coordinateUncertaintyInMeters > MAX_UNC_M are dropped) so the model isn't
trained on labels whose true location might be outside its prediction
region (~115 m half-width). Records with missing uncertainty are kept and
filtered later by the QC water-in-region check.

Optional: --species-targets <json_path> — stop fetching when each target
species reaches its chip count target. JSON is {species_name: target_count}.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chip_common import QC_STRICT_PRESENT, fetch_chips

MAX_UNC_M = 115.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates")
    ap.add_argument("out_dir")
    ap.add_argument("--species-targets", type=str, default=None,
                    help="JSON path with {species: target_chip_count} for early stop")
    args = ap.parse_args()

    raw = list(csv.DictReader(open(args.candidates)))
    kept, dropped = [], 0
    for r in raw:
        try: u = float(r.get("coordinateUncertaintyInMeters","") or 0)
        except (ValueError, TypeError): u = 0
        if u > MAX_UNC_M:
            dropped += 1; continue
        kept.append(r)
    print(f"pre-filter: kept {len(kept)}/{len(raw)} candidates (dropped {dropped} with unc > {MAX_UNC_M}m)")

    targets = None
    if args.species_targets:
        targets = json.load(open(args.species_targets))
        print(f"species targets: {targets}")

    fetch_chips(kept, QC_STRICT_PRESENT, args.out_dir,
                class_tag="present", max_scenes_per_record=1,
                species_targets=targets)
