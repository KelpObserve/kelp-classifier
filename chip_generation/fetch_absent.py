"""Thin wrapper: fetch ABSENT chips within ±30 days (up to eight scenes).

Modes:
  --scenes N       : up to N scenes per record (default 8 for multi-scene)
  --single         : single scene per record (equivalent to --scenes 1)

If --exclude-existing <chips_metadata.csv> is passed, previously fetched
sceneIDs are skipped so only NEW scenes are added.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chip_generation.chip_common import QC_RELAXED_ABSENT, fetch_chips

# No coordinate-uncertainty cutoff: very few ABSENT candidates are available.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", help="path to candidate CSV")
    ap.add_argument("out_dir", help="output directory")
    ap.add_argument("--max-chips", type=int, default=None,
                    help="stop when this many OK chips have been written across all workers")
    ap.add_argument("--scenes", type=int, default=8, help="max scenes per record (default: 8)")
    ap.add_argument("--single", action="store_true", help="shortcut for --scenes 1")
    ap.add_argument("--exclude-existing", type=str,
                    help="path to previously-fetched chips_metadata.csv "
                         "to derive sceneIDs to skip per gbifID")
    ap.add_argument("--disable-snap", action="store_true",
                    help="hard-off snap-to-water for this run")
    ap.add_argument("--workers", type=int, default=6,
                    help="concurrent worker threads (default: 6)")
    args = ap.parse_args()

    if args.single: args.scenes = 1
    if not 1 <= args.scenes <= 8:
        ap.error("--scenes must be between 1 and 8")
    with open(args.candidates) as stream:
        candidates = list(csv.DictReader(stream))
    print(f"loaded {len(candidates)} candidates (no unc pre-filter for ABSENT)")

    for r in candidates:
        r["allowed_max_days"] = 30

    exclude = defaultdict(set)
    if args.exclude_existing:
        for r in csv.DictReader(open(args.exclude_existing)):
            gid = r["gbifID"].rsplit("_s", 1)[0]
            if r.get("sceneID"):
                exclude[gid].add(r["sceneID"])
        print(f"loaded {sum(len(v) for v in exclude.values())} existing sceneIDs "
              f"across {len(exclude)} gbifIDs to skip")

    fetch_chips(candidates, QC_RELAXED_ABSENT, args.out_dir,
                class_tag="absent", max_scenes_per_record=args.scenes,
                exclude_scene_ids=dict(exclude),
                disable_snap=args.disable_snap,
                max_chips=args.max_chips,
                workers=args.workers)


if __name__ == "__main__":
    main()
