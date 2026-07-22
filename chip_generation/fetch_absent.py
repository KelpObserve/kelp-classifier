"""Thin wrapper: fetch ABSENT chips (relaxed QC).

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
# Locate repo root by walking up until config.py is found
import sys as _sys
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chip_common import QC_RELAXED_ABSENT, fetch_chips

# No coord-uncertainty pre-filter for ABSENT. An observer's "no kelp seen"
# over an area is robust to positional imprecision — unlike a PRESENT record,
# which pinpoints a specific kelp bed. Originally added for P/A symmetry
# (reviewer C2) but reverted because it costs ABSENT coverage without
# meaningfully improving label quality.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidates", help="path to candidate CSV")
    ap.add_argument("out_dir", help="output directory")
    ap.add_argument("--scenes", type=int, default=8, help="max scenes per record (default: 8)")
    ap.add_argument("--single", action="store_true", help="shortcut for --scenes 1")
    ap.add_argument("--exclude-existing", type=str,
                    help="path to previously-fetched chips_metadata.csv "
                         "to derive sceneIDs to skip per gbifID")
    ap.add_argument("--absent-windows", type=str,
                    default=str(config.ABSENT_WINDOWS_JSON),
                    help="path to absent_windows.json — per-gbifID "
                         "allowed_max_days lookup produced by "
                         "01_annotate_absent_windows.py. Missing entries "
                         "default to 30 (strict).")
    ap.add_argument("--disable-snap", action="store_true",
                    help="hard-off snap-to-water for this run (e.g. A2 top-ups)")
    args = ap.parse_args()

    if args.single: args.scenes = 1
    candidates = list(csv.DictReader(open(args.candidates)))
    print(f"loaded {len(candidates)} candidates (no unc pre-filter for ABSENT)")

    # Stamp allowed_max_days on each candidate from the pre-computed lookup.
    # Missing → 30 (strict default), so a stale/absent lookup doesn't silently
    # widen the fetch window.
    import json as _json
    try:
        windows = _json.load(open(args.absent_windows))
        print(f"loaded {len(windows)} absent-window annotations from {args.absent_windows}")
    except FileNotFoundError:
        print(f"WARN: {args.absent_windows} not found — every candidate defaults to allowed_max_days=30")
        windows = {}
    default_days = 30
    for r in candidates:
        gid = r.get("gbifID", "")
        r["allowed_max_days"] = int(windows.get(gid, default_days))
    n_extended = sum(1 for r in candidates if r["allowed_max_days"] == 60)
    n_strict   = len(candidates) - n_extended
    print(f"per-record windows: {n_strict} at ±30 (strict), {n_extended} at ±60 (extended)")

    exclude = defaultdict(set)
    if args.exclude_existing:
        for r in csv.DictReader(open(args.exclude_existing)):
            gid = r["gbifID"].rsplit("_s", 1)[0]
            if r.get("sceneID"):
                exclude[gid].add(r["sceneID"])
        print(f"loaded {sum(len(v) for v in exclude.values())} existing sceneIDs "
              f"across {len(exclude)} gbifIDs to skip")

    fetch_chips(candidates, QC_RELAXED_ABSENT, args.out_dir,
                class_tag="absent_real", max_scenes_per_record=args.scenes,
                exclude_scene_ids=dict(exclude),
                disable_snap=args.disable_snap)


if __name__ == "__main__":
    main()
