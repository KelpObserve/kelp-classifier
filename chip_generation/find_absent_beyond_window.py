"""Stage 1b (new rule): for each ABSENT chip in the current merged dataset,
check whether |sceneDate − eventDate| exceeds its allowed_max_days
(from absent_windows.json). Chips that exceed → flagged for drop.

If a chip's gbifID isn't in absent_windows.json (missing from raw sources),
fall back to the default 30-day cap (strict, safer for label quality).
"""
import csv, json
import datetime as dt
from collections import Counter
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

WIN = config.ABSENT_WINDOWS_JSON
META = config.CHIPS_META
OUT  = config.TMP_DIR / "absent_beyond_window.csv"

DEFAULT_DAYS = 30

windows = json.load(WIN.open())
rows = list(csv.DictReader(META.open()))
assert rows and "species" in rows[0]

flagged = []
counts = {"absent_total": 0, "missing_from_windows": 0, "within_window": 0,
          "beyond_window": 0}

for i, r in enumerate(rows):
    if r.get("occurrenceStatus") != "ABSENT": continue
    counts["absent_total"] += 1
    base_gid = r["gbifID"].rsplit("_s", 1)[0]
    allowed = windows.get(base_gid)
    if allowed is None:
        counts["missing_from_windows"] += 1
        allowed = DEFAULT_DAYS
    try:
        ev = dt.date.fromisoformat(r["eventDate"][:10])
        sc = dt.date.fromisoformat(r["sceneDate"][:10])
    except (ValueError, KeyError):
        continue
    days_off = abs((sc - ev).days)
    if days_off <= allowed:
        counts["within_window"] += 1
        continue
    counts["beyond_window"] += 1
    flagged.append({
        "row": i, "gbifID": r["gbifID"], "base_gbifID": base_gid,
        "species": r.get("species", ""),
        "eventDate": r["eventDate"][:10], "sceneDate": r["sceneDate"][:10],
        "sceneID": r.get("sceneID", ""),
        "days_off": days_off, "allowed_max_days": allowed,
    })

print(f"ABSENT chips total:            {counts['absent_total']}")
print(f"  missing from windows lookup: {counts['missing_from_windows']} (fell back to 30)")
print(f"  within window (kept):        {counts['within_window']}")
print(f"  beyond window (drop):        {counts['beyond_window']}")

by_species = Counter(p["species"] for p in flagged)
print(f"\ntop 10 species affected:")
for sp, n in by_species.most_common(10):
    print(f"  {n:>4}  {sp}")

by_allowed = Counter(p["allowed_max_days"] for p in flagged)
print(f"\nby allowed_max_days: {dict(by_allowed)}")

if flagged:
    OUT.parent.mkdir(exist_ok=True, parents=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(flagged[0].keys()))
        w.writeheader()
        for p in flagged: w.writerow(p)
    print(f"\nwrote {OUT}")
