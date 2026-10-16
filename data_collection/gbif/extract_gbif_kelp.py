"""Extract kelp PRESENT + ABSENT from the GBIF Europe master download.

Input:  the configured data location (already downloaded from GBIF)
Output: kelp_gbif_europe_present.csv, kelp_gbif_europe_absent.csv

If you need to re-download the raw data, use GBIF's occurrence download interface:
  https://www.gbif.org/occurrence/download
  Filter: taxonKey ∈ our kelp species, continent = EUROPE
  Format: SIMPLE_CSV
Then place the file at the configured data location
"""

# Discover the configuration module; data locations come from config.json.
import sys as _path_sys
from pathlib import Path as _ConfigPath
_path_root = _ConfigPath(__file__).resolve().parent
while not (_path_root / "example_config.json").is_file() and _path_root != _path_root.parent:
    _path_root = _path_root.parent
_path_sys.path.insert(0, str(_path_root))
import config as _path_config
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import is_kelp_species, stats

HERE = _path_config.get_path("GBIF_DIR")
SRC = _path_config.get_path("GBIF_MASTER")


def main():
    all_rows = list(csv.DictReader(SRC.open()))
    kelp = [r for r in all_rows if is_kelp_species(r.get("species", ""))]
    print(f"master rows: {len(all_rows)}, kelp rows: {len(kelp)}", flush=True)
    fields = list(all_rows[0].keys())

    present = [r for r in kelp if r.get("occurrenceStatus") == "PRESENT"]
    absent = [r for r in kelp if r.get("occurrenceStatus") == "ABSENT"]
    for name, subset in [("present", present), ("absent", absent)]:
        out = _path_config.get_path("GBIF_" + name.upper())
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(subset)
        print(f"wrote {out}: {len(subset)}", flush=True)

    # Stats
    rows = [stats(present, "GBIF Europe (kelp)", "PRESENT"),
            stats(absent, "GBIF Europe (kelp)", "ABSENT")]
    return rows


if __name__ == "__main__":
    rows = main()
    for r in rows:
        print(f"  {r['source']:35s}  {r['class']:>7s}  N={r['n_records']:>6d}  post2016={r['year_post2016']:>5d}")
