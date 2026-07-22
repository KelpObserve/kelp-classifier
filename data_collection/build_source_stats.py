"""Aggregate stats across all data sources into a single CSV."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import stats, write_stats_csv, STATS_FIELDS

HERE = Path(__file__).resolve().parent

SOURCE_FILES = [
    ("GBIF Europe (kelp)", "PRESENT", HERE / "gbif/kelp_gbif_europe_present.csv"),
    ("GBIF Europe (kelp)", "ABSENT",  HERE / "gbif/kelp_gbif_europe_absent.csv"),
    ("OBIS Europe (kelp; 2016+)",         "PRESENT", HERE / "obis/kelp_obis_europe_present.csv"),
    ("OBIS Europe (kelp; 2016+)",         "ABSENT",  HERE / "obis/kelp_obis_europe_absent.csv"),
    ("JNCC Marine Recorder v52 (2016+)",  "PRESENT", HERE / "jncc/kelp_jncc_present.csv"),
    ("JNCC Marine Recorder v52 (2016+)",  "ABSENT",  HERE / "jncc/kelp_jncc_absent.csv"),
]


def main():
    all_stats = []
    for src, klass, path in SOURCE_FILES:
        if not path.exists():
            print(f"missing: {path}")
            continue
        rows = list(csv.DictReader(path.open()))
        s = stats(rows, src, klass)
        all_stats.append(s)
        print(f"  {src:35s}  {klass:>7s}  N={s['n_records']:>6d}  post2016={s['year_post2016']:>5d}")

    out = HERE / "source_statistics.csv"
    write_stats_csv(all_stats, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
