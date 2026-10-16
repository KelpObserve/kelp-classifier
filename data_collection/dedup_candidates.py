"""Deduplicate configured GBIF, OBIS and JNCC class exports by location and date.

Use build_observation_summary_tables.py for the complete current selection policy.
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
import re
from collections import defaultdict

DC = _path_config.get_path("DATA_COLLECTION_DIR")

MIN_EVENT_YEAR = 2016

SOURCE_PRIORITY = {"gbif": 0, "obis": 1, "jncc": 2}

# id_column, id_prefix
SOURCES = {
    "gbif": ("gbifID",     ""),
    "obis": ("obisID",     "obis:"),
    "jncc": ("sample_ref", "jncc:"),
}


def parse_lat_lon(r):
    try:
        lat = float(r.get("decimalLatitude", "") or 0)
        lon = float(r.get("decimalLongitude", "") or 0)
    except (ValueError, TypeError):
        return None
    if lat == 0 and lon == 0: return None
    return lat, lon


def parse_event_date(r):
    ed = (r.get("eventDate") or "").strip()
    if not ed: return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", ed)
    if not m: return None
    y = int(m.group(1))
    if y < MIN_EVENT_YEAR: return None
    return ed[:10]


def normalize_row(r, src):
    id_col, prefix = SOURCES[src]
    raw_id = (r.get(id_col) or "").strip()
    if not raw_id: return None
    ll = parse_lat_lon(r)
    if ll is None: return None
    lat, lon = ll
    ed = parse_event_date(r)
    if ed is None: return None
    species = (r.get("species") or "").strip()
    unc = (r.get("coordinateUncertaintyInMeters") or "").strip()
    return {
        "gbifID": f"{prefix}{raw_id}",
        "decimalLatitude":  lat,
        "decimalLongitude": lon,
        "eventDate":        ed,
        "species":          species,
        "source":           src,
        "coordinateUncertaintyInMeters": unc,
    }


def load_source(csv_path, src):
    if not csv_path.exists():
        print(f"  {csv_path.name}: MISSING")
        return []
    rows = list(csv.DictReader(csv_path.open()))
    kept = []; dropped = 0
    for r in rows:
        n = normalize_row(r, src)
        if n is None: dropped += 1
        else:         kept.append(n)
    print(f"  {csv_path.name}: {len(rows)} raw -> {len(kept)} kept ({dropped} dropped: bad coord/date/pre-2016/missing id)")
    return kept


def dedup(rows):
    """Group by (round(lat, 4), round(lon, 4), eventDate). Combine species,
    pick source-priority + alphabetical winner as representative."""
    groups = defaultdict(list)
    for r in rows:
        key = (round(r["decimalLatitude"], 4), round(r["decimalLongitude"], 4), r["eventDate"])
        groups[key].append(r)

    out = []
    n_collapsed = 0
    for key, members in groups.items():
        if len(members) > 1: n_collapsed += 1
        # combined species (unique, alphabetical, non-empty)
        species = sorted({m["species"] for m in members if m["species"]})
        combined_species = ", ".join(species)
        # winner
        def pick_key(m): return (SOURCE_PRIORITY[m["source"]], m["gbifID"])
        winner = min(members, key=pick_key)
        row = dict(winner)
        row["species"] = combined_species
        out.append(row)

    print(f"  input rows: {len(rows)}  -> unique groups: {len(groups)}  ({n_collapsed} groups had >=2 members collapsed)")
    return out


def write_csv(rows, out_path, status):
    fields = ["gbifID", "decimalLatitude", "decimalLongitude", "eventDate",
              "species", "occurrenceStatus", "source", "coordinateUncertaintyInMeters"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in fields}
            row["occurrenceStatus"] = status
            w.writerow(row)
    print(f"  wrote {out_path} ({len(rows)} rows)")


def main():
    print("=== PRESENT ===")
    p_rows = []
    p_rows += load_source(_path_config.get_path("GBIF_PRESENT"), "gbif")
    p_rows += load_source(_path_config.get_path("OBIS_PRESENT"), "obis")
    p_rows += load_source(_path_config.get_path("JNCC_PRESENT"),        "jncc")
    p_dedup = dedup(p_rows)
    write_csv(p_dedup, DC / "deduped_present_candidates.csv", "PRESENT")

    print()
    print("=== ABSENT ===")
    a_rows = []
    a_rows += load_source(_path_config.get_path("GBIF_ABSENT"), "gbif")
    a_rows += load_source(_path_config.get_path("OBIS_ABSENT"), "obis")
    a_rows += load_source(_path_config.get_path("JNCC_ABSENT"),        "jncc")
    a_dedup = dedup(a_rows)
    write_csv(a_dedup, DC / "deduped_absent_candidates.csv", "ABSENT")


if __name__ == "__main__":
    main()
