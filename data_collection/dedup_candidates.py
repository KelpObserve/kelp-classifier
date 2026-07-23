"""dedup_candidates.py — collapse the 6 raw source CSVs (GBIF/OBIS/JNCC × P/A)
into two deduplicated candidate CSVs (present + absent), grouping any rows
that share (occurrenceStatus, round(lat, 4), round(lon, 4), eventDate) even
across sources.

Rules:
  1. Key = (occurrenceStatus, round(decimalLatitude, 4), round(decimalLongitude, 4), eventDate[:10]).
     4-decimal rounding = ~11 m precision; tighter than the 50 m location-safety
     radius, so we only merge rows that really point at the same spot on the
     same day.
  2. Species names of all rows in a group are combined into a comma-separated
     list (unique, alphabetically sorted).
  3. Representative row: preferred source order GBIF > OBIS > JNCC, then
     alphabetically-first ID within the winning source.
  4. Rows with missing/blank eventDate or lat/lon are dropped.
  5. Filter: eventDate must be >= 2016-01-01 (Sentinel-2 coverage window).

Output columns (fetch_present/fetch_absent expect these):
  gbifID (source-prefixed for OBIS/JNCC), decimalLatitude, decimalLongitude,
  eventDate, species, occurrenceStatus, source, coordinateUncertaintyInMeters.

Reads:
  data_collection/gbif/kelp_gbif_europe_{present,absent}.csv
  data_collection/obis/kelp_obis_europe_{present,absent}.csv
  data_collection/jncc/kelp_jncc_{present,absent}.csv

Writes:
  data_collection/deduped_present_candidates.csv
  data_collection/deduped_absent_candidates.csv
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DC = REPO / "data_collection"

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
    print(f"  wrote {out_path.relative_to(REPO)} ({len(rows)} rows)")


def main():
    print("=== PRESENT ===")
    p_rows = []
    p_rows += load_source(DC / "gbif" / "kelp_gbif_europe_present.csv", "gbif")
    p_rows += load_source(DC / "obis" / "kelp_obis_europe_present.csv", "obis")
    p_rows += load_source(DC / "jncc" / "kelp_jncc_present.csv",        "jncc")
    p_dedup = dedup(p_rows)
    write_csv(p_dedup, DC / "deduped_present_candidates.csv", "PRESENT")

    print()
    print("=== ABSENT ===")
    a_rows = []
    a_rows += load_source(DC / "gbif" / "kelp_gbif_europe_absent.csv", "gbif")
    a_rows += load_source(DC / "obis" / "kelp_obis_europe_absent.csv", "obis")
    a_rows += load_source(DC / "jncc" / "kelp_jncc_absent.csv",        "jncc")
    a_dedup = dedup(a_rows)
    write_csv(a_dedup, DC / "deduped_absent_candidates.csv", "ABSENT")


if __name__ == "__main__":
    main()
