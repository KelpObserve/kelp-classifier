"""Shared functions for kelp record filtering, dedup, and statistics.

Used by:
  data_collection/gbif/extract_gbif_kelp.py
  data_collection/obis/extract_obis_kelp.py
  data_collection/jncc/extract_jncc_kelp.py
"""
import csv
from collections import Counter
from pathlib import Path


KELP_SPECIES = [
    # 7 target species for European kelp detection + 3 synonyms
    "Laminaria hyperborea", "Laminaria digitata", "Saccharina latissima",
    "Alaria esculenta", "Saccorhiza polyschides", "Laminaria ochroleuca",
    "Chorda filum", "Undaria pinnatifida", "Phyllariopsis brevipes",
    "Laminaria saccharina",  # synonym for S. latissima
]

# Europe bounding box (rough) — used to filter records with valid coords
EUROPE_BBOX = dict(lat_min=30, lat_max=72, lon_min=-25, lon_max=45)


def is_kelp_species(scientific_name: str) -> bool:
    """True if the scientific name starts with one of our target kelp species."""
    if not scientific_name: return False
    return any(scientific_name.startswith(k) for k in KELP_SPECIES)


def in_europe(lat, lon) -> bool:
    try:
        lat = float(lat); lon = float(lon)
    except (ValueError, TypeError):
        return False
    return (EUROPE_BBOX["lat_min"] <= lat <= EUROPE_BBOX["lat_max"] and
            EUROPE_BBOX["lon_min"] <= lon <= EUROPE_BBOX["lon_max"])


def has_valid_coord(row, latk="decimalLatitude", lonk="decimalLongitude"):
    try:
        float(row.get(latk, "")); float(row.get(lonk, ""))
        return True
    except (ValueError, TypeError):
        return False


def dedup_key(row, latk="decimalLatitude", lonk="decimalLongitude",
              datek="eventDate", spk="species", precision=3):
    """Return a canonical key (species, lat, lon, date) for deduplication."""
    try:
        lat = round(float(row.get(latk, "")), precision)
        lon = round(float(row.get(lonk, "")), precision)
    except (ValueError, TypeError):
        return None
    sp = (row.get(spk) or row.get("scientificName") or "").strip().lower()
    date = (row.get(datek) or "")[:10]
    if not sp or not date: return None
    return (sp, lat, lon, date)


def stats(records, source_name: str, klass: str) -> dict:
    """Compute descriptive stats for a set of records."""
    yh = Counter((r.get("eventDate") or "")[:4] for r in records)
    y_valid = [y for y in yh if y.isdigit()]
    sp_ct = Counter(r.get("species") or r.get("scientificName") or "?" for r in records)
    lats = []; lons = []
    for r in records:
        try:
            lats.append(float(r["decimalLatitude"])); lons.append(float(r["decimalLongitude"]))
        except (KeyError, ValueError, TypeError):
            pass
    return {
        "source": source_name,
        "class": klass,
        "n_records": len(records),
        "n_species": len(sp_ct),
        "top_species": ";".join(f"{k}({v})" for k, v in sp_ct.most_common(5)),
        "year_min": min(y_valid) if y_valid else "",
        "year_max": max(y_valid) if y_valid else "",
        "year_post2016": sum(v for k, v in yh.items() if k.isdigit() and int(k) >= 2016),
        "geo_bbox": (f"lat[{min(lats):.2f},{max(lats):.2f}] "
                     f"lon[{min(lons):.2f},{max(lons):.2f}]") if lats else "",
    }


STATS_FIELDS = ["source","class","n_records","n_species","top_species",
                "year_min","year_max","year_post2016","geo_bbox"]


def write_stats_csv(stats_rows: list, out_path: Path) -> None:
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATS_FIELDS)
        w.writeheader()
        w.writerows(stats_rows)
