"""Extract kelp PRESENT + ABSENT from OBIS via their API.

Endpoint: configs/services.json, obis_occurrences.
No API key required. Rate-limits are generous for occurrence queries.

Output: kelp_obis_europe_present.csv, kelp_obis_europe_absent.csv
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
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import KELP_SPECIES

BASE = _path_config.get_service("obis_occurrences")
EUROPE_WKT = "POLYGON ((-25 30, 45 30, 45 72, -25 72, -25 30))"
START_DATE = "2016-01-01"  # S2 era

# Fields to keep in the output CSVs (OBIS returns many more)
OUT_FIELDS = [
    "obisID","aphiaID","species","scientificName","decimalLatitude","decimalLongitude",
    "coordinateUncertaintyInMeters","eventDate","country","countryCode","stateProvince",
    "waterBody","basisOfRecord","occurrenceStatus","individualCount",
    "datasetName","datasetKey","recordedBy","minimumDepthInMeters",
]


def fetch_species(sci: str, absence_flag: bool) -> list:
    """Paginate through all OBIS occurrences of `sci` in Europe from START_DATE."""
    records = []
    after = None
    while True:
        params = {
            "scientificname": sci,
            "geometry": EUROPE_WKT,
            "startdate": START_DATE,
            "size": 500,
        }
        if absence_flag:
            params["absence"] = "true"
        if after is not None:
            params["after"] = after
        url = BASE + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as r:
            j = json.load(r)
        batch = j.get("results", [])
        records.extend(batch)
        if len(batch) < 500:
            break
        after = batch[-1].get("id")
        if after is None: break
        time.sleep(0.2)
    return records


def project(o: dict) -> dict:
    """Map an OBIS record to our canonical schema."""
    return {
        "obisID": o.get("id"),
        "aphiaID": o.get("aphiaID"),
        "species": o.get("scientificName"),
        "scientificName": o.get("scientificName"),
        "decimalLatitude": o.get("decimalLatitude"),
        "decimalLongitude": o.get("decimalLongitude"),
        "coordinateUncertaintyInMeters": o.get("coordinateUncertaintyInMeters"),
        "eventDate": o.get("eventDate"),
        "country": o.get("country"),
        "countryCode": o.get("countryCode"),
        "stateProvince": o.get("stateProvince"),
        "waterBody": o.get("waterBody"),
        "basisOfRecord": o.get("basisOfRecord"),
        "occurrenceStatus": ("ABSENT" if o.get("absence") else "PRESENT"),
        "individualCount": o.get("individualCount"),
        "datasetName": o.get("datasetName"),
        "datasetKey": o.get("dataset_id"),
        "recordedBy": o.get("recordedBy"),
        "minimumDepthInMeters": o.get("minimumDepthInMeters"),
    }


def main():
    # The seven marine target species include Chorda filum.
    target = KELP_SPECIES[:7]
    for klass, absence in [("present", False), ("absent", True)]:
        all_records = []
        for sp in target:
            recs = fetch_species(sp, absence)
            print(f"  OBIS {klass:>7s}  {sp}: {len(recs)}", flush=True)
            all_records.extend(recs)
        rows = [project(o) for o in all_records]
        out = _path_config.get_path("OBIS_" + klass.upper())
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"wrote {out}: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
