"""Build June/July/September S2 field events using explicit source matches.

Requires openpyxl. Retains measurement dates/depths and source coordinates;
fails before writing if source keys, labels or coordinates do not agree.
"""
import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def number(value):
    return int(str(value).strip().removeprefix("T").removeprefix("Q").removesuffix("m"))


def source_rows(path, sheet):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        return [(i, row) for i, row in enumerate(workbook[sheet].values, 1)
                if i > 1 and row[0] is not None]
    finally:
        workbook.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--coordinates", type=Path)
    parser.add_argument("--september-coordinates", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.measurements is None:
        args.measurements = config.get_path("FIELD_MEASUREMENTS")
    if args.coordinates is None:
        args.coordinates = config.get_path("JUNE_JULY_COORDINATES")
    if args.september_coordinates is None:
        args.september_coordinates = config.get_path("SEPTEMBER_COORDINATES")
    if args.output is None:
        args.output = config.get_path("FIELD_OBSERVATIONS")
    measurements = defaultdict(list)
    for _, row in source_rows(args.measurements, "Folha1"):
        key = (row[0].date().isoformat(), *(number(x) for x in row[1:4]))
        measurements[key].append(str(row[4] or "").strip().lower())
    gps = {}
    for row_number, row in source_rows(args.coordinates, "Transetos Jul26"):
        key = (row[0].date().isoformat(), *(number(x) for x in row[1:4]))
        if key in gps:
            raise ValueError(f"Duplicate GPS key: {key}")
        gps[key] = (float(row[4]), float(row[5]), row_number, args.coordinates.name, key[0])
    with sqlite3.connect(f"file:{args.september_coordinates.resolve()}?mode=ro", uri=True) as connection:
        rows = connection.execute('SELECT fid, Date, Depth, "Transect number", Quadrat, "GPS Lat", "GPS Long" FROM transects WHERE Date = ?', ("01/set",))
        for fid, date, depth, transect, quadrat, lat, lon in rows:
            key = ("2026-09-01", number(depth), number(transect), number(quadrat))
            if key in gps:
                raise ValueError(f"Duplicate GPS key: {key}")
            gps[key] = (float(lat), float(lon), fid, args.september_coordinates.name, date)
    fields = ["lat", "lon", "date", "depth", "T", "Q", "label"]
    events = [dict(date=key[0], depth=key[1], T=key[2], Q=key[3],
                   label=int(any(species != "empty" for species in records)))
              for key, records in sorted(measurements.items())]
    used, seen = set(), set()
    for event in events:
        key = (event["date"], *(int(event[f]) for f in ("depth", "T", "Q")))
        date, depth, transect, quadrat = key
        if date == "2026-06-02":
            gps_date, gps_depth, match = date, depth, "exact"
        elif date in ("2026-07-06", "2026-07-07") and depth in (7, 16):
            gps_date = {"2026-07-06": "2026-07-07", "2026-07-07": "2026-07-08"}[date]
            gps_depth = {7: 7, 16: 17}[depth]
            match = "july_date_alignment" if depth == 7 else "july_date_and_depth_alignment"
        elif date == "2026-09-02" and depth in (6, 15):
            gps_date, gps_depth = "2026-09-01", {6: 7, 15: 17}[depth]
            match = "september_date_and_depth_alignment"
        else:
            raise ValueError(f"No explicit matching rule for {key}")
        gps_key = (gps_date, gps_depth, transect, quadrat)
        if key in seen or gps_key in used:
            raise ValueError(f"Match is not one-to-one: {key} -> {gps_key}")
        species = measurements[key]
        if not species or any(not s for s in species):
            raise ValueError(f"Missing species label: {key}")
        label = int(any(s != "empty" for s in species))
        lat, lon, row_number, source, source_date = gps[gps_key]
        if label != int(event["label"]) or ("lat" in event and max(abs(lat - float(event["lat"])), abs(lon - float(event["lon"]))) > 1e-9):
            raise ValueError(f"Existing label or coordinates disagree: {key}")
        if "lat" not in event:
            event.update(lat=lat, lon=lon)
        event.update(gps_date=gps_date, gps_depth_m=gps_depth,
                     coordinate_match=match, gps_source=source, gps_source_date=source_date,
                     gps_source_row_or_fid=row_number)
        seen.add(key)
        used.add(gps_key)
    if seen != set(measurements) or used != set(gps):
        raise ValueError("Unmatched measurement events or GPS rows")
    for field in ("gps_date", "gps_depth_m", "coordinate_match", "gps_source", "gps_source_date", "gps_source_row_or_fid"):
        if field not in fields:
            fields.append(field)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(events)
    report = {
        "events": len(events),
        "matches": dict(Counter(e["coordinate_match"] for e in events)),
        "sources": {name: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "sheet": sheet}
                    for name, path, sheet in (("measurements", args.measurements, "Folha1"),
                                              ("coordinates", args.coordinates, "Transetos Jul26"),
                                              ("september_coordinates", args.september_coordinates, "transects"))},
        "note": "date/depth retain measurement values; gps_date/gps_depth_m retain coordinate source values. Explicit alignment does not establish which source date or depth is correct. September GPS year is taken from the named 2026 source; original date text is retained in gps_source_date.",
    }
    args.output.with_suffix(".provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
