"""check_pneo_coverage.py — bulk-check OneAtlas archive for PNEO (and
optionally Pléiades 1A/1B) coverage over our training candidates.

Uses Airbus's OneAtlas Data API. Search is FREE — you only pay if you
actually order/download imagery. Requires:

  - An Airbus OneAtlas account (register at oneatlas.airbus.com)
  - An API key (generated in account settings)

Set the API key in env:
  export ONEATLAS_API_KEY=<your-key>

Output CSV: one row per unique location cell with columns:
  cell_id, lat, lon, n_representative_gbif_ids, n_scenes,
  best_cloud_pct, best_scene_date, best_scene_id, best_constellation

Usage:
  python check_pneo_coverage.py \
      --candidates data_collection/deduped_present_candidates.csv \
      --target-species "saccorhiza polyschides,laminaria hyperborea,laminaria ochroleuca,phyllariopsis brevipes" \
      --date-from 2021-04-01 --date-to 2026-12-31 \
      --max-cloud 15 \
      --constellations PNEO,PHR \
      --cell-deg 0.01 \
      --out data_collection/pneo_coverage.csv
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

TOKEN_URL   = "https://authenticate.foundation.api.oneatlas.airbus.com/auth/realms/IDP/protocol/openid-connect/token"
SEARCH_URL  = "https://search.foundation.api.oneatlas.airbus.com/api/v2/opensearch"

# Airbus constellation codes as used in the OneAtlas /opensearch API
CONSTELLATION_MAP = {
    "PNEO": "PNEO",         # Pléiades Neo 3/4/5/6 (0.30 m PAN, 1.2 m MS)
    "PHR":  "PHR",          # Pléiades 1A/1B (0.5 m PAN, 2 m MS)
    "SPOT": "SPOT",         # SPOT 6/7 (1.5 m PAN, 6 m MS)
}


def get_token(api_key):
    """OAuth2 client_credentials-style grant using OneAtlas API key."""
    r = requests.post(
        TOKEN_URL,
        data={"grant_type": "api_key", "client_id": "IDP", "apikey": api_key},
        timeout=30,
    )
    if r.status_code != 200:
        raise RuntimeError(f"auth failed {r.status_code}: {r.text[:300]}")
    return r.json()["access_token"]


def search_cell(token, lon, lat, radius_km, date_from, date_to, max_cloud, constellations):
    """Search OneAtlas archive over a small bbox around (lon, lat).

    Returns list of scenes: [{acquisitionDate, cloudCover, constellation,
    id, ...}, ...].
    """
    dl = radius_km / 111.0
    dp = radius_km / (111.0 * abs(pow(1.0 - (lat / 90.0) ** 2, 0.5) or 1e-9))  # lon degrees at this lat
    # Simpler: use cos(lat) directly
    import math
    dp = radius_km / (111.0 * max(0.05, math.cos(math.radians(lat))))
    bbox = f"{lon - dp},{lat - dl},{lon + dp},{lat + dl}"

    params = {
        "bbox": bbox,
        "acquisitionDate": f"[{date_from}T00:00:00Z,{date_to}T23:59:59Z]",
        "cloudCover": f"[0,{max_cloud}]",
        "constellation": ",".join(constellations),
        "itemsPerPage": 20,
        "sortBy": "-acquisitionDate",
    }
    headers = {"Authorization": f"Bearer {token}", "Cache-Control": "no-cache"}
    r = requests.get(SEARCH_URL, headers=headers, params=params, timeout=60)
    if r.status_code == 401:
        raise RuntimeError("401 — token expired, re-auth needed")
    if r.status_code != 200:
        raise RuntimeError(f"search {r.status_code}: {r.text[:300]}")
    return r.json().get("features", [])


def load_candidates(csv_path, target_species):
    """Return list of (gbifID, lat, lon) filtered to target species (or all)."""
    rows = []
    target = {s.strip().lower() for s in target_species.split(",")} if target_species else None
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            sp = (r.get("species") or "").strip().lower()
            if target and not any(t in {x.strip() for x in sp.split(",")} for t in target):
                continue
            try:
                rows.append((r["gbifID"], float(r["decimalLatitude"]), float(r["decimalLongitude"])))
            except (ValueError, KeyError):
                pass
    return rows


def dedupe_by_cell(rows, cell_deg):
    """Group by (round(lat/cell_deg), round(lon/cell_deg)) → cell centre.
    Returns list of (cell_id, lat_centre, lon_centre, [gbifIDs])."""
    by_cell = defaultdict(list)
    for gid, lat, lon in rows:
        key = (round(lat / cell_deg), round(lon / cell_deg))
        by_cell[key].append((gid, lat, lon))
    out = []
    for i, (key, members) in enumerate(sorted(by_cell.items())):
        lat_c = sum(m[1] for m in members) / len(members)
        lon_c = sum(m[2] for m in members) / len(members)
        out.append((f"cell_{i:05d}", lat_c, lon_c, [m[0] for m in members]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--target-species", default="saccorhiza polyschides,laminaria hyperborea,laminaria ochroleuca,phyllariopsis brevipes",
                    help="Comma-separated species to keep (empty = all)")
    ap.add_argument("--date-from", default="2021-04-01", help="PNEO 3 first light was 2021-04")
    ap.add_argument("--date-to",   default="2026-12-31")
    ap.add_argument("--max-cloud", type=float, default=15.0)
    ap.add_argument("--constellations", default="PNEO",
                    help="Comma-separated: PNEO, PHR (Pléiades), SPOT")
    ap.add_argument("--cell-deg", type=float, default=0.01,
                    help="Dedupe cell size in degrees (~1 km at mid-latitudes)")
    ap.add_argument("--radius-km", type=float, default=2.0)
    ap.add_argument("--rate-sleep-s", type=float, default=0.2,
                    help="Sleep between API calls to stay under rate limits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only check first N cells (for a smoke test)")
    args = ap.parse_args()

    api_key = os.environ.get("ONEATLAS_API_KEY")
    if not api_key:
        sys.exit("ERROR: set ONEATLAS_API_KEY environment variable to your OneAtlas API key")

    print("loading candidates...")
    rows = load_candidates(args.candidates, args.target_species)
    print(f"  {len(rows)} candidate rows (target species = {args.target_species!r})")

    cells = dedupe_by_cell(rows, args.cell_deg)
    print(f"  → {len(cells)} unique {args.cell_deg}° cells")
    if args.limit:
        cells = cells[:args.limit]
        print(f"  limited to first {len(cells)} for this run")

    print("authenticating with OneAtlas...")
    token = get_token(api_key)
    token_age = time.time()

    constellations = [c.strip() for c in args.constellations.split(",")]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.out, "w", newline="")
    w = csv.writer(out_f)
    w.writerow(["cell_id", "lat", "lon", "n_gbifIDs", "n_scenes",
                "best_cloud_pct", "best_scene_date", "best_scene_id",
                "best_constellation", "any_gbifID_sample"])
    out_f.flush()

    errors = 0
    for i, (cell_id, lat, lon, gids) in enumerate(cells):
        # Refresh token every 50 min
        if time.time() - token_age > 50 * 60:
            token = get_token(api_key); token_age = time.time()
        try:
            scenes = search_cell(token, lon, lat, args.radius_km,
                                  args.date_from, args.date_to, args.max_cloud,
                                  constellations)
        except Exception as e:
            print(f"  {cell_id} error: {e}")
            scenes = []
            errors += 1
            if errors > 20:
                sys.exit("too many API errors — aborting")

        best = None
        if scenes:
            best = min(scenes, key=lambda s: s["properties"].get("cloudCover", 100))
        w.writerow([
            cell_id, f"{lat:.5f}", f"{lon:.5f}", len(gids), len(scenes),
            f"{best['properties'].get('cloudCover', ''):.2f}" if best else "",
            (best['properties'].get('acquisitionDate') or "")[:10] if best else "",
            best.get("id", "") if best else "",
            best['properties'].get('constellation', '') if best else "",
            gids[0] if gids else "",
        ])
        out_f.flush()
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(cells)}  ({sum(1 for s in [] if s):d})  errors={errors}")
        time.sleep(args.rate_sleep_s)
    out_f.close()

    # Post-summary
    with open(args.out) as f:
        out_rows = list(csv.DictReader(f))
    n_any = sum(1 for r in out_rows if int(r["n_scenes"]) > 0)
    print()
    print(f"wrote {args.out}")
    print(f"  cells checked:            {len(out_rows)}")
    print(f"  cells with any coverage:  {n_any} ({100*n_any/max(1,len(out_rows)):.1f}%)")
    if n_any:
        print(f"  best-cloud distribution (of cells with coverage):")
        clouds = sorted(float(r["best_cloud_pct"]) for r in out_rows if r["best_cloud_pct"])
        for pct in (10, 25, 50, 75, 90):
            i = int(len(clouds) * pct / 100)
            print(f"    p{pct:>2}: {clouds[i]:.2f}%")


if __name__ == "__main__":
    main()
