# European kelp polygons — downloaded from EMODnet Seabed Habitats

Source: EMODnet Seabed Habitats WFS
`https://ows.emodnet-seabedhabitats.eu/geoserver/emodnet_open/wfs`
Downloaded 2026-07-07.

## Files

| File | Layer | Records | Coverage | Type |
|---|---|---|---|---|
| `niva_kelpforestnorway.zip` | `emodnet_open:niva_kelpforestnorway` | 11,611 shp rows / 7,228 unique lokalids | Norwegian coast — kelp forests (NIVA). **Mixed: 78.6% modelled, 5.3% field-validated, 16.1% pre-flag-era observations** | polygons, mixed |
| `niva_kelpforestnorway_observed.zip` | filtered subset (`modellert != '1'`) | 3,290 shp rows / **1,544 unique observed polygons** | Same but only actual observations — modelled polygons dropped | polygons, observed |
| `niva_kelpforestnorway_dates.csv` | per-shp-row date audit | 11,611 rows | Adds `datafangstdato`, `registreringsDato`, `modellert`, `noeyaktighetsklasse`, `leveranseId`, `innledning` from the Miljødirektoratet ArcGIS layer | metadata |
| `niva_kelpforestnorway_observed_dates.csv` | date audit for the observed subset | 3,290 rows | Same fields, observed subset only | metadata |
| `ospar2025_points_kelp.zip` | `emodnet_open:ospar2025_points` (kelp filter) | 5,946 | Northeast Atlantic — OSPAR kelp-forest habitat point records | points, observed |
| `macroalgae_kelp_uk_ireland.zip` | `emodnet_open:macroalgae_eov_poly_2025` (kelp filter, UK/IE bbox) | 374 | UK + Ireland | polygons, observed |
| `macroalgae_iberia_all.zip` | `emodnet_open:macroalgae_eov_poly_2025` (Iberian bbox) | 8 | Iberian Atlantic coast (Portugal/Spain) | polygons, observed |
| `gbif_kelp_europe.csv` | GBIF occurrence API (`continent=EUROPE`, 7 kelp taxa) | **201,699 total, 51,785 in S2 era, incl. 6,899 observed ABSENCES (1,220 in S2 era)** | Europe-wide, all years, aggregated from DASSH, NBN Atlas / Seasearch UK, JNCC Marine Recorder, MarLIN, iNaturalist etc. via GBIF | points, observed |
| `fetch_gbif_kelp.py` + `fetch_gbif_kelp_extra.py` | download scripts (reproducible, concurrent) | — | Paginate GBIF API 300 records/page for *Laminaria*, *Saccharina*, *Alaria*, *Saccorhiza polyschides* (base) and *Phyllariopsis*, *Undaria pinnatifida*, *Chorda* (extra) | script |
| `build_prototype_labels.py` | filter GBIF → prototype label subset | — | Reads `gbif_kelp_europe.csv`, filters to S2-era + coord ≤100 m + P/A + marine (Natural Earth land polygon test), caps per country + genus, writes `gbif_prototype_3k.csv` | script |
| `gbif_prototype_3k.csv` | filtered label subset | 2,982 (2,809 PRESENT + 173 ABSENT) | Balanced across Norway/UK/Finland/Sweden/Ireland/France/Portugal/Germany + 7 kelp genera | points, observed |
| `fetch_s2_chips_for_gbif.py` | S2 chip fetch script (runs on server) | — | For each row: STAC-searches Element84 `sentinel-2-l2a` for closest cloud-free scene within ±30 days of `eventDate`, opens 8 COGs on S3 via rasterio, crops a 32×32 window on the 10 m grid centred on the point, writes to HDF5 with LZF compression + metadata CSV | script |

Kelp filter used on `macroalgae_eov_poly_2025` and `ospar2025_points`:
```
habtype ILIKE '%kelp%' OR eunis_name ILIKE '%kelp%' OR eunis_name ILIKE '%Laminaria%'
  OR eunis_name ILIKE '%Saccharina%' OR eunis_name ILIKE '%Alaria%'
  OR eunis_name ILIKE '%Saccorhiza%' OR eunis_name ILIKE '%ochroleuca%'
```

All shapefile geometries are in ETRS89 / LAEA Europe (EPSG:3035) except the
OSPAR points which are stored in multiple projections in the same zip.

## Not downloaded (and why)

- **France (Brittany) macroalgae kelp polygons — 404k features, ~356k just
  in Molène-Ouessant.** Ifremer digitizes kelp at extremely fine resolution;
  each individual patch is a separate polygon. The WFS 500-errors on any
  single-call fetch of the full set. Would need paginated download in
  25k-feature batches (~15+ pages) — skipped for time. Rerun with
  `startIndex`/`count` pagination if you need the full Brittany subset.
- **Portugal FindKelp dataset (388 sites, 6 kelp species, Portuguese
  mainland, observed).** Not in EMODnet's open macroalgae layer (Iberia
  bbox returned only 8 features). Held at `gis.ices.dk` record
  `eab61c22-a7b9-4685-8a50-7e11e8157cb3`.
  Metadata: https://gis.ices.dk/geonetwork/srv/api/records/eab61c22-a7b9-4685-8a50-7e11e8157cb3
  Human-facing catalogue page:
  https://gis.ices.dk/geonetwork/srv/eng/catalog.search#/metadata/eab61c22-a7b9-4685-8a50-7e11e8157cb3

  **What's actually in the shapefile** (from the Assis et al. 2009
  Journal of Coastal Research paper — the methods paper):
  - **Points, not polygons.** GIS output is a point layer of
    ~388 georeferenced sites reported by volunteer scuba divers +
    56 groundtruth-validation transect points. "GIS query point
    analyses" per the paper's methods.
  - **Attributes per site**: Site name, coordinates, **date of
    observation**, depth, confidence level (1–3), species report
    (one of 10 categories including `No species recorded`).
  - **Positives + negatives in one file**: 338 sites reported kelp,
    **50 sites recorded no kelp** (confirmed negatives). This is the
    only file in this folder that ships true observed negatives.
  - **Survey period: 15 May – 31 August 2008** — all points fall in
    that summer window.
  - **Not Sentinel-2 compatible** — S2 didn't exist until June 2015.
    Time-matching requires **Landsat 5 TM** (operational in 2008)
    or Landsat 7 ETM+ (with SLC-off gaps since May 2003).
  - Historical archive, no updates planned.

  **Cannot be downloaded anonymously via API.** The file endpoint
  `http://gis.ices.dk/geonetwork/srv/en/resources.get?id=402&access=private`
  responds `302 Found` with `Location: /geonetwork/signin` — deliberate
  access control (metadata is public, file is not). The metadata licence
  field reads "Restricted access requiring request to the FindKelp Project
  and Centre of Marine Sciences (CCMAR)."
  To obtain the shapefile: either (a) request an ICES GeoNetwork account
  with permission granted by CCMAR/FindKelp, or (b) ask CCMAR (the
  KelpObserve partner in Portugal, Ester Serrão / Jorge Assis group)
  directly — fastest path since they own the data.
  Paper (open-access PDF): http://maree.ccmar.ualg.pt/pdf/Assisetal2009JCR.pdf
- **Full pan-European `ospar2025_polygons` kelp subset — 680k polygons.**
  Same WFS-payload problem as France; would need pagination.

## Caveats for use as evaluation data

- These are **observation-derived polygons** but survey coverage is uneven —
  the polygon layers themselves do not tell you which pixels of the coast
  were **surveyed and found empty**. Absence is inferred, not observed.
- Each polygon has a `det_date` / `startdate` / `enddate` — check per-record
  age, as kelp distribution has shifted measurably in the last 10–20 years
  (Portuguese *L. hyperborea* range contraction, Norwegian urchin-barren
  dynamics, UK *L. ochroleuca* northward expansion).
- Licence: EMODnet Seabed Habitats data are generally openly downloadable;
  check per-layer metadata in the EMODnet catalogue for attribution
  requirements before publishing derivatives.

## Time-matching to satellite imagery (audit of the four files)

For training a supervised kelp detector we want each label to have a
survey date so we can pull a Sentinel-2 / Landsat scene from the same
window. Actual state of the date fields in the downloaded shapefiles:

| Dataset | Records | Date field | ≥2016 (S2 era) | ≥2013 (L8 era) | Notes |
|---|---:|---|---:|---:|---|
| `niva_kelpforestnorway` (raw) | 11,611 rows / 7,228 unique | none in DBF; recovered via Miljødir. ArcGIS `naturtyper/MapServer/1` join on `lokalid = marinNaturtypeId` | 5,697 (79%) | 5,978 (83%) | **Misleading — most are `modellert=1` batch-stamped 2020-03-01, not observations.** See "observed subset" row below. |
| `niva_kelpforestnorway_observed` (filtered `modellert != '1'`) | 1,544 unique | `datafangstdato` | **13 (0.8%)** | 294 (19.7%) | Real observed polygons only. Year histogram: 2002:399, 2003:570, 2004:38, 2011:192, 2015:279, plus scattered. Mostly Landsat-era. |
| `macroalgae_kelp_uk_ireland` | 374 | `det_date` | 0 (0.0%) | 3 (0.8%) | All 2002–2014, peak **2010-04-11** (240 records). Needs Landsat 5/7 archive to time-match. |
| `macroalgae_iberia_all` | 8 | `det_date` | 0 | 5 | Only 2012 (×3) and 2013 (×5). Needs Landsat 7/8. |
| `ospar2025_points_kelp` | 5,946 | `detdate` (+ start/end) | **5,149 (86.6%)** | 5,161 (86.8%) | **5,103 points all dated 2022-07-13** — a single Europe-wide survey day fully in the S2 era. Remainder scattered 1983–2024 → mixed Landsat + S2. |
| **FindKelp Portugal** (points, not held here — CCMAR request required) | 388 | per-site `Date` attribute in shapefile | 0 | 0 | Points, not polygons. **338 positives + 50 confirmed negatives** (the only file with observed absence). All observations 15 May – 31 Aug 2008 → **Landsat 5 TM only**. |
| **GBIF Europe 7 kelp taxa** (`gbif_kelp_europe.csv`) | 201,699 | `eventDate` per record | **51,785 (25.7%)** | 71,378 (35.4%) | Points. 7 taxa: *Laminaria*, *Saccharina*, *Alaria*, *Saccorhiza polyschides*, *Phyllariopsis*, *Undaria pinnatifida* (invasive), *Chorda*. Aggregated from DASSH, NBN Atlas (Seasearch UK), JNCC Marine Recorder, MarLIN, iNaturalist etc. **6,899 absence records**, 1,220 in S2 era — the only observed-absence source at scale in Europe. Phyllariopsis genus alone is ~50% absences (Portuguese species-checklist surveys). |

### Practical consequences

- The **OSPAR 2022-07-13 batch (5,103 records)** is the only large,
  cleanly S2-time-matchable set here. But these are **points, not
  polygons** — each labels one 10 m S2 pixel (or a ~3×3-pixel buffer
  depending on GPS accuracy). Fine for a pixel classifier, not for
  segmentation of contiguous kelp patches. Filter by the `accuracy`
  field before use; drop or heavily buffer site-level accuracy points.
- **Only ~13 observed polygons in the ≥2016 S2 era** across UK/IE +
  Iberia + NIVA-observed combined (all 13 in NIVA). The rest are either
  modelled predictions (dropped) or pre-Sentinel-2 fieldwork. For polygon
  labels in the S2 era we depend on:
  a) CCMAR KelpObserve June/July/Sept 2026 transects (upcoming), or
  b) using the older observed polygons with **Landsat** archive
  time-matching instead of S2.
- **Landsat archive** (5 TM, 7 ETM+ SLC-off, 8 OLI) is required to
  time-match the UK/IE and Iberian polygons and the older half of the
  OSPAR points. All free via USGS / Element84 STAC. Adds 30 m imagery
  to the training pool for pre-2015 labels.

### Prototype pipeline (in flight — 2026-07-07)

**Filter → chip fetch pipeline running on the server:**

1. `build_prototype_labels.py` (local) — filtered `gbif_kelp_europe.csv`
   (201,699 rows) → `gbif_prototype_3k.csv` (2,982 points).
   - S2 era (year ≥ 2016)
   - coord uncertainty ≤ 100 m
   - marine-only (Natural Earth 10m land-polygon test)
   - all ABSENT records kept (173 in filtered set)
   - PRESENT capped at 500 per country + 700 per genus → geographic and
     taxonomic diversity
2. `fetch_s2_chips_for_gbif.py` (server, in `$KELP_DATA_DIR/../scripts/`)
   — for each row, STAC-searches Element84 `sentinel-2-l2a` for the
   closest cloud-free (≤30 %) scene within ±30 days of `eventDate`, opens
   the 8 COGs on S3, crops a 32×32 window (~320 m) on the 10 m grid
   centred on the point, writes to
   `$KELP_DATA_DIR/../scripts/data/gbif_s2_chips/chips.h5` +
   `chips_metadata.csv`.
3. Bands captured (same 8 as the Sesimbra pipeline):
   B01 coastal / B02 blue / B03 green / B04 red / B05 rededge1 /
   B08 NIR / B11 SWIR1 / SCL scene classification.
4. Server venv at `$KELP_DATA_DIR/../.venv`
   (rasterio 1.4.4, pystac-client 0.9.0, h5py 3.16.0, numpy 2.4.6).

Verified during smoke test: for a Portuguese Laminaria hyperborea absence
record on 2022-07-11, chip fetched from S2A tile 29TNF acquired
2022-07-08 (3 days off, cloud 0.011 %) — 8×32×32 chip with realistic
per-band water reflectances and SCL class 6 (water).

Rate on the live run: ~30 chips/min, ~90 % success rate; estimated
completion ~1–2 h for the full 2,982 candidates.

### Suggested first training pull

1. **GBIF Europe kelp records** — from `gbif_kelp_europe.csv`, 7 taxa:
   - Positives: `occurrenceStatus=PRESENT`, `year ≥ 2016`. **~50,500 records.**
   - Negatives: `occurrenceStatus=ABSENT`, `year ≥ 2016`. **1,220 records** —
     the only observed-absence set at scale in Europe.
   - Filter for coord uncertainty ≤ 10 m for pixel-quality labels; ≤ 100 m
     is workable with a 2–3-pixel buffer.
   - Group by S2 MGRS tile + acquisition month → pull cloud-free S2 L2A
     scene closest to each `eventDate`. **This is now the primary
     Europe-wide labelled set**, both P and N.
   - **For per-species binary "kelp / no-kelp" labelling**, aggregate
     records by (lat, lon, eventDate, datasetKey) — a pixel is a
     true kelp-negative only when *every* record for that site
     across all 7 taxa is ABSENT.
2. **OSPAR points, `detdate = 2022-07-13`, accuracy ≤ 10 m** →
   group by S2 MGRS tile → pull cloud-free S2 L2A scenes within
   ±14 days per tile. Europe-wide, one-date supplementary positive set.
3. **NIVA Norway polygons** against a recent (2023–2025) S2 mosaic,
   flagged as "date unknown, treat as weak label".
3. **UK/IE + Iberia polygons** paired with Landsat 5/7/8 archive
   at their `det_date` (mostly 2010, 2012–2013) — separate 30 m
   training pass.
4. **FindKelp Portugal points** paired with Landsat 5 TM archive
   at each site's summer-2008 `Date` attribute — the only European
   file with observed positives *and* observed negatives together,
   valuable both for training and for calibrating false-positive
   thresholds. Requires CCMAR data-sharing agreement first.
5. **CCMAR Sesimbra transects** once uploaded — the only species-
   correct polygon-label pass for *L. ochroleuca*.
