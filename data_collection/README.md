# Data collection

Extract labelled kelp presence/absence records from public databases.

## Kelp species we target

Defined in [`common.py`](common.py) as `KELP_SPECIES`:

- Laminaria hyperborea
- Laminaria digitata
- Saccharina latissima *(a.k.a. Laminaria saccharina — synonym)*
- Alaria esculenta
- Saccorhiza polyschides
- Laminaria ochroleuca
- Chorda filum
- Undaria pinnatifida *(invasive)*
- Phyllariopsis brevipes

## Cross-source dedup (pre-fetch)

The six raw source CSVs (GBIF/OBIS/JNCC × PRESENT/ABSENT) frequently contain
the same physical observation logged in more than one database. The pre-fetch
dedup consolidates them so we don't fetch duplicates:

| Script | Role |
|---|---|
| [`dedup_candidates.py`](dedup_candidates.py) | Groups all six sources by `(occurrenceStatus, round(lat, 4), round(lon, 4), eventDate)`, combines species names, and picks a representative (source priority: GBIF > OBIS > JNCC, then alphabetically-first ID). Filters to eventDate ≥ 2016 (Sentinel-2 coverage). Emits `deduped_present_candidates.csv` and `deduped_absent_candidates.csv`. |
| [`select_present_candidates.py`](select_present_candidates.py) | Narrows the deduped PRESENT pool to the candidates geographically closest to any ABSENT candidate: keeps all forced-P (within 50 m of an A, will co-cluster under the 50 m location-safety rule) plus the top-N free-P by nearest-A distance. Rationale: the split algorithm matches P to A within the same water bin, and geographic proximity is a strong proxy for water-bin membership. |
| [`build_high_precision_candidates.py`](build_high_precision_candidates.py) | Restricts the PRESENT pool further to records with `coordinateUncertaintyInMeters <= 10` m — well inside the 115 m half-width of the model's prediction region. All ABSENT records are kept (no unc filter on ABSENT). Emits `high_precision_candidates.csv` (used as the input to `chip_generation/fetch_present.py`). |
| [`annotate_absent_windows.py`](annotate_absent_windows.py) | For every raw ABSENT record from GBIF/OBIS/JNCC, computes `allowed_max_days = 30` (strict) or `60` (extended) based on whether any PRESENT record within 1 km AND ±90 days exists. Consulted at fetch time to reject ABSENT chips whose scene date is too far from the reported observation date. Output: `$KELP_DATA_DIR/absent_windows.json` (keyed by per-source ID: plain gbifID, `obis:<obisID>`, or `jncc:<sample_ref>`). |

## Submerged candidates (Sesimbra / Arrábida-focused subset)

The Sesimbra validation site targets four deep-water kelp species with
sub-surface blades:

- *Saccorhiza polyschides*
- *Laminaria hyperborea*
- *Laminaria ochroleuca*
- *Phyllariopsis brevipes*

Chip-side subsetting (see [`chip_generation/assemble_submerged.py`](../chip_generation/assemble_submerged.py))
can subset the merged pool by species without refetching. But when new
records exist in raw sources that were never fetched into the merged pool,
we need candidate CSVs to feed the chip fetchers. That's
[`build_submerged_candidates.py`](build_submerged_candidates.py):

| Script | Role |
|---|---|
| [`build_submerged_candidates.py`](build_submerged_candidates.py) | Emits candidate CSVs for the four target species from `deduped_{present,absent}_candidates.csv`, **excluding** any observation whose gbifID (with or without `_sN` scene suffix) already appears in the merged chip pool. PRESENT filter matches `build_high_precision_candidates.py` (`unc <= P_MAX_UNC`, default 10 m; blank unc dropped unless `--include-blank-unc`). ABSENT: no unc filter (a "no kelp seen" claim over an area is robust to a noisy anchor point). Emits `submerged_present_candidates.csv` + `submerged_absent_candidates.csv`. |

Feed the CSVs to `chip_generation/fetch_present.py` + `fetch_absent.py` to
grow the merged pool, then run `chip_generation/assemble_submerged.py` to
carve out the species-filtered submerged subset.

## Ad-hoc probes (not part of the training pipeline)

| Script | Role |
|---|---|
| [`check_pneo_coverage.py`](check_pneo_coverage.py) | Bulk query Airbus OneAtlas Data API for PNEO/Pléiades archive coverage over training candidates. Free (search only; ordering costs money). Requires `ONEATLAS_API_KEY` env var. Groups candidates into `--cell-deg` geographic cells to avoid duplicate queries. Useful for scoping which locations *could* be re-imaged at ~1 m for a follow-up validation study — not consumed by any training script. |

## Sources

Every source produces two CSVs: `..._present.csv` and `..._absent.csv`.

### GBIF Europe

- Master file: `data_collection/gbif/gbif_kelp_europe.csv` (bundled in-repo, 49 MB) (downloaded manually from [GBIF](https://www.gbif.org/occurrence/download); 201,699 rows, all seven kelp taxa, continent=EUROPE).
- Script: [`gbif/extract_gbif_kelp.py`](gbif/extract_gbif_kelp.py) — filters to kelp species, splits by `occurrenceStatus`.
- Outputs: [`gbif/kelp_gbif_europe_present.csv`](gbif/) (153,970), [`gbif/kelp_gbif_europe_absent.csv`](gbif/) (3,962).

### OBIS

- Fetched live via the [OBIS API](https://api.obis.org/v3/occurrence).
- Filter: kelp species × Europe polygon × `startdate >= 2016-01-01` (S2 era).
- Script: [`obis/extract_obis_kelp.py`](obis/extract_obis_kelp.py) — pages through PRESENT and ABSENT (via `absence=true` flag).
- Outputs: [`obis/kelp_obis_europe_present.csv`](obis/) (18,307), [`obis/kelp_obis_europe_absent.csv`](obis/) (578).
- OBIS shares many datasets with GBIF — expect ~50% overlap by `(species, lat, lon, date)`.

### JNCC Marine Recorder (UK)

- Snapshot v52 (2022-01-24): `$KELP_DATA_DIR/jncc_mr/SnapshotDatav52_PUBLIC_20220124.mdb` (~1.7 GB unpacked; original zip archived alongside). Originally downloaded from [JNCC data portal](https://data.jncc.gov.uk/data/b9934e31-39b6-41f9-9364-d1e93db68307/SnapshotDatav52-PUBLIC-20220124.zip).
- Format: two MS Access `.mdb` files. Requires `mdbtools` (`sudo apt install mdbtools`). Absence is encoded as `PA='N'` or `SACFORN='N'` on the `SampleSpecies` table; coords come from the `Sample` table joined on `SAMPLE_REFERENCE`.
- Script: [`jncc/extract_jncc_kelp.py`](jncc/extract_jncc_kelp.py) — filters to kelp species × post-2016 date.
- Outputs: [`jncc/kelp_jncc_present.csv`](jncc/) (7,553), [`jncc/kelp_jncc_absent.csv`](jncc/) (174 unique after dedup vs GBIF/OBIS).
- UK-only; heavily weighted to 2017 (one big survey wave).

## Aggregated statistics

Built by [`build_source_stats.py`](build_source_stats.py) → [`source_statistics.csv`](source_statistics.csv).

Current totals:

| Source | Class | Records | Post-2016 |
|---|---|---:|---:|
| GBIF Europe | PRESENT | 153,970 | 49,762 |
| GBIF Europe | ABSENT | 3,962 | 1,201 |
| OBIS (2016+) | PRESENT | 18,307 | 18,307 |
| OBIS (2016+) | ABSENT | 578 | 578 |
| JNCC MR v52 (2016+) | PRESENT | 7,553 | 7,553 |
| JNCC MR v52 (2016+) | ABSENT | 174 | 174 |

## Sources we tried and dropped

Notes are in [`other_sources_tried/`](other_sources_tried/).

- **DASSH IPT** ([www.dassh.ac.uk/ipt](https://www.dassh.ac.uk/ipt)) — pulled 16 kelp-relevant Darwin Core Archives (Seasearch, Natural England, NRW, etc). All 361 kelp records had `occurrenceStatus: present`. Seasearch methodology records species observed, not species surveyed but absent. **Zero new ABSENT.** Script preserved as [`other_sources_tried/pull_dassh.py`](other_sources_tried/pull_dassh.py).
- **PANGAEA** ([pangaea.de](https://www.pangaea.de/)) — checked top 20 kelp datasets. Almost all are physiological/laboratory experiments (Bergen, Bodø, Helgoland, Locmariaquer, Ny-Ålesund). One transect study (Porsangerfjorden, 4 stations) is under moratorium until Nov 2026. **Not useful.**
- **NIVA Brage / Nasjonalt vitenarkiv** ([sikt.no NVA](https://sikt.no/en/tjenester/nasjonalt-vitenarkiv-nva)) — Norwegian NIVA institutional repo. 15 hits for the sukkertareprosjektet monitoring series, all `ReportResearch` (PDFs). Station tables are inside the PDFs — extractable but manual. **Not automatable in the time budget.**
- **iNaturalist**, **BODC**, **IFREMER SEXTANT**, **AZTI**, **IPMA/MARE-Ciimar** — not tried; likely low yield beyond what GBIF/OBIS already carry.

## Adding a new source

1. Create `data_collection/<source>/extract_<source>_kelp.py`.
2. Reuse `common.is_kelp_species`, `common.stats`, `common.dedup_key`.
3. Write `kelp_<source>_present.csv` and `kelp_<source>_absent.csv` into the same directory.
4. Add the file paths to `SOURCE_FILES` in [`build_source_stats.py`](build_source_stats.py) and rerun.
