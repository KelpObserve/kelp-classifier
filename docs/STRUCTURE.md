# Repository structure

Only the current workflow and its scientific evidence belong in the active tree.
Large image arrays, caches and model binaries are ignored. New runs belong under
`runs/`, separate from the included result tables.

```text
README.md                     Start here: purpose, quick start and workflow index
technical_report.md           Paper-style study report with figures and references
config.py                     Lazy path resolution and service configuration
example_config.json           Portable defaults and input availability
requirements*.txt / *.lock    Reporting, data and locked CUDA environments
configs/
  metadata.yaml               Frozen Clay band statistics
  species.yaml                Current PRESENT species allowlist
  services.json               Runtime service endpoints
  references.json             Source-reference inventory
scripts/                      Repository/input checks, resource links and fresh-run setup
tests/                        Configuration, consolidation and evidence regression checks
docs/
  SCIENCE.md                  Detailed methods, results and limitations
  REPRODUCIBILITY.md          Environments, artifacts and ordered commands
  STRUCTURE.md                This guide
  figures/                    Supporting study-report figures
data_collection/
  gbif/, obis/, jncc/         Source extraction; raw exports are external
  field_map/                  Published observation map and geographic source assets
  sesimbra_s2_campaigns/      Saved scene metadata, labels and downloader
  eligible_chips.csv          1,198-chip inventory before final QC
  split_initialization.csv   ABSENT initialization for the partition search
chip_generation/              S2 acquisition, array consolidation and visual QC
training/
  full_nuisance_features/     Eligible whole-chip features and quality audit
  full_nuisance_balance/      Current partitions, metrics and classifier comparison
  final_models/              Final train/validation manifests, metrics and catalogs
  sesimbra_evaluation/        Current field predictions, maps and verification evidence
  pneo_september/            Current PNEO feature evaluation
  report_templates/         Maintained narrative inputs for generated reports
```

## Data entry points

| Script | Purpose |
|---|---|
| `data_collection/gbif/extract_gbif_kelp.py` | Split saved GBIF records by explicit presence/absence |
| `data_collection/obis/extract_obis_kelp.py` | Obtain source records from the OBIS API |
| `data_collection/jncc/download_snapshot.py` | Obtain the pinned public database snapshot |
| `data_collection/jncc/extract_jncc_kelp.py` | Extract records from the JNCC Access snapshot |
| `data_collection/dedup_candidates.py` | Basic source-export deduplication |
| `data_collection/build_observation_summary_tables.py` | Current source, uncertainty, species and absence-consistency policy |
| `data_collection/match_sesimbra_field_coordinates.py` | Match all three field campaigns directly from source files |
| `data_collection/count_training_pixel_dates.py` | Count observations/chips/pixel-dates in current selected manifests |
| `data_collection/sesimbra_s2_campaigns/download_scene.py` | Reconstruct saved campaign image crops |
| `data_collection/sesimbra_s2_campaigns/count_pixel_dates.py` | Rebuild combined S2/PNEO labels using image geometry |
| `chip_generation/fetch_present.py`, `fetch_absent.py` | Download quality-filtered acquisition arrays |
| `chip_generation/consolidate_chips.py` | Deduplicate arrays and preserve observation associations |
| `chip_generation/audit_boundaries.py` | Check inventory footprints against source raster extents |
| `chip_generation/render_chips.py` | Visual inspection of acquisition-format outputs |

`data_collection/common.py` shares source species recognition and statistics.
`chip_generation/chip_common.py` shares acquisition and quality-control logic.
`chip_generation/tile_coverage.py` selects complete overlapping tiles and preserves
the requested grid. The acquisition README owns these methods;
Coverage reports from `chip_generation/audit_boundaries.py` are generated locally
and excluded from Git.

## Training entry points

| Stage | Scripts in `training/` | Output |
|---|---|---|
| Prepare features | `prepare_nuisance_pool.py` | Full-chip feature table and quality exclusions |
| Select partitions | `balance_absent_first.py` | Seed manifests, balance evidence and baseline metrics |
| Fit comparison models | `train_frozen_sesimbra.py`, `train_simple_clay_heads.py`, `run_location_baselines.py`, `build_model_catalog.py` | Embeddings, checkpoints, predictions and metrics |
| Verify comparison | `verify_full_nuisance_trial.py`, `verify_simple_clay_heads.py` | Verification evidence |
| Report comparison | `summarize_full_nuisance_trial.py` | Report, union manifest and figures |
| Fit/verify/report final heads | `train_final_models.py`, `verify_final_models.py`, `summarize_final_models.py` | Final split, heads, catalog and report |
| Apply/verify/report field campaigns | `evaluate_sesimbra_campaigns.py`, `verify_sesimbra_campaigns.py`, `report_sesimbra_campaigns.py` | Field predictions, raster outputs, metrics and maps |
| PNEO analysis | `extract_pneo_september.py`, `evaluate_pneo_september.py` | Spectral features, embeddings and held-out-transect results |

Shared modules are `baselines.py` (fixed baseline fitters), `split_groups.py`
(observation/duplicate/location grouping), `model_head.py` (MLP and loss),
`patch_pool.py` (central token slices), `encoder_geometry.py` (derived S2 grid
spacing and metadata validation), `final_partition.py` (new final split optimization), `final_model_io.py` (reusable heads), and `report_links.py` (portable report links).
Each retained module serves the current pipeline; executable scripts are not
assumed to be unused merely because nothing imports them.

## Documentation ownership

Keep complete methods and result tables in their canonical homes. Other pages
summarize the conclusion and link to the detail; key limitations can remain
visible in the entry-point README.

| Content | Canonical home |
|---|---|
| Purpose, workflow navigation and prominent limitations | [Root README](../README.md) |
| Field survey design, label aggregation and scientific interpretation | [Scientific account](SCIENCE.md) |
| Source selection, species policy, acquisition filters and observation counts | [Data collection](../data_collection/README.md) |
| Complete chip footprints, overlapping tiles and consolidation | [Chip acquisition](../chip_generation/README.md) |
| Split construction, model inputs and comparison training methods | [Training methods](../training/README.md) |
| Complete comparison metrics | [Generated comparison report](../training/full_nuisance_balance/README.md) and linked CSVs |
| Final fit, selection and metrics | [Generated final-model report](../training/final_models/README.md) and linked manifests |
| Field application, maps and PNEO analysis | [Generated campaign report](../training/sesimbra_evaluation/README.md) and linked evidence |
| Environments, input availability, commands and inference example | [Reproduction guide](REPRODUCIBILITY.md) |
| Directory and entry-point inventory | This guide |

## Evidence and generated files

CSV/JSON manifests and metrics are reviewable evidence. Report scripts regenerate
README tables and PNG figures from those inputs. Checkpoint catalogs store
relative paths and hashes, not machine locations. Source hashes describe the
run that produced an artifact; editing a script does not retroactively change
that recorded hash. New runs must produce new provenance.

The campaign generator reads the maintained
[PNEO narrative](../training/report_templates/pneo_september.md), never the
previously generated README. Edit that source for the published September
analysis, then run `python training/report_sesimbra_campaigns.py` with the data
environment installed and campaign evidence configured. It regenerates the
campaign README and map from included CSV/JSON/GeoTIFF evidence without model
weights or a GPU. Links are adjusted for custom report destinations.

The PNEO narrative describes the saved experiment, including its numerical
interpretation; it is not automatically recalculated from a new PNEO run.
Review it against the relevant result tables before publishing a new experiment.
S2 report prose lives in the generator; its tables are read from the configured
campaign metrics. Do not edit generated READMEs as the sole source of a change.

Fresh-run commands and required configuration keys are documented in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). The included scientific reports describe
saved experiments; successful syntax or table checks do not certify GPU replay.
