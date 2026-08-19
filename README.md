# kelp-observe-classifier

Detect kelp presence/absence in European coastal waters from Sentinel-2 L2A
imagery, using [Clay v1.5](https://clay-foundation.github.io/model/) foundation-model
embeddings plus a light MLP classification head.

## Repository structure

- [`data_collection/`](data_collection/) — extract labelled kelp records from public
  databases (GBIF, OBIS, JNCC).
- [`chip_generation/`](chip_generation/) — pull Sentinel-2 chips for each record
  (strict QC for PRESENT, relaxed for ABSENT, up to 8 scenes per ABSENT record)
  and merge into a single HDF5.
- [`training/`](training/) — build fixed matched splits, precompute Clay embeddings,
  train the MLP head, evaluate.
- [`configs/metadata.yaml`](configs/metadata.yaml) — Clay v1.5 band-stat file (kept
  in the repo because it is required by every Clay forward pass).
- [`config.py`](config.py) — single point of truth for filesystem paths
  (driven by env vars, defaults documented below).

## Setup

Python 3.11 in a virtualenv, then:

```bash
git clone https://github.com/OceanStreamIO/kelp-observe-classifier.git
cd kelp-observe-classifier
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Environment variables

Set these in your shell (`export …`) or in a `.env` file at the repo root (loaded with e.g. `set -a; source .env; set +a`; the file is git-ignored).

| Variable | Default | Purpose |
|---|---|---|
| `KELP_DATA_DIR` | `~/kelp-data` | Root for all data + intermediate artefacts (chips, embeddings, checkpoints) |
| `KELP_CLAY_CKPT` | `$KELP_DATA_DIR/clay/v1.5/clay-v1.5.ckpt` | Path to the Clay v1.5 checkpoint (download from the [Clay Foundation release page](https://github.com/Clay-foundation/model/releases)) |
| `KELP_LOGS_DIR` | `$KELP_DATA_DIR/../logs` | Where nohup / long-running scripts write logs |
| `KELP_TMP_DIR`  | `$KELP_DATA_DIR/../tmp`  | Where one-off scripts drop scratch files |
| `KELP_POOL_SIZE` | `5` | Centre-pool size (patches per side) for the head's spatial embedding. One of `1`, `2`, `3`, `5`, `7`. Smaller = tighter effective footprint around the chip centre pixel. See [`training/README.md`](training/README.md#centre-pool-size). |

### Expected `KELP_DATA_DIR` layout

```
$KELP_DATA_DIR/
├── clay/v1.5/clay-v1.5.ckpt         # Clay foundation-model weights (~4.9 GB)
├── s2_chips_merged/
│   ├── chips.h5                     # Merged 128×128 S2 chips (8 bands)
│   └── chips_metadata.csv           # One row per chip
├── clay_cache/identity.npz          # Cached Clay CLS + centre 7×7 patches (pool-independent)
├── absent_windows.json              # Per-A time-window annotations
└── jncc_mr/                         # Optional: JNCC MDB snapshot (~1.7 GB)
```

The repo does not ship data; the extractors in [`data_collection/`](data_collection/)
build the CSVs from public sources, [`chip_generation/`](chip_generation/) turns them
into HDF5 chips, and [`training/`](training/) does the rest.

## Pipeline

```
GBIF ┐
OBIS ┼─▶ data_collection/  ─▶  per-source CSVs of PRESENT & ABSENT records
JNCC ┘                              │
                                    ▼
                          chip_generation/   ─▶  128×128 S2 chips in HDF5, 8 bands
                                    │              strict QC for PRESENT (1 scene)
                                    │              relaxed QC for ABSENT (up to 8 scenes)
                                    ▼
                              training/      ─▶  make_splits.py  (3 fixed splits, no location
                                                                   leakage; A stratified 75/15/15 by
                                                                   (cloud, water) with 50 m safety; P
                                                                   subselected per split to match A on
                                                                   water exactly + b02_range greedily)
                                                    precompute_clay_embeddings.py  (Clay v1.5 backbone)
                                                    train_head.py  (MLP head, focal loss,
                                                                    val-tuned threshold calibration)
```

## Reference dataset

The reference run of the pipeline (as of the initial public release) produced:

- **PRESENT**: 4,089 chips (single-scene, cleanest S2 scene in each event window)
- **ABSENT**: 1,614 chips (up to 8 multi-scene per unique observation, 243 unique locations)
- **Sources**: GBIF Europe (2016+), OBIS, JNCC Marine Recorder v52 (UK, 2016+)

See [`data_collection/DATA_SOURCES.md`](data_collection/DATA_SOURCES.md) for the full
source manifest.

## Reference results (3-seed mean, MLP head, matched splits)

| Metric | Value |
|---|---|
| **Test ROC-AUC** | **0.893 ± 0.009** |
| Test balanced_acc @ t=0.5 | 0.816 ± 0.031 |
| Test macro_F1 @ 0.5 | 0.812 ± 0.021 |
| Test ABSENT recall @ 0.5 | 0.761 ± 0.082 |
| Test ABSENT precision @ 0.5 | 0.736 ± 0.012 |
| Test PRESENT recall @ 0.5 | 0.871 ± 0.021 |

**ROC-AUC is the primary metric** — threshold-independent, stable across seeds.
`balanced_acc` and `macro_F1` are threshold-sensitive; the wobble reflects
calibration differences across the geographically-different splits, not model quality.

See [`training/README.md`](training/README.md) for the full method, splits algorithm
description, and per-seed detail.

## Model catalog

Four trained heads live in the repo. All share the same MLP architecture
(dropout 0.3, focal loss α=0.5 γ=2.0); they differ in *training subset*,
*input dimensionality*, and *effective spatial footprint*.

| Model | Subset | Input dim | Footprint | Purpose |
|---|---|---:|---|---|
| `merged_p1` | merged (all species) | 2048 | ~46 m | General Europe-wide model. |
| `submerged_p1` | submerged (4 deep-water spp.) | 2048 | ~46 m | Sharpens response to *Saccorhiza polyschides* et al. |
| `submerged_p5` | submerged | 2048 | ~229 m | Best on submerged pool sweep + Sesimbra CCMar. Deployment-recommended for wide-area maps. |
| `submerged_pixel` | submerged | 2054 | ~46 m Clay context + 10 m raw S2 | Runs at every 10 m pixel during rasterisation. Sharper outputs; worse on the tiny CCMar test. |

Full training-vs-test results and Sesimbra CCMar pixel-aggregated evaluation
are in [`training/README.md`](training/README.md#model-catalog).
