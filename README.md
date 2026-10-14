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

## Dataset and run status (verified 2026-10-14)

The current training-server metadata contains:

| Pool | PRESENT | ABSENT | Total |
|---|---:|---:|---:|
| Merged, all species | 6,077 | 1,533 | 7,610 |
| Submerged, four target species | 2,202 | 1,533 | 3,735 |

These are **current pool inventories**, not counts of chips used by every
checkpoint. Matching, splitting and balancing select smaller training sets.
The archived Sesimbra experiment page records an earlier submerged snapshot
of 3,733 chips (2,200 P / 1,533 A). The 5,703-chip pool from 17 July is historical.
Sources are GBIF, OBIS and JNCC; see the
[source manifest](data_collection/DATA_SOURCES.md).

## Reference results (merged, pool=5, seeds 42/123/456)

Recomputed from the server's saved `head_seed{42,123,456}_p5/results.json`
files on 2026-10-14; these are existing trained runs, not a new training run.
The 0.893 initial baseline has been superseded. Machine-readable evidence is
in [`training/release_run_summary.json`](training/release_run_summary.json).

| Metric | Mean ± population std |
|---|---|
| Test ROC-AUC | 0.974 ± 0.005 |
| Test balanced accuracy @ 0.5 | 0.903 ± 0.011 |
| Test macro F1 @ 0.5 | 0.907 ± 0.007 |
| Test ABSENT recall @ 0.5 | 0.856 ± 0.031 |
| Test ABSENT precision @ 0.5 | 0.898 ± 0.014 |
| Test PRESENT recall @ 0.5 | 0.950 ± 0.010 |
| Test PRESENT precision @ 0.5 | 0.928 ± 0.013 |

ROC-AUC is threshold-independent; the other metrics use threshold 0.5.
These merged-pool scores must not be attributed to the submerged heads.
For the submerged pool, the committed three-seed sweep reports ROC-AUC
**0.866 ± 0.025** for pool=1 and **0.891 ± 0.009** for pool=5.
The separate Sesimbra evaluation has only 34 unique S2 pixels (32 P / 2 A),
so its high pool=5 ROC-AUC of 0.984 is not a Europe-wide performance estimate.
See the [training documentation](training/README.md) for the evaluation protocols.

## Model catalog

Four deployment heads are versioned as explicit exceptions to the `*.pt` ignore rule.
See the [checkpoint paths and checksums](training/model_manifest.json) for Zenodo packaging. All share the same MLP architecture
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
