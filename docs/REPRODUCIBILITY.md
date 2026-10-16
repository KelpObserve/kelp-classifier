# Reproduction and required inputs

Use matching arrays, manifests and models: chip IDs depend on pixel values.
The [acquisition procedure](../chip_generation/README.md) defines complete-grid
reads, overlapping-tile handling and quality filtering.

## Reproducibility matrix

Choose the level of reproduction before configuring inputs. Included evidence
supports inspection without the external image arrays or model binaries.
The table distinguishes artifact-dependent execution from checks that a source
checkout can run on its own.

| Goal | Included evidence | Additional inputs / environment | Status and boundary |
|---|---|---|---|
| Verify saved tables and regenerate comparison/final reports | Partitions, predictions, metrics, catalogs and report scripts | Python 3.11; reporting dependencies for figures, data dependencies for regression tests and baseline refitting | Supported from included evidence; verifies consistency, not model inference. See [table checks](#verify-included-evidence-without-imagery). |
| Replay saved checkpoints and field outputs | Manifests, encoder protocols, model hashes and verification scripts | Matching weights, backbone, arrays/caches and campaign imagery; CUDA environment | Supported when external artifacts are supplied; See [input inventory](#input-inventory). |
| Retrain comparison models on saved splits | Three seed manifests, split initialization and selection protocol | Canonical arrays, backbone, CUDA environment and a fresh output directory | Supported using copied published partitions. Exact numerical equality can depend on runtime/hardware. See [comparison retraining](#retrain-the-exact-published-partitions). |
| Refit final heads | Selected union, saved final split, metrics and training script | Matching embedding cache and encoder protocol, CUDA environment and a fresh output directory | Training replays the included final split with identical membership and row order. It validates the source union and encoder geometry before fitting. See [final fitting](#fit-final-heads-and-apply-them). |
| Reconstruct from original sources | Extractors, scene metadata, field matching rules and prepared eligible inventory | Source exports, measurement/coordinate originals, imagery, backbone and licensed PNEO inputs where applicable | Partial: GBIF download DOI, complete depth preparation and a country-lookup rebuilding command are missing; live sources may change. New consolidation does not recreate published HDF5 indices. |

## Environments

Use Python **3.11**. Install the smallest environment needed:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reporting.txt  # reports and figures
# Or, for acquisition, baseline checks and tests:
python -m pip install -r requirements-data.txt
```

Training and checkpoint replay use **Linux with an NVIDIA CUDA GPU**. The
primary training and field-evaluation entry points use CUDA. Install the exact
131-package environment closure captured from the verified execution runtime:

```sh
python -m pip install -r requirements.txt
python -m pip check
```

`requirements-cuda.lock` includes PyTorch 2.4.0 and torchvision 0.19.0 CUDA 12.1
builds with the correct wheel index, plus the pinned Clay Git revision and
transitive dependencies. See
[PyTorch's version-specific instructions](https://pytorch.org/get-started/previous-versions/).
The data/reporting dependencies have their own `requirements-data.lock` for
lightweight installations. GPU drivers and hardware still affect exact replay.

JNCC extraction additionally requires the `mdb-export` executable from
[mdbtools](https://github.com/mdbtools/mdbtools). GBIF extraction needs the original
occurrence export; OBIS acquisition queries a live service and may return changed
records. Spreadsheet coordinate matching requires `openpyxl`, included in the
data requirements. PNEO input imagery requires a separate license.

## Configuration

`example_config.json` contains portable defaults and resource descriptions.
`config.py` loads these defaults and overlays `config.json` when present.
Alternatively, `KELP_CONFIG` selects a separate JSON file; it must exist.
Keep private overrides in the Git-ignored `config.json` in the repository root.
Copy `example_config.json` to `config.json`, then edit its `paths`; it is loaded
automatically without setting an environment variable. Use `KELP_CONFIG` only
when selecting an alternate configuration. Do not commit private overrides or
include them in source distributions.
Unknown path keys and circular references fail explicitly. `resources` describes
input provenance and availability; `python scripts/show_resources.py` displays
configured resources. Paths can reference other keys, such as
`{FULL_NUISANCE_BALANCE_OUTPUT}/selected.csv`. Relative paths resolve against the
repository root. Only a workflow's required paths are resolved, so reporting does
not require imagery configuration. Run commands from the repository root in the
appropriate Python environment.

Use a minimal override file to keep included evidence separate from new outputs:

```json
{
  "paths": {
    "CANONICAL_CHIPS_H5": "data/chips/chips.h5",
    "CLAY_CKPT": "data/clay/clay-v1.5.ckpt",
    "FULL_NUISANCE_BALANCE_OUTPUT": "runs/experiment",
    "FINAL_MODELS_DIR": "runs/final_models"
  }
}
```

The output paths above must not already contain completed runs. Final training
creates `FINAL_MODELS_DIR` itself and refuses an existing directory.
`FULL_NUISANCE_EMBEDDINGS` defaults to the experiment's cache; overriding it is
honoured by final training and verification. `FINAL_CHECKPOINTS_DIR` defaults to
`FINAL_MODELS_DIR` and can be overridden for separately supplied weights.

## Input inventory

| Input | Configuration | Availability and verification |
|---|---|---|
| JNCC source database | `JNCC_DATABASE` | Pinned v20220124 primary database downloaded and tables checked; source and SHA-256 provenance stored alongside it |
| Eligible inventory before central QC | `ELIGIBLE_CHIPS` | Included 1,198-row CSV with array hashes, indices, labels and prepared depth evidence |
| ABSENT split initialization | `SPLIT_INITIALIZATION` | Included assignments for the three partition-search seeds |
| Full-chip candidate features | `FULL_NUISANCE_MATCHING_POOL` | Included 1,196-row table; verifier recomputes these from canonical arrays |
| Published partitions and metrics | `FULL_NUISANCE_BALANCE_OUTPUT` | Included; see selected manifests and split-count tables for current counts |
| Canonical training arrays | `CANONICAL_CHIPS_H5` | Externally supplied path; compare array hashes against manifests |
| Clay v1.5 large backbone | `CLAY_CKPT` | External; compare SHA-256 with the saved encoder protocol |
| Band normalization | `CLAY_META` | Included `configs/metadata.yaml`; hash recorded in encoder protocol |
| Full-chip embeddings | `FULL_NUISANCE_EMBEDDINGS` | External generated cache; chip IDs, hashes and encoder settings must match |
| Evaluation checkpoints | `CHECKPOINTS_DIR` | External; relative paths and SHA-256 in `training/model_manifest.json` |
| Final checkpoints | `FINAL_CHECKPOINTS_DIR` | External; relative paths and SHA-256 in final-model catalog |
| Field observations | `FIELD_OBSERVATIONS` | Included 90-event CSV and source-file hashes |
| Measurement and coordinate originals | `FIELD_MEASUREMENTS`, `JUNE_JULY_COORDINATES`, `SEPTEMBER_COORDINATES` | External; rebuild directly from these three files |
| Campaign scene arrays | `S2_CAMPAIGNS_DIR` | Generated by included downloader using saved scene metadata |
| PNEO imagery | `PNEO_DIR`, `PNEO_SEPTEMBER_DIR` | Licensed external imagery; September directory contains paired RGB/NED JP2 files |

**Artifact access status:** the source checkout does not provide a public download
location for the canonical training arrays, trained heads or embedding caches.
They are unavailable from this checkout and must be obtained separately from the
artifact holder; no access is implied by the example paths. Their catalogs and
protocols provide checksums, not retrieval locations. Original field workbooks,
coordinate files and the unidentified GBIF export are likewise not distributed.
Licensed PNEO imagery requires separate access. Thus GPU replay is conditional
on separately supplied artifacts, not executable from the source checkout alone.

Public inputs have these retrieval routes:

- Campaign S2 crops: the three `download_scene.py` commands below use included
  scene metadata and public asset URLs.
- JNCC database: `python data_collection/jncc/download_snapshot.py` obtains the
  pinned public snapshot.
- Clay backbone: obtain v1.5 large from the [official weights collection](https://huggingface.co/made-with-clay),
  then compare its SHA-256 with the included encoder protocol. Do not substitute
  a different checkpoint merely because its filename matches.

**Supplying artifacts:** supply the canonical HDF5, frozen backbone, caches,
checkpoints, campaign images and licensed PNEO files at your configured paths.
They are not distributed with a source checkout. Verify the applicable hashes
before replay. The original GBIF DOI is not recorded; upstream depth evidence is
supplied as prepared inputs rather than inferred again from source records.
Preserve these inputs and their hashes when transferring the experiment.
The consolidation command makes acquisition outputs usable, but does not
reconstruct published HDF5 indices or independently validate depth estimates.

## Artifact inventory and access

[configs/artifacts.json](../configs/artifacts.json) is the versioned inventory
for the published study. Entries identify inputs by configuration key, optional
relative filename, SHA-256, and access route. Model catalogs retain the complete
architecture and selection metadata. The inventory identifies the published
artifacts, not new training outputs.

For project-controlled arrays, caches, checkpoints and original field files,
request the inventory's artifact IDs and checksums from the project artifact
holder. No public download location or designated contact is recorded. Store
any supplied private access locations in the ignored configuration's
`resources` metadata; `python scripts/show_resources.py` displays them. Public
campaign imagery has downloader commands in the inventory; licensed imagery
still requires separate access.

After configuring the supplied files, verify their identities:

```sh
python scripts/check_artifacts.py --group training --group comparison --group final
python scripts/check_artifacts.py --group field
python scripts/check_artifacts.py --group sources
```

A missing file or checksum mismatch fails the command. The missing original
GBIF DOI, depth-preparation recipe and country-lookup reconstruction remain
source-reconstruction gaps; an inventory cannot recreate that provenance.
The field-map assets have their own checksum inventory and an offline
[rendering command](../data_collection/field_map/README.md).

## Verify included evidence without imagery

```sh
python scripts/check_repository.py
python -m unittest discover -s tests -v
python scripts/check_inputs.py reports
python training/run_location_baselines.py --check --auc-tolerance 0.002
python data_collection/count_training_pixel_dates.py
```

`check_repository.py` checks syntax, Markdown files/anchors, script references,
configuration references and prohibited machine-specific wording. Add
`--external` to probe public web links. The network check also covers public URLs in configuration, saved scene metadata,
Python, HTML, CSV and dependency files. It uses HEAD requests to avoid downloading
large imagery and resolves pinned GitHub dependency revisions to commit pages. Access-denied responses and timeouts are
unverified links, not proof of missing pages. Tests also check the saved
partition counts, labels and recomputed metrics. Baseline refitting defaults to
an absolute tolerance of 1e-12. A refit with the same package versions on a
different operating system reproduced all classification metrics but differed
by 0.001461 AUC for one nuisance-only random forest. For that cross-platform
check, use `python training/run_location_baselines.py --check --auc-tolerance 0.002`;
it reports the actual difference and keeps classification-metric checks strict.
The saved metrics remain those from the recorded execution environment.

## Retrain the exact published partitions

First configure external arrays/backbone and a new experiment destination. Copy
only the seed manifests into that destination:

```sh
python scripts/prepare_run.py runs/experiment
# Set FULL_NUISANCE_BALANCE_OUTPUT to runs/experiment in your override file.
python scripts/check_inputs.py training
python training/train_frozen_sesimbra.py --pools 1 2 3 5
python training/run_location_baselines.py
python training/train_simple_clay_heads.py
python training/verify_full_nuisance_trial.py
python training/verify_simple_clay_heads.py
python training/build_model_catalog.py
python training/summarize_full_nuisance_trial.py
```

`prepare_run.py` also copies the selection protocol and balance table required
for verification and reports. It never copies weights or previous predictions.
The training command creates `clay_full_sweep` and refuses to overwrite it.
An optional `--embedding-cache` reuses a cache only after matching IDs, array
hashes and encoder protocol. This still requires the backbone file for hash
verification and a CUDA runtime for head training.

To generate new partitions from the included eligible inventory, configure
fresh `FULL_NUISANCE_MATCHING_POOL` and `FULL_NUISANCE_BALANCE_OUTPUT` paths,
then run `prepare_nuisance_pool.py` and `balance_absent_first.py` under `training/`.
The initializer is included, but time-limited solver reruns can select different
equivalent partitions. Do not present a newly optimized split as byte-identical
to the published one.

## Fit final heads and apply them

Final training replays the included `training/final_models/selected.csv` and
`split.json`, checking them against the selected union. It does not re-optimize
partitions. The embedding cache must have a companion `protocol.json` recording
128-pixel native chips at 10 m, a 224-pixel encoder input and derived GSD
`10 × 128 / 224`. New checkpoints carry the matching GSD and resize metadata;
inference rejects incompatible settings.

```sh
python scripts/check_inputs.py final
python training/train_final_models.py
python training/verify_final_models.py
python training/summarize_final_models.py
```

For field evaluation, use the saved scene metadata in the default campaign
folder or copy it to your configured folder. Download all three scene arrays:

```sh
python data_collection/sesimbra_s2_campaigns/download_scene.py june
python data_collection/sesimbra_s2_campaigns/download_scene.py july
python data_collection/sesimbra_s2_campaigns/download_scene.py september
python scripts/check_inputs.py field
python training/evaluate_sesimbra_campaigns.py
python training/verify_sesimbra_campaigns.py
python training/report_sesimbra_campaigns.py
```

Set `SESIMBRA_EVALUATION_DIR` to a fresh destination before evaluation. The
included `pixel_date_labels.csv` supplies field labels; rebuilding the combined
S2/PNEO label file also needs licensed PNEO imagery. To rebuild the observation
CSV, run `data_collection/match_sesimbra_field_coordinates.py --output runs/field.csv`
with the three original source files configured. Labels are derived directly
from the measurement workbook and matched one-to-one to coordinates.

For the separate PNEO analysis, configure a fresh `PNEO_SEPTEMBER_INPUTS` and
`PNEO_SEPTEMBER_RESULTS`, then run `training/extract_pneo_september.py` followed
by `training/evaluate_pneo_september.py`. Feature and embedding paths default
to the extractor's output names. This analysis does not apply the final S2 heads.

## Predict from saved embeddings

The checkpoints and embedding cache are external inputs. Set
`FINAL_CHECKPOINTS_DIR` and `FULL_NUISANCE_EMBEDDINGS` as described in
[Configuration](#configuration). This example applies saved heads on CPU;
creating new embeddings still requires the documented encoder environment.

The [reuse loader](../training/final_model_io.py) restores the head and applies its
saved threshold. This example runs from the repository root and predicts from
the configured full-chip Clay embedding cache. Pooling is applied to Clay's
encoded patch tokens; each output corresponds to the same row of `chip_ids`.

```python
import numpy as np
import config
from training.final_model_io import load_head, predict_embeddings
from training.patch_pool import cache_local_slice

with np.load(config.get_path("FULL_NUISANCE_EMBEDDINGS")) as cache:
    for name in ("small_mlp_pool5", "mlp_pool1"):
        path = config.get_path("FINAL_CHECKPOINTS_DIR") / name / "model.pt"
        model, checkpoint = load_head(path, device="cpu")
        lo, hi = cache_local_slice(checkpoint["pool_size"])
        pooled = cache["centre_patches"][:, lo:hi, lo:hi].mean(axis=(1, 2))
        embeddings = np.concatenate([cache["cls"], pooled], axis=1)
        probabilities, present = predict_embeddings(model, checkpoint, embeddings)
        print(name, cache["chip_ids"], probabilities.numpy(), present.numpy())
```

For new imagery, use the frozen backbone, bands, normalization and pooling
recorded in the checkpoint. See the [encoder implementation](../training/train_frozen_sesimbra.py)
and [final-model catalog](../training/final_models/model_manifest.json).

## Acquiring a new collection

Follow the [chip acquisition README](../chip_generation/README.md) for complete
footprints, overlapping tiles, quality checks, consolidation and coverage audits.
Configure the resulting arrays and prepared eligible inventory before feature
preparation and selection. To fit a newly selected union, use
`python training/train_final_models.py --new-split`; omit the flag to replay the
included final split. Regenerate embeddings whenever their source arrays change.

## Automated maintenance checks

CI installs the pinned data environment, runs the regression tests and baseline
check, checks public links, and runs Ruff and high-confidence Vulture checks.
It regenerates the three scientific reports and field observation map, then
requires all non-PNG tracked files to remain unchanged. PNGs are rendered as a
smoke check; font and rasterization differences between platforms can change
image bytes, so those bytes are not required to match. Regenerate published
outputs when their maintained inputs change; do not edit generated reports alone.
