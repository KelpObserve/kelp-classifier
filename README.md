# kelp-observe-classifier

Kelp presence/absence classification for the Arrábida coast near Sesimbra,
Portugal, using Sentinel-2 imagery, frozen Clay v1.5 features and classifier
heads. The encoder uses resize-adjusted grid spacing of approximately 5.714 m
for each 128→224 input; the native imagery and output grid remain 10 m.
CCMAR diver observations provide the field evaluation reference.
Pléiades Neo imagery supports a separate exploratory analysis.

The three-seed comparison uses 50 m location groups and a 2:1 PRESENT/ABSENT
ratio. Two final heads fit the 725-chip union. They predict
PRESENT at all 52 labelled field pixel-dates, so it does not establish
presence/absence discrimination. See the [campaign evaluation](training/sesimbra_evaluation/README.md).
Read the [technical study report](technical_report.md) for a paper-style account
of the data, methods, results, figures and conclusions.
See [methods and scientific results](docs/SCIENCE.md) for the full account.

## What the pool sizes mean

Clay converts each image into a grid of learned feature vectors, called **patch
tokens**, together with a summary of the whole image. **Central pooling** selects
features around the observation and averages them into one feature vector:

- **Pool 1:** one central token.
- **Pool 2:** the average of a central 2×2 square, containing 4 tokens.
- **Pool 3:** the average of a central 3×3 square, containing 9 tokens.
- **Pool 5:** the average of a central 5×5 square, containing 25 tokens.

The classifier receives both this central feature vector and the whole-image
summary. Larger pools summarize a wider central area. The numbers describe
learned feature tokens, **not satellite pixels**: Clay processes the full image
before pooling, so even pool 1 can use surrounding image context. Pooling does
not change the original image footprint or create finer spatial resolution.

## Limitations

- **Small dataset, especially ABSENT observations.** The training union reuses
  51 distinct ABSENT location–observation-date records across image dates.
  [Current counts](data_collection/README.md#current-experiment-observations-and-s2-pixel-date-labels) distinguish chips from independent observations. Multiple image dates for one observation do not create new
  independent field evidence. The local field dataset has only 90 survey events
  (73 PRESENT, 17 ABSENT), collected along nearby, repeatedly surveyed transects.
- **Possible overfitting.** Frozen features reduce the number of fitted weights,
  but the heads still learn from few independent observations. Repeated model,
  regularization and threshold selection can overfit validation data. The final
  models have no independent internal test set; the three split seeds reuse
  observations and are not three independent experiments.
- **Sentinel-2 collapses different Sesimbra observations into one pixel.** The 90 field
  events become 52 labelled 10 m pixel-dates: 47 PRESENT and only 5 ABSENT.
  Under the any-presence rule, an empty quadrat sharing a pixel with an occupied
  quadrat does not supply an ABSENT pixel label. Fine-scale variation is lost,
  and an empty sampled quadrat does not prove absence across the whole pixel.
  Bands B05 and B11 are native 20 m measurements resampled to the 10 m grid;
  resampling does not add spatial detail (see the
  [Sentinel-2 mission specification](https://sentiwiki.copernicus.eu/web/s2-mission)).
  Clay tokens also use whole-chip context.
- **Spatial context crosses partitions.** Shared scenes and overlapping chip
  footprints were allowed because the small dataset, particularly the limited
  independent ABSENT observations, would otherwise not support the required
  class-balanced train/validation/test partitions. This was a practical
  compromise: the 50 m grouping rule keeps nearby observations together but
  does not isolate the full image context. Reported split performance may
  therefore be optimistic relative to evaluation on entirely new scenes or
  geographically separated sites.
- **Limited transfer evidence.** The [field evaluation](training/sesimbra_evaluation/README.md)
  has only five ABSENT pixel-dates. Both heads predict PRESENT
  for all 52 labels (ABSENT recall 0; Macro F1 0.475), so this does not establish
  reliable operational presence/absence mapping. The 50 m grouping rule
  allows shared scenes and overlapping 1.28 km chip footprints across splits.
  New-site and new-campaign evaluation, with more independent absences and
  uncertainty intervals computed over independent groups, remains necessary.
- **Input consistency check.** All Sentinel-2 images use the same model
  normalization. Source-image scaling and offset conventions have not been
  independently verified across the complete collection; no resulting bias
  has been demonstrated.

## Quick start: inspect and reproduce reports

Use Python 3.11. Reporting does not need imagery, model weights or a GPU.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reporting.txt
python scripts/check_repository.py
python training/summarize_full_nuisance_trial.py
python training/summarize_final_models.py
```

The report commands regenerate their README and figures from included tables.
The default report paths work without a local configuration. For execution,
`config.json` supplies local data/model paths; inspect configured resources with
`python scripts/show_resources.py`. For overrides, copy
`example_config.json` to the ignored `config.json`, or set `KELP_CONFIG` to a
separate JSON file containing a `paths` object. Relative paths are resolved
against the repository root, not the shell's working directory.

## Choose a workflow

| Goal | Instructions | Additional inputs |
|---|---|---|
| Understand labels, model design and limitations | [Scientific account](docs/SCIENCE.md) | None |
| Inspect sources and collection rules | [Data collection](data_collection/README.md) | None for saved tables |
| Fetch and consolidate new S2 chips | [Chip generation](chip_generation/README.md) | Source candidates and network access |
| Train or verify current classifiers | [Training](training/README.md) | Canonical arrays, Clay weights and CUDA |
| Apply final classifiers to field campaigns | [Campaign evaluation](training/sesimbra_evaluation/README.md) | Campaign images and final weights |
| Supply required artifacts and configure paths | [Reproduction and inputs](docs/REPRODUCIBILITY.md) | Depends on workflow |
| Find scripts and understand outputs | [Repository structure](docs/STRUCTURE.md) | None |

## Reproducibility status

The [reproducibility matrix](docs/REPRODUCIBILITY.md#reproducibility-matrix)
distinguishes table checks, checkpoint replay, retraining and source reconstruction.

Saved partitions, selection inventories, predictions, metrics, model catalogs
and scientific reports are included. Image arrays, embeddings and model binaries
must be supplied separately at the paths configured in `config.json`.
The full CUDA environment is pinned in `requirements-cuda.lock`.

The included results cover twelve original-architecture MLPs, twelve simpler
heads, two final heads and field predictions/rasters. Chip coverage and quality
handling are part of the [acquisition procedure](chip_generation/README.md).
Source preparation remains distinct from replaying these
artifacts: the original GBIF export DOI is not recorded, and rebuilding upstream
depth evidence requires the supplied preparation inputs. See
[reproduction and inputs](docs/REPRODUCIBILITY.md) for commands and required inputs.

## Development checks

```sh
python -m pip install -r requirements-data.txt
python scripts/check_repository.py
python -m unittest discover -s tests -v
python scripts/check_inputs.py reports
python training/run_location_baselines.py --check --auc-tolerance 0.002
```

The baseline command reports any cross-platform AUC drift; classification metrics
remain strict. Omit the tolerance option for exact refitting in the recorded
execution environment. See [reproduction details](docs/REPRODUCIBILITY.md).

Source citations and internal documentation links are retained. Runtime service
endpoints are configured in `configs/services.json`. Run
`python scripts/check_repository.py --external` for a network link check;
access blocks and timeouts are reported separately from missing pages.
