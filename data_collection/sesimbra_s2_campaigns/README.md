# Sesimbra campaign-specific Sentinel-2 images

Configure the campaign imagery using the descriptions and references in
[example_config.json](../../example_config.json).

| Campaign | Observation dates | Selected scene | Acquisition date | SCL at 30 observation locations |
|---|---|---|---|---|
| June | 2.06.2026 | S2B_29SMC_20260602_0_L2A | 2.06.2026 | 30 water |
| July | 6–7.07.2026 | S2A_29SMC_20260709_0_L2A | 9.07.2026 | 30 water |
| September | 2.09.2026 | S2B_29SMC_20260903_0_L2A | 3.09.2026 | 28 water, 2 unclassified |

Each `june/`, `july/`, and `september/` directory contains `scene.npz`,
`scene_item.json`, and `acquisition.json`. The latter records the grid,
source metadata, point-level quality counts and NPZ SHA-256. All crops use the
same 333 × 570 grid at 10 m spacing in EPSG:32629, over the existing benchmark
extent. The seven layers are blue, green, red, red-edge 1, near infrared,
shortwave infrared 1, and scene classification (SCL). Source digital numbers
are retained; scale/offset metadata are in each scene item. Spectral bands use
bilinear resampling and SCL uses nearest-neighbour resampling. Coarser source
bands retain their native resolving power despite the 10 m output grid.

The selected images have no cloud, cirrus, cloud shadow or missing-data flags
at the observation locations. This is an automated point-level check; the two
unclassified September locations are retained in evaluation. Full-chip quality
measurements are recorded in the [chip manifest](../../training/sesimbra_evaluation/chip_manifest.csv).
The other 2 June acquisition (Sentinel-2A) has cirrus at 26 June locations.
The closer 5 July acquisition has cirrus at 21 July locations and cloud at 5,
so 9 July was selected. The catalogue has no acquisition over the site on
1 or 2 September. Candidate dates and tile cloud percentages are recorded in
`catalogue_candidates.json`; tile-wide cloud cover was not used as a substitute
for checking observation locations.

`field_observations.csv` is the matched 90-event observation snapshot for these
inputs. Group observations by campaign and pixel for evaluation; do not merge
labels from different campaigns into a single pixel label.
These campaign inputs are evaluated with both final classifiers in the
[Sesimbra campaign observations prediction](../../training/sesimbra_evaluation/README.md).
The [current training results](../../training/full_nuisance_balance/README.md)
report only the held-out partitions of the wider training dataset.

Reproduce downloads with Python, numpy and rasterio, from the repository root after configuring `config.json`:

```sh
python data_collection/sesimbra_s2_campaigns/download_scene.py june
python data_collection/sesimbra_s2_campaigns/download_scene.py july
python data_collection/sesimbra_s2_campaigns/download_scene.py september
```

The NPZ image arrays are external inputs described in [example_config.json](../../example_config.json). Metadata and the download script
are kept in the repository. Verify a transfer against `SHA256SUMS` with
`sha256sum -c SHA256SUMS` alongside the image files.

Create the combined PNEO/S2 pixel-date labels with
`python data_collection/sesimbra_s2_campaigns/count_pixel_dates.py`.
This reads `S2_CAMPAIGNS_DIR` and `PNEO_DIR` from `config.json`.
