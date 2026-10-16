## September Pléiades Neo analysis

[Pléiades Neo](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/missions/pleiadesneo) is an Airbus optical Earth-observation satellite constellation.
We abbreviate it as **PNEO**. Its six-band multispectral imagery has approximately
1.2 m pixels; the separately available panchromatic imagery has 0.3 m pixels.
This analysis uses the six-band multispectral product.

The image acquired on 3.09.2026 is compared with the 30 quadrats surveyed on
2.09.2026: 20 PRESENT and 10 ABSENT in six depth-by-transect groups. Each quadrat
occupies a separate native pixel. Feature and embedding exports were freshly
extracted and their labels and coordinates checked against the current matched
field file. This run recalculates the analysis from those exports.

### Features and evaluation

[Extraction code](../extract_pneo_september.py) preserves the PNEO exploration's
feature definitions: means, standard deviations and ranges in 5×5 windows for
red, green, blue, near infrared, red edge and deep blue; six Lyzenga band-pair
indices; and frozen Clay embeddings from native 224×224 chips at approximately
1.2 m spacing. Lyzenga indices combine log-transformed bands using a coefficient
estimated from low-NIR water pixels to reduce water-column effects. They do not
guarantee removal of depth or seabed differences.

The table compares different **inputs to the same type of classifier**:

- **Spectral statistics:** for each of the six bands, calculate the mean,
  standard deviation and range (maximum minus minimum) in a 5×5-pixel window
  around the quadrat. These 18 values describe local brightness and variation
  over approximately 6×6 m. This is a simple feature baseline implemented in the
  [extraction script](../extract_pneo_september.py).
- **Lyzenga indices:** water absorbs different colours at different rates, so
  a depth change can alter the measured signal even if the bottom stays the same.
  Following [Lyzenga (1978), *Passive remote sensing techniques for mapping water
  depth and bottom features*, Applied Optics 17, 379–383](https://doi.org/10.1364/AO.17.000379),
  we subtract an estimated deep-water signal, take logarithms, and combine pairs
  of bands using a scene-estimated attenuation ratio. We average each resulting
  index over the same 5×5 window. The six pairs are deep blue/blue, deep blue/green,
  blue/green, green/red, blue/red and deep blue/red. These six inputs aim to
  reduce depth-related variation, but do not guarantee its removal or isolate kelp.
- **Spectral statistics + Lyzenga:** concatenate the 18 spectral statistics and
  six window-mean indices into **24 input values** for one classifier. This tests
  whether local brightness/variation and the depth-adjusted indices complement
  each other; it is not an average of two models' predictions.
- **Clay CLS + pool 1 / pool 5:** the frozen
  [Clay v1.5 model](https://clay-foundation.github.io/model/release-notes/specification.html)
  encodes the full 224×224 chip. We concatenate its global image summary (CLS)
  with either the central patch token (pool 1) or the mean of the central 5×5
  patch tokens (pool 5), producing 2,048 input values. These patch tokens can use
  information from the whole chip; they are not individual native PNEO pixels.

All five feature sets feed [logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression),
a classifier that learns a weighted combination of its inputs and converts it
to a PRESENT probability. The [evaluation code](../evaluate_pneo_september.py)
standardizes each input using training-only means and standard deviations before
fitting the classifier.

PNEO Clay preprocessing uses scene-water band means and standard deviations,
geographic and observation-week encodings, and the original central-token
convention. It differs from S2 preprocessing. Scene-wide water statistics use
unlabelled image pixels, including the held-out area. These PNEO embeddings must
not be passed directly to the final Sentinel-2 heads.

The [evaluation script](../evaluate_pneo_september.py) fits a separate logistic
regression for each feature set: C=0.01, balanced class weights and threshold
0.5, all fixed. StandardScaler is fitted on training quadrats only. Each of the
six folds holds out all five quadrats of one depth-by-transect group; every
quadrat receives exactly one held-out prediction. The Lyzenga classifier uses
the six window-mean indices. The spectral classifier uses 18 statistics.

| Features | PRESENT precision | PRESENT recall | ABSENT precision | ABSENT recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Spectral statistics | 0.632 | 0.600 | 0.273 | 0.300 | 0.451 |
| Lyzenga indices | 0.714 | 0.500 | 0.375 | 0.600 | 0.525 |
| Spectral statistics + Lyzenga | 0.500 | 0.400 | 0.143 | 0.200 | 0.306 |
| Clay CLS + pool 1 | 0.583 | 0.700 | 0.000 | 0.000 | 0.318 |
| Clay CLS + pool 5 | 0.609 | 0.700 | 0.143 | 0.100 | 0.384 |

Always predicting PRESENT gives Macro F1 0.400. This exploratory comparison
reuses one image and six adjacent transects, with overlapping image windows;
it is not independent geographic validation. The best result, 0.525 for
Lyzenga indices, provides limited discrimination on this small sample.

### Within-transect comparisons

For each feature, subtracting the median of its transect removes a shared
offset. It does not remove all environmental differences. We compare PRESENT
and ABSENT anomalies and separately compare their median difference across
six transects. Benjamini–Hochberg correction is applied across the 30 features
separately for each test family.

The strongest unadjusted anomaly comparison is deep-blue/green Lyzenga:
Mann–Whitney p=0.032, adjusted p=0.418. Its PRESENT-minus-ABSENT median difference
is positive in five of six transects, but the paired Wilcoxon p=0.094 becomes
0.772 after correction. No feature is significant at adjusted p<0.05.
The quadrats are spatially related, further limiting interpretation.

### Files and reproduction

[Features](../pneo_september/features.csv), [held-out predictions](../pneo_september/predictions.csv),
[overall metrics](../pneo_september/metrics.csv), [fold metrics](../pneo_september/fold_metrics.csv),
[within-transect effects](../pneo_september/within_transect_effects.csv), and
[protocol/input hashes](../pneo_september/protocol.json) preserve this analysis.

Configure `PNEO_SEPTEMBER_FEATURES`, `PNEO_CLAY_METADATA`,
`PNEO_CLAY_EMBEDDINGS`, `FIELD_OBSERVATIONS` and a fresh
`PNEO_SEPTEMBER_RESULTS` directory using [example_config.json](../../example_config.json):

```sh
python training/evaluate_pneo_september.py
```

To regenerate inputs from licensed imagery, configure the imagery and Clay
resources plus a fresh `PNEO_SEPTEMBER_INPUTS` directory, then run:

```sh
python training/extract_pneo_september.py
```

Point the three feature/metadata/embedding config keys at the generated
`features_per_quadrat.csv`, `meta.csv` and `identity_pneo_field.npz`, respectively,
before evaluation.
Source imagery and binary embeddings are external inputs. No licensed RGB
background figures are generated or included.
