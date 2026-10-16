# Training classifier

The [current experiment report](full_nuisance_balance/README.md) documents
whole-chip nuisance-balanced partitions, whole-chip baselines and frozen Clay
with central token pools 1, 2, 3 and 5. All models use the same saved partitions.

## Dataset and partitions

The train/validation/test partitions are constructed by
[`balance_absent_first.py`](balance_absent_first.py), using the grouping rules in
[`split_groups.py`](split_groups.py). This is a two-stage procedure: first divide
all eligible ABSENT chips into location groups and partitions, then select
PRESENT chips to match the ABSENT distribution within each partition. The same
saved partitions are used for every comparison model.

### 1. Prepare eligible chips and balancing features

[`prepare_nuisance_pool.py`](prepare_nuisance_pool.py) reads the
[eligible inventory](../data_collection/eligible_chips.csv): 984 PRESENT and
214 ABSENT chips. It verifies each array's SHA-256 and applies the same central
5×5 quality exclusion to both labels: reject a chip if any SCL pixel in
`[62:67, 62:67]` is missing (0), saturated/defective (1), shadow (3), cloud (8/9)
or cirrus (10). This excludes two ABSENT chips, leaving 984 PRESENT candidates
and 212 ABSENT chips.

The splitter retains rows with depth from −20 to 0 m inclusive,
`depth_disagreement <= 1`, no ambiguous label, and an absolute image/observation
date offset of at most 10 days for PRESENT or 30 days for ABSENT. It balances
four features: observation depth and the fractions of the **whole native
128×128 chip** flagged as cloud/shadow/cirrus, missing, or saturated/defective.
Whole-chip fractions may be nonzero even though the central quality check passes.

### 2. Build groups that cannot cross partitions

[`group_rows()` in `split_groups.py`](split_groups.py) joins rows sharing an
observation ID, chip ID or array SHA-256. It also joins neighbours within 50 m
using observation coordinates and, separately, chip-centre coordinates. These
connections are transitive: a chain of neighbours forms one group even if its
endpoints are more than 50 m apart. Groups are built across both labels before
partitioning. All selected members of a group must belong to the same partition;
a PRESENT candidate may be omitted, but cannot be assigned to a different
partition from retained members of its group.

### 3. Partition all ABSENT chips first

For each seed **42, 123 and 456**, `partition_absent()` in
[`balance_absent_first.py`](balance_absent_first.py) starts from the included
[ABSENT initialization assignments](../data_collection/split_initialization.csv).
It keeps all 212 eligible ABSENT chips and searches over whole-group assignments
with nominal **70% / 15% / 15%** train/validation/test targets. These are approximate
size targets, not exact proportions: the bounds are the rounded target counts
with allowances of ±10 chips for training and ±5 for validation/test. Each
partition must contain at least five ABSENT groups.

The search compares each partition's feature distributions with the complete
ABSENT population. It measures empirical cumulative-distribution differences
at every unique depth and at up to 31 quantile cut points for each quality
fraction. Its score is the mean plus the maximum of the per-feature/per-partition
maximum differences, with a small count-deviation penalty (weight 0.02).
Five seeded annealing restarts each make 8,000 proposals, swapping two group
assignments or moving one group. This is a heuristic search, without an
optimality guarantee.

### 4. Select and assign matching PRESENT chips

With ABSENT assignments fixed, the same script uses SciPy's HiGHS mixed-integer
optimizer to select PRESENT candidates and assign them to partitions. Every
partition must contain **exactly two PRESENT chips per ABSENT chip**. Each chip
can be selected at most once, each group can occupy at most one partition, and
groups containing ABSENT chips are fixed to their ABSENT partition. Thus each
seed retains 424 PRESENT and all 212 ABSENT chips: **636 chips total**.

The objective matches PRESENT to ABSENT **within each partition**, using
standardized means (clipped to ±5 standard deviations) and empirical cumulative
distributions at up to 21 quantile cut points per feature. Depth also uses every
ABSENT depth and a cut immediately below it. The objective penalizes weighted
absolute mean/CDF differences, the largest CDF difference for each feature, and
the largest CDF difference overall. Depth terms receive 3× weight and
cloud/shadow/cirrus terms 1.5× weight; train, validation and test have equal
priority. A tiny seeded cost breaks ties between selections. Feature combinations
and correlations are not matched.

The solver has a 180-second limit and a 1.5% relative optimality-gap target. If
it finds no feasible solution, it retries for 600 seconds with a 2.5% gap target.
Per-seed achieved gaps are recorded in the [selection protocol](full_nuisance_balance/protocol.json).
These are optimized approximate distribution matches, not identical feature distributions.

### Saved partitions and balance evidence

The [split-count table](full_nuisance_balance/split_counts.csv) records the
current train, validation and test class counts for every seed.

The [seed-aware manifest](full_nuisance_balance/selected.csv) records every
assignment; [split counts](full_nuisance_balance/split_counts.csv) and the
[selection protocol](full_nuisance_balance/protocol.json) record counts,
objectives and solver results. [Feature-balance measurements](full_nuisance_balance/feature_balance.csv)
report PRESENT-versus-ABSENT Kolmogorov–Smirnov distances and standardized mean
differences for each feature, seed and partition. The
[experiment report](full_nuisance_balance/README.md#distribution-balance) summarizes
them. [`verify_full_nuisance_trial.py`](verify_full_nuisance_trial.py) checks the
source features, grouping and class ratios against the saved evidence.

Training fits the classifier; validation selects checkpoints and decision
thresholds; test measures performance after those choices. **Classifier
predictions and scores are not used to select chips or partitions.** However,
all-pool covariates, including evaluation-partition covariates, inform the
balancing procedure. These are exploratory, covariate-balanced partitions;
repeated test inspection means they are not an untouched independent test.

Spectral appearance, acquisition/processing metadata and SCL land/water
fractions are not balanced. Independent land-overlap and reliable sunglint
measurements remain unavailable. Scene IDs do not define groups, so shared
acquisitions and overlapping 1.28 km chip footprints can cross partitions.

For exact reproduction, use the saved manifests. Rerunning the time-limited
optimizer can produce a different equivalent selection; see [Reproduction](#reproduction).
The complete deduplicated union across seeds is subsequently used
for a **separate final train/validation fit**, with no independent test set;
see [Final reusable models](#final-reusable-models) and its
[split/training script](train_final_models.py).

## Full-chip Clay and baselines

Clay receives six bands (B02, B03, B04, B05, B08 and B11) from the whole native
128×128 chip, bilinearly resized to 224×224 and normalized with `CLAY_META`.
Encoder GSD is derived from the resize: `10 × 128 / 224 ≈ 5.714 m`.
The native chip footprint remains 1,280 m; resampling adds no spatial detail.
Training and field inference share `encoder_geometry.py`; cache and checkpoint
metadata must match that geometry.
Time and geographic encodings are zero. The frozen backbone produces a CLS
vector and a 28×28 patch grid. The MLP receives CLS concatenated with the average
of central patch tokens. Pooling happens after whole-chip encoding, so every
pool can use the surrounding image. Depth and quality fields are not Clay inputs.

| Pool | Slice on 28×28 patch grid | Nominal pooled footprint |
|---:|---|---|
| 1 | `[13:14,13:14]` | approximately 46×46 m |
| 2 | `[13:15,13:15]` | approximately 92×92 m |
| 3 | `[12:15,12:15]` | approximately 137×137 m |
| 5 | `[11:16,11:16]` | approximately 229×229 m |

These footprints describe pooled tokens, not the encoder's full receptive field.
The head is 2048 → 512 → 128 → 1, ReLU and dropout 0.3. Training uses focal loss
(alpha 0.5, gamma 2), Adam (learning rate 0.001, weight decay 0.0001), batch size
64, at most 100 epochs and patience 25. All training rows are used. Checkpoints
maximize validation Macro F1 at 0.5; thresholds maximize validation Macro F1
over 0.05–0.95 in steps of 0.01, preferring 0.5 on ties.

**LR (logistic regression)** learns a linear decision rule from the supplied
features and converts its output to a PRESENT probability. **RF (random forest)**
combines many decision trees and can capture nonlinear relationships between
features. In the baseline tables, their inputs are whole-chip band statistics or
depth and quality measurements. **Clay + logistic regression** is a different
model: its inputs are the frozen Clay embeddings, not those image statistics.

Whole-chip image baselines use 30 band-statistic features. Nuisance baselines use
depth and three whole-chip SCL quality fractions. Logistic regression uses
training-only scaling, C=0.1 and balanced class weights; random forests use 120
trees, maximum depth 3, minimum leaf size 3 and balanced class weights. Missing
values use training medians. Their fixed decision threshold is 0.5.

The [simpler-head runner](train_simple_clay_heads.py) fits logistic regression
and a 2048 → 32 → 1 MLP on the same cached Clay embeddings, for pools 1 and 5.
The small MLP has one hidden layer of 32 units and fewer trainable parameters
than the original MLP, testing whether a simpler classifier suits this small dataset.
It selects logistic C from 0.0001, 0.001, 0.01, 0.1, 1 and 10, and small-MLP
weight decay from 0.1, 0.01 and 0.0001, by validation Macro F1 at 0.5. Ties prefer
stronger regularization; thresholds are then selected on validation.
[Results](full_nuisance_balance/simple_heads/metrics.csv) and
[selection evidence](full_nuisance_balance/simple_heads/validation_candidates.csv)
record all choices.

## Results

The [current experiment report](full_nuisance_balance/README.md) contains the
canonical three-seed metric tables. The [scientific account](../docs/SCIENCE.md)
explains interpretation, possible confounders and field performance.

## Reproduction

Use [Reproduction and required inputs](../docs/REPRODUCIBILITY.md) for environment
setup, configuration and the ordered commands. It distinguishes
[training on published partitions](../docs/REPRODUCIBILITY.md#retrain-the-exact-published-partitions)
from re-optimizing a new selection. The saved manifests are authoritative for
the reported results; time-limited optimization can return different selections.

The verifiers check source arrays and features, group separation, saved
checkpoints and validation choices. Report generation alone checks neither
encoder inference nor checkpoint replay; see the
[reproducibility matrix](../docs/REPRODUCIBILITY.md#reproducibility-matrix).

## Saved models

The 24 evaluation models comprise eighteen PyTorch checkpoints and six serialized
logistic-regression pipelines. These files are excluded from Git. The
[model catalog](model_manifest.json) records paths relative to `CHECKPOINTS_DIR`,
hashes, seeds, pools and validation thresholds. Checkpoints, the frozen backbone,
chip arrays and generated embedding cache are described and referenced in
[`example_config.json`](../example_config.json). Two final reusable heads are described below.
Inference must use the matching full-chip preprocessing, CLS concatenation and
central pooling. A 10 m output grid does not establish 10 m independent resolution.
The separate [Sesimbra campaign evaluation](sesimbra_evaluation/README.md) applies
the two final models to June, July and September images.

## Final reusable models

[train_final_models.py](train_final_models.py) fits the small MLP with pool 5
and the original MLP with pool 1 on the selected union. Both use grouped
train/validation partitions; there is no independent test partition.

The [final-model report](final_models/README.md) owns partition counts,
hyperparameters, selected thresholds and metrics. The
[checkpoint catalog](final_models/model_manifest.json) records hashes, and
[final_model_io.py](final_model_io.py) loads each architecture and applies its
saved threshold to matching embeddings. Both final checkpoints are included in
Git under `training/final_models/`; comparison checkpoints remain external.
See the [final fitting commands](../docs/REPRODUCIBILITY.md#fit-final-heads-and-apply-them)
and [inference example](../docs/REPRODUCIBILITY.md#predict-from-saved-embeddings).

## Sesimbra field application

[evaluate_sesimbra_campaigns.py](evaluate_sesimbra_campaigns.py) extracts the
52 campaign-matched chips, applies both final heads and rasterizes pool 1.
Predictions, class recalls and Macro F1 are regenerated from the fitted heads. See the
[Sesimbra campaign observations prediction](sesimbra_evaluation/README.md) for the map, metrics and commands.
[verify_sesimbra_campaigns.py](verify_sesimbra_campaigns.py) validates crops,
head replay, raster reconstruction and metrics;
[report_sesimbra_campaigns.py](report_sesimbra_campaigns.py) builds the report.

The separate [September PNEO exploration](sesimbra_evaluation/README.md#september-pléiades-neo-analysis) restores
spectral/Lyzenga features and Clay embeddings for 30 quadrats and evaluates
PNEO-specific logistic regression by held-out transect.
