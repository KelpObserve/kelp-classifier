# Data collection

## Sources

These sources provide observations of kelp presence and recorded absence. They
are separate from the CCMAR Sesimbra surveys used for field evaluation.

### GBIF — Global Biodiversity Information Facility

[GBIF](https://www.gbif.org/what-is-gbif) brings together species occurrence
records from research institutions, monitoring programmes and other publishers.
The input is the saved European kelp occurrence download, obtained through the
[GBIF download service](https://www.gbif.org/occurrence/download).
[`extract_gbif_kelp.py`](gbif/extract_gbif_kelp.py) selects kelp records and splits
them by `occurrenceStatus`. Input and output locations are configured with
`GBIF_MASTER` and `GBIF_DIR`. The saved master has 201,699 rows; these are raw
records of different dates and taxa, not independent training observations.
A download-specific GBIF DOI is not recorded in the repository; the portal link
is a source reference, not a substitute for that dataset citation.

### OBIS — Ocean Biodiversity Information System

[OBIS](https://obis.org/about/visionmission/) is the marine biodiversity data
network under IOC-UNESCO's oceanographic data programme.
[`extract_obis_kelp.py`](obis/extract_obis_kelp.py) queries the
[OBIS occurrence API](https://api.obis.org/v3/occurrence) for the kelp taxa within
the Europe polygon, with observation dates from 1.01.2016. Recorded absences
are requested explicitly using the API's absence filter. Outputs use `OBIS_DIR`.
Many contributing datasets are also published through GBIF.

### JNCC — Joint Nature Conservation Committee, Marine Recorder

The [Marine Recorder public UK snapshot v20220124](https://jncc.gov.uk/resources/b9934e31-39b6-41f9-9364-d1e93db68307)
contains UK seabed surveys, species, habitats and physical attributes.
[`extract_jncc_kelp.py`](jncc/extract_jncc_kelp.py) reads the v52 Access database
configured as `JNCC_DATABASE`, using `mdbtools`, and writes to `JNCC_DIR`.
It joins species records to survey coordinates and dates. Absence is explicitly
encoded by `PA='N'` or `SACFORN='N'`; a species not listed is not automatically absent.

If the original database is missing, run
`python data_collection/jncc/download_snapshot.py`. It downloads the pinned public
snapshot, extracts its single Access database to `JNCC_DATABASE`, records SHA-256
provenance and refuses to overwrite an existing database.

## Deduplication and observation selection

[`dedup_candidates.py`](dedup_candidates.py) consolidates the six source exports
(three sources × two classes PRESENT and ABSENT). The current observation tables are reproduced by
[`build_observation_summary_tables.py`](build_observation_summary_tables.py), which
uses the same key and adds the current source, precision, species and absence rules.

An observation is one **status + location + calendar date**. Latitude and
longitude are rounded to four decimal places for deduplication. Species names
are combined within a group; representatives prefer GBIF, then OBIS, then JNCC.
Original source IDs are retained. Observations on different days remain separate;
neither nearby dates nor a distance radius merge separate visits.

Only valid coordinates and full calendar dates on or after 1.01.2016 qualify. PRESENT records must
report coordinate uncertainty ≤10 m or have unknown uncertainty; known values
above 10 m are excluded. Unknown uncertainty is retained, not assumed to be zero.
ABSENT records have no coordinate-uncertainty cutoff because there are very few
candidate ABSENT observations, since applying a cutoff would further reduce this
already limited class.

The eight-species list below applies to PRESENT only. ABSENT is a shared class
across the source kelp taxa. A deduplicated absence is excluded if any contributing
source member has any recorded kelp presence within 10 m and ±5 calendar days.
This check uses original coordinates and all available presence evidence,
including taxa outside the training list.
The exclusion is one-way: an absence never removes a presence.

## Obtaining Sentinel-2 chips

The collection starts from **8,275 PRESENT and 219 ABSENT candidate observations**
selected below. ABSENT observations are the limiting class, so the collection
strategy is to search for usable images for all ABSENT candidates first, then
collect PRESENT chips toward a target of **two PRESENT chips per ABSENT chip**.
The target concerns accepted chips, not the number of candidate observations:
not every observation has a usable image, and an ABSENT observation can supply
several image dates. We therefore do not need to download images for every
PRESENT candidate. The per-class image limits and quality rules are listed below.

[`fetch_present.py`](../chip_generation/fetch_present.py) and
[`fetch_absent.py`](../chip_generation/fetch_absent.py) call the shared
[`chip_common.py`](../chip_generation/chip_common.py) downloader. It searches
[Earth Search](https://earth-search.aws.element84.com/v1) for Sentinel-2 Level-2A
images and reads a 128×128-pixel crop at 10 m spacing around each observation.
The stored layers are coastal aerosol, blue, green, red, red-edge 1, near infrared,
shortwave infrared 1 and the scene-classification layer (SCL). The chip supplies
image context; its observed location supplies the label.

For the collection used by the current models:

- PRESENT retains at most one accepted image per observation, within ±10 days.
- ABSENT retains at most eight accepted image dates per observation, within ±30 days.
- Images must have valid identity, metadata, shape and type. The central 23×23
  region requires cloud/shadow/cirrus ≤40%, water ≥30%, missing data ≤70%, and
  blue-band P95−P05 spread ≥30. The central 5×5 region requires at least 24 valid
  water pixels, with SCL water and blue/green values above 10.
- Candidate images are ranked by date distance and quality. Identical stored
  arrays are deduplicated while retaining their observation associations.

The fetchers enforce the current rules directly: PRESENT uncertainty ≤10 m
or unknown, PRESENT images within ±10 days, and ABSENT images within ±30 days.
Both classes must pass the regional and central 5×5 water checks. Species,
depth and label-consistency checks still determine eligibility for training.
The downloader supports a shift to nearby water and records the offset;
shifted coordinates are not treated as the original survey location.

The consolidated collection stores arrays and their provenance separately:

| File | Role |
|---|---|
| `chips.h5` | The unique image arrays, stored as `uint16` with eight bands and 128×128 pixels per chip. |
| `chips_metadata.csv` | One record per stored chip, including its array index, identity/hash, image scene and quality information. |
| `observation_chips.csv` | Links observations to chips and image dates, retaining the original and chip coordinates and any recorded shift. Multiple observations can refer to the same stored array. |

The fetchers produce chip arrays and acquisition metadata; `chip_generation/consolidate_chips.py` builds
the unique-array collection and its observation-to-chip links. Count observations
from those links, not simply from the number of arrays in `chips.h5`.

The quality filters also have limitations. The central-water check can reject
shoreline or mixed land/water pixels that SCL does not classify as water, even
when the observation is valid and the image is not cloudy. It tests the acquired
chip centre, including any shift, rather than necessarily the original survey
coordinate. The blue-band spread check rejects nearly flat image values, but
can also reject naturally uniform water. It is not a cloud detector: it measures
the difference between the 95th and 5th percentiles of nonzero blue-band values
in the central 23×23 region, with at least 21 nonzero samples required.

The [chip visualizer](../chip_generation/render_chips.py) supports manual
inspection of selected canonical arrays and their central SCL flags.

## Observation statistics

Counts below are **unique location-date observations within each class**, not
raw database rows or chips. The S2 era is defined here as dates from 1.01.2016.

| Cumulative source-selection step | PRESENT | ABSENT | Removed P / A at this step |
|---|---:|---:|---:|
| TOTAL S2-era observations with valid coordinates and a full date, deduplicated across sources | 32,773 | 411 | — |
| Exclude iNaturalist candidates | 29,640 | 411 | 3,133 / 0 |
| PRESENT uncertainty ≤10 m or unknown; no uncertainty cutoff for ABSENT | 21,239 | 411 | 8,401 / 0 |
| Relevant eight-species PRESENT selection | 8,275 | 411 | 12,964 / 0 |
| Exclude ABSENT contradicted by nearby kelp presence | 8,275 | 219 | 0 / 192 |

The source exclusion removes 3,651 raw iNaturalist PRESENT rows but 3,133 unique
observations; these quantities must not be substituted for one another. Of the
8,275 relevant PRESENT observations, 1,739 have a qualifying source record with
uncertainty ≤10 m and 6,536 have unknown uncertainty only. These counts precede
image availability, image quality and depth filtering.

### Sesimbra species selection and CCMAR evidence

The goal is to detect kelp presence at Sesimbra, where *Saccorhiza polyschides*
is the only named kelp in the inspected CCMAR workbooks. The
[eight-species selection](../configs/species.yaml) supplements this target
with European kelps to provide more training examples than the local survey alone:

- *Saccorhiza polyschides* provides examples of the target species itself.
- *Laminaria ochroleuca*, *Phyllariopsis brevipes* and *P. purpurascens* provide
  regional comparisons: they occur along the Portuguese coast, with recorded
  depth ranges overlapping those of *Saccorhiza*. This makes them relevant
  examples of kelp habitats in the target region
  ([Portuguese Findkelp survey](https://sapientia.ualg.pt/entities/publication/870b80f6-491a-4ada-8015-f2f932475461)).
- *Undaria pinnatifida* provides a growth-cycle comparison: like *Saccorhiza*, it
  is a fast-growing annual canopy-forming kelp. Field experiments in southwest
  Britain found similar seasonal cycles and depth distributions
  ([Epstein et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6584561/)).
- *Laminaria hyperborea*, *Saccharina latissima* and *Alaria esculenta* broaden
  the training pool with additional European kelp observations. Their role is
  to help learn a broader kelp-presence class from more locations; they are
  less direct regional comparisons to Sesimbra.

This is a focused European kelp selection rather than a selection of all
seaweeds. The project's policy explicitly excluded *Laminaria digitata* to
reduce intertidal examples when targeting submerged kelp. The exact eight-species
list is a modelling choice, not a demonstrated optimum: we have not tested each
species' contribution to Sesimbra predictions or established that their satellite
appearance is identical.

| Species | Qualifying PRESENT observations | Role |
|---|---:|---|
| *Saccorhiza polyschides* | 2,619 | Directly observed Sesimbra target |
| *Laminaria ochroleuca* | 241 | Southern-European kelp training group |
| *Phyllariopsis brevipes* | 27 | Southern-European kelp training group |
| *Phyllariopsis purpurascens* | 0 | Allowed by the policy; no qualifying source records |
| *Undaria pinnatifida* | 1,536 | Broader regional kelp training group |
| *Laminaria hyperborea* | 4,389 | Additional northern-European kelp examples |
| *Saccharina latissima*, including *Laminaria saccharina* | 5,266 | Additional northern-European kelp examples |
| *Alaria esculenta* | 318 | Additional northern-European kelp examples |

Species counts overlap at mixed-species observations and must not be summed.
The resulting candidate pool has 8,275 PRESENT and 219 ABSENT records. The broader ingestion list in
[`common.py`](common.py) also retains other taxa as evidence for absence checks;
it is not the final model's PRESENT species list.

## Image availability and filtering

The starting pool is **8,275 PRESENT and 219 ABSENT observations**, from the
source-selection table above. Collection prioritises all ABSENT candidates and
then adds PRESENT chips toward the 2:1 target.

ABSENT acquisition accepts up to eight scenes per observation within ±30 days;
PRESENT acquisition accepts at most one scene within ±10 days. Both use complete
footprints and the quality checks in the [acquisition README](../chip_generation/README.md).
Prepared depth evidence restricts the inventory to 0–20 m below sea level with
depth estimates agreeing within 1 m; ambiguous labels are excluded.

| Current chip-selection stage | PRESENT | ABSENT | Total |
|---|---:|---:|---:|
| Eligible inventory after acquisition, depth and label checks | 984 | 214 | 1,198 |
| Central 5×5 cloud/shadow/missing/defective exclusion | 984 | 212 | 1,196 |
| Selected union across three seeds | 513 | 212 | 725 |

The obtained collection contains more PRESENT chips than the 2:1 target.
The [splitter](../training/balance_absent_first.py) partitions ABSENT groups and
selects 424 PRESENT chips per seed to match depth and whole-chip cloud/shadow,
missing-data and defective-pixel fractions. All 212 eligible ABSENT chips are
retained. Spectral appearance, acquisition/processing metadata and SCL land/water
fractions are not matched. Independent land-overlap and sunglint measurements
are unavailable. The union row is sampling, not a quality failure.

## Current experiment observations and S2 pixel-date labels

The [chip manifest](../training/full_nuisance_balance/selected.csv) is the single
seed-aware list used in the experiment. Each row includes `seed`, `split`, label,
observation and image dates, coordinates, array identifier and features. A chip
may appear in multiple seeds; it appears once within a seed. `train`, `val` and
`test` membership is explicit. The table includes all three partitions, not just
rows used to fit weights.

| Scope and unit | PRESENT | ABSENT | Total |
|---|---:|---:|---:|
| Seed 42: unique location–observation-date records | 424 | 51 | 475 |
| Seed 42: selected chips | 424 | 212 | 636 |
| Seed 42: native 10 m pixel–image-date labels | 422 | 212 | 634 |
| Seed 123: unique location–observation-date records | 424 | 51 | 475 |
| Seed 123: selected chips | 424 | 212 | 636 |
| Seed 123: native 10 m pixel–image-date labels | 421 | 212 | 633 |
| Seed 456: unique location–observation-date records | 424 | 51 | 475 |
| Seed 456: selected chips | 424 | 212 | 636 |
| Seed 456: native 10 m pixel–image-date labels | 421 | 212 | 633 |
| Union across seeds: unique location–observation-date records | 513 | 51 | 564 |
| Union across seeds: distinct chips | 513 | 212 | 725 |
| Union across seeds: native 10 m pixel–image-date labels | 510 | 212 | 722 |

Observation coordinates are rounded to four decimal places, consistently with
source deduplication. Pixel-date keys use UTM CRS, the containing native 10 m
cell at the chip centre and acquisition calendar date. Different image dates
remain separate. Three PRESENT pixel-date groups in the union contain duplicate
labels; none has contradictory labels. Several ABSENT image dates can correspond
to one observation, so 51 observations supply 212 labels. Of these 51 observations,
36 supply more than one image date; none has a duplicate acquisition date. These
images share field evidence and location, so they stay in the same partition. These are point-derived
labels, not labels assigned to every pixel in a chip. Secondary source associations
are provenance, not additional representative observations in these counts.

Each seed has exactly two PRESENT chips per ABSENT chip. The deduplicated union
has a different ratio because seeds choose different PRESENT subsets. The complete union is also used for the [final train/validation fit](../training/final_models/README.md),
whose [split protocol](../training/final_models/split_protocol.json) and
[report](../training/final_models/README.md) give the current class counts.

The [counting script](count_training_pixel_dates.py) reads `TRAINING_SELECTION`
from `config.json`, pointing to the current manifest, and prints union/per-seed
counts without creating another dataset:

```sh
python data_collection/count_training_pixel_dates.py
```

See the [training README](../training/README.md) for partition counts, full-chip
Clay encoding, baseline scripts and current results, and the [root README](../README.md)
for configuration.

## Prepared inputs and reconstruction limits

`eligible_chips.csv` records 1,198 eligible chips before the final central-quality
exclusion. It preserves array hashes, indices, observation evidence and prepared
depth estimates. `output_crs` records the chip grid independently of the source
tile. Complete-footprint reads and overlapping-tile handling are described in the
[acquisition procedure](../chip_generation/README.md).
`split_initialization.csv` supplies the ABSENT initialization
assignments used by the current partition search. Both are active reproduction
inputs. The exact upstream depth preparation is not fully reconstructible here;
see `docs/REPRODUCIBILITY.md` before trying to build these inputs from new downloads.

`match_sesimbra_field_coordinates.py` builds all 90 field events directly
from the measurement workbook and two coordinate files. It requires explicit
matching rules, a complete one-to-one match and nonempty species labels.
