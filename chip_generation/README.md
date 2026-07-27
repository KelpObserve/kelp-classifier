# Chip generation

Pull 128×128 Sentinel-2 L2A chips for each labelled kelp record.

## Chip format

- **Size**: 128×128 pixels (10 m Sentinel-2 native → 1.28 × 1.28 km footprint on the ground)
- **Bands** (dim 0): `B01, B02, B03, B04, B05, B08, B11, SCL` — coastal, blue, green, red, red-edge-1, NIR, SWIR16, scene-classification
- **Dtype**: uint16
- **Storage**: HDF5 with `chunks=(1, 8, 128, 128)`, `compression="lzf"`
- **Metadata**: one CSV row per chip alongside `chips.h5` with sceneID, sceneDate, cloud/water/fill metrics, snap fields (see below), gbifID, class_tag.

## Prediction region

All QC operates on the **23×23-pixel prediction region** at chip centre (rows/cols 50:73). This region maps to Clay's centre-pool patches `[11:16, 11:16]` on the 28×28 patch grid after 128→224 resize. Aligning QC with what the model actually looks at prevents ABSENT chips from passing on flat-zero centres and PRESENT chips from landing on land in the region the model emphasises.

## QC profiles ([`chip_common.py`](chip_common.py))

Both profiles use the same chip reader; only the time window and downstream fetch policy differ.

| Parameter | STRICT (PRESENT) | RELAXED (ABSENT) |
|---|---|---|
| Scene cloud filter (STAC) | ≤ 60 % | ≤ 60 % |
| Region cloud reject | > 40 % SCL cloud/shadow/cirrus | > 40 % (same) |
| **Region water minimum (`region_water_min`)** | **≥ 30 %** | **≥ 30 % (fixed to match PRESENT)** |
| Region fill reject | > 70 % no-data pixels | > 70 % (same) |
| Region B02 dynamic range | ≥ 30 uint16 P05-P95 spread | ≥ 30 (same) |
| Time window around eventDate | ± 30 d | ± 60 d |
| Chips per record (default) | 1 (best scene) | up to 8 (multi-scene) |
| **Snap cap (m)** | **300** | **300** |

### Post-review symmetry fix

Earlier the ABSENT profile had `region_water_min = 0.0` — no water requirement — which meant ABSENT chips could land anywhere including entirely on land, while PRESENT chips were pre-filtered to sit clearly over water. A code-review sanity check (`L2`) showed a logistic classifier trained on QC metadata alone (no image content) predicted PRESENT vs ABSENT at ROC-AUC **0.859** — higher than the model's headline 0.841 — because the water-fraction distributions of PRESENT and ABSENT were disjoint. Symmetrising `region_water_min` to 0.30 closes that leak on the water axis.

## Snap-to-water (`snap_to_water`)

When a candidate's reported coord fails the water QC (`region_water_min < 0.30`), the pipeline tries a single snap step before rejecting:

1. Read the item's SCL band over a widened window (`128 + 2·cap_px` pixels, e.g. 188×188 for a 300 m cap).
2. Find the nearest water pixel (`SCL == 6`) to the chip centre inside the cap.
3. If none exists inside the cap → drop the scene as `snap_no_water`.
4. Otherwise re-read the full 8-band chip centred on the snapped water pixel and re-run QC.
5. If the snapped chip passes → keep it with metadata `snap_dist_m`, `snap_lat`, `snap_lon` recording the shift; original observer coords are preserved unchanged in `decimalLatitude` / `decimalLongitude`.

### Why it's sound

A shore observer's *"no kelp seen"* claim is really about the water they were looking at, not the rock they stood on. Symmetrically, an observer at the top of a rocky Portuguese cliff logging a kelp bed offshore has a valid observation, just anchored at their standpoint. Snapping ≤ 300 m to a defined water pixel re-anchors the chip on the water surface the observation actually described.

Enabled by default via `snap_cap_m=300` in both QC dicts; set to 0 to disable.

### Metadata columns added

| Column | Meaning |
|---|---|
| `snap_dist_m` | Distance (m) between reported and snapped chip centre. `0.0` if no snap was needed. |
| `snap_lat`, `snap_lon` | Snapped coords (WGS-84). Blank if no snap was applied. |

## Coordinate uncertainty filter

- **PRESENT** (`fetch_present.py`): candidates with `coordinateUncertaintyInMeters > 115` are dropped pre-fetch. 115 m is the half-width of the 23×23 prediction region — a PRESENT record whose true location is outside the region the model looks at is a noisy label. In practice we go tighter: the current PRESENT candidate list (`high_precision_candidates.csv`) restricts to `coordinateUncertaintyInMeters <= 10 m`, so the fetch operates well inside the 115 m safety filter.
- **ABSENT** (`fetch_absent.py`): no unc pre-filter. An "no kelp seen" claim over an area is robust to positional imprecision — unlike a PRESENT record, which pinpoints a specific kelp bed. (Reviewer C2 originally symmetrised this filter for P/A parity; the intentional revert trades a small symmetry cost for meaningful ABSENT coverage.)

## Usage

### Present chips

```
python fetch_present.py <candidates.csv> <out_dir>
```

Reads candidates with columns `gbifID, decimalLatitude, decimalLongitude, eventDate`. Writes `<out_dir>/chips.h5` and `<out_dir>/chips_metadata.csv`.

### Absent chips (up to 8 scenes)

```
python fetch_absent.py <candidates.csv> <out_dir> [--scenes 8] [--single] \
       [--exclude-existing <prev_chips_metadata.csv>]
```

- `--scenes N` sets the maximum scenes per record (default 8).
- `--single` shortcut for `--scenes 1`.
- `--exclude-existing` skips sceneIDs already recorded in a previous metadata CSV — used to top up an existing dataset with fresh scenes without duplicating.

Under the hood both wrappers call [`chip_common.fetch_chips(...)`](chip_common.py), which:
1. Queries STAC (Element84 Earth-Search) for candidate scenes in the record's time window.
2. Sorts by scene-level cloud coverage.
3. Reads the 128×128 chip at the record's coord for each candidate scene.
4. Runs `qc_chip(...)`. If `bad_water` and `snap_cap_m > 0`, tries `snap_to_water(...)` + re-read + re-QC.
5. Writes passing chips (and one metadata row each) to the output HDF5/CSV.

### Merging fetches

Individual fetches land in per-source directories under `$KELP_DATA_DIR/`. [`merge_all_chips.py`](merge_all_chips.py) picks up **any** directory whose name contains `s2_chips_` (with a valid `chips.h5`), so future OBIS / JNCC / snap-fetch dirs are folded in automatically. It applies a `gbifID`-based dedup at merge time to collapse same-source refetches of the same scene.

Cross-source duplicates (same observation reported in both GBIF and OBIS) are handled *pre-fetch* — see [`../data_collection/dedup_candidates.py`](../data_collection/dedup_candidates.py) which unifies the six raw source CSVs into deduplicated PRESENT/ABSENT candidate lists.

Field names in the output CSV are the **union** across all source rows, so columns present in only some sources (e.g. `snap_lat/snap_lon/snap_dist_m` from snap-enabled fetches) are preserved rather than silently dropped. Chip data is **streamed** — each source `chips.h5` is opened once and chips are written to the merged file one at a time, so peak RAM stays O(one chip) instead of O(all chips).

- **Merged output**: `$KELP_DATA_DIR/s2_chips_merged/chips.h5` (+ `chips_metadata.csv`)

This is the file consumed by [`../training/`](../training/) (via `precompute_clay_embeddings.py`).
## Current chip counts (2026-07-17)

Merged file at `$KELP_DATA_DIR/s2_chips_merged/chips.h5`:

| Class | Chips |
|---|---:|
| PRESENT | 4,089 |
| ABSENT (single + multi-scene) | 1,614 |
| **Total** | **5,703** |

These are the counts consumed by the training pipeline. Cross-source dedup runs at candidate time (`data_collection/dedup_candidates.py`); QC metadata (`wf_region`, `cloud_region`, `fill_region`, `b02_range`) is written by `chip_common.qc_chip` at fetch time.
