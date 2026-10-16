# Sentinel-2 chip acquisition

`fetch_present.py` and `fetch_absent.py` use the shared `chip_common.py` engine.
Install `requirements-data.txt`. Candidate CSVs use source columns including
`gbifID`, `decimalLatitude`, `decimalLongitude`, `eventDate` and `species`.
The [data-collection policy](../data_collection/README.md#obtaining-sentinel-2-chips)
describes date windows, quality checks and retained labels.

```sh
python chip_generation/fetch_absent.py data_collection/deduped_absent_candidates.csv runs/absent
python chip_generation/fetch_present.py data_collection/deduped_present_candidates.csv runs/present
python chip_generation/consolidate_chips.py --output runs/collection runs/absent runs/present
```

Use fresh acquisition and consolidation directories. Fetchers preserve the
original observation identifier in `observation_id`, separately from the
scene-specific output ID. Acquisition depends on current STAC availability;
it does not recreate a fixed published dataset automatically.

## Consolidated files

| File | Meaning |
|---|---|
| `chips.h5` | Unique uint16 arrays, eight bands, 128×128 pixels; deduplicated by SHA-256 |
| `chips_metadata.csv` | Array index, hash-based chip ID and contradictory-label flag |
| `observation_chips.csv` | Every accepted observation/scene association, including source quality fields and coordinates |
| `consolidation.json` | Counts and input collection names |

Contradictory labels are flagged, not silently discarded or resolved. HDF5 indices
follow input order. Do not apply published manifests to this new file unless
all referenced indices and array hashes match. Depth preparation and eligible
inventory construction require the evidence described in the
[reproduction guide](../docs/REPRODUCIBILITY.md).

`render_chips.py` visualizes selected canonical arrays using their explicit
`h5_index` and SHA-256. It writes RGB/SCL panels under `runs/chip_inspection/`.
Run `python chip_generation/render_chips.py --limit 8` after configuring the
canonical arrays.

## Complete footprints and overlapping tiles

Every chip represents a 1,280 m square on a 128×128 grid at 10 m spacing.
`chip_common.py` checks all eight source raster windows before reading them.
An incomplete window is never clipped and resized into a full chip.

The normal acquisition procedure is:

1. Search within the observation's class-specific date window and scene-cloud
   threshold. Keep overlapping tiles in the candidate list until a usable chip
   is found; accept at most one scene per calendar day.
2. Read the requested footprint. If it crosses a tile boundary,
   `tile_coverage.py` tries a fully covering overlapping tile with the same
   platform, datatake and processing baseline and matching band scale, offset,
   nodata and data-type metadata. This does not move the observation or change
   its label or acquisition date. Missing acquisition identity prevents substitution.
3. Preserve the requested projected grid. For an adjacent UTM zone, reproject
   onto that grid using bilinear interpolation for spectral bands and nearest
   neighbour for categorical SCL. Validate coverage in both STAC metadata and
   the opened source raster. Record the actual `sceneID` and `output_crs`;
   the source tile's CRS can differ from the output grid.
4. Apply the same regional and central-water quality checks to every chip,
   including overlapping-tile reads. If there is no complete tile, reject that
   candidate and continue the normal scene search. Water snapping remains a
   separate, explicitly recorded fallback for water-QC failures, capped at
   300 m; its wider search window must also be complete.
5. Consolidate accepted arrays by SHA-256, keeping observation associations and
   flagging conflicting labels. Apply the additional training central-SCL
   exclusion during feature preparation. Any pixel change requires new hashes,
   features, partitions, embeddings and model fits; caches from different arrays
   are not interchangeable.

The current eligible inventory contains 1,198 chips (984 PRESENT, 214 ABSENT).
The training central-SCL exclusion leaves 1,196 (984 PRESENT, 212 ABSENT).
The coverage audit checked all 1,198 eligible footprints; none extends beyond
its source assets. Audit reports are generated locally and excluded from Git.
To audit a configured collection:

```sh
python chip_generation/audit_boundaries.py --output runs/boundary_audit.json
# Optional: --manifest path/to/chip_inventory.csv
```

The inventory must provide `chip_id`, `sceneID`, `chip_center_lon`,
`chip_center_lat` and, when different from the source grid, `output_crs`.
This metadata audit checks coverage, not internal nodata, radiometric consistency
across acquisitions or source-asset immutability. See the
[scientific limitations](../docs/SCIENCE.md#additional-unresolved-assumptions).
