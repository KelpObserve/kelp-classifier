"""chip_generation/assemble_submerged.py — subset the merged chip pool to a
deep-submerged-kelp training set.

PRESENT filter: must be a species-specific observation of one of the four
deep-submerged kelps (blade sub-surface at 5-30 m):
  Saccorhiza polyschides
  Laminaria hyperborea
  Laminaria ochroleuca
  Phyllariopsis brevipes

ABSENT filter: no species filter — for an "any target-species kelp" detector,
an ABSENT chip is valid whenever the surveyor reported no kelp of any species
at the location; whether they specifically checked for L. hyperborea vs
Saccharina latissima doesn't change the fact that the chip contains no
visible kelp of interest. Deduped ABSENT rows with comma-separated species
lists are the CLEANEST negatives — a JNCC survey that checked 8 species and
found none of them is a strong "no kelp here" signal.

Reads:
  $KELP_DATA_DIR/s2_chips_merged/chips.h5
  $KELP_DATA_DIR/s2_chips_merged/chips_metadata.csv

Writes:
  $KELP_DATA_DIR/s2_chips_submerged/chips.h5
  $KELP_DATA_DIR/s2_chips_submerged/chips_metadata.csv

No refetch — chips are copied from the merged pool by matching gbifID. If a
merged-CSV row is missing a species label, we cross-check against the raw
source CSVs in data_collection/ before deciding whether to keep it.

Usage:
  python chip_generation/assemble_submerged.py
"""
import csv
import re
import sys as _sys
from pathlib import Path
from pathlib import Path as _P

import h5py
import numpy as np

_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

TARGET = {
    "saccorhiza polyschides",
    "laminaria hyperborea",
    "laminaria ochroleuca",
    "phyllariopsis brevipes",
}

SRC_DIR = config.DATA_DIR / "s2_chips_merged"
DST_DIR = config.DATA_DIR / "s2_chips_submerged"
SRC_H5 = SRC_DIR / "chips.h5"
SRC_CSV = SRC_DIR / "chips_metadata.csv"
DST_H5 = DST_DIR / "chips.h5"
DST_CSV = DST_DIR / "chips_metadata.csv"

RAW_CSVS = [
    config.REPO_ROOT / "data_collection" / "gbif" / "kelp_gbif_europe_present.csv",
    config.REPO_ROOT / "data_collection" / "gbif" / "kelp_gbif_europe_absent.csv",
    config.REPO_ROOT / "data_collection" / "obis" / "kelp_obis_europe_present.csv",
    config.REPO_ROOT / "data_collection" / "obis" / "kelp_obis_europe_absent.csv",
    config.REPO_ROOT / "data_collection" / "jncc" / "kelp_jncc_present.csv",
    config.REPO_ROOT / "data_collection" / "jncc" / "kelp_jncc_absent.csv",
]


def base_id(gid):
    """Strip _sN scene-index suffix and OBIS URI prefix, return the raw source ID."""
    stripped = re.sub(r"_s\d+$", "", gid)
    return stripped[5:] if stripped.startswith("obis:") else stripped


def build_species_lookup():
    """Build {source_id → species (lower)} from every raw source CSV."""
    lookup = {}
    for p in RAW_CSVS:
        if not p.exists():
            print(f"  [skip] {p}")
            continue
        with p.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                sp = (row.get("species") or "").strip().lower()
                if not sp:
                    continue
                for key in ("gbifID", "id", "occurrenceID"):
                    if key in row and row[key]:
                        lookup[str(row[key])] = sp
                        break
    return lookup


def main():
    print(f"src: {SRC_CSV}")
    print(f"src: {SRC_H5}")
    print(f"dst: {DST_CSV}")
    print(f"dst: {DST_H5}")
    print(f"target species: {sorted(TARGET)}")

    print("\nbuilding species lookup from raw source CSVs...")
    sp_of = build_species_lookup()
    print(f"  {len(sp_of)} unique source IDs indexed")

    print("\nscanning merged metadata...")
    kept_rows = []
    fieldnames = None
    n_present_all = n_absent_all = n_present_kept = n_absent_kept = 0
    with SRC_CSV.open() as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            status = row.get("occurrenceStatus", "")
            sp_raw = (row.get("species") or "").strip().lower()
            if not sp_raw:
                sp_raw = sp_of.get(base_id(row["gbifID"]), "")
            # PRESENT: species-specific filter — an observation of one of the
            # four target species. Species field may be a comma-list (multi-
            # species survey) — accept if ANY listed species is in TARGET.
            # ABSENT: species-agnostic — a chip is a valid negative for
            # "any target-species kelp here" whenever a surveyor recorded no
            # kelp at the location, regardless of which species they checked.
            if status == "PRESENT":
                n_present_all += 1
                listed = {s.strip() for s in sp_raw.split(",") if s.strip()}
                match = listed & TARGET
                if not match:
                    continue
                # Populate species with the FIRST target species found (stable
                # ordering) so downstream per-species stats/filters are simple.
                row["species"] = sorted(match)[0]
                kept_rows.append(row)
                n_present_kept += 1
            elif status == "ABSENT":
                n_absent_all += 1
                # Keep the species field verbatim (may be comma-list) — useful
                # metadata but not used for filtering.
                kept_rows.append(row)
                n_absent_kept += 1
            # unknown status: skip
    print(f"  PRESENT: {n_present_kept}/{n_present_all} kept (target-species match)")
    print(f"  ABSENT:  {n_absent_kept}/{n_absent_all} kept (all species — species-agnostic)")
    print(f"  kept {len(kept_rows)} rows (of {sum(1 for _ in SRC_CSV.open()) - 1} total)")

    # Chip HDF5 is indexed by the CSV row order — the row's `merged_index`
    # column stores the row's position in chips.h5. Verify.
    src_indices = []
    for row in kept_rows:
        mi = row.get("merged_index")
        if mi in (None, ""):
            raise RuntimeError(f"row without merged_index: gbifID={row.get('gbifID')}")
        src_indices.append(int(mi))
    src_indices_arr = np.array(src_indices, dtype=np.int64)
    print(f"  src merged_index range: {src_indices_arr.min()}..{src_indices_arr.max()}  "
          f"(unique: {len(np.unique(src_indices_arr))})")

    print("\ncopying chip tensors...")
    DST_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(SRC_H5, "r") as src, h5py.File(DST_H5, "w") as dst:
        src_ds = src["chips"]
        N = len(src_indices_arr)
        chip_shape = src_ds.shape[1:]
        dst_ds = dst.create_dataset(
            "chips", shape=(N, *chip_shape), dtype=src_ds.dtype,
            chunks=(1, *chip_shape), compression="lzf",
        )
        # h5py fancy-indexing requires sorted unique indices; go one at a time.
        for out_i, src_i in enumerate(src_indices_arr):
            dst_ds[out_i] = src_ds[int(src_i)]
            if (out_i + 1) % 100 == 0 or out_i + 1 == N:
                print(f"  {out_i + 1}/{N}")

    print("\nwriting metadata CSV with re-indexed merged_index...")
    for i, row in enumerate(kept_rows):
        # Overwrite merged_index so it matches the row's slot in the NEW h5.
        row["merged_index"] = i
    with DST_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in kept_rows:
            w.writerow(row)
    print(f"wrote {DST_CSV} ({len(kept_rows)} rows)")
    print(f"wrote {DST_H5}  (N={len(kept_rows)}, chip shape={chip_shape})")

    # Summary by species/status
    from collections import Counter
    ct = Counter((r["species"], r["occurrenceStatus"]) for r in kept_rows)
    print("\nsummary:")
    for (sp, st), n in sorted(ct.items()):
        print(f"  {sp:<28} {st:<8} {n:>5}")
    print(f"  {'TOTAL':<28} {'':<8} {sum(ct.values()):>5}")


if __name__ == "__main__":
    main()
