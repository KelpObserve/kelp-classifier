"""Extract kelp PRESENT + ABSENT from the JNCC Marine Recorder snapshot.

Snapshot: SnapshotDatav52-PUBLIC-20220124.zip (~293 MB)
  https://data.jncc.gov.uk/data/b9934e31-39b6-41f9-9364-d1e93db68307/SnapshotDatav52-PUBLIC-20220124.zip

Requires `mdbtools` package (Ubuntu: `sudo apt-get install mdbtools`).

Output: kelp_jncc_present.csv, kelp_jncc_absent.csv
"""
import csv
import io
import subprocess
import sys
from pathlib import Path
# Locate repo root by walking up until config.py is found
import sys as _sys
from pathlib import Path as _P
_here = _P(__file__).resolve().parent
while _here != _here.parent:
    if (_here / "config.py").exists():
        _sys.path.insert(0, str(_here)); break
    _here = _here.parent
import config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import KELP_SPECIES

HERE = Path(__file__).resolve().parent
MDB = str(config.DATA_DIR / "jncc_mr" / "SnapshotDatav52_PUBLIC_20220124.mdb")


# Rejection counter for JNCC date parses that fall through the format branches.
# Populated by parse_mmddyy(); dumped once at end of main() so silent drops become
# visible (reviewer finding #3).
_DATE_REJECTS = {"empty": 0, "unparseable": 0, "samples": []}

def parse_mmddyy(s: str) -> str:
    """MDB dates come out as 'MM/DD/YY HH:MM:SS'. Return YYYY-MM-DD.

    Tracks failures in _DATE_REJECTS so upstream drops in the >= 2016-01-01
    filter can be attributed to date-parse failures vs pre-S2-era records.
    """
    if not s:
        _DATE_REJECTS["empty"] += 1
        return ""
    try:
        parts = s.split()[0].split("/")
        m, d, y = parts
        y = int(y)
        if y < 100:
            y = 2000 + y if y < 50 else 1900 + y
        # Otherwise assume 4-digit year already
        return f"{y:04d}-{int(m):02d}-{int(d):02d}"
    except Exception:
        _DATE_REJECTS["unparseable"] += 1
        if len(_DATE_REJECTS["samples"]) < 5:
            _DATE_REJECTS["samples"].append(s)
        return ""


def mdb_export(mdb: str, table: str):
    """Yield dict rows from an Access table using mdbtools."""
    p = subprocess.run(["mdb-export", mdb, table],
                       capture_output=True, text=True, check=True)
    yield from csv.DictReader(io.StringIO(p.stdout))


def classify_pa(row: dict) -> str | None:
    """JNCC Marine Recorder convention:
       PA field: 'P' = Present, 'N' = Not present (i.e. Absent)
       SACFORN abundance scale: N = Not seen (i.e. Absent), P/S/A/C/F/O/R = present"""
    pa = (row.get("PA") or "").strip().upper()
    sacf = (row.get("SACFORN") or "").strip().upper()
    if pa == "P" or sacf in ("P", "S", "SA", "A", "C", "F", "O", "R"):
        return "PRESENT"
    if pa == "N" or sacf == "N":
        return "ABSENT"
    return None  # unknown / missing


def main():
    # 1. Scan SampleSpecies for kelp rows
    print("scanning SampleSpecies...", flush=True)
    kelp_ss = {"PRESENT": [], "ABSENT": []}
    for r in mdb_export(MDB, "SampleSpecies"):
        sn = r.get("SpeciesName", "")
        if not any(sn.startswith(k) for k in KELP_SPECIES): continue
        klass = classify_pa(r)
        if not klass: continue
        kelp_ss[klass].append({
            "sample_ref": r.get("SAMPLE_REFERENCE"),
            "species": sn,
            "pa": r.get("PA"),
            "sacf": r.get("SACFORN"),
            "aphia": r.get("AphiaId"),
        })
    print(f"kelp records: PRESENT={len(kelp_ss['PRESENT'])}  ABSENT={len(kelp_ss['ABSENT'])}", flush=True)

    # 2. Load Sample coords keyed by SAMPLE_REFERENCE
    print("loading Sample coords...", flush=True)
    sample_meta = {}
    for r in mdb_export(MDB, "Sample"):
        try:
            lat = float(r.get("LatWGS84") or 0); lon = float(r.get("LongWGS84") or 0)
        except (ValueError, TypeError):
            continue
        if lat == 0 and lon == 0: continue
        sample_meta[r.get("SAMPLE_REFERENCE")] = {
            "decimalLatitude": lat, "decimalLongitude": lon,
            "eventDate": parse_mmddyy(r.get("Date") or r.get("VAGUE_DATE_START", "")),
            "sampling_notes": r.get("Description") or "",
        }
    print(f"samples with coords: {len(sample_meta)}", flush=True)

    # 3. Join + write per class
    for klass in ("PRESENT", "ABSENT"):
        joined = []
        for ss in kelp_ss[klass]:
            s = sample_meta.get(ss["sample_ref"])
            if not s: continue
            joined.append({**ss, **s, "occurrenceStatus": klass})
        # Filter post-2016 (S2 era)
        post = [j for j in joined if j["eventDate"] >= "2016-01-01"]
        out = HERE / f"kelp_jncc_{klass.lower()}.csv"
        fields = ["sample_ref","species","scientificName","decimalLatitude","decimalLongitude",
                  "eventDate","occurrenceStatus","pa","sacf","aphia","sampling_notes",
                  "datasetKey","basisOfRecord"]
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for j in post:
                w.writerow({**j, "scientificName": j["species"],
                            "datasetKey": "jncc-mr-v52-20220124",
                            "basisOfRecord": "HumanObservation"})
        print(f"wrote {out}: {len(post)} rows (post-2016)", flush=True)



# atexit: emit JNCC date-parse totals once, even if mdb_export raises earlier.
import atexit as _atx
def _emit_date_rejects():
    if any(_DATE_REJECTS.get(k, 0) for k in ("empty", "unparseable")):
        print(f"[jncc] date-parse rejects (total across classes): "
              f"empty={_DATE_REJECTS['empty']} "
              f"unparseable={_DATE_REJECTS['unparseable']} "
              f"samples={_DATE_REJECTS['samples'][:5]}")
_atx.register(_emit_date_rejects)


if __name__ == "__main__":
    main()
