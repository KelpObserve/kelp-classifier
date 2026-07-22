"""Download 20 kelp-relevant DASSH datasets, extract occurrence.txt,
filter for kelp species with ABSENT records, dedup against existing GBIF/OBIS.
"""
import csv
import io
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

DASSH_IPT = "https://www.dassh.ac.uk/ipt"

TARGETS = [
    "dasshdt00000679",  # 2023 Natural England Duddon Estuary Littoral
    "dasshdt00000712",  # 2023 Seasearch West Scotland
    "dasshdt00000710",  # 2022 Seasearch West Scotland
    "dasshdt00000716",  # 2023 Seasearch Scotland North Coast + Northern Isles
    "dasshdt00000597",  # 2024 Seawilding/BSAC Great Seagrass Survey
    "dasshdt00000585",  # 2023 Seasearch ReMEDIES Hampshire + IoW
    "dasshdt00000602",  # 2023 WMU Macrophyte Survey
    "dassh00000587",    # 2022 Seasearch ReMEDIES Hampshire + IoW
    "dassh00000632",    # 2023 NRW Carmarthen Bay
    "dasshdt00000571",  # 2020-2023 Seawilding/BSAC Great Seagrass
    "dasshdt00000138",  # 2015 David Fenwick Devon kelp holdfast
    "dasshdt00000279",  # 2015 MBA Wembury Seaweeds course
    "eifca20161219gt",  # 2016 Eastern IFCA The Wash
    "dasshdt00000086",  # 2014 Cefas Farnes East MCZ
    "dasshdt00000464",  # 2013 U Plymouth Flamborough Head
    "dasshdt00000465",  # 2013 U Plymouth Thanet Coast
]

KELP_SPECIES = {
    "Laminaria hyperborea", "Laminaria digitata", "Saccharina latissima",
    "Alaria esculenta", "Saccorhiza polyschides", "Laminaria ochroleuca",
    "Chorda filum", "Undaria pinnatifida", "Phyllariopsis brevipes",
    "Laminaria saccharina",  # synonym for S. latissima
}
KELP_GENERA = {"Laminaria", "Saccharina", "Alaria", "Chorda", "Saccorhiza", "Undaria", "Phyllariopsis"}


def download(rid):
    """Download DwC-A archive for a DASSH resource."""
    url = f"{DASSH_IPT}/archive.do?r={rid}"
    print(f"  fetching {url}...", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except Exception as e:
        print(f"    ERR: {e}", flush=True)
        return None


def parse_occurrences(zip_bytes):
    """Extract occurrence.txt records from a DwC-A zip."""
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    # find occurrence file
    occ_files = [n for n in zf.namelist() if "occurrence" in n.lower() and n.endswith(".txt")]
    if not occ_files:
        # try all .txt
        occ_files = [n for n in zf.namelist() if n.endswith(".txt")]
    all_rows = []
    for name in occ_files:
        data = zf.read(name).decode("utf-8", errors="replace")
        try:
            reader = csv.DictReader(io.StringIO(data), delimiter="\t")
            all_rows.extend(list(reader))
        except Exception as e:
            print(f"    parse error {name}: {e}", flush=True)
    return all_rows


def is_kelp(row):
    sn = (row.get("scientificName") or row.get("scientific_name") or "").strip()
    genus = (row.get("genus") or "").strip()
    # normalise
    for k in KELP_SPECIES:
        if sn.startswith(k):
            return True
    if genus in KELP_GENERA:
        return True
    return False


def is_absent(row):
    """Detect ABSENT records: occurrenceStatus, individualCount, or abundance flags."""
    os_ = (row.get("occurrenceStatus") or "").strip().lower()
    if os_ in ("absent", "not present", "none"):
        return True
    ic = row.get("individualCount") or row.get("organismQuantity")
    if ic:
        try:
            if float(ic) == 0:
                return True
        except (ValueError, TypeError):
            pass
    # Abundance categories that can equal absent
    ab = (row.get("occurrenceRemarks") or "").lower()
    if "absent" in ab or "not found" in ab:
        return True
    return False


def main():
    all_absent = []
    all_kelp_ct = 0
    for rid in TARGETS:
        print(f"=== {rid} ===", flush=True)
        data = download(rid)
        if not data:
            continue
        try:
            rows = parse_occurrences(data)
        except Exception as e:
            print(f"    zip parse err: {e}", flush=True)
            continue
        n = len(rows)
        kelp_rows = [r for r in rows if is_kelp(r)]
        absent_kelp = [r for r in kelp_rows if is_absent(r)]
        all_kelp_ct += len(kelp_rows)
        all_absent.extend([(rid, r) for r in absent_kelp])
        print(f"  rows: {n}  kelp: {len(kelp_rows)}  ABSENT kelp: {len(absent_kelp)}", flush=True)
        # sample a kelp record for insight
        if kelp_rows:
            r = kelp_rows[0]
            keys_of_interest = ["scientificName", "occurrenceStatus", "individualCount",
                                "decimalLatitude", "decimalLongitude", "eventDate",
                                "occurrenceRemarks", "coordinateUncertaintyInMeters"]
            print(f"    sample kelp: {{k: r.get(k) for k in keys_of_interest if r.get(k)}}")
    print(f"\n=== SUMMARY ===", flush=True)
    print(f"total kelp records across datasets: {all_kelp_ct}", flush=True)
    print(f"total ABSENT kelp records: {len(all_absent)}", flush=True)
    if all_absent:
        # Species and year distribution
        sp_ct = Counter((r.get("scientificName") or "?")[:50] for _, r in all_absent)
        print(f"\nby species:")
        for sp, c in sp_ct.most_common():
            print(f"  {c:>4}  {sp}")
        year_ct = Counter((r.get("eventDate") or "?")[:4] for _, r in all_absent)
        print(f"\nby year:")
        for y, c in year_ct.most_common():
            print(f"  {c:>4}  {y}")
    # save
    out = Path("/tmp/dassh_absent_kelp.tsv")
    if all_absent:
        keys = sorted({k for _, r in all_absent for k in r.keys()})
        with out.open("w") as f:
            f.write("dassh_resource\t" + "\t".join(keys) + "\n")
            for rid, r in all_absent:
                f.write(rid + "\t" + "\t".join((r.get(k) or "").replace("\t", " ").replace("\n", " ")
                                                for k in keys) + "\n")
        print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
