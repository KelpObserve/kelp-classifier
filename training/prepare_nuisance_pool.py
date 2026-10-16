"""Apply central quality exclusions and measure full-chip nuisances and baseline features.

No independent land overlap or glint measurement is available in this pool.
SCL vegetation/water fractions are retained for auditing, not used for selection.
"""
from pathlib import Path
import sys, json, hashlib, argparse
import pandas as pd, numpy as np, h5py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    source = config.get_path('ELIGIBLE_CHIPS')
    dest = config.get_path('FULL_NUISANCE_MATCHING_POOL')
    f = pd.read_csv(source)
    rows = []
    with h5py.File(config.get_path('CANONICAL_CHIPS_H5'), 'r') as h:
        for r in f.itertuples():
            a = h['chips'][int(r.h5_index)]
            assert hashlib.sha256(a.tobytes()).hexdigest() == r.sha256
            central = a[7, 62:67, 62:67]
            scl = a[7]
            d = dict(chip_id=r.chip_id, nuisance_cloud=float(np.isin(scl, [3, 8, 9, 10]).mean()), nuisance_missing=float((scl == 0).mean()), nuisance_saturated_defective=float((scl == 1).mean()), quality_pass=not np.isin(central, [0, 1, 3, 8, 9, 10]).any())
            for i, band in enumerate(['B02', 'B03', 'B04', 'B05', 'B08', 'B11'], start=1):
                v = a[i].astype(float).ravel()
                lo, hi = np.percentile(v, [10, 90])
                d.update({f'full_{band}_mean': v.mean(), f'full_{band}_std': v.std(), f'full_{band}_p10': lo, f'full_{band}_p90': hi, f'full_{band}_range': hi - lo})
            rows.append(d)
    f = f.merge(pd.DataFrame(rows), on='chip_id', validate='one_to_one')
    cols = ['nuisance_cloud', 'nuisance_missing', 'nuisance_saturated_defective']
    dest.parent.mkdir(parents=True, exist_ok=True)
    f[['chip_id', 'label'] + cols + ['quality_pass']].to_csv(dest.with_name('quality_audit.csv'), index=False)
    eligible = f[f.quality_pass].copy()
    eligible.to_csv(dest, index=False)
    protocol = dict(input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), quality_exclusion_slice='[62:67,62:67]', matching_scope='full native 128x128', baseline_scope='Six-band whole-chip statistics', quality_rule='No central pixels flagged by SCL as cloud, cirrus, shadow, missing, or saturated/defective; identical rule for both classes.', matched=['depth_m', 'nuisance_cloud', 'nuisance_missing', 'nuisance_saturated_defective'], effective_matching='Depth and whole-chip cloud/shadow, missing and defective fractions', unavailable=['Independent land overlap', 'Reliable sunglint flag'], not_balanced=['Spectral reflectance statistics', 'SCL water/land fractions', 'Acquisition and processing metadata'], inherited='Starts from the existing depth-eligible fetched pool; original acquisition quality filters remain inherited.', before=f.label.value_counts().to_dict(), after=eligible.label.value_counts().to_dict())
    dest.with_suffix('.json').write_text(json.dumps(protocol, indent=2))
    print(json.dumps(protocol, indent=2), flush=True)
if __name__ == '__main__':
    main()
