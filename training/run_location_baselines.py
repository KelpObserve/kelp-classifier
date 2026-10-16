"""Fit whole-chip and nuisance baselines on the current saved partitions.

Use --check to reproduce the metrics without modifying saved evidence.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
from split_groups import group_rows
from baselines import evaluate


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');ap.add_argument('--auc-tolerance',type=float,default=1e-12,help='Explicit absolute AUC tolerance for cross-platform refits; classification metrics remain strict');args=ap.parse_args()
    if args.auc_tolerance < 0:ap.error('--auc-tolerance must be nonnegative')
    root=config.get_path('FULL_NUISANCE_BALANCE_OUTPUT');records=[]
    for seed in [42,123,456]:
        f=pd.read_csv(root/f'seed_{seed}/selected.csv');saved=json.loads((root/f'seed_{seed}/split_seed{seed}.json').read_text())
        assert f.chip_id.is_unique
        g=group_rows(f);assert pd.DataFrame({'group':g,'split':f.split}).groupby('group').split.nunique().max()==1
        for split in ['train','val','test']:
            q=f[f.split.eq(split)];assert q.label.sum()==2*q.label.eq(0).sum();assert set(q.chip_id)==set(saved[split+'_chip_ids'])
        assert len([c for c in f if c.startswith('full_')])==30
        for split in ['val','test']:
            records.extend(evaluate(f,split,seed,'balanced',image_prefix='full_',image_family='full'))
            records.extend(r for r in evaluate(f,split,seed,'balanced',image_prefix='nuisance_',image_family='quality') if r['family']!='depth_only')
    result=pd.DataFrame(records);path=root/'baseline_metrics.csv'
    if args.check:
        old=pd.read_csv(path);old=old[old.variant.eq('balanced')]
        keys=['seed','split','family','model'];a=result.set_index(keys).sort_index();b=old.set_index(keys).sort_index()
        assert a.index.equals(b.index)
        for col in ['macro_f1','present_precision','present_recall','absent_precision','absent_recall']:assert np.allclose(a[col],b[col],atol=1e-12,rtol=0),col
        delta=float(np.max(np.abs(a.roc_auc-b.roc_auc)))
        assert delta <= args.auc_tolerance, f'roc_auc maximum difference {delta:.6g} exceeds {args.auc_tolerance:.6g}'
        print(f'Baseline classification metrics reproduced; maximum AUC difference {delta:.6g} (allowed {args.auc_tolerance:.6g}).')
    else:
        result.to_csv(path,index=False);print('Saved',path)


if __name__=='__main__':main()
