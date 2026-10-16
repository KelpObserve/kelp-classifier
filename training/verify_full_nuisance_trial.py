"""Verify whole-chip nuisance partitions and replay saved full-chip Clay heads."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,torch,h5py
from sklearn.metrics import f1_score,roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
from split_groups import group_rows
from model_head import MLPHead
from patch_pool import cache_local_slice
from encoder_geometry import validate_geometry

def main():
    root=config.get_path('FULL_NUISANCE_BALANCE_OUTPUT');pool=pd.read_csv(config.get_path('FULL_NUISANCE_MATCHING_POOL')).set_index('chip_id');protocol=json.loads((root/'protocol.json').read_text())
    assert set(protocol['predictors'])=={'depth_m','nuisance_cloud','nuisance_missing','nuisance_saturated_defective'}
    with h5py.File(config.get_path('CANONICAL_CHIPS_H5'),'r') as h:
        for row in pool.reset_index().itertuples():
            a=h['chips'][int(row.h5_index)];assert hashlib.sha256(a.tobytes()).hexdigest()==row.sha256
            assert not np.isin(a[7,62:67,62:67],[0,1,3,8,9,10]).any()
            for field,classes in [('cloud',[3,8,9,10]),('missing',[0]),('saturated_defective',[1])]:assert np.isclose(getattr(row,'nuisance_'+field),np.isin(a[7],classes).mean())
            for i,band in enumerate(['B02','B03','B04','B05','B08','B11'],1):
                v=a[i].astype(float);lo,hi=np.percentile(v,[10,90])
                for stat,value in [('mean',v.mean()),('std',v.std()),('p10',lo),('p90',hi),('range',hi-lo)]:assert np.isclose(getattr(row,f'full_{band}_{stat}'),value)
    frames={}
    for seed in [42,123,456]:
        f=pd.read_csv(root/f'seed_{seed}/selected.csv');frames[seed]=f
        assert len(f)==3*int(pool.label.eq(0).sum()) and f.chip_id.is_unique and f.label.sum()==2*int(pool.label.eq(0).sum())
        assert set(f[f.label.eq(0)].chip_id)==set(pool[pool.label.eq(0)].index)
        g=group_rows(f);assert pd.DataFrame({'group':g,'split':f.split}).groupby('group').split.nunique().max()==1
        sp=json.loads((root/f'seed_{seed}/split_seed{seed}.json').read_text())
        for split in ['train','val','test']:
            q=f[f.split.eq(split)];assert q.label.sum()==2*q.label.eq(0).sum();assert set(q.chip_id)==set(sp[split+'_chip_ids'])
        for col in [c for c in f if c.startswith(('full_','nuisance_'))]+['depth_m']:assert np.allclose(f[col],pool.loc[f.chip_id,col],equal_nan=True)
    cachepath=config.get_path('FULL_NUISANCE_EMBEDDINGS')
    encoder=json.loads((root/'clay_full_sweep/protocol.json').read_text());cached_encoder=json.loads(cachepath.with_name('protocol.json').read_text())
    validate_geometry(cached_encoder)
    for key in ['checkpoint_sha256','clay_metadata_sha256','gsd_m','bands','scope']:assert encoder[key]==cached_encoder[key],key
    out=root/'clay_full_sweep';p=pd.read_csv(out/'predictions.csv');m=pd.read_csv(out/'metrics.csv');cache=np.load(cachepath);idx={v:i for i,v in enumerate(cache['chip_ids'])}
    assert set(idx)==set(pd.concat(frames.values()).chip_id)
    assert np.array_equal(cache['sha256'],pool.loc[cache['chip_ids'],'sha256'].to_numpy())
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('medium');worst=0
    for size in sorted(p.pool_size.unique()):
        lo,hi=cache_local_slice(int(size));embedding=np.concatenate([cache['cls'],cache['centre_patches'][:,lo:hi,lo:hi].mean((1,2))],axis=1)
        for seed in [42,123,456]:
            folder=out/f'pool_{size}'/f'seed_{seed}';state=torch.load(config.get_path('CHECKPOINTS_DIR')/'clay_full_sweep'/f'pool_{size}'/f'seed_{seed}'/'model.pt',map_location='cpu',weights_only=True)
            assert state['scope']=='full'
            head=MLPHead().cuda().eval();head.load_state_dict(state['state_dict']);history=json.loads((folder/'history.json').read_text());assert history[state['best_epoch']-1]['val_macro_f1']==max(h['val_macro_f1'] for h in history)
            qv=p[p.pool_size.eq(size)&p.seed.eq(seed)&p.split.eq('val')];grid=[.5]+[float(t) for t in np.linspace(.05,.95,91) if not np.isclose(t,.5)];chosen=max(grid,key=lambda t:f1_score(qv.label,qv.probability.ge(t),average='macro'));assert np.isclose(chosen,state['threshold'])
            for split in ['train','val','test']:
                q=p[p.pool_size.eq(size)&p.seed.eq(seed)&p.split.eq(split)];assert set(q.chip_id)==set(frames[seed].loc[frames[seed].split.eq(split),'chip_id'])
                x=torch.tensor(embedding[[idx[v] for v in q.chip_id]],device='cuda')
                with torch.no_grad():pred=torch.sigmoid(head(x)).cpu().numpy()
                delta=float(np.max(np.abs(pred-q.probability)));worst=max(worst,delta);assert np.allclose(pred,q.probability,atol=1e-7)
                z=m[m.pool_size.eq(size)&m.seed.eq(seed)&m.split.eq(split)&m.threshold_mode.eq('validation_tuned')].iloc[0]
                assert np.isclose(f1_score(q.label,q.probability.ge(z.threshold),average='macro'),z.macro_f1);assert np.isclose(roc_auc_score(q.label,q.probability),z.roc_auc)
    (root/'audit.json').write_text(json.dumps(dict(no_spectral_matching=True,eligible_absent_retained=int(pool.label.eq(0).sum()),selected_present_per_seed=2*int(pool.label.eq(0).sum()),grouping_verified=True,full_chip_source_features_recomputed=True,central_quality_exclusions_verified=True,checkpoint_replay_maximum_difference=worst,metrics_recomputed=True,validation_selection_verified=True,uncontrolled=['Independent land contamination','Sunglint']),indent=2))
    print('All source, partition, full-chip feature, checkpoint and selection checks passed.',flush=True)

if __name__=='__main__':main()
