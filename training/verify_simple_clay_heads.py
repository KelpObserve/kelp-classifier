"""Replay simpler Clay heads and verify validation selection on current partitions."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,json
from pathlib import Path
import numpy as np,pandas as pd,torch,joblib
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
from train_simple_clay_heads import metrics
from patch_pool import cache_local_slice
from encoder_geometry import validate_geometry


def main():
    root=config.get_path('FULL_NUISANCE_BALANCE_OUTPUT');out=root/'simple_heads'
    cachepath=config.get_path('FULL_NUISANCE_EMBEDDINGS')
    encoder=json.loads((root/'clay_full_sweep/protocol.json').read_text());cached_encoder=json.loads(cachepath.with_name('protocol.json').read_text())
    validate_geometry(cached_encoder)
    for key in ['checkpoint_sha256','clay_metadata_sha256','gsd_m','bands','scope']:assert encoder[key]==cached_encoder[key],key
    pred=pd.read_csv(out/'predictions.csv');saved=pd.read_csv(out/'metrics.csv');candidates=pd.read_csv(out/'validation_candidates.csv');cache=np.load(cachepath);idx={str(v):i for i,v in enumerate(cache['chip_ids'])}
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('medium');worst=0
    for (family,pool,seed),rows in pred.groupby(['family','pool','seed']):
        folder=out/f'{family}_p{pool}_seed{seed}';selection=json.loads((folder/'selection.json').read_text());cs=candidates[candidates.family.eq(family)&candidates.pool.eq(pool)&candidates.seed.eq(seed)];best=cs.iloc[np.argmax(cs.val_macro_f1.to_numpy())];assert np.isclose(best.regularization,selection['regularization'])
        if family=='small_mlp':assert selection['history'][selection['best_epoch']-1]['val_macro_f1']==max(h['val_macro_f1'] for h in selection['history'])
        lo,hi=cache_local_slice(int(pool));x=np.concatenate([cache['cls'],cache['centre_patches'][:,lo:hi,lo:hi].mean((1,2))],1)
        weights=config.get_path('CHECKPOINTS_DIR')/'simple_heads'/folder.name
        if family=='logistic':model=joblib.load(weights/'model.joblib')
        else:
            model=nn.Sequential(nn.Linear(2048,32),nn.ReLU(),nn.Dropout(.3),nn.Linear(32,1)).cuda().eval();model.load_state_dict(torch.load(weights/'model.pt',map_location='cpu',weights_only=True)['state_dict'])
        f=pd.read_csv(root/f'seed_{seed}/selected.csv').set_index('chip_id')
        assert np.array_equal(cache['sha256'][[idx[c] for c in f.index]].astype(str),f.sha256.to_numpy(str))
        v=rows[rows.split.eq('val')];grid=[.5]+[float(t) for t in np.linspace(.05,.95,91) if not np.isclose(t,.5)];assert np.isclose(max(grid,key=lambda t:metrics(v.label.to_numpy(),v.probability.to_numpy(),t)['macro_f1']),selection['threshold'])
        for split,q in rows.groupby('split'):
            assert set(q.chip_id)==set(f[f.split.eq(split)].index);assert np.array_equal(q.label.to_numpy(),f.loc[q.chip_id,'label'].to_numpy())
            xx=x[[idx[c] for c in q.chip_id]]
            if family=='logistic':p=model.predict_proba(xx)[:,1]
            else:
                with torch.no_grad():p=torch.sigmoid(model(torch.tensor(xx,device='cuda')).squeeze(1)).cpu().numpy()
            worst=max(worst,float(np.max(np.abs(p-q.probability))));assert np.allclose(p,q.probability,atol=1e-7)
            for mode,threshold in [('default',.5),('validation_tuned',selection['threshold'])]:
                row=saved[saved.family.eq(family)&saved.pool.eq(pool)&saved.seed.eq(seed)&saved.split.eq(split)&saved['mode'].eq(mode)].iloc[0]
                for key,value in metrics(q.label.to_numpy(),q.probability.to_numpy(),threshold).items():assert np.isclose(value,row[key]),key
    (out/'audit.json').write_text(json.dumps(dict(models_verified=12,checkpoint_replay_maximum_difference=worst,metrics_recomputed=True,partitions_match=True,validation_regularization_and_threshold_verified=True),indent=2));print('All twelve simpler heads replayed; partitions, metrics and validation selection verified.')

if __name__=='__main__':main()
