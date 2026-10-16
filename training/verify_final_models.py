"""Verify final union split, saved thresholds, metrics and reusable checkpoint replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
from split_groups import group_rows
from final_model_io import load_head,predict_embeddings
from train_frozen_sesimbra import metrics
from patch_pool import cache_local_slice
from encoder_geometry import validate_geometry


def main():
    root=config.get_path('FINAL_MODELS_DIR');source=config.get_path('FULL_NUISANCE_BALANCE_OUTPUT');f=pd.read_csv(root/'selected.csv');original=pd.concat([pd.read_csv(source/f'seed_{s}/selected.csv') for s in [42,123,456]]).drop_duplicates('chip_id').set_index('chip_id')
    assert f.chip_id.is_unique and set(f.chip_id)==set(original.index)
    assert set(f.split)=={'train','val'}
    g=group_rows(f);assert pd.DataFrame({'group':g,'split':f.split}).groupby('group').split.nunique().max()==1
    for col in ['label','depth_m','nuisance_cloud','nuisance_missing','nuisance_saturated_defective']:assert np.allclose(f[col],original.loc[f.chip_id,col])
    split=json.loads((root/'split.json').read_text())
    counts=json.loads((root/'split_protocol.json').read_text())['counts']
    for sp in ['train','val']:
        q=f[f.split.eq(sp)];assert int(q.label.sum())==counts[sp]['1'] and int(q.label.eq(0).sum())==counts[sp]['0'];assert set(q.chip_id)==set(split[sp+'_chip_ids'])
    validate_geometry(json.loads(config.get_path('FULL_NUISANCE_EMBEDDINGS').with_name('protocol.json').read_text()))
    cache=np.load(config.get_path('FULL_NUISANCE_EMBEDDINGS'));idx={str(cid):i for i,cid in enumerate(cache['chip_ids'])};manifest=json.loads((root/'model_manifest.json').read_text());pred=pd.read_csv(root/'predictions.csv');saved=pd.read_csv(root/'metrics.csv')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('medium');worst=0
    for item in manifest['models']:
        name=item['model'];path=config.get_path('FINAL_CHECKPOINTS_DIR')/item['relative_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];model,ck=load_head(path,'cuda');assert ck['selected_manifest_sha256']==hashlib.sha256((root/'selected.csv').read_bytes()).hexdigest();assert ck['split_sha256']==hashlib.sha256((root/'split.json').read_bytes()).hexdigest()
        selection=json.loads((root/item['relative_path']).with_name('selection.json').read_text());best=max(selection['candidates'],key=lambda c:c['val_macro_f1']);assert best['weight_decay']==ck['weight_decay'];assert best['best_epoch']==ck['best_epoch'];assert best['history'][ck['best_epoch']-1]['val_macro_f1']==max(h['val_macro_f1'] for h in best['history'])
        lo,hi=cache_local_slice(ck['pool_size']);x=np.concatenate([cache['cls'],cache['centre_patches'][:,lo:hi,lo:hi].mean((1,2))],1)
        v=pred[pred.model.eq(name)&pred.split.eq('val')];grid=[.5]+[float(t) for t in np.linspace(.05,.95,91) if not np.isclose(t,.5)];assert np.isclose(max(grid,key=lambda t:metrics(v.label.to_numpy(),v.probability.to_numpy(),t)['macro_f1']),ck['threshold'])
        for sp in ['train','val']:
            q=pred[pred.model.eq(name)&pred.split.eq(sp)];assert set(q.chip_id)==set(f.loc[f.split.eq(sp),'chip_id']);assert np.array_equal(q.label.to_numpy(),original.loc[q.chip_id,'label'].to_numpy());prob,_=predict_embeddings(model,ck,x[[idx[c] for c in q.chip_id]]);prob=prob.cpu().numpy();worst=max(worst,float(np.max(np.abs(prob-q.probability))));assert np.allclose(prob,q.probability,atol=1e-7)
            for mode,t in [('default',.5),('validation_tuned',ck['threshold'])]:
                row=saved[saved.model.eq(name)&saved.split.eq(sp)&saved.threshold_mode.eq(mode)].iloc[0]
                for k,value in metrics(q.label.to_numpy(),q.probability.to_numpy(),t).items():assert np.isclose(row[k],value),k
    (root/'audit.json').write_text(json.dumps(dict(all_selected_chips_retained=True,selected_chip_count=len(f),group_separation_verified=True,source_labels_and_matching_features_verified=True,train_present=counts['train']['1'],train_absent=counts['train']['0'],val_present=counts['val']['1'],val_absent=counts['val']['0'],checkpoint_hashes_verified=True,checkpoint_replay_maximum_difference=worst,validation_selection_verified=True,metrics_recomputed=True,no_independent_test_set=True),indent=2));print('Final split, both reusable checkpoints, validation selection and metrics verified.')

if __name__=='__main__':main()
