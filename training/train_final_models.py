"""Fit two reusable Clay heads on a replayed or newly optimized final split.

All selected chips are retained. The included split was balanced on depth and
whole-chip quality; default training preserves its membership and row order.
Use --new-split after changing the selected union.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,json,hashlib,random,argparse
import numpy as np,pandas as pd,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
from split_groups import group_rows
from model_head import sigmoid_focal_loss
from final_model_io import make_head,load_head,predict_embeddings
from train_frozen_sesimbra import metrics
from patch_pool import cache_local_slice
from encoder_geometry import ENCODER_SIZE, validate_geometry


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def split_union(f):
    source = config.REPO_ROOT / 'training/final_models'
    saved = pd.read_csv(source / 'selected.csv')
    expected = f.set_index('chip_id')
    assert saved.chip_id.is_unique and set(saved.chip_id) == set(expected.index)
    for col in ['label', 'sha256', 'h5_index', 'depth_m', 'nuisance_cloud', 'nuisance_missing', 'nuisance_saturated_defective']:
        assert np.array_equal(saved[col].to_numpy(), expected.loc[saved.chip_id, col].to_numpy()), col
    groups = group_rows(saved)
    assert pd.DataFrame({'group': groups, 'split': saved.split}).groupby('group').split.nunique().max() == 1
    split = json.loads((source / 'split.json').read_text())
    for name in ['train', 'val']:
        assert saved.loc[saved.split.eq(name), 'chip_id'].tolist() == split[name + '_chip_ids']
    protocol = json.loads((source / 'split_protocol.json').read_text())
    replay = dict(mode='published_split_replay', selected_sha256=sha(source / 'selected.csv'), split_sha256=sha(source / 'split.json'), original_solver=protocol['solver'].get('original_solver', protocol['solver']))
    return saved, pd.read_csv(source / 'feature_balance.csv').to_dict('records'), replay


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--new-split',action='store_true',help='Optimize a new grouped split for a changed selected union');args=parser.parse_args()
    source=config.get_path('FULL_NUISANCE_BALANCE_OUTPUT');out=config.get_path('FINAL_MODELS_DIR');out.mkdir(parents=True,exist_ok=False)
    frames=[pd.read_csv(source/f'seed_{seed}/selected.csv') for seed in [42,123,456]]
    f=pd.concat(frames).drop_duplicates('chip_id').sort_values('chip_id').reset_index(drop=True);assert f.chip_id.is_unique and set(f.label)=={0,1}
    lineage={f'seed_{seed}/selected.csv':sha(source/f'seed_{seed}/selected.csv') for seed in [42,123,456]}
    if args.new_split:
        from final_partition import optimize_final_split
        f,balance,solver=optimize_final_split(f)
    else:
        f,balance,solver=split_union(f)
    f.to_csv(out/'selected.csv',index=False);(out/'split.json').write_text(json.dumps({sp+'_chip_ids':f.loc[f.split.eq(sp),'chip_id'].tolist() for sp in ['train','val']},indent=2));pd.DataFrame(balance).to_csv(out/'feature_balance.csv',index=False)
    (out/'split_protocol.json').write_text(json.dumps(dict(source_hashes=lineage,all_selected_chips_retained=True,counts={sp:{str(lab):int(((f.split==sp)&(f.label==lab)).sum()) for lab in [0,1]} for sp in ['train','val']},solver=solver,limitation='Internal validation on previously explored observations; no independent test set.'),indent=2))
    import shutil
    for name in ([] if args.new_split else ['selected.csv', 'split.json', 'feature_balance.csv']):
        shutil.copy2(config.REPO_ROOT / 'training/final_models' / name, out / name)
    cachepath=config.get_path('FULL_NUISANCE_EMBEDDINGS');cache=np.load(cachepath);idx={str(v):i for i,v in enumerate(cache['chip_ids'])};indices=[idx[v] for v in f.chip_id]
    assert set(idx)==set(f.chip_id);assert np.array_equal(cache['sha256'][indices].astype(str),f.sha256.to_numpy(str))
    encoder=json.loads(cachepath.with_name('protocol.json').read_text());assert encoder['scope']=='full';validate_geometry(encoder)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('medium');records=[];predictions=[];catalog=[]
    for family,pool,settings in [('small_mlp',5,[.1,.01,.0001]),('mlp',1,[.0001])]:
        lo,hi=cache_local_slice(pool);x=np.concatenate([cache['cls'],cache['centre_patches'][:,lo:hi,lo:hi].mean((1,2))],1)[indices];assert np.isfinite(x).all()
        train=f.split.eq('train').to_numpy();val=f.split.eq('val').to_numpy();xt=torch.tensor(x[train],device='cuda');yt=torch.tensor(f.label.to_numpy(np.float32)[train],device='cuda');xv=torch.tensor(x[val],device='cuda');yv=f.label.to_numpy()[val]
        best=-1;chosen=None;candidates=[]
        for wd in settings:
            random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);head=make_head(family).cuda();opt=torch.optim.Adam(head.parameters(),lr=.001,weight_decay=wd);bestscore=-1;stale=0;history=[]
            for epoch in range(1,101):
                head.train();order=torch.randperm(len(yt),device='cuda');total=0
                for start in range(0,len(yt),64):
                    batch=order[start:start+64];loss=sigmoid_focal_loss(head(xt[batch]).reshape(-1),yt[batch]);opt.zero_grad();loss.backward();opt.step();total+=float(loss)*len(batch)
                head.eval()
                with torch.no_grad():vp=torch.sigmoid(head(xv).reshape(-1)).cpu().numpy()
                score=metrics(yv,vp,.5)['macro_f1'];history.append(dict(epoch=epoch,train_loss=total/len(yt),val_macro_f1=score))
                if score>bestscore+1e-5:bestscore=score;bestepoch=epoch;state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};stale=0
                else:stale+=1
                if stale>=25:break
            head.load_state_dict(state);head.eval()
            with torch.no_grad():vp=torch.sigmoid(head(xv).reshape(-1)).cpu().numpy()
            score=metrics(yv,vp,.5)['macro_f1'];candidates.append(dict(weight_decay=wd,val_macro_f1=score,best_epoch=bestepoch,history=history))
            if score>best+1e-12:best=score;chosen=(state,wd,bestepoch,vp)
        state,wd,epoch,vp=chosen;grid=[.5]+[float(t) for t in np.linspace(.05,.95,91) if not np.isclose(t,.5)];threshold=max(grid,key=lambda t:metrics(yv,vp,t)['macro_f1']);name=f'{family}_pool{pool}';folder=out/name;folder.mkdir()
        checkpoint=dict(format_version=1,architecture=family,state_dict=state,pool_size=pool,threshold=threshold,seed=42,best_epoch=epoch,weight_decay=wd,input_dim=2048,encoder=dict(backbone='Clay v1.5 large',checkpoint_sha256=encoder['checkpoint_sha256'],metadata_sha256=encoder['clay_metadata_sha256'],bands=encoder['bands'],scope='full128',resize=[ENCODER_SIZE,ENCODER_SIZE],interpolation='bilinear align_corners=False',normalization='Resize then normalize with CLAY_META',gsd=encoder['gsd_m'],time='zeros',latlon='zeros',features='CLS concatenated with mean of central patch tokens'),selected_manifest_sha256=sha(out/'selected.csv'),split_sha256=sha(out/'split.json'))
        torch.save(checkpoint,folder/'model.pt');model,reloaded=load_head(folder/'model.pt','cuda');(folder/'selection.json').write_text(json.dumps(dict(weight_decay=wd,threshold=threshold,best_epoch=epoch,candidates=candidates),indent=2))
        with torch.no_grad():rp,_=predict_embeddings(model,reloaded,x[val])
        assert np.allclose(rp.cpu().numpy(),vp,atol=1e-7)
        for split,mask in [('train',train),('val',val)]:
            prob,_=predict_embeddings(model,reloaded,x[mask]);prob=prob.cpu().numpy();y=f.label.to_numpy()[mask]
            for mode,t in [('default',.5),('validation_tuned',threshold)]:records.append(dict(model=name,split=split,threshold_mode=mode,N=len(y),P=int(y.sum()),A=int((y==0).sum()),**metrics(y,prob,t)))
            predictions.extend(dict(model=name,split=split,chip_id=cid,label=int(label),probability=float(p)) for cid,label,p in zip(f.loc[mask,'chip_id'],y,prob))
        catalog.append(dict(model=name,path_key='FINAL_CHECKPOINTS_DIR',relative_path=f'{name}/model.pt',sha256=sha(folder/'model.pt'),bytes=(folder/'model.pt').stat().st_size,architecture=family,pool_size=pool,threshold=threshold,best_epoch=epoch,weight_decay=wd,format_version=1))
        print(name,'epoch',epoch,'threshold',threshold,'validation Macro F1',metrics(yv,vp,threshold)['macro_f1'],flush=True)
    pd.DataFrame(records).to_csv(out/'metrics.csv',index=False);pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False);(out/'model_manifest.json').write_text(json.dumps(dict(models=catalog,embedding_cache_sha256=sha(cachepath),source_hashes=lineage,script_sha256=sha(__file__),replay_checked=True,no_test_set=True),indent=2));assert all(sha(source/p)==v for p,v in lineage.items());print('Saved both reusable final heads.',flush=True)

if __name__=='__main__':main()
