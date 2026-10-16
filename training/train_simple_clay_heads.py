"""Compare smaller heads on immutable full-chip Clay features; writes only --out.

Regularization chosen by validation macro F1 at .5, then threshold tuned on val.
Test predictions are computed only after selection. No field data used.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse,sys
import hashlib
import json
import random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
from split_groups import group_rows
from encoder_geometry import validate_geometry


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def metrics(y, p, t):
    tn, fp, fn, tp = confusion_matrix(y, p >= t, labels=[0,1]).ravel()
    f = f1_score(y, p >= t, labels=[0,1], average=None, zero_division=0)
    return dict(threshold=float(t), roc_auc=float(roc_auc_score(y,p)), macro_f1=float(f.mean()), f1_absent=float(f[0]),
                f1_present=float(f[1]), present_precision=float(tp/(tp+fp)) if tp+fp else 0.,
                absent_precision=float(tn/(tn+fn)) if tn+fn else 0.,
                present_recall=float(tp/(tp+fn)) if tp+fn else 0.,
                absent_recall=float(tn/(tn+fp)) if tn+fp else 0., tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=Path)
    ap.add_argument('--cache', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--device', default='cuda', choices=['cpu','cuda'])
    args = ap.parse_args()
    args.dataset=args.dataset or config.get_path('FULL_NUISANCE_BALANCE_OUTPUT')
    args.cache=args.cache or args.dataset/'clay_full_sweep/embeddings.npz'
    args.out=args.out or args.dataset/'simple_heads'
    root, out = args.dataset.resolve(), args.out.resolve()
    assert not out.exists(), 'Output already exists; refusing to overwrite'
    assert root != out, 'Do not overwrite input manifests'
    out.mkdir(parents=True)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('medium')
    files = [args.cache] + [root/f'seed_{s}/{n}' for s in [42,123,456]
                           for n in ['selected.csv',f'split_seed{s}.json']]
    hashes = {str(p):sha(p) for p in files}
    protocol = dict(seeds=[42,123,456], pools=[1,5], selection='Validation macro F1 at .5; ties prefer stronger regularization. Then tune threshold on validation .05:.95 step .01, ties prefer .5.',
        logistic_C=[.0001,.001,.01,.1,1.,10.], small_mlp_weight_decay=[.1,.01,.0001],
        logistic='Training-only StandardScaler; L2 logistic, unweighted cross entropy, no class resampling, lbfgs max_iter=10000',
        small_mlp='2048->32 ReLU dropout .3->1; unscaled cached features as in original MLP; same focal alpha .5 gamma 2; Adam .001; batch64 max100 patience25; checkpoint by val macroF1@.5',
        parameter_counts={'logistic':2049,'small_mlp':65601,'reference_mlp':1114881},
        limitations='Exploratory comparison after examining existing tests; not a fresh confirmatory test. No field-label tuning. Linear head also changes loss and scaling; small MLP isolates capacity more closely.',
        device=args.device, input_sha256={('embedding_cache' if Path(p).resolve()==args.cache.resolve() else str(Path(p).relative_to(root))):h for p,h in hashes.items()}, script_sha256=sha(__file__))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    cache=np.load(args.cache)
    n=len(cache['chip_ids']);assert cache['cls'].shape==(n,1024) and cache['centre_patches'].shape==(n,7,7,1024)
    source_protocol=json.loads(args.cache.with_name('protocol.json').read_text());assert source_protocol['scope']=='full';validate_geometry(source_protocol)
    protocol['encoder_protocol_sha256']=sha(args.cache.with_name('protocol.json'))
    protocol['encoder_scope']=source_protocol['spatial_input']
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    idx={str(c):i for i,c in enumerate(cache['chip_ids'])}
    assert len(idx)==n
    records=[]; predictions=[]; candidates=[]
    for pool in [1,5]:
        lo,hi=(3,4) if pool==1 else (1,6)
        x=np.concatenate([cache['cls'],cache['centre_patches'][:,lo:hi,lo:hi].mean((1,2))],1)
        assert np.isfinite(x).all()
        for seed in [42,123,456]:
            f=pd.read_csv(root/f'seed_{seed}/selected.csv')
            assert f.chip_id.is_unique
            assert pd.DataFrame({'group':group_rows(f),'split':f.split}).groupby('group').split.nunique().max()==1
            split=json.loads((root/f'seed_{seed}/split_seed{seed}.json').read_text())
            indices=np.array([idx[str(c)] for c in f.chip_id])
            assert np.array_equal(cache['sha256'][indices].astype(str),f.sha256.to_numpy(str))
            parts={s:f[f.split.eq(s)].copy() for s in ['train','val','test']}
            for s,z in parts.items():
                assert set(z.chip_id)==set(split[s+'_chip_ids'])
                assert int(z.label.sum())==2*int(z.label.eq(0).sum())
            xx={s:x[[idx[str(c)] for c in z.chip_id]] for s,z in parts.items()}
            yy={s:z.label.to_numpy(int) for s,z in parts.items()}
            for family in ['logistic','small_mlp']:
                best=-1.; chosen=None
                settings=protocol['logistic_C'] if family=='logistic' else protocol['small_mlp_weight_decay']
                for value in settings:
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
                    epoch_best=None;history=[]
                    if family=='logistic':
                        model=make_pipeline(StandardScaler(),LogisticRegression(C=value,solver='lbfgs',max_iter=10000,random_state=seed))
                        model.fit(xx['train'],yy['train'])
                        assert model[-1].n_iter_.max()<10000, 'Logistic regression did not converge'
                        vp=model.predict_proba(xx['val'])[:,1]
                    else:
                        model=nn.Sequential(nn.Linear(2048,32),nn.ReLU(),nn.Dropout(.3),nn.Linear(32,1)).to(args.device)
                        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=value)
                        xt=torch.tensor(xx['train'],dtype=torch.float32,device=args.device)
                        yt=torch.tensor(yy['train'],dtype=torch.float32,device=args.device)
                        xv=torch.tensor(xx['val'],dtype=torch.float32,device=args.device)
                        score_best=-1.;stale=0
                        for epoch in range(1,101):
                            model.train();order=torch.randperm(len(yt),device=args.device)
                            for start in range(0,len(yt),64):
                                b=order[start:start+64];logits=model(xt[b]).squeeze(1)
                                ce=F.binary_cross_entropy_with_logits(logits,yt[b],reduction='none')
                                p=torch.sigmoid(logits);pt=p*yt[b]+(1-p)*(1-yt[b])
                                loss=(.5*(1-pt).pow(2)*ce).mean()
                                opt.zero_grad();loss.backward();opt.step()
                            model.eval()
                            with torch.no_grad():vp=torch.sigmoid(model(xv).squeeze(1)).cpu().numpy()
                            score=metrics(yy['val'],vp,.5)['macro_f1']
                            history.append(dict(epoch=epoch,val_macro_f1=score))
                            if score>score_best+1e-5:
                                score_best=score;epoch_best=epoch;stale=0
                                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                            else:stale+=1
                            if stale>=25:break
                        model.load_state_dict(state);model.eval()
                        with torch.no_grad():vp=torch.sigmoid(model(xv).squeeze(1)).cpu().numpy()
                    score=metrics(yy['val'],vp,.5)['macro_f1']
                    candidates.append(dict(pool=pool,seed=seed,family=family,regularization=value,val_macro_f1=score,best_epoch=epoch_best))
                    if score>best+1e-12:
                        best=score;chosen=(model,value,vp.copy(),epoch_best,history)
                model,value,vp,epoch_best,history=chosen
                grid=[.5]+[float(t) for t in np.linspace(.05,.95,91) if not np.isclose(t,.5)]
                threshold=max(grid,key=lambda t:metrics(yy['val'],vp,t)['macro_f1'])
                dest=out/f'{family}_p{pool}_seed{seed}';dest.mkdir()
                if family=='logistic':
                    joblib.dump(model,dest/'model.joblib');reloaded=joblib.load(dest/'model.joblib')
                    predict=lambda a:reloaded.predict_proba(a)[:,1]
                else:
                    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},threshold=threshold,pool=pool),dest/'model.pt')
                    model.load_state_dict(torch.load(dest/'model.pt',weights_only=True,map_location=args.device)['state_dict']);model.eval()
                    def predict(a):
                        with torch.no_grad():return torch.sigmoid(model(torch.tensor(a,dtype=torch.float32,device=args.device)).squeeze(1)).cpu().numpy()
                assert np.allclose(predict(xx['val']),vp,atol=1e-7)
                (dest/'selection.json').write_text(json.dumps(dict(regularization=value,threshold=threshold,best_epoch=epoch_best,history=history),indent=2))
                for s,z in parts.items():
                    p=predict(xx[s])
                    for mode,t in [('default',.5),('validation_tuned',threshold)]:
                        records.append(dict(family=family,pool=pool,seed=seed,split=s,mode=mode,regularization=value,**metrics(yy[s],p,t)))
                    for row,prob in zip(z.itertuples(),p):predictions.append(dict(family=family,pool=pool,seed=seed,split=s,chip_id=row.chip_id,label=row.label,probability=float(prob)))
                pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)
                pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
                pd.DataFrame(candidates).to_csv(out/'validation_candidates.csv',index=False)
                print(f'Finished {family} pool={pool} seed={seed} regularization={value} threshold={threshold:.2f}',flush=True)
    assert all(sha(p)==h for p,h in hashes.items())
    (out/'verification.json').write_text(json.dumps(dict(inputs_unchanged=True,cache_hashes_match=True,checkpoint_reload_predictions_match=True,selected_models=12,field_data_used=False),indent=2))
    print('Complete',flush=True)

if __name__=='__main__':main()
