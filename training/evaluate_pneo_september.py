"""Restore September PNEO features and evaluate exploratory transect-held-out LR.

This fits PNEO-specific diagnostic heads; it does not apply the final S2 heads.
"""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from scipy.stats import mannwhitneyu,wilcoxon
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
from evaluate_sesimbra_campaigns import scores

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    out=config.get_path('PNEO_SEPTEMBER_RESULTS');out.mkdir(parents=True,exist_ok=False)
    f=pd.read_csv(config.get_path('PNEO_SEPTEMBER_FEATURES'));f=f[f.date_iso.eq('2026-09-02')].sort_values(['depth_m','transect','quadrat']).reset_index(drop=True);f['y']=f.is_present.astype(int);f['group']=f.depth_m.astype(str)+'_T'+f.transect.astype(str)
    field=pd.read_csv(config.get_path('FIELD_OBSERVATIONS'));field=field[field.date.eq('2026-09-02')].sort_values(['gps_depth_m','T','Q']).reset_index(drop=True);assert len(f)==30 and f.y.sum()==20 and np.array_equal(f.y,field.label) and np.allclose(f[['lat','lon']],field[['lat','lon']],atol=1e-8,rtol=0)
    meta=pd.read_csv(config.get_path('PNEO_CLAY_METADATA'));cache=np.load(config.get_path('PNEO_CLAY_EMBEDDINGS'));lookup={(r.date_iso,int(r.depth_m),int(r.transect),int(r.quadrat)):i for i,r in enumerate(meta.itertuples())};ids=[lookup[(r.date_iso,int(r.depth_m),int(r.transect),int(r.quadrat))] for r in f.itertuples()];assert np.array_equal(meta.iloc[ids].is_present.astype(int),f.y);assert np.allclose(meta.iloc[ids][['lat','lon']],f[['lat','lon']],atol=1e-8,rtol=0)
    raw=[c for c in f if c.endswith(('_mean','_std','_range'))];dii=[c for c in f if c.startswith('DII_') and c.endswith('_w5')];featurecols=raw+[c for c in f if c.startswith('DII_')]
    cls=cache['cls'][ids];patch=cache['centre_patches'][ids];assert cls.shape==(30,1024) and patch.shape==(30,7,7,1024)
    specs={'spectral_statistics':f[raw].to_numpy(),'lyzenga_indices':f[dii].to_numpy(),'spectral_plus_lyzenga':f[raw+dii].to_numpy(),'clay_cls_pool1':np.concatenate([cls,patch[:,3,3]],1),'clay_cls_pool5':np.concatenate([cls,patch[:,1:6,1:6].mean((1,2))],1)}
    predictions=[];metrics=[];folds=[];y=f.y.to_numpy();groups=f.group.to_numpy()
    for name,x in specs.items():
        p=np.full(len(f),np.nan)
        for train,test in LeaveOneGroupOut().split(x,y,groups):
            assert len(np.unique(y[train]))==2;model=make_pipeline(StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=10000,random_state=42));model.fit(x[train],y[train]);assert model[-1].n_iter_.max()<10000;p[test]=model.predict_proba(x[test])[:,1];folds.append(dict(model=name,held_out_group=groups[test][0],**scores(y[test],p[test],.5)))
        assert np.isfinite(p).all();metrics.append(dict(model=name,**scores(y,p,.5)))
        for i,r in f.iterrows():predictions.append(dict(model=name,group=r['group'],depth_m=r.depth_m,transect=r.transect,quadrat=r.quadrat,label=int(y[i]),probability=float(p[i]),predicted_label=int(p[i]>=.5)))
    effects=[]
    for col in featurecols:
        anomaly=f[col]-f.groupby('group')[col].transform('median');f[col+'_anomaly']=anomaly;p=anomaly[f.y.eq(1)];a=anomaly[f.y.eq(0)];diff=[]
        for _,q in f.groupby('group'):diff.append(q.loc[q.y.eq(1),col].median()-q.loc[q.y.eq(0),col].median())
        effects.append(dict(feature=col,median_present_anomaly=float(p.median()),median_absent_anomaly=float(a.median()),p_mann_whitney=float(mannwhitneyu(p,a,alternative='two-sided').pvalue),n_transects=6,positive_transect_differences=int((np.array(diff)>0).sum()),median_transect_difference=float(np.median(diff)),p_wilcoxon=float(wilcoxon(diff,zero_method='pratt').pvalue)))
    effects=pd.DataFrame(effects)
    for col in ['p_mann_whitney','p_wilcoxon']:
        order=np.argsort(effects[col].to_numpy());pv=effects[col].to_numpy()[order];q=np.minimum.accumulate((pv*len(pv)/np.arange(1,len(pv)+1))[::-1])[::-1];corrected=np.empty(len(q));corrected[order]=np.minimum(q,1);effects[col+'_bh']=corrected
    f.to_csv(out/'features.csv',index=False);effects.to_csv(out/'within_transect_effects.csv',index=False);pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False);pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False);pd.DataFrame(folds).to_csv(out/'fold_metrics.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(scene='000618890',image_date='2026-09-03',observation_date='2026-09-02',N=30,P=20,A=10,transects=6,field_labels_and_coordinates_verified=True,input_hashes={key:sha(config.get_path(key)) for key in ['PNEO_SEPTEMBER_FEATURES','PNEO_CLAY_METADATA','PNEO_CLAY_EMBEDDINGS','FIELD_OBSERVATIONS']},source='September feature and embedding exports; checked against current matched field observations.',features='Six bands R G B NIR RE DB; 5x5 means/std/ranges plus Lyzenga log-difference indices. Full-scene water statistics for unsupervised normalization. Clay native224x224, GSD1.2, scene-water normalization, geographic/week encodings and central token14 preserve the PNEO exploration conventions.',evaluation='Leave one depth-by-transect group out; train-only StandardScaler; LR C0.01 class_weight balanced; threshold0.5 fixed; every September quadrat evaluated once.',limitations='Exploratory same-scene evaluation on six adjacent transects; overlapping image windows and reused analysis. Not the final S2 models or independent geographic validation. Transect centring removes a shared offset, not all confounders. Multiple feature tests corrected with Benjamini-Hochberg separately per test family.',script_sha256=sha(__file__)),indent=2));print(pd.DataFrame(metrics).to_string(index=False),flush=True)

if __name__=='__main__':main()
