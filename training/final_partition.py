"""Construct a grouped final train/validation split using depth and quality only."""
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from scipy.stats import ks_2samp
from split_groups import group_rows


def optimize_final_split(f):
    cols=['depth_m','nuisance_cloud','nuisance_missing','nuisance_saturated_defective']
    f=f.copy();f['group']=group_rows(f);groups=np.unique(f.group);G=len(groups);target={lab:max(1,round(.15*int(f.label.eq(lab).sum()))) for lab in [0,1]}
    sizes={lab:np.array([((f.group==g)&(f.label==lab)).sum() for g in groups]) for lab in [0,1]}
    counts={lab:int(f.label.eq(lab).sum()) for lab in [0,1]}
    # A common grid checks both sides of ABSENT steps, preserving the depth audit convention.
    features=[];owners=[]
    for col in cols:
        v=f[col].to_numpy(float);assert np.isfinite(v).all()
        sd=v.std();features.append((v-v.mean())/(sd if sd else 1));owners.append(col)
        av=f.loc[f.label.eq(0),col].to_numpy();cuts=np.unique(np.r_[np.quantile(v,np.linspace(0,1,21)),av,np.nextafter(av,-np.inf)])
        for cut in cuts:features.append((v<=cut).astype(float));owners.append(col)
    M=np.column_stack(features);K=M.shape[1]
    sums={lab:np.array([M[(f.group==g)&f.label.eq(lab)].sum(0) for g in groups]) for lab in [0,1]}
    # Four blocks: P/A discrepancy in val and train, and each class's val-vs-full discrepancy.
    expressions=[]
    expressions.append((sums[1]/target[1]-sums[0]/target[0],np.zeros(K)))
    trP=counts[1]-target[1];trA=counts[0]-target[0]
    expressions.append((-sums[1]/trP+sums[0]/trA,sums[1].sum(0)/trP-sums[0].sum(0)/trA))
    for lab in [0,1]:expressions.append((sums[lab]/target[lab],-sums[lab].sum(0)/counts[lab]))
    V=G+len(expressions)*K+len(expressions)*len(cols);ri=[];ci=[];vv=[];lower=[];upper=[]
    def add(ids,vals,lo=-np.inf,hi=np.inf):
        row=len(lower);ri.extend([row]*len(ids));ci.extend(ids);vv.extend(vals);lower.append(lo);upper.append(hi)
    for lab in [0,1]:add(list(range(G)),sizes[lab].tolist(),lo=target[lab],hi=target[lab])
    ag=(sizes[0]>0).astype(int);add(list(range(G)),ag.tolist(),lo=5,hi=int(ag.sum())-5)
    c=np.zeros(V)
    for block,(coef,constant) in enumerate(expressions):
        weight=1 if block<2 else .5
        for k,col in enumerate(owners):
            e=G+block*K+k;z=G+len(expressions)*K+block*len(cols)+cols.index(col)
            nz=np.flatnonzero(coef[:,k]);add(nz.tolist()+[e],coef[nz,k].tolist()+[-1],hi=-constant[k]);add(nz.tolist()+[e],(-coef[nz,k]).tolist()+[-1],hi=constant[k]);add([e,z],[1,-1],hi=0)
            c[e]=weight*(3 if col=='depth_m' else 1)/len(owners)/4
        for j,col in enumerate(cols):c[G+len(expressions)*K+block*len(cols)+j]=weight*(3 if col=='depth_m' else 1)/len(cols)
    c[:G]=np.random.default_rng(42).uniform(0,1e-8,G);integrality=np.zeros(V);integrality[:G]=1;ub=np.full(V,np.inf);ub[:G]=1
    result=milp(c,integrality=integrality,bounds=Bounds(np.zeros(V),ub),constraints=LinearConstraint(coo_matrix((vv,(ri,ci)),shape=(len(lower),V)).tocsr(),lower,upper),options={'time_limit':180,'mip_rel_gap':.015})
    assert result.x is not None,result.message
    val=set(groups[result.x[:G]>.5]);f['split']=np.where(f.group.isin(val),'val','train')
    assert f.groupby('group').split.nunique().max()==1
    for lab in [0,1]:assert int((f.split.eq('val')&f.label.eq(lab)).sum())==target[lab]
    audit=[]
    for split in ['train','val']:
        for col in cols:
            p=f.loc[f.split.eq(split)&f.label.eq(1),col];a=f.loc[f.split.eq(split)&f.label.eq(0),col];audit.append(dict(split=split,feature=col,ks=float(ks_2samp(p,a).statistic),present_mean=float(p.mean()),absent_mean=float(a.mean())))
    return f,audit,dict(status=int(result.status),message=result.message,objective=float(result.fun),relative_gap=float(result.mip_gap),features=cols,seed=42,objective_description='PRESENT/ABSENT means and CDF differences in both partitions, plus each class validation distribution versus full union; no spectral or acquisition variables.',target_validation=target)
