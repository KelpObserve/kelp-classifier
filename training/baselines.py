"""Shared fixed-hyperparameter baseline fitting for the current partitions."""
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score, precision_recall_fscore_support

def evaluate(f,split,seed,variant,image_prefix="ctx_",image_family="context"):
    train=f[f.split.eq('train')];held=f[f.split.eq(split)];rows=[]
    ctx=[c for c in f if c.startswith(image_prefix)]
    assert ctx, 'No image predictors found'
    for family,cols in [(image_family,ctx),('depth_only',['depth_m']),(image_family+'_depth',ctx+['depth_m'])]:
        median=train[cols].median().fillna(0)
        for kind in ['lr','rf']:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=3000,class_weight='balanced',random_state=seed)) if kind=='lr' else RandomForestClassifier(n_estimators=120,max_depth=3,min_samples_leaf=3,class_weight='balanced',random_state=seed,n_jobs=1)
            model.fit(train[cols].fillna(median),train.label)
            p=model.predict_proba(held[cols].fillna(median))[:,1];y=held.label.to_numpy();pred=p>=.5
            precision,recall,_,_=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
            rows.append(dict(variant=variant,seed=seed,split=split,family=family,model=kind,macro_f1=f1_score(y,pred,average='macro'),roc_auc=roc_auc_score(y,p),balanced_accuracy=balanced_accuracy_score(y,pred),present_precision=precision[1],present_recall=recall[1],absent_precision=precision[0],absent_recall=recall[0],N=len(y)))
    return rows
