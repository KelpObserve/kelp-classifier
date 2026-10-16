"""Report full-chip nuisance balancing and frozen Clay pool sweep."""
import csv,sys,statistics,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
from report_links import relocate_links

def read(p):
    with p.open() as f:return list(csv.DictReader(f))

def main():
    root=config.get_path('FULL_NUISANCE_BALANCE_OUTPUT')
    assignments=[]
    for seed in [42,123,456]:assignments.extend(dict(seed=seed,**row) for row in read(root/f'seed_{seed}/selected.csv'))
    with (root/'selected.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(assignments[0]));writer.writeheader();writer.writerows(assignments)
    pool_protocol=json.loads(config.get_path('FULL_NUISANCE_MATCHING_POOL').with_suffix('.json').read_text())
    before=pool_protocol['before'];after=pool_protocol['after']
    train_sizes=[sum(r['split']=='train' for r in assignments if r['seed']==seed) for seed in [42,123,456]]
    b=read(root/'baseline_metrics.csv');c=[r for r in read(root/'clay_full_sweep/metrics.csv') if r['threshold_mode']=='validation_tuned']
    pools=sorted({int(r['pool_size']) for r in c})
    groups=[]
    for fam in ['quality_depth','full']:
        for model in ['lr','rf']:
            groups.append((f'{fam.replace("_"," ")} {model.upper()}',[r for r in b if r['variant']=='balanced' and r['family']==fam and r['model']==model]))
    for pool in pools:groups.append((f'Clay + MLP pool {pool}',[r for r in c if int(r['pool_size'])==pool]))
    simple_path=root/'simple_heads/metrics.csv'
    if simple_path.exists():
        simple=[r for r in read(simple_path) if r['mode']=='validation_tuned']
        for family,title in [('small_mlp','Clay + small MLP'),('logistic','Clay + logistic regression')]:
            for pool in [1,5]:groups.append((f'{title} pool {pool}',[r for r in simple if r['family']==family and int(r['pool'])==pool]))
    def avg(rows,split,key):return statistics.mean(float(r[key]) for r in rows if r['split']==split)
    fig,axs=plt.subplots(1,2,figsize=(19,6));x=np.arange(len(groups))
    for ax,key,title in zip(axs,['macro_f1','roc_auc'],['Macro F1','ROC AUC']):
        for split,offset,color in [('val',-.18,'#197b92'),('test',.18,'#c56027')]:
            bars=ax.bar(x+offset,[avg(rs,split,key) for _,rs in groups],.35,color=color,label=split)
            ax.bar_label(bars,fmt='%.3f',fontsize=7,padding=2,rotation=90)
        ax.set_xticks(x,[n for n,_ in groups],rotation=35,ha='right');ax.set_ylim(0,1);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axs[0].legend();fig.tight_layout();fig.savefig(root/'model_comparison.png',dpi=160);plt.close(fig)
    d=read(root/'feature_balance.csv');features=list(dict.fromkeys(r['feature'] for r in d));fig,axs=plt.subplots(len(features),3,figsize=(11,3*len(features)),squeeze=False)
    for i,feature in enumerate(features):
        for j,split in enumerate(['train','val','test']):
            ax=axs[i,j]
            for seed,style in zip([42,123,456],['-', '--', ':']):
                rows=read(root/f'seed_{seed}/selected.csv')
                for lab,color in [('1','#197b92'),('0','#c56027')]:
                    values=np.sort([float(r[feature]) for r in rows if r['split']==split and r['label']==lab])
                    ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,linestyle=style,label=f'{"P" if lab=="1" else "A"} seed {seed}')
            ax.set_title(f'{split}: {feature}');ax.grid(alpha=.2);ax.set_ylim(0,1.02)
    axs[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(root/'nuisance_distributions.png',dpi=160);plt.close(fig)
    lines=['# Whole-chip nuisance balancing','','Clay encodes the full native 128×128 chip, resized to 224×224. Encoder GSD is derived as 128 × 10 / 224 ≈ 5.714 m; the native footprint remains 1,280 m. The MLP receives the global CLS token concatenated with the average of the selected central patch tokens (pools 1, 2, 3 and 5). Central pooling happens after full-image encoding. Clay is frozen; time/location inputs are constant zeros.','','## Selection and baselines','',f'The starting pool has {before["1"]} PRESENT and {before["0"]} ABSENT chips. Central 5×5 quality exclusion leaves {after["1"]} PRESENT and {after["0"]} ABSENT candidates. We retain all {after["0"]} eligible ABSENT chips, repartition their spatial/observation/duplicate groups, then select {2*after["0"]} PRESENT chips for each seed at 2:1 in every partition. Matching uses observation depth and full-chip SCL cloud/shadow/cirrus, missing-data and saturated/defective fractions. Whole-chip flags are matched, not used as a zero-tolerance exclusion.','','Spectral brightness, colour, texture, SCL land/water fractions, acquisition dates and processing metadata do not enter matching. Independent land-overlap and reliable sunglint measurements are unavailable: this experiment does not establish control of all confounders. Existing acquisition filters remain inherited. Each ABSENT observation supplies images from up to eight different acquisition dates within ±30 days. No observation has duplicate acquisition dates. These images share one field label/location and remain grouped in the same partition.','','LR means logistic regression: a linear decision rule on the supplied features. RF means random forest: an ensemble of decision trees that can capture nonlinear relationships. Baseline LR/RF use image statistics or nuisance measurements; Clay + logistic regression uses frozen Clay embeddings. Whole-chip image baselines use mean, standard deviation, 10th/90th percentiles and their difference for all pixels in each of the same six bands Clay receives. These baseline predictors are not matching variables. The nuisance baseline uses only depth and the three quality fractions. Baselines use fixed hyperparameters and threshold 0.5; each Clay checkpoint and threshold is chosen using validation only. These are exploratory partitions, not untouched external validation.','','## Classifier results','','Three-seed means. Every model uses the same selected chips and partitions. All pool results are shown; no pool is selected using test performance.','','| Model | Validation Macro F1 | Validation AUC | Test Macro F1 | Test AUC |','| --- | ---: | ---: | ---: | ---: |']
    for name,rs in groups:lines.append('| '+name+' | '+' | '.join(f'{avg(rs,sp,k):.3f}' for sp in ['val','test'] for k in ['macro_f1','roc_auc'])+' |')
    lines+=['','![Classifier comparison](model_comparison.png)','','| Clay pool | Split | PRESENT precision | PRESENT recall | ABSENT precision | ABSENT recall | Macro F1 |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for pool in pools:
        rs=[r for r in c if int(r['pool_size'])==pool]
        for sp in ['train','val','test']:lines.append(f'| {pool} | {sp} | '+' | '.join(f'{avg(rs,sp,k):.3f}' for k in ['present_precision','present_recall','absent_precision','absent_recall','macro_f1'])+' |')
    if simple_path.exists():
        lines+=['','## Simpler Clay heads','','Logistic regression (2,049 parameters) and a 2048 → 32 → 1 MLP (65,601 parameters) use the same full-chip embeddings and partitions, for pools 1 and 5. Logistic C and small-MLP weight decay are selected by validation Macro F1 at 0.5, followed by validation-only threshold selection. No test results select regularization. [Runner](../train_simple_clay_heads.py), [metrics](simple_heads/metrics.csv), [selection candidates](simple_heads/validation_candidates.csv) and [protocol](simple_heads/protocol.json).','','| Model | Split | PRESENT precision | PRESENT recall | ABSENT precision | ABSENT recall | Macro F1 |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for name,rs in groups[8:]:
            for sp in ['train','val','test']:lines.append('| '+name+' | '+sp+' | '+' | '.join(f'{avg(rs,sp,k):.3f}' for k in ['present_precision','present_recall','absent_precision','absent_recall','macro_f1'])+' |')
    lines+=['',f'Original MLP pool 5 has mean validation Macro F1 {avg([r for r in c if int(r["pool_size"])==5], "val", "macro_f1"):.3f}, test Macro F1 {avg([r for r in c if int(r["pool_size"])==5], "test", "macro_f1"):.3f}. Depth/quality RF test Macro F1 is {avg(groups[1][1], "test", "macro_f1"):.3f}; the always-PRESENT baseline is 0.400. Measured nuisance predictability must be considered when interpreting performance. With {min(train_sizes)}–{max(train_sizes)} training chips and about 1.1 million trainable parameters in the original MLP, overfitting is possible. Shared scenes, overlapping footprints and repeated test inspection also limit generalization claims. The full selected union also supports the [two final reusable models](../final_models/README.md), fitted on a separate train/validation split.','','## Distribution balance','','Mean class KS distance across three seeds; smaller means closer marginal distributions. Only the current selected partitions are shown.','','| Feature | Split | KS |','| --- | --- | ---: |']
    for feature in features:
        for split in ['train','val','test']:lines.append(f'| {feature} | {split} | '+' | '.join(f'{statistics.mean(float(r["ks"]) for r in d if r["feature"]==feature and r["split"]==split and r["variant"]==v):.3f}' for v in ['balanced'])+' |')
    lines+=['','![Nuisance distributions](nuisance_distributions.png)','','## Reproduction','','Configure the inputs and outputs described and linked in [example_config.json](../../example_config.json), overriding defaults in untracked config.json and using fresh output directories.','','```sh','python training/prepare_nuisance_pool.py','python training/balance_absent_first.py','python training/train_frozen_sesimbra.py --pools 1 2 3 5','python training/train_simple_clay_heads.py','python training/verify_full_nuisance_trial.py','python training/verify_simple_clay_heads.py','python training/build_model_catalog.py','python training/summarize_full_nuisance_trial.py','```','','[Seed-aware chip manifest](selected.csv), [split counts](split_counts.csv), [selection protocol](protocol.json), [baseline results](baseline_metrics.csv) (`variant=balanced`), [Clay results](clay_full_sweep/metrics.csv), [Clay predictions](clay_full_sweep/predictions.csv), [Clay protocol](clay_full_sweep/protocol.json), and [feature balance](feature_balance.csv). The 24 trained classifier models are excluded from Git. The [model catalog](../model_manifest.json) records hashes and paths relative to CHECKPOINTS_DIR. Checkpoints, other external inputs and generated caches are described and linked in [example_config.json](../../example_config.json).','']
    (root/'README.md').write_text(relocate_links('\n'.join(lines), root, config.REPO_ROOT/'training/full_nuisance_balance'))
    for name,rs in groups:print(name,*(f'{avg(rs,sp,k):.3f}' for sp in ['val','test'] for k in ['macro_f1','roc_auc']))

if __name__=='__main__':main()
