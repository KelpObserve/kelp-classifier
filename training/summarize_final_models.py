"""Render final-model train/validation metrics and nuisance-balance evidence."""
from pathlib import Path
import sys,csv,json
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
    root=config.get_path('FINAL_MODELS_DIR');rows=read(root/'selected.csv');metrics=read(root/'metrics.csv');balance=read(root/'feature_balance.csv');manifest=json.loads((root/'model_manifest.json').read_text());features=['depth_m','nuisance_cloud','nuisance_missing','nuisance_saturated_defective']
    counts={sp:{lab:sum(r['split']==sp and r['label']==lab for r in rows) for lab in ['0','1']} for sp in ['train','val']}
    solver=json.loads((root/'split_protocol.json').read_text())['solver'];solver=solver.get('original_solver',solver)
    gap=solver.get('relative_gap')
    solver_note=f' The partition optimizer reports a relative objective gap of {gap:.1%}; the measured balance is not proof of global optimality.' if gap is not None else ''
    fig,axs=plt.subplots(2,4,figsize=(14,6))
    for i,sp in enumerate(['train','val']):
        for j,feature in enumerate(features):
            ax=axs[i,j]
            for label,name,color in [('1','PRESENT','#197b92'),('0','ABSENT','#c56027')]:
                values=np.sort([float(r[feature]) for r in rows if r['split']==sp and r['label']==label]);ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=name,color=color)
            ax.set_title(f'{sp}: {feature}',fontsize=9);ax.grid(alpha=.2);ax.set_ylim(0,1.02)
    axs[0,0].legend();fig.tight_layout();fig.savefig(root/'balance.png',dpi=160);plt.close(fig)
    lines=['# Final reusable Clay models','',f'Two models are fitted on the complete {len(rows)}-chip selected union: **small MLP, pool 5** and **original MLP, pool 1**. All {sum(r["label"]=="1" for r in rows)} PRESENT and {sum(r["label"]=="0" for r in rows)} ABSENT chips are retained. Both models use the same grouped train/validation split.','','## Partitions','','| Partition | PRESENT | ABSENT | Total |','| --- | ---: | ---: | ---: |',f'| Train | {counts["train"]["1"]} | {counts["train"]["0"]} | {sum(counts["train"].values())} |',f'| Validation | {counts["val"]["1"]} | {counts["val"]["0"]} | {sum(counts["val"].values())} |','','Observation IDs, identical arrays and connected locations within 50 m stay together. The split balances depth and whole-chip cloud/shadow/cirrus, missing and defective fractions, including both class differences and validation representativeness. Spectral appearance, acquisition and processing variables do not enter its objective. All chips are retained, so the class ratio follows the selected union rather than exactly 2:1. Training replays the included split exactly. Independent land and glint measurements remain unavailable.'+solver_note,'','Validation selects checkpoints and thresholds; it is not an independent test. The union has already been explored in model comparisons, so these are fitting diagnostics, not new estimates of generalization. The [three-seed comparison](../full_nuisance_balance/README.md) remains the classifier comparison.','','## Models and selection','','Clay v1.5 remains frozen and encodes the full chip with derived GSD 128 × 10 / 224 ≈ 5.714 m. The heads receive CLS concatenated with centrally pooled patch tokens. The original MLP is 2048 → 512 → 128 → 1; the small MLP is 2048 → 32 → 1. Both use seed 42, focal loss, Adam learning rate 0.001, batch size 64, at most 100 epochs and patience 25. Original-MLP weight decay is fixed at 0.0001; small-MLP weight decay is selected from 0.1, 0.01 and 0.0001 on validation Macro F1 at 0.5. Thresholds are then selected on validation over 0.05–0.95 in steps of 0.01.','','| Model | Best epoch | Weight decay | Decision threshold |','| --- | ---: | ---: | ---: |']
    for item in manifest['models']:lines.append(f'| {item["model"]} | {item["best_epoch"]} | {item["weight_decay"]} | {item["threshold"]:.2f} |')
    lines+=['','| Model | Split | PRESENT precision | PRESENT recall | ABSENT precision | ABSENT recall | Macro F1 |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in metrics:
        if row['threshold_mode']=='validation_tuned':lines.append('| '+row['model']+' | '+row['split']+' | '+' | '.join(f'{float(row[k]):.3f}' for k in ['present_precision','present_recall','absent_precision','absent_recall','macro_f1'])+' |')
    lines+=['','## Balance','','| Partition | Feature | Class KS distance |','| --- | --- | ---: |']
    for row in balance:lines.append(f'| {row["split"]} | {row["feature"]} | {float(row["ks"]):.3f} |')
    lines+=['','![Final train/validation balance](balance.png)','','## Reproduction and reuse','','[Selected chips](selected.csv), [split](split.json), [split protocol](split_protocol.json), [metrics](metrics.csv), [model catalog](model_manifest.json) identify the reusable models. The two final checkpoints are included in Git: [small_mlp_pool5/model.pt](small_mlp_pool5/model.pt) and [mlp_pool1/model.pt](mlp_pool1/model.pt). Their SHA-256 checksums are recorded in the model catalog. The pretrained backbone and embedding cache remain external inputs described in [example_config.json](../../example_config.json).','','Configure a fresh FINAL_MODELS_DIR and the full-chip embedding cache configured as FULL_NUISANCE_EMBEDDINGS, then run:','','```sh','python training/train_final_models.py','python training/verify_final_models.py','python training/summarize_final_models.py','```','','For reuse, [final_model_io.py](../final_model_io.py) provides `load_head(path, device)` and `predict_embeddings(model, checkpoint, embeddings)`. Supply N×2048 embeddings from the recorded frozen encoder, normalization and matching central pool. The returned labels use the saved validation-selected threshold. Each checkpoint includes architecture, pool, threshold, encoder settings/hashes and the training-manifest/split hashes. `FINAL_CHECKPOINTS_DIR` resolves the relative model paths in the catalog.','']
    (root/'README.md').write_text(relocate_links('\n'.join(lines), root, config.REPO_ROOT/'training/final_models'))
    for row in metrics:
        if row['threshold_mode']=='validation_tuned':print(row['model'],row['split'],'Macro F1',row['macro_f1'])

if __name__=='__main__':main()
