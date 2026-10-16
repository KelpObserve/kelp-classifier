"""Plot campaign pool-1 maps and document fixed-threshold field evaluation."""
from pathlib import Path
import csv,json,sys
import numpy as np,rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
from report_links import relocate_links

def display_date(value):
    year, month, day = value.split('-')
    return f'{int(day)}.{month}.{year}'

def read(p):
    with p.open() as f:return list(csv.DictReader(f))

def main():
    root=config.get_path('SESIMBRA_EVALUATION_DIR');metrics=read(root/'metrics.csv');audit=read(root/'chip_manifest.csv');protocol=json.loads((root/'protocol.json').read_text())
    thresholds = {r['model']: float(r['threshold']) for r in metrics if r['campaign']=='all' and r['method']=='centred_chip'}
    pool1_threshold = thresholds['mlp_pool1']
    pool5_threshold = thresholds['small_mlp_pool5']
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained');image=None
    for ax,scene in zip(axes,protocol['campaigns']):
        name=scene['campaign'];events=read(root/name/'events.csv')
        with rasterio.open(root/name/'pool1_probability.tif') as ds:a=ds.read(1);bounds=ds.bounds
        with rasterio.open(root/name/'scl.tif') as ds:scl=ds.read(1)
        extent=[bounds.left,bounds.right,bounds.bottom,bounds.top];ax.set_facecolor('#edf2f5');land=np.ma.masked_where(~np.isin(scl,[4,5,11]),np.ones(scl.shape));ax.imshow(land,extent=extent,origin='upper',cmap='Greys',vmin=0,vmax=2,interpolation='nearest');image=ax.imshow(a,extent=extent,origin='upper',cmap='viridis',vmin=0,vmax=1,interpolation='nearest')
        xs=np.array([float(r['x']) for r in events]);ys=np.array([float(r['y']) for r in events]);labels=np.array([int(r['label']) for r in events])
        ax.scatter(xs[labels==1],ys[labels==1],s=45,facecolors='white',edgecolors='#12312a',linewidths=1,zorder=5);ax.scatter(xs[labels==0],ys[labels==0],s=55,marker='x',color='#e73556',linewidths=2,zorder=6)
        cx=(xs.min()+xs.max())/2;cy=(ys.min()+ys.max())/2;half=max(np.ptp(xs),np.ptp(ys))/2+45;ax.set_xlim(cx-half,cx+half);ax.set_ylim(cy-half,cy+half);ax.set_aspect('equal');ax.ticklabel_format(useOffset=False,style='plain');ax.tick_params(labelsize=7);ax.set_xlabel('UTM easting (m)');ax.set_title(name.capitalize()+' · '+display_date(scene['image_date']));ax.plot([cx-half+12,cx-half+62],[cy-half+12]*2,'k-',lw=3);ax.text(cx-half+37,cy-half+16,'50 m',ha='center',fontsize=8)
    axes[0].set_ylabel('UTM northing (m)');fig.colorbar(image,ax=axes,shrink=.75,label=f'Pool-1 model score (threshold {pool1_threshold:.2f})');fig.legend(handles=[Line2D([],[],marker='o',ls='',markerfacecolor='white',markeredgecolor='#12312a',label='Observed PRESENT'),Line2D([],[],marker='x',ls='',color='#e73556',label='Observed ABSENT')],loc='outside lower center',ncol=2);fig.suptitle('Sesimbra campaign rasters with the 30 field quadrats per campaign');fig.savefig(root/'pool1_observation_maps.png',dpi=180);plt.close(fig)
    names={'mlp_pool1':'Clay + MLP, pool 1','small_mlp_pool5':'Clay + small MLP, pool 5'}
    lines=['# Sesimbra campaign observations prediction','',f'Both final S2 heads are applied to the image matching each field campaign. Model checkpoints and thresholds are fixed from training validation (pool 1: {pool1_threshold:.2f}; small MLP pool 5: {pool5_threshold:.2f}). Field labels are not used for tuning.','','## Inputs and chips','','| Campaign | Image date | Field events | PRESENT pixels | ABSENT pixels |','| --- | --- | ---: | ---: | ---: |']
    for s in protocol['campaigns']:lines.append(f'| {s["campaign"]} | '+display_date(s['image_date'])+f' | {s["events"]} | {s["P"]} | {s["A"]} |')
    centred=[r for r in metrics if r['method']=='centred_chip' and r['campaign']=='all']
    field_summary=' '.join(f"{names[r['model']]} has PRESENT recall {float(r['present_recall']):.3f}, ABSENT recall {float(r['absent_recall']):.3f} and Macro F1 {float(r['macro_f1']):.3f}." for r in centred)
    raster=next(r for r in metrics if r['method']=='raster' and r['campaign']=='all')
    raster_summary=f"Pool-1 rasterization has ABSENT recall {float(raster['absent_recall']):.3f} and overall Macro F1 {float(raster['macro_f1']):.3f}."
    passed=sum(r['passes_training_central24']=='True' for r in audit)
    lines+=['','The 90 events become **52 distinct pixel-date labels: 47 PRESENT and 5 ABSENT**. Within each campaign, a pixel is PRESENT if any quadrat in it is PRESENT; ABSENT requires all quadrats to be empty. June has no ABSENT pixel labels after this aggregation.','','[The evaluation script](../evaluate_sesimbra_campaigns.py) extracts one native 128×128 chip per pixel-date from the prepared campaign image, placing the containing pixel at chip row/column 64. Six spectral bands enter the frozen Clay encoder after the same 128→224 resize and normalization used for training. Encoder grid spacing is 128 × 10 / 224 ≈ 5.714 m, preserving the native 1,280 m footprint. There is no snapping or padding. Chips and embeddings are external generated files described in [example_config.json](../../example_config.json); [chip_manifest.csv](chip_manifest.csv) preserves extraction indices and hashes.',f'',f'{passed} of 52 chips pass the training central-water check (at least 24 of 25 central pixels classified as water with valid blue/green values). All 52 are retained for transparent field evaluation; the [quality audit](chip_manifest.csv) records flags and full-chip contamination fractions. Passing a point-level cloud check does not guarantee that an entire 1.28 km chip is uncontaminated.','','## Observation-centred predictions','','| Model | PRESENT precision | PRESENT recall | ABSENT precision | ABSENT recall | Macro F1 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in metrics:
        if row['method']=='centred_chip' and row['campaign']=='all':lines.append('| '+names[row['model']]+' | '+' | '.join(f'{float(row[k]):.3f}' for k in ['present_precision','present_recall','absent_precision','absent_recall','macro_f1'])+' |')
    lines+=['',field_summary+' The always-PRESENT Macro F1 reference is 0.475. Thresholds are fixed from training validation and are not adjusted to improve field scores.','','## Pool-1 rasterization','','For each campaign, 40 overlapping 128×128 windows cover the 333×570 grid at stride 64. The pool-1 head receives CLS plus each patch token on the 28×28 Clay grid. Scores are bilinearly resized to 128×128 and averaged where windows overlap. The output grid is 10 m; this does not establish 10 m independent resolving power. Moving patch predictions and overlap averaging differ from a chip centred on each observation.','','The published probability GeoTIFF keeps SCL water (6) and unclassified (7) pixels, including the September unclassified observation locations. Other SCL classes are masked; the raw unmasked raster is also saved. No depth clipping or field-derived mask is applied. Grey areas in the map indicate SCL land classes; other masked areas show the pale background. Scores are not established calibrated probabilities.','','![Pool-1 raster predictions with field observations](pool1_observation_maps.png)','','Markers show the 30 actual quadrat positions in each campaign. Several markers can share one S2 pixel. Evaluation samples the containing pixel from the saved raster; it uses the same 52 pixel-date labels as the centred-chip comparison.','','| Campaign | N | PRESENT | ABSENT | Pool-1 centred Macro F1 | Pool-1 raster Macro F1 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for campaign in ['june','july','september','all']:
        rows={r['method']:r for r in metrics if r['campaign']==campaign and r['model']=='mlp_pool1'};r=rows['raster'];values=[f'{float(rows[k]["macro_f1"]):.3f}' if rows[k]['macro_f1'] else '—' for k in ['centred_chip','raster']];lines.append(f'| {campaign} | {r["N"]} | {r["P"]} | {r["A"]} | '+ ' | '.join(values)+' |')
    lines+=['',raster_summary+' The continuous scores can differ between methods even where binary decisions agree. June Macro F1 is omitted because its aggregated labels contain only one class. Only five ABSENT pixel-dates and repeated nearby locations limit inference about wider performance.','','## Reproduction and evidence','','Configure the inputs described in [example_config.json](../../example_config.json) and use a fresh evaluation output directory:','','```sh','python training/evaluate_sesimbra_campaigns.py','python training/verify_sesimbra_campaigns.py','python training/report_sesimbra_campaigns.py','```','','[Predictions](predictions.csv), [metrics](metrics.csv), [protocol](protocol.json), and [map/report script](../report_sesimbra_campaigns.py). Raster files: [June](june/pool1_probability.tif), [July](july/pool1_probability.tif), [September](september/pool1_probability.tif). The [verification script](../verify_sesimbra_campaigns.py) checks exact scene crops, saved-head replay, reconstruction from patch-score grids, masks and metrics.','']
    lines[2:2] = ['Generated by [report_sesimbra_campaigns.py](../report_sesimbra_campaigns.py). Edit the maintained [PNEO narrative](../report_templates/pneo_september.md) for that section; regenerate this README after edits.', '']
    s2 = '\n'.join(lines)
    s2 = s2.replace('\nBoth final S2 heads', '\nThis report covers Sentinel-2 predictions for June, July and September and the\nseparate September Pléiades Neo analysis.\n\n## Sentinel-2 campaign predictions\n\nBoth final S2 heads', 1)
    s2 = s2.replace('\n## ', '\n### ').replace('\n### Sentinel-2 campaign predictions', '\n## Sentinel-2 campaign predictions')
    # Maintained prose is an input, independent of any previously generated report.
    # The template and canonical report directories are siblings, so external
    # relative links have the same base depth before relocate_links is applied.
    source = Path(__file__).resolve().parent / 'report_templates' / 'pneo_september.md'
    pneo = source.read_text()
    if not pneo.startswith('## September Pléiades Neo analysis\n'):
        raise ValueError('PNEO narrative must start with its section heading')
    (root/'README.md').write_text(relocate_links(s2.rstrip() + '\n\n' + pneo, root, config.REPO_ROOT/'training/sesimbra_evaluation'))


if __name__=='__main__':main()
