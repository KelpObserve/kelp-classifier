"""Verify field chips, frozen-head predictions and pool-1 raster reconstruction."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,h5py,torch,torch.nn.functional as F,rasterio
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
from final_model_io import load_head,predict_embeddings
from evaluate_sesimbra_campaigns import scores

def main():
    root=config.get_path('SESIMBRA_EVALUATION_DIR');source=config.get_path('S2_CAMPAIGNS_DIR');audit=pd.read_csv(root/'chip_manifest.csv');p=pd.read_csv(root/'predictions.csv');m=pd.read_csv(root/'metrics.csv');cache=np.load(root/'embeddings.npz');idx={str(v):i for i,v in enumerate(cache['chip_ids'])};torch.set_float32_matmul_precision('medium');torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    assert len(audit)==52 and audit.label.sum()==47 and audit.n_events.sum()==90
    with h5py.File(root/'observation_chips.h5','r') as h:
        for campaign,q in audit.groupby('campaign'):
            scene=np.load(source/campaign/'scene.npz')['cube']
            for r in q.itertuples():
                chip=h['chips'][r.h5_index];assert hashlib.sha256(chip.tobytes()).hexdigest()==r.sha256;assert np.array_equal(chip,scene[:,r.chip_top:r.chip_top+128,r.chip_left:r.chip_left+128])
    maxdiff=0
    for name in ['small_mlp_pool5','mlp_pool1']:
        model,ck=load_head(config.get_path('FINAL_CHECKPOINTS_DIR')/name/'model.pt','cuda');q=p[p.model.eq(name)&p.method.eq('centred_chip')];xx=cache[name][[idx[v] for v in q.chip_id]];prob,_=predict_embeddings(model,ck,xx);prob=prob.cpu().numpy();maxdiff=max(maxdiff,float(np.max(np.abs(prob-q.probability))));assert np.allclose(prob,q.probability,atol=1e-6);assert np.allclose(q.threshold,ck['threshold'])
    rasterdiff=0
    for campaign in ['june','july','september']:
        z=np.load(root/campaign/'raster_tokens.npz');counts=z['coverage'];total=np.zeros(counts.shape,np.float32);recount=np.zeros(counts.shape,np.uint16)
        with torch.no_grad():native=F.interpolate(torch.tensor(z['probabilities'])[:,None],size=(128,128),mode='bilinear',align_corners=False)[:,0].numpy()
        for v,(y,x) in zip(native,z['anchors']):total[y:y+128,x:x+128]+=v;recount[y:y+128,x:x+128]+=1
        assert np.array_equal(counts,recount);raw=total/counts
        with rasterio.open(root/campaign/'pool1_probability_raw.tif') as ds:saved=ds.read(1);assert np.allclose(saved,raw,atol=1e-7);rasterdiff=max(rasterdiff,float(np.max(np.abs(saved-raw))))
        with rasterio.open(root/campaign/'pool1_probability.tif') as ds:masked=ds.read(1)
        scene=np.load(source/campaign/'scene.npz')['cube'];valid=np.isin(scene[6],[6,7]);assert np.array_equal(np.isfinite(masked),valid)
        for row in p[p.method.eq('raster')&p.campaign.eq(campaign)].itertuples():assert np.isclose(row.probability,masked[row.row,row.col])
    for row in m.itertuples():
        q=p[p.model.eq(row.model)&p.method.eq(row.method)]
        if row.campaign!='all':q=q[q.campaign.eq(row.campaign)]
        result=scores(q.label,q.probability,row.threshold)
        for key,v in result.items():
            if v is None:assert pd.isna(getattr(row,key))
            else:assert np.isclose(getattr(row,key),v),key
    (root/'audit.json').write_text(json.dumps(dict(chips_verified=52,field_events=90,present_pixel_dates=47,absent_pixel_dates=5,chip_arrays_exact_scene_crops=True,checkpoint_replay_maximum_difference=maxdiff,raster_reconstruction_maximum_difference=rasterdiff,raster_mask_verified=True,metrics_recomputed=True,field_threshold_tuning=False),indent=2));print('All field chips, saved-head predictions, rasters and metrics verified.')

if __name__=='__main__':main()
