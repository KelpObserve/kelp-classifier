"""Extract campaign-matched field chips, predict final heads and rasterize pool 1.

Field labels never select checkpoints or thresholds. All image paths use config.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,torch,torch.nn.functional as F,h5py,yaml,rasterio
from rasterio.transform import Affine,rowcol,xy
from rasterio.warp import transform
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
from final_model_io import load_head,predict_embeddings
from patch_pool import centre_pool_slice
from encoder_geometry import ENCODER_SIZE, validate_geometry

CAMPAIGNS=[('june','06'),('july','07'),('september','09')]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def scores(y,p,t):
    y=np.asarray(y,int);pred=np.asarray(p)>=t;pr,re,f1,_=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(N=len(y),P=int(y.sum()),A=int((y==0).sum()),threshold=t,macro_f1=float(f1.mean()) if len(np.unique(y))==2 else None,present_precision=float(pr[1]),present_recall=float(re[1]),absent_precision=float(pr[0]),absent_recall=float(re[0]) if (y==0).any() else None,tp=int(tp),fn=int(fn),fp=int(fp),tn=int(tn))

def write_tif(path,a,tf,crs):
    with rasterio.open(path,'w',driver='GTiff',height=a.shape[0],width=a.shape[1],count=1,dtype='float32',transform=tf,crs=crs,nodata=np.nan,compress='deflate') as ds:ds.write(a.astype(np.float32),1)

def main():
    root=config.get_path('S2_CAMPAIGNS_DIR');out=config.get_path('SESIMBRA_EVALUATION_DIR');out.mkdir(parents=True,exist_ok=False)
    events=pd.read_csv(config.get_path('FIELD_OBSERVATIONS'));labels=pd.read_csv(root/'pixel_date_labels.csv');models={};checkpoints={};modelhash={}
    for name in ['small_mlp_pool5','mlp_pool1']:
        path=config.get_path('FINAL_CHECKPOINTS_DIR')/name/'model.pt';models[name],checkpoints[name]=load_head(path,'cuda');modelhash[name]=sha(path)
    gsds = {validate_geometry(ck['encoder'], checkpoint=True) for ck in checkpoints.values()}
    if len(gsds) != 1:
        raise ValueError('Final heads must share one encoder geometry')
    encoder_gsd = gsds.pop()
    ckpath=config.get_path('CLAY_CKPT');metadata=config.get_path('CLAY_META')
    for ck in checkpoints.values():assert ck['encoder']['checkpoint_sha256']==sha(ckpath) and ck['encoder']['metadata_sha256']==sha(metadata)
    from claymodel.module import ClayMAEModule
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('medium')
    encoder=ClayMAEModule.load_from_checkpoint(str(ckpath),model_size='large',shuffle=False,mask_ratio=0.,metadata_path=str(metadata),map_location='cuda').cuda().eval()
    stats=yaml.safe_load(metadata.read_text())['sentinel-2-l2a']['bands'];keys=['blue','green','red','rededge1','nir','swir16'];mean=torch.tensor([stats['mean'][k] for k in keys],device='cuda')[None,:,None,None];std=torch.tensor([stats['std'][k] for k in keys],device='cuda')[None,:,None,None];waves=torch.tensor([.493,.560,.665,.704,.842,1.610],device='cuda')
    def encode(raw):
        pix=F.interpolate(torch.tensor(raw.astype(np.float32),device='cuda'),size=(ENCODER_SIZE,ENCODER_SIZE),mode='bilinear',align_corners=False);pix=(pix-mean)/std;n=len(pix)
        return encoder.model.encoder(dict(pixels=pix,time=torch.zeros(n,4,device='cuda'),latlon=torch.zeros(n,4,device='cuda'),gsd=torch.tensor(encoder_gsd,device='cuda'),waves=waves,platform='sentinel-2-l2a'))[0]
    prediction_rows=[];audits=[];provenance=[];embedding_ids=[];embeddings={name:[] for name in models};allchips=[]
    with torch.inference_mode():
        for campaign,month in CAMPAIGNS:
            directory=out/campaign;directory.mkdir();scene=root/campaign/'scene.npz';meta=json.loads((root/campaign/'acquisition.json').read_text());assert sha(scene)==meta['sha256'];z=np.load(scene);a=z['cube'];tf=Affine(*z['transform'][:6]);crs=str(z['crs']);H,W=a.shape[1:];date=meta['acquisition_datetime'][:10]
            ev=events[events.date.str[5:7].eq(month)].copy();xs,ys=transform('EPSG:4326',crs,ev.lon.tolist(),ev.lat.tolist());rr,cc=rowcol(tf,xs,ys);ev['row']=rr;ev['col']=cc;ev['x']=xs;ev['y']=ys;ev['campaign']=campaign;ev['image_date']=date;ev.to_csv(directory/'events.csv',index=False)
            groups=ev.groupby(['row','col']).agg(label=('label','max'),n_events=('label','size')).reset_index().sort_values(['row','col']).reset_index(drop=True)
            expected=labels[labels.sensor.eq('S2')&labels.image_date.eq(date)].sort_values(['row','col']);assert np.array_equal(groups[['row','col','n_events']].to_numpy(),expected[['row','col','n_events']].to_numpy());assert np.array_equal(groups.label.to_numpy(),expected.label.eq('PRESENT').astype(int).to_numpy())
            chips=[]
            for r in groups.itertuples():
                row,col=int(r.row),int(r.col);top,left=row-64,col-64;assert top>=0 and left>=0 and top+128<=H and left+128<=W
                chip=a[:,top:top+128,left:left+128];chips.append(chip[:6]);allchips.append(chip);key=f'{campaign}_r{row}_c{col}';embedding_ids.append(key);scl=chip[6];central=scl[62:67,62:67];valid=(central==6)&(chip[0,62:67,62:67]>=10)&(chip[1,62:67,62:67]>=10)
                audits.append(dict(chip_id=key,campaign=campaign,image_date=date,row=row,col=col,chip_top=top,chip_left=left,label=int(r.label),n_events=int(r.n_events),h5_index=len(allchips)-1,sha256=hashlib.sha256(chip.tobytes()).hexdigest(),centre_scl=int(a[6,row,col]),central_valid_water=int(valid.sum()),central_cloud_shadow=int(np.isin(central,[3,8,9,10]).sum()),full_cloud_shadow_fraction=float(np.isin(scl,[3,8,9,10]).mean()),full_missing_fraction=float((scl==0).mean()),passes_training_central24=bool(valid.sum()>=24)))
            centred={name:[] for name in models}
            for start in range(0,len(chips),4):
                e=encode(np.stack(chips[start:start+4]));patch=e[:,1:].reshape(-1,28,28,1024)
                for name,head in models.items():
                    lo,hi=centre_pool_slice(checkpoints[name]['pool_size']);emb=torch.cat([e[:,0],patch[:,lo:hi,lo:hi].mean((1,2))],1);p,_=predict_embeddings(head,checkpoints[name],emb);centred[name].extend(p.cpu().tolist());embeddings[name].append(emb.cpu().numpy())
            anchors_y=list(range(0,H-127,64));anchors_x=list(range(0,W-127,64))
            if anchors_y[-1]!=H-128:anchors_y.append(H-128)
            if anchors_x[-1]!=W-128:anchors_x.append(W-128)
            anchors=[(y,x) for y in anchors_y for x in anchors_x];total=np.zeros((H,W),np.float32);counts=np.zeros((H,W),np.uint16);token_scores=[]
            for start in range(0,len(anchors),4):
                batch=anchors[start:start+4];e=encode(np.stack([a[:6,y:y+128,x:x+128] for y,x in batch]));n=len(batch);patch=e[:,1:].reshape(n,28,28,1024);cls=e[:,0,None,None,:].expand(-1,28,28,-1);emb=torch.cat([cls,patch],-1).reshape(-1,2048);prob,_=predict_embeddings(models['mlp_pool1'],checkpoints['mlp_pool1'],emb);grids=prob.reshape(n,28,28);token_scores.append(grids.cpu().numpy());native=F.interpolate(grids[:,None],size=(128,128),mode='bilinear',align_corners=False)[:,0].cpu().numpy()
                for i,(y,x) in enumerate(batch):total[y:y+128,x:x+128]+=native[i];counts[y:y+128,x:x+128]+=1
            assert (counts>0).all();raster=total/counts;valid=np.isin(a[6],[6,7]);masked=np.where(valid,raster,np.nan);write_tif(directory/'pool1_probability_raw.tif',raster,tf,crs);write_tif(directory/'pool1_probability.tif',masked,tf,crs);write_tif(directory/'scl.tif',a[6],tf,crs);np.savez_compressed(directory/'raster_tokens.npz',anchors=np.array(anchors),probabilities=np.concatenate(token_scores),coverage=counts)
            for i,r in enumerate(groups.itertuples()):
                key=f'{campaign}_r{int(r.row)}_c{int(r.col)}';x,y=xy(tf,int(r.row),int(r.col))
                for name in models:
                    prediction_rows.append(dict(chip_id=key,campaign=campaign,image_date=date,row=int(r.row),col=int(r.col),x=x,y=y,label=int(r.label),n_events=int(r.n_events),model=name,method='centred_chip',probability=centred[name][i],threshold=checkpoints[name]['threshold']))
                assert np.isfinite(masked[int(r.row),int(r.col)])
                prediction_rows.append(dict(chip_id=key,campaign=campaign,image_date=date,row=int(r.row),col=int(r.col),x=x,y=y,label=int(r.label),n_events=int(r.n_events),model='mlp_pool1',method='raster',probability=float(masked[int(r.row),int(r.col)]),threshold=checkpoints['mlp_pool1']['threshold']))
            provenance.append(dict(campaign=campaign,scene_id=meta['scene_id'],image_date=date,scene_sha256=sha(scene),events=len(ev),pixels=len(groups),P=int(groups.label.sum()),A=int(groups.label.eq(0).sum()),raster_windows=len(anchors),raster_sha256=sha(directory/'pool1_probability.tif')));print(campaign,'centred pixels',len(groups),'raster windows',len(anchors),flush=True)
    with h5py.File(out/'observation_chips.h5','w') as h:h.create_dataset('chips',data=np.stack(allchips),compression='gzip');h.attrs['bands']='blue,green,red,rededge1,nir,swir16,scl'
    pd.DataFrame(audits).to_csv(out/'chip_manifest.csv',index=False);np.savez_compressed(out/'embeddings.npz',chip_ids=np.array(embedding_ids),**{name:np.concatenate(v) for name,v in embeddings.items()})
    predictions=pd.DataFrame(prediction_rows);predictions['predicted_label']=(predictions.probability>=predictions.threshold).astype(int);predictions.to_csv(out/'predictions.csv',index=False);results=[]
    for (model,method),group in predictions.groupby(['model','method']):
        for campaign in ['june','july','september','all']:
            q=group if campaign=='all' else group[group.campaign.eq(campaign)];results.append(dict(model=model,method=method,campaign=campaign,**scores(q.label,q.probability,float(q.threshold.iloc[0]))))
    pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)
    (out/'protocol.json').write_text(json.dumps(dict(encoder_gsd_m=encoder_gsd,campaigns=provenance,model_sha256=modelhash,field_observations_sha256=sha(config.get_path('FIELD_OBSERVATIONS')),pixel_labels_sha256=sha(root/'pixel_date_labels.csv'),script_sha256=sha(__file__),threshold_source='Saved training-validation thresholds; no field tuning',centred_method='Containing native pixel at chip64,64; full128->224 Clay with zero time/location; CLS plus training pool slice',raster_method='Full-chip windows stride64; CLS plus each of 28x28 patch tokens, bilinear28->128 probability interpolation, equal overlap averaging. No depth clipping. Display/evaluation mask keeps SCL6 water and7 unclassified; other SCL classes masked. Raw raster also saved.',qc_policy='Audit central and full-chip flags; retain all observed pixels to avoid label-dependent filtering. No coordinate snapping or padding.',labels='Any PRESENT event makes its campaign pixel PRESENT; ABSENT requires all events in pixel absent. Pixel-date is evaluation unit.',single_class='June has no ABSENT pixel labels; Macro F1 and ABSENT recall omitted for June.',limitations='Only5 ABSENT pixel-dates across campaigns; repeated locations and overlapping windows are not independent. Scores are model outputs, not established calibrated probabilities. Dense raster deployment differs from centre-supervised fitting.'),indent=2));print(pd.DataFrame(results).to_string(index=False),flush=True)

if __name__=='__main__':main()
