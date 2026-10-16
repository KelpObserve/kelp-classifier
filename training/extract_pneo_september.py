"""Reproduce September PNEO spectral/Lyzenga features and frozen Clay embeddings.

Methods for the PNEO exploration; labels use the matched field table.
No licensed imagery or RGB-background figures are exported for redistribution.
"""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np,pandas as pd,rasterio,torch
from rasterio.warp import transform as warp_transform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
BAND_NAMES=['R','G','B','NIR','RE','DB']
LYZ_PAIRS=[('DB','B'),('DB','G'),('B','G'),('G','R'),('B','R'),('DB','R')]
KERNEL=5
CHIP_PX=224
GSD_M=1.2
def load_stack(pneo_dir: Path):
    rgb_path = next(pneo_dir.glob("IMG_*_RGB_*.JP2"))
    ned_path = next(pneo_dir.glob("IMG_*_NED_*.JP2"))
    with rasterio.open(rgb_path) as rgb, rasterio.open(ned_path) as ned:
        assert rgb.crs == ned.crs and rgb.transform == ned.transform
        stack = np.concatenate([rgb.read().astype(np.float32),
                                 ned.read().astype(np.float32)], axis=0)
        return stack, rgb.crs, rgb.transform

def water_mask(stack, nir_percentile=25):
    NIR = stack[BAND_NAMES.index("NIR")]
    thr = np.percentile(NIR[NIR > 0], nir_percentile)
    return (NIR < thr) & (NIR > 0)

def estimate_rsw(stack, mask, pct=5):
    rsw = np.zeros(len(BAND_NAMES), dtype=np.float64)
    for i in range(len(BAND_NAMES)):
        rsw[i] = float(np.percentile(stack[i][mask], pct))
    return rsw

def estimate_kij(X_i, X_j, mask, sample_size=200_000):
    xi = X_i[mask]; xj = X_j[mask]
    ok = np.isfinite(xi) & np.isfinite(xj)
    xi = xi[ok]; xj = xj[ok]
    if len(xi) > sample_size:
        idx = np.random.default_rng(0).choice(len(xi), size=sample_size, replace=False)
        xi = xi[idx]; xj = xj[idx]
    var_i = xi.var(ddof=1); var_j = xj.var(ddof=1)
    cov_ij = np.cov(xi, xj, ddof=1)[0, 1]
    num = 2 * cov_ij
    den = var_j - var_i + math.sqrt((var_j - var_i) ** 2 + 4 * cov_ij ** 2)
    return float(num / den) if den != 0 else 1.0

def compute_X(stack, rsw, eps=1e-3):
    X = np.empty_like(stack, dtype=np.float32)
    for i in range(len(BAND_NAMES)):
        diff = stack[i] - rsw[i]
        X[i] = np.log(np.maximum(diff, eps))
    return X

def sample_kernel(arr3d, transform, crs, lon, lat, kernel=KERNEL):
    xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
    x, y = xs[0], ys[0]
    row, col = rasterio.transform.rowcol(transform, x, y)
    half = kernel // 2
    H, W = arr3d.shape[1:]
    r0, r1 = max(0, row - half), min(H, row + half + 1)
    c0, c1 = max(0, col - half), min(W, col + half + 1)
    win = arr3d[:, r0:r1, c0:c1].astype(np.float64)
    return win.mean(axis=(1, 2)), win.std(axis=(1, 2)), win.max(axis=(1, 2)) - win.min(axis=(1, 2))

def compute_scene_stats(stack, nir_percentile=25):
    """Per-band mean/std over the water mask of this scene."""
    NIR = stack[3]   # channel index for NIR
    thr = np.percentile(NIR[NIR > 0], nir_percentile)
    water = (NIR < thr) & (NIR > 0)
    means = np.array([float(stack[i][water].mean()) for i in range(6)], dtype=np.float32)
    stds = np.array([float(stack[i][water].std()) for i in range(6)], dtype=np.float32)
    return means, stds

def extract_chip(stack, crs, transform, lon, lat, size=CHIP_PX):
    """Extract a size x size window centred on (lon, lat) in stack native CRS."""
    xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
    x, y = xs[0], ys[0]
    # convert to pixel indices (nearest)
    col = int(round((x - transform.c) / transform.a))
    row = int(round((y - transform.f) / transform.e))
    r0 = row - size // 2
    c0 = col - size // 2
    H, W = stack.shape[1:]
    if r0 < 0 or c0 < 0 or r0 + size > H or c0 + size > W:
        return None
    return stack[:, r0:r0 + size, c0:c0 + size].astype(np.float32)

def run_clay_batch(model, pixels, waves, gsd, latlons, weeks, device):
    """pixels: (N, 6, 224, 224) float32
       returns cls (N, 1024) and centre_patches (N, 7, 7, 1024)"""
    with torch.no_grad():
        n = pixels.shape[0]
        cls_all = []; patches_all = []
        BATCH = 8
        for i in range(0, n, BATCH):
            j = min(i + BATCH, n)
            px = torch.from_numpy(pixels[i:j]).to(device)
            latlon = torch.from_numpy(latlons[i:j]).to(device)
            time_vec = torch.from_numpy(weeks[i:j]).to(device)
            gsd_t = torch.tensor(gsd, device=device, dtype=torch.float32)
            waves_t = torch.tensor(waves, device=device, dtype=torch.float32)
            datacube = dict(
                pixels=px, time=time_vec, latlon=latlon,
                gsd=gsd_t, waves=waves_t, platform="pneo",
            )
            out = model.model.encoder(datacube)
            if isinstance(out, tuple):
                out = out[0]
            # out shape: (B, 1 + n_patches, D). CLS at index 0. Rest are patches.
            cls = out[:, 0, :].cpu().numpy()
            # Grid is 224/patch_size (Clay v1.5 large uses 8 -> 28x28 = 784 patches)
            n_patches_flat = out.shape[1] - 1
            grid = int(round(math.sqrt(n_patches_flat)))
            patches = out[:, 1:, :].cpu().numpy().reshape(-1, grid, grid, out.shape[-1])
            # Take the centre 7x7 patches
            c = grid // 2
            centre = patches[:, c - 3:c + 4, c - 3:c + 4, :]
            cls_all.append(cls); patches_all.append(centre)
        return np.concatenate(cls_all), np.concatenate(patches_all)
def main():
    out=config.get_path('PNEO_SEPTEMBER_INPUTS');out.mkdir(parents=True,exist_ok=False)
    field=pd.read_csv(config.get_path('FIELD_OBSERVATIONS'));q=field[field.date.eq('2026-09-02')].copy();assert len(q)==30 and q.label.sum()==20
    stack,crs,tf=load_stack(config.get_path('PNEO_SEPTEMBER_DIR'));wm=water_mask(stack);rsw=estimate_rsw(stack,wm);X=compute_X(stack,rsw);dii={}
    for a,b in LYZ_PAIRS:dii[f'DII_{a}_{b}']=X[BAND_NAMES.index(a)]-estimate_kij(X[BAND_NAMES.index(a)],X[BAND_NAMES.index(b)],wm)*X[BAND_NAMES.index(b)]
    rows=[];chips=[];latlon=[];time=[];means,std=compute_scene_stats(stack)
    for r in q.itertuples():
        rec=dict(date_iso=r.date,depth_m=r.gps_depth_m,transect=r.T,quadrat=r.Q,lat=r.lat,lon=r.lon,is_present=bool(r.label),label='PRESENT' if r.label else 'ABSENT');m,s,w=sample_kernel(stack,tf,crs,r.lon,r.lat)
        for i,b in enumerate(BAND_NAMES):rec.update({b+'_mean':m[i],b+'_std':s[i],b+'_range':w[i]})
        xs,ys=warp_transform('EPSG:4326',crs,[r.lon],[r.lat]);row,col=rasterio.transform.rowcol(tf,xs[0],ys[0]);assert row>=2 and col>=2
        for name,a in dii.items():rec[name+'_w5']=float(a[row-2:row+3,col-2:col+3].mean());rec[name+'_px']=float(a[row,col])
        rows.append(rec);chip=extract_chip(stack,crs,tf,r.lon,r.lat);assert chip is not None;chips.append((chip-means[:,None,None])/(std[:,None,None]+1e-6));la,lo=np.radians([r.lat,r.lon]);latlon.append([np.sin(la),np.cos(la),np.sin(lo),np.cos(lo)]);week=pd.Timestamp(r.date).isocalendar().week;time.append([np.sin(2*np.pi*week/52),np.cos(2*np.pi*week/52),0,0])
    f=pd.DataFrame(rows);f.to_csv(out/'features_per_quadrat.csv',index=False);f[['date_iso','depth_m','transect','quadrat','lat','lon','is_present']].to_csv(out/'meta.csv',index=False)
    from claymodel.module import ClayMAEModule
    model=ClayMAEModule.load_from_checkpoint(str(config.get_path('CLAY_CKPT')),model_size='large',shuffle=False,mask_ratio=0.,metadata_path=str(config.get_path('CLAY_META')),map_location='cuda').cuda().eval()
    cls,patch=run_clay_batch(model,np.stack(chips).astype(np.float32),[.660,.560,.480,.800,.725,.425],1.2,np.array(latlon,np.float32),np.array(time,np.float32),'cuda');np.savez_compressed(out/'identity_pneo_field.npz',cls=cls,centre_patches=patch)
    (out/'extraction.json').write_text(json.dumps(dict(scene='000618890',image_date='2026-09-03',observation_date='2026-09-02',source_resolution_m=float(tf.a),field_sha256=hashlib.sha256(config.get_path('FIELD_OBSERVATIONS').read_bytes()).hexdigest(),spectral_window=5,clay_input=224,clay_central_token=14,clay_normalization='Per-scene low-NIR water-mask mean/std',clay_metadata='Coordinate sine/cosine and observation week encodings, retaining PNEO exploration convention'),indent=2))

if __name__=='__main__':main()
