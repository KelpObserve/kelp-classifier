import csv,json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config
from collections import defaultdict,Counter
import rasterio
from rasterio.warp import transform
from rasterio.transform import Affine,rowcol
root=config.get_path('S2_CAMPAIGNS_DIR')
events=list(csv.DictReader((root/'field_observations.csv').open()))
summary=[]; pixels=[]
for sensor,month,date,source in [('S2','06','2026-06-02','june'),('S2','07','2026-07-09','july'),('S2','09','2026-09-03','september'),('PNEO','07','2026-07-08','000598128'),('PNEO','09','2026-09-03','000618890')]:
 if sensor=='S2':
  meta=json.loads((root/source/'acquisition.json').read_text());tf=Affine(*meta['transform'][:6]);crs=meta['crs'];height,width=meta['shape'][1:];ds=None
 else:
  path=next(config.get_path('PNEO_DIR').glob(source+'*/IMG_01_PNEO3_MS-FS/*RGB*.JP2'))
  ds=rasterio.open(path);tf=ds.transform;crs=ds.crs;height,width=ds.shape
 subset=[e for e in events if e['date'][5:7]==month]; groups=defaultdict(list);outside=0;nodata=0
 for e in subset:
  xs,ys=transform('EPSG:4326',crs,[float(e['lon'])],[float(e['lat'])]);r,c=rowcol(tf,xs[0],ys[0]);r,c=int(r),int(c)
  if not (0<=r<height and 0<=c<width):outside+=1;continue
  if ds and not ds.read_masks(1,window=rasterio.windows.Window(c,r,1,1))[0,0]:nodata+=1
  groups[r,c].append(e)
 for (r,c),ee in groups.items():pixels.append(dict(sensor=sensor,image_date=date,row=r,col=c,label='PRESENT' if any(int(e['label']) for e in ee) else 'ABSENT',n_events=len(ee)))
 counts=Counter(p['label'] for p in pixels if p['sensor']==sensor and p['image_date']==date)
 summary.append(dict(sensor=sensor,image_date=date,events=len(subset),pixels=len(groups),counts=dict(counts),outside=outside,nodata_events=nodata))
 if ds:ds.close()
print(json.dumps(summary,indent=2))
with (root/'pixel_date_labels.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(pixels[0]));w.writeheader();w.writerows(pixels)
(root/'pixel_date_counts.json').write_text(json.dumps(summary,indent=2)+'\n')
