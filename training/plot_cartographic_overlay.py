"""Plot predictions/observations over OSM land and EMODnet depth, without imagery.

The coastline is vector geometry; observations retain their geographic coordinates.
Bathymetry is ~115 m source data resampled to the prediction grid, not fine-scale
survey depth. No-data predictions are not used to infer land or water.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform, transform_geom
from plot_sesimbra_map import load_ccmar, add_scale_bar


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raster', required=True)
    ap.add_argument('--bathy', required=True)
    ap.add_argument('--land', required=True)
    ap.add_argument('--ccmar', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--pool-label', default='pool=1')
    ap.add_argument('--observations-only', action='store_true')
    ap.add_argument('--zoom-around', nargs=3, type=float, metavar=('LAT','LON','RADIUS_M'))
    args = ap.parse_args()
    with rasterio.open(args.raster) as src:
        prob = src.read(1, masked=True).filled(np.nan)
        crs, tf, bounds = src.crs, src.transform, src.bounds
    with rasterio.open(args.bathy) as src:
        if src.crs != crs or src.transform != tf or src.shape != prob.shape:
            raise ValueError('Bathymetry must be aligned to the prediction grid')
        depth = -src.read(1, masked=True).filled(np.nan)
    geo = json.loads(Path(args.land).read_text())
    land = [transform_geom('EPSG:4326', crs, f['geometry']) for f in geo['features']]
    land_mask = geometry_mask(land, out_shape=prob.shape, transform=tf, invert=True)
    depth[land_mask | (depth < 0)] = np.nan
    quads = load_ccmar(args.ccmar)
    xs, ys = transform('EPSG:4326', crs, [q['lon'] for q in quads], [q['lat'] for q in quads])
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    view = extent
    if args.zoom_around:
        lat, lon, radius = args.zoom_around
        xx, yy = transform('EPSG:4326', crs, [lon], [lat])
        view = (xx[0]-radius, xx[0]+radius, yy[0]-radius, yy[0]+radius)
    x = tf.c + (np.arange(prob.shape[1])+.5)*tf.a
    y = tf.f + (np.arange(prob.shape[0])+.5)*tf.e
    variants = ['observations_map'] if args.observations_only else ['over_map', 'with_ccmar']
    for variant in variants:
        fig, ax = plt.subplots(figsize=(8,8) if args.zoom_around else (12,7))
        fig.subplots_adjust(left=.09, right=.87, bottom=.16, top=.88)
        ax.set_facecolor('#dfedf2')
        bathy = ax.imshow(depth, extent=extent, cmap='Blues', vmin=0, vmax=30 if args.zoom_around else 130,
                         alpha=.55, interpolation='bilinear', zorder=1)
        if not args.observations_only:
            im = ax.imshow(prob, extent=extent, cmap='Reds', vmin=0, vmax=1,
                           alpha=np.where(np.isfinite(prob), .65*np.clip(prob,0,1), 0),
                           interpolation='nearest', zorder=2)
            fig.colorbar(im, ax=ax, fraction=.035, pad=.02, label='P(kelp)')
        else:
            fig.colorbar(bathy, ax=ax, fraction=.035, pad=.02, label='Depth below sea level (m)')
        cs = ax.contour(x, y, depth, levels=[5,10,20,30,50,75,100],
                        colors='#526c80', linewidths=.65, alpha=.9, zorder=3)
        ax.clabel(cs, fmt=lambda v:f'{v:g} m', fontsize=7)
        for g in land:
            polygons = [g['coordinates']] if g['type']=='Polygon' else g['coordinates']
            for p in polygons:
                ax.add_patch(PolygonPatch(p[0], facecolor='#e6e2d6', edgecolor='#686b60', lw=.7, zorder=4))
        if variant != 'over_map':
            # No raster-cell aggregation or snapping: retain each distinct coordinate.
            for present in [True,False]:
                idx = [i for i,q in enumerate(quads) if q['has_kelp']==present]
                ax.scatter(np.array(xs)[idx], np.array(ys)[idx], s=22,
                           facecolors='#13965a' if present else 'white',
                           edgecolors='#13432d' if present else '#ba2543', linewidths=.8,
                           zorder=6, label='Kelp observed' if present else 'Empty quadrat')
            ax.legend(loc='lower right', fontsize=8, framealpha=.95)
        ax.set(xlim=view[:2], ylim=view[2:], aspect='equal',
               xlabel=f'Easting (m, {crs})', ylabel='Northing (m)')
        width=view[1]-view[0]; height=view[3]-view[2]
        add_scale_bar(ax, view[0]+width*.04, view[2]+height*.06,
                      length_m=50 if width<300 else 100 if width<1000 else 1000,
                      height_m=height*.008)
        ax.text(.97,.95,'N ↑',transform=ax.transAxes,ha='right',fontsize=12,zorder=7)
        title = 'CCMar quadrats — geographic coordinates' if args.observations_only else f'S2 submerged head {args.pool_label} — P(kelp)'
        ax.set_title('Arrábida | '+title+'\nVector coastline, land/water and depth contours',fontsize=13)
        fig.text(.09,.055,'Coastline © OpenStreetMap contributors · openstreetmap.org/copyright (ODbL)\n'
                 'Bathymetry: EMODnet, ~115 m source grid (resampled); contours are approximate.',fontsize=8,color='#46545d')
        out=Path(f'{args.out_prefix}_{variant}.png'); out.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(out,dpi=140); plt.close(fig); print(out)


if __name__ == '__main__':
    main()
