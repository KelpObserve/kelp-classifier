"""Render matched field observations from the included coastline and elevation assets."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as PlotPath
import numpy as np
import rasterio
from rasterio.warp import transform, transform_geom

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

ASSETS = Path(__file__).resolve().parent


def render(output):
    with config.get_path('FIELD_OBSERVATIONS').open() as stream:
        rows = list(csv.DictReader(stream))
    with rasterio.open(ASSETS / 'arrabida_elevation_emodnet.tif') as raster:
        depth = -raster.read(1, masked=True)
        crs, bounds, affine = raster.crs, raster.bounds, raster.transform
    xs, ys = map(np.array, transform('EPSG:4326', crs,
                                   [float(r['lon']) for r in rows],
                                   [float(r['lat']) for r in rows]))
    fig, ax = plt.subplots(figsize=(10, 10), layout='constrained')
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    image = ax.imshow(depth, extent=extent, origin='upper', cmap='Blues',
                      vmin=0, vmax=30, interpolation='nearest', alpha=.65)
    grid_x = affine.c + (np.arange(depth.shape[1]) + .5) * affine.a
    grid_y = affine.f + (np.arange(depth.shape[0]) + .5) * affine.e
    ax.contour(grid_x, grid_y, depth, levels=[5, 10, 15, 20, 25, 30],
               colors='#607d8b', linewidths=.7)
    land = json.loads((ASSETS / 'arrabida_land_osm.geojson').read_text())
    for feature in land['features']:
        geometry = transform_geom('EPSG:4326', crs, feature['geometry'])
        polygons = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
        for polygon in polygons:
            vertices, codes = [], []
            for ring in polygon:
                vertices.extend(ring)
                codes.extend([PlotPath.MOVETO] + [PlotPath.LINETO] * (len(ring) - 2) + [PlotPath.CLOSEPOLY])
            ax.add_patch(PathPatch(PlotPath(vertices, codes), facecolor='#e6e2d6',
                                   edgecolor='#68746b', linewidth=.8))
    for month, title, marker in [('06', 'June', 's'), ('07', 'July', 'o'), ('09', 'September', '^')]:
        for label, description, face, edge in [(1, 'Kelp observed', '#159963', '#174d38'),
                                                (0, 'Empty quadrat', 'white', '#dc2349')]:
            selected = np.array([r['date'][5:7] == month and int(r['label']) == label for r in rows])
            ax.scatter(xs[selected], ys[selected], marker=marker, s=50,
                       facecolors=face, edgecolors=edge, linewidths=1,
                       label=f'{title} — {description} ({selected.sum()})', zorder=5)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(np.ptp(xs), np.ptp(ys)) / 2 + 20
    ax.set(xlim=(cx-half, cx+half), ylim=(cy-half, cy+half), aspect='equal',
           xlabel='Easting (m, EPSG:32629)', ylabel='Northing (m)',
           title='Arrábida | CCMAR field observations\nJune, July and September 2026')
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.tick_params(labelsize=8)
    ax.text(.96, .96, 'N ↑', transform=ax.transAxes, ha='right', va='top')
    ax.plot([cx-half+8, cx-half+58], [cy-half+12]*2, color='#20313c', linewidth=3)
    ax.text(cx-half+33, cy-half+15, '50 m', ha='center', fontsize=9)
    ax.legend(loc='lower right', fontsize=8)
    fig.colorbar(image, ax=ax, shrink=.75, label='Depth below sea level (m)')
    fig.supxlabel('Coastline © OpenStreetMap contributors (ODbL)\n'
                  'EMODnet bathymetry: approximately 115 m source grid; contours are approximate', fontsize=9)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ASSETS / 'sesimbra_quadrats_observations_map.png')
    render(parser.parse_args().output)
