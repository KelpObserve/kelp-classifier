"""Complete-tile selection and fixed-grid reads used during chip acquisition."""
import numpy as np
import pystac
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.transform import Affine, from_origin
from rasterio.warp import transform, transform_bounds
from rasterio.windows import from_bounds
from chip_generation.chip_common import BANDS, IncompleteChipError, require_complete_window, read_chip


def source_crs(properties, asset=None):
    """Read STAC projection metadata, including PySTAC's normalized proj:code."""
    for metadata in (asset or {}, properties):
        if metadata.get('proj:code'):
            return rasterio.crs.CRS.from_user_input(metadata['proj:code'])
        if metadata.get('proj:epsg'):
            return rasterio.crs.CRS.from_epsg(metadata['proj:epsg'])
    raise ValueError('Source projection metadata is missing')


def complete(item, lon, lat, epsg):
    """Require complete source coverage of the requested projected footprint."""
    x, y = transform('EPSG:4326', f'EPSG:{epsg}', [lon], [lat])
    target_bounds = (x[0]-640, y[0]-640, x[0]+640, y[0]+640)
    for band in BANDS:
        asset = item['assets'][band]
        crs = source_crs(item['properties'], asset)
        bounds = transform_bounds(f'EPSG:{epsg}', crs, *target_bounds, densify_pts=21)
        window = from_bounds(*bounds, Affine(*asset['proj:transform'][:6]))
        height, width = asset['proj:shape']
        try:
            require_complete_window(window, width, height)
        except IncompleteChipError:
            return False
    return True


def read_on_grid(item, lon, lat, epsg):
    if not complete(item, lon, lat, epsg):
        raise IncompleteChipError('No complete source coverage of requested grid')
    if source_crs(item['properties']).to_epsg() == epsg:
        return read_chip(pystac.Item.from_dict(item), lon, lat)
    x, y = transform('EPSG:4326', f'EPSG:{epsg}', [lon], [lat])
    target = from_origin(x[0]-640, y[0]+640, 10, 10)
    array = np.zeros((8, 128, 128), dtype=np.uint16)
    for index, band in enumerate(BANDS):
        method = rasterio.enums.Resampling.nearest if band == 'scl' else rasterio.enums.Resampling.bilinear
        with rasterio.open(item['assets'][band]['href']) as source:
            bounds = transform_bounds(f'EPSG:{epsg}', source.crs, x[0]-640, y[0]-640, x[0]+640, y[0]+640, densify_pts=21)
            require_complete_window(from_bounds(*bounds, source.transform), source.width, source.height)
            with WarpedVRT(source, crs=f'EPSG:{epsg}', transform=target,
                           width=128, height=128, resampling=method, nodata=0) as vrt:
                array[index] = vrt.read(1)
    return array


def read_acquisition_chip(item, candidates, lon, lat):
    """Keep the requested grid when an overlapping tile supplies full coverage.

    Only tiles with the same platform, datatake, processing baseline and band
    radiometry may substitute. Missing acquisition identity is not a match.
    All candidates are supplied by the normal date-window STAC search.
    """
    epsg = source_crs(item.properties).to_epsg()
    if epsg is None:
        raise ValueError('An EPSG code is required for the acquisition grid')
    try:
        return item, read_chip(item, lon, lat), epsg
    except IncompleteChipError:
        pass
    original = item.to_dict()
    keys = ('platform', 's2:datatake_id', 's2:processing_baseline')
    for _, candidate in candidates:
        if candidate.id == item.id:
            continue
        other = candidate.to_dict()
        if not all(original['properties'].get(k) is not None and
                   original['properties'][k] == other['properties'].get(k) for k in keys):
            continue
        if not all(original['assets'][band].get('raster:bands', [{}])[0].get(k) ==
                   other['assets'][band].get('raster:bands', [{}])[0].get(k)
                   for band in BANDS for k in ('scale', 'offset', 'nodata', 'data_type')):
            continue
        if complete(other, lon, lat, epsg):
            return candidate, read_on_grid(other, lon, lat, epsg), epsg
    raise IncompleteChipError('No matching acquisition tile covers the requested grid')
