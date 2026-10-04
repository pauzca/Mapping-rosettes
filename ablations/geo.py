"""Shared geospatial helpers for the ablation pipeline.

Inference always runs on the mosaic's native pixel grid (the same grid the
training tiles were cut from). Outputs are reprojected to a metric CRS
(UTM 18N by default) so areas and IoU are computed in metres.
"""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

METRIC_CRS = "EPSG:32618"  # UTM 18N, same CRS as the 1 cm ground truth


def iter_windows(width, height, tile, overlap):
    """Yield tile windows covering the raster; the last row/col is snapped to the border."""
    stride = max(1, int(round(tile * (1 - overlap))))
    xs = list(range(0, max(width - tile, 0) + 1, stride))
    ys = list(range(0, max(height - tile, 0) + 1, stride))
    if xs[-1] + tile < width:
        xs.append(width - tile)
    if ys[-1] + tile < height:
        ys.append(height - tile)
    for y in ys:
        for x in xs:
            yield Window(x, y, min(tile, width), min(tile, height))


def read_rgb(src, window):
    """Read an RGB tile (H, W, 3) uint8 plus the fraction of valid pixels (from alpha if present)."""
    img = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
    img = np.moveaxis(img, 0, -1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if src.count >= 4:
        alpha = src.read(4, window=window, boundless=True, fill_value=0)
        valid = float((alpha > 0).mean())
    else:
        valid = float((img.sum(axis=-1) > 0).mean())
    return img, valid


def pixel_boxes_to_geoms(boxes_xyxy, window, transform):
    """Convert tile-local pixel boxes (N, 4) to polygons in the raster CRS."""
    geoms = []
    # float64 is required: in float32 a coordinate like -72.97 deg only resolves ~0.8 m
    for x1, y1, x2, y2 in np.asarray(boxes_xyxy, dtype=np.float64).tolist():
        gx1, gy1 = transform * (window.col_off + x1, window.row_off + y1)
        gx2, gy2 = transform * (window.col_off + x2, window.row_off + y2)
        geoms.append(box(min(gx1, gx2), min(gy1, gy2), max(gx1, gx2), max(gy1, gy2)))
    return geoms


def to_metric_boxes(gdf, crs=METRIC_CRS):
    """Reproject to a metric CRS and replace each geometry by its axis-aligned envelope."""
    out = gdf.to_crs(crs)
    out["geometry"] = out.geometry.envelope
    return out


def native_gsd_m(src, crs=METRIC_CRS):
    """Pixel size in metres at the raster centre (works for geographic CRSs too)."""
    from pyproj import Transformer
    tr = Transformer.from_crs(src.crs, crs, always_xy=True)
    cx, cy = src.width / 2, src.height / 2
    x0, y0 = tr.transform(*(src.transform * (cx, cy)))
    x1, y1 = tr.transform(*(src.transform * (cx + 1, cy)))
    x2, y2 = tr.transform(*(src.transform * (cx, cy + 1)))
    return (np.hypot(x1 - x0, y1 - y0) + np.hypot(x2 - x0, y2 - y0)) / 2


def read_gt(path, layer=None):
    gdf = gpd.read_file(path, layer=layer)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)


def open_mosaic(path):
    return rasterio.open(path)
