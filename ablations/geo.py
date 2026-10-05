"""Shared geospatial helpers: tile pixel boxes -> polygons, metric reprojection, ground truth."""
import geopandas as gpd
import numpy as np
from shapely.geometry import box

METRIC_CRS = "EPSG:32618"  # UTM 18N, same CRS as the 1 cm ground truth


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


def read_gt(path, layer=None):
    gdf = gpd.read_file(path, layer=layer)
    return gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
