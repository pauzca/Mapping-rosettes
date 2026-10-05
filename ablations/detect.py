"""Tile an orthomosaic, run a detector on each tile and merge detections into one layer.

Same protocol as the paper: geodataset tiles at 1.36 cm/px, 512 px, 20% overlap, then a global greedy
NMS across tiles (IoU 0.2). Detections are kept down to min_score; the operating threshold is applied
at evaluation time.
"""
import os
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torchvision.ops import nms

from ablations.geo import METRIC_CRS, pixel_boxes_to_geoms, to_metric_boxes

TILES_ROOT = Path(os.environ.get("ROSETTES_TILES", "outputs/tiles"))


def tile_mosaic(mosaic, tile=512, overlap=0.2, gsd=0.0136):
    """Tiles exactly as the paper's pipeline: geodataset RasterTilerizer (reprojects geographic rasters
    to UTM and resamples to gsd). Cached per mosaic and setting; a file lock avoids concurrent writers."""
    import fcntl

    from geodataset.tilerize import RasterTilerizer

    out = TILES_ROOT / f"{Path(mosaic).stem}_t{tile}_o{overlap}_gr{gsd or 'native'}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (out / "done").exists():
            RasterTilerizer(raster_path=mosaic, output_path=out, tile_size=tile, tile_overlap=overlap,
                            ground_resolution=gsd or None, ignore_black_white_alpha_tiles_threshold=0.8,
                            temp_dir=out / "tmp").generate_tiles()
            (out / "done").touch()
    return sorted(p for p in out.rglob("*.tif") if "tiles" in p.parts)


def detect_mosaic(predict, mosaic, tile=512, overlap=0.2, gsd=0.0136, min_score=0.01, nms_iou=0.2):
    """Run predict(img_rgb, min_score) on every tile and merge with a global greedy NMS, which keeps the same
    boxes above any score threshold as geodataset's Aggregator ('iou' algorithm) used in the paper."""
    tiles = tile_mosaic(mosaic, tile, overlap, gsd)
    t0 = time.time()
    geoms, scores, crs = [], [], None
    for f in tiles:
        with rasterio.open(f) as src:
            img = np.moveaxis(src.read([1, 2, 3]), 0, -1)
            crs = crs or src.crs
            b, s = predict(img, min_score)
            geoms += pixel_boxes_to_geoms(b, Window(0, 0, src.width, src.height), src.transform)
            scores += s.tolist()
    elapsed = time.time() - t0

    det = to_metric_boxes(gpd.GeoDataFrame({"score": scores}, geometry=geoms, crs=crs))
    if len(det):
        b = torch.tensor(np.array([g.bounds for g in det.geometry]), dtype=torch.float64)
        keep = nms(b, torch.tensor(det.score.values, dtype=torch.float64), nms_iou).numpy()
        det = det.iloc[np.sort(keep)].reset_index(drop=True)
    meta = dict(n_tiles=len(tiles), n_raw=len(scores), n_after_nms=len(det), seconds=round(elapsed, 1),
                crs=METRIC_CRS)
    return det, meta
