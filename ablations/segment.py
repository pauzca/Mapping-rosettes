"""Turn detection boxes into plant masks.

Methods:
  sam2_{t,s,b+,l}  SAM2.1 with the box as prompt (bboxes_to_sam2_masks.ipynb uses sam2_l)
  box              the detection box itself (no segmentation)
  ellipse          ellipse inscribed in the box

SAM runs on the native pixel grid in large tiles; each box is assigned to a tile
that fully contains it, so no box is cut. Output masks are saved in a metric CRS.

Example:
    python -m ablations.segment --mosaic data/mosaics/rosettes_gsd_3cm_mosaic.tif \
        --boxes outputs/3cm/yolov11/detections.gpkg --min-score 0.4 --method sam2_l \
        --out outputs/3cm/yolov11/masks_sam2_l.gpkg
"""
import argparse
import json
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from affine import Affine
from rasterio.features import shapes
from shapely import affinity
from shapely.geometry import MultiPolygon, Point, shape

from ablations.geo import METRIC_CRS, iter_windows, native_gsd_m, open_mosaic, read_rgb

SAM2_MODELS = {
    "sam2_t": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "sam2_s": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "sam2_b+": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "sam2_l": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}


def ellipse_in(geom):
    x1, y1, x2, y2 = geom.bounds
    return affinity.scale(Point((x1 + x2) / 2, (y1 + y2) / 2).buffer(1, 64), (x2 - x1) / 2, (y2 - y1) / 2)


def mask_to_geom(mask, transform):
    parts = [shape(g) for g, v in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform) if v > 0]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def run_sam(src, boxes_px, method, ckpt_dir, device, tile):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    cfg, ckpt = SAM2_MODELS[method]
    predictor = SAM2ImagePredictor(build_sam2(cfg, str(Path(ckpt_dir) / ckpt), device=device))

    # tiles overlap by more than the largest box, so every box fits whole in some tile
    max_box = float(np.max(boxes_px[:, 2:] - boxes_px[:, :2])) if len(boxes_px) else 0
    overlap = min(0.5, (max_box + 16) / tile)
    windows = list(iter_windows(src.width, src.height, tile, overlap))
    assign = {}
    for k, (x1, y1, x2, y2) in enumerate(boxes_px):
        for w_i, w in enumerate(windows):
            if w.col_off <= x1 and x2 <= w.col_off + w.width and w.row_off <= y1 and y2 <= w.row_off + w.height:
                assign.setdefault(w_i, []).append(k)
                break
        else:
            raise RuntimeError(f"box {k} ({x2 - x1:.0f}x{y2 - y1:.0f} px) does not fit in any tile")

    geoms, sam_scores = [None] * len(boxes_px), [np.nan] * len(boxes_px)
    use_amp = device.startswith("cuda")
    for w_i, ks in assign.items():
        w = windows[w_i]
        img, _ = read_rgb(src, w)
        local = boxes_px[ks] - np.array([w.col_off, w.row_off, w.col_off, w.row_off])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            predictor.set_image(img)
            masks, scores, _ = predictor.predict(box=local, multimask_output=False)
        masks = masks.reshape(len(ks), *masks.shape[-2:])
        scores = np.asarray(scores).reshape(len(ks))
        transform = src.transform * Affine.translation(w.col_off, w.row_off)
        for k, m, s in zip(ks, masks, scores):
            geoms[k] = mask_to_geom(m > 0, transform)
            sam_scores[k] = float(s)
    return geoms, sam_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mosaic", required=True)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--boxes-layer", default=None)
    ap.add_argument("--min-score", type=float, default=None, help="filter detections by 'score' first")
    ap.add_argument("--method", default="sam2_l", choices=list(SAM2_MODELS) + ["box", "ellipse"])
    ap.add_argument("--ckpt-dir", default="/disk/guille/mapping-rosettes/models")
    ap.add_argument("--tile", type=int, default=1024, help="SAM input size; native tile = tile * gsd / native_gsd")
    ap.add_argument("--gsd", type=float, default=0.0136, help="effective GSD in m seen by SAM (0 = native)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    boxes = gpd.read_file(args.boxes, layer=args.boxes_layer)
    if args.min_score is not None:
        boxes = boxes[boxes["score"] >= args.min_score]
    boxes = boxes.reset_index(drop=True)
    boxes["box_id"] = boxes.index

    t0 = time.time()
    with open_mosaic(args.mosaic) as src:
        native = boxes.to_crs(src.crs)
        if args.method in ("box", "ellipse"):
            geoms = list(native.geometry.envelope) if args.method == "box" else [ellipse_in(g) for g in native.geometry]
            sam_scores = [np.nan] * len(geoms)
        else:
            inv = ~src.transform
            px = []
            for g in native.geometry:
                x1, y1, x2, y2 = g.bounds
                c1, r1 = inv * (x1, y2)  # top-left
                c2, r2 = inv * (x2, y1)  # bottom-right
                px.append([min(c1, c2), min(r1, r2), max(c1, c2), max(r1, r2)])
            px = np.clip(np.array(px).reshape(-1, 4), 0, [src.width, src.height, src.width, src.height])
            # SAM resizes every input to 1024 px, so a smaller native tile = upsampling
            scale = native_gsd_m(src) / args.gsd if args.gsd else 1.0
            tile_native = min(int(round(args.tile / scale)), src.width, src.height)
            geoms, sam_scores = run_sam(src, px, args.method, args.ckpt_dir, args.device, tile_native)
        crs = src.crs
    elapsed = time.time() - t0

    cols = [c for c in boxes.columns if c != "geometry"]
    masks = gpd.GeoDataFrame(boxes[cols].assign(seg_score=sam_scores), geometry=geoms, crs=crs)
    n_empty = int(masks.geometry.isna().sum())
    masks = masks[masks.geometry.notna()].to_crs(METRIC_CRS)
    masks["area_m2"] = masks.geometry.area

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    masks.to_file(out, driver="GPKG")
    meta = dict(vars(args), n_boxes=len(boxes), n_masks=len(masks), n_empty=n_empty, seconds=round(elapsed, 1))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
