"""Evaluate any detector on a YOLO-format validation split (tile level).

Runs every model through the same predictor interface (ablations.models.load_predictor) and the
Ultralytics-equivalent metrics (ablations.tile_metrics), overall and per source mosaic
(file-name prefix: agosto, dic1cm, dic3cm, ...). The confidence that maximises F1 on the
validation tiles is stored as the model's operating threshold for the mosaics.

Example:
    python -m ablations.val_tiles --model run:outputs/train/mixed/frcnn_r50_s123 \
        --data /disk/guille/mapping-rosettes/data/mixed-resolution/data.yaml --out outputs/eval/mixed/frcnn_r50_s123/val_tiles.json
"""
import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from ablations.models import SCORE_FLOOR, load_predictor
from ablations.tile_metrics import TileStats


def source_of(name):
    m = re.match(r"([a-z0-9]+)_tile_", name) or re.match(r"tile_\d+_\d+_([a-z0-9]+)\.", name)
    return m.group(1) if m else None


def read_labels(path, w, h):
    if not path.exists() or not path.read_text().strip():
        return np.zeros((0, 4))
    a = np.loadtxt(path, ndmin=2)[:, 1:5] * [w, h, w, h]
    return np.c_[a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2]


def split_dir(data, split):
    cfg = yaml.safe_load(open(data))
    return Path(cfg.get("path") or Path(data).parent) / cfg[split]


def collect(predict, data, split="val"):
    """Predictions and ground truth of every tile: list of (source, boxes, scores, gt), boxes in px."""
    files = sorted(p for p in split_dir(data, split).iterdir() if p.suffix.lower() in (".png", ".jpg", ".tif"))
    out = []
    for f in files:
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        gt = read_labels(f.parent.parent / "labels" / f"{f.stem}.txt", img.shape[1], img.shape[0])
        boxes, scores = predict(img, SCORE_FLOOR)
        out.append((source_of(f.name), np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float), gt))
    return out


def evaluate_split(predict, data, split="val", records=None):
    """Ultralytics-equivalent mAP and F1-optimal confidence, overall and per source."""
    t0 = time.time()
    records = records if records is not None else collect(predict, data, split)
    stats = {"all": TileStats()}
    for src, boxes, scores, gt in records:
        for key in {"all", src} - {None}:
            stats.setdefault(key, TileStats()).add(boxes, scores, gt)
    res = {k: {m: round(v, 5) if isinstance(v, float) else v for m, v in s.compute().items()} for k, s in stats.items()}
    res["meta"] = dict(data=data, split=split, n_tiles=len(records), seconds=round(time.time() - t0, 1))
    return res


def iou(a, b):
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def prf_at(records, thr, iou_thr=0.5):
    """Tile-level P / R / F1 at a score threshold, per source and overall.

    Per tile, predictions above the threshold are matched by descending score to the unmatched ground-truth
    box with the highest IoU (>= iou_thr), the same rule as the mosaic-level evaluation.
    """
    acc = {}
    for src, boxes, scores, gt in records:
        keep = scores >= thr
        b, s = boxes[keep], scores[keep]
        tp = 0
        if len(b) and len(gt):
            m = iou(b[np.argsort(-s)], gt)
            taken = np.zeros(len(gt), bool)
            for row in m:
                row = np.where(taken, -1, row)
                j = row.argmax()
                if row[j] >= iou_thr:
                    taken[j] = True
                    tp += 1
        for key in {"all", src} - {None}:
            a = acc.setdefault(key, [0, 0, 0])
            a[0] += tp; a[1] += len(b); a[2] += len(gt)
    res = {}
    for key, (tp, n_pred, n_gt) in acc.items():
        p = tp / n_pred if n_pred else 0.0
        r = tp / n_gt if n_gt else 0.0
        res[key] = dict(thr=round(float(thr), 4), tp=tp, fp=n_pred - tp, fn=n_gt - tp, n_gt=n_gt,
                        P=round(p, 4), R=round(r, 4), F1=round(2 * p * r / (p + r) if p + r else 0.0, 4))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model spec, see ablations.models.load_predictor")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    res = evaluate_split(load_predictor(args.model, args.device), args.data, args.split)
    res["meta"]["model"] = args.model
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
