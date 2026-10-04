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


def evaluate_split(predict, data, split="val"):
    cfg = yaml.safe_load(open(data))
    img_dir = Path(cfg["path"]) / cfg[split]
    stats, t0 = {"all": TileStats()}, time.time()
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".tif"))
    for f in files:
        img = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        gt = read_labels(Path(str(f.parent).replace("images", "labels")) / f"{f.stem}.txt", img.shape[1], img.shape[0])
        boxes, scores = predict(img, SCORE_FLOOR)
        for key in {"all", source_of(f.name)} - {None}:
            stats.setdefault(key, TileStats()).add(boxes, scores, gt)
    res = {k: {m: round(v, 5) if isinstance(v, float) else v for m, v in s.compute().items()} for k, s in stats.items()}
    res["meta"] = dict(data=data, split=split, n_tiles=len(files), seconds=round(time.time() - t0, 1))
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
