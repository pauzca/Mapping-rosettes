"""Evaluate one detector end to end: validation tiles, then every available evaluation mosaic.

1. Tile-level mAP on the validation split (Ultralytics-equivalent) and the F1-optimal confidence.
2. For each mosaic in ablations/mosaics.yaml whose raster exists: tiled inference + global NMS,
   then RP / RR / RF1 at the paper's fixed threshold (0.5) and at the validation-selected one,
   plus AP50 and AP50:95.

Example:
    python -m ablations.evaluate_model --model run:outputs/train/mixed/frcnn_r50_s123:last \
        --data /disk/guille/mapping-rosettes/data/mixed-resolution/data.yaml --out outputs/eval/mixed/frcnn_r50_s123_last
"""
import argparse
import json
from pathlib import Path

import torch
import yaml

from ablations.detect import detect_mosaic
from ablations.evaluate_detection import evaluate
from ablations.geo import read_gt
from ablations.models import load_predictor
from ablations.val_tiles import evaluate_split

MOSAICS = Path(__file__).with_name("mosaics.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model spec, see ablations.models.load_predictor")
    ap.add_argument("--data", required=True, help="data.yaml whose val split selects the operating threshold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixed-thr", type=float, default=0.5)
    ap.add_argument("--mosaics", nargs="*", default=None, help="subset of mosaics.yaml keys")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    predict = load_predictor(args.model, args.device)

    val = evaluate_split(predict, args.data, "val")
    val["meta"]["model"] = args.model
    (out / "val_tiles.json").write_text(json.dumps(val, indent=2))
    thrs = [args.fixed_thr, val["all"]["best_conf"]]
    print(f"val tiles: mAP50 {val['all']['map50']:.4f} mAP50-95 {val['all']['map']:.4f} "
          f"best_conf {val['all']['best_conf']:.3f}", flush=True)

    for key, m in yaml.safe_load(MOSAICS.read_text()).items():
        if (args.mosaics and key not in args.mosaics) or not Path(m["raster"]).exists():
            continue
        det, meta = detect_mosaic(predict, m["raster"], gsd=m["gsd"])
        det.to_file(out / f"{key}_detections.gpkg", driver="GPKG")
        res = evaluate(det, read_gt(m["gt"], m.get("layer")), score_thrs=thrs)
        res.update(mosaic=key, name=m["name"], resolution=m["resolution"], model=args.model,
                   threshold_rule={"fixed": thrs[0], "val_f1_optimal": thrs[1]}, inference=meta)
        (out / f"{key}_metrics.json").write_text(json.dumps(res, indent=2))
        fx, vs = res["at_iou50"]
        print(f"{key}: fixed {fx['RF1']:.3f}/{fx['RP']:.3f}/{fx['RR']:.3f}  val-thr {vs['RF1']:.3f}/{vs['RP']:.3f}/"
              f"{vs['RR']:.3f}  AP50 {res['AP50']:.3f}  ({meta['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
