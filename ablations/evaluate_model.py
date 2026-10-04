"""Evaluate one detector end to end: validation tiles, test tiles, then the evaluation mosaics.

1. Validation tiles: Ultralytics-equivalent mAP and the F1-optimal confidence (the operating threshold).
2. Test tiles (if the data has a test split): P / R / F1 per source at that threshold, plus mAP.
3. Each mosaic in ablations/mosaics.yaml whose raster exists: tiled inference + global NMS, then
   RP / RR / RF1 at that threshold, plus AP50 and AP50:95.

Example:
    python -m ablations.evaluate_model --model run:outputs/train_v2/mixed/frcnn_r50_s123 \
        --data <kaggle_export>/mixed-resolution-frailejones/dataset/data.yaml --out outputs/eval_v2/mixed/frcnn_r50_s123
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
from ablations.val_tiles import collect, evaluate_split, prf_at

MOSAICS = Path(__file__).with_name("mosaics.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model spec, see ablations.models.load_predictor")
    ap.add_argument("--data", required=True, help="data.yaml whose val split selects the operating threshold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mosaics", nargs="*", default=None, help="subset of mosaics.yaml keys")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    predict = load_predictor(args.model, args.device)

    val = evaluate_split(predict, args.data, "val")
    val["meta"]["model"] = args.model
    (out / "val_tiles.json").write_text(json.dumps(val, indent=2))
    thr = val["all"]["best_conf"]
    print(f"val tiles: mAP50 {val['all']['map50']:.4f} mAP50-95 {val['all']['map']:.4f} best_conf {thr:.3f}", flush=True)

    if "test" in yaml.safe_load(open(args.data)):
        records = collect(predict, args.data, "test")
        test = dict(prf=prf_at(records, thr), map=evaluate_split(predict, args.data, "test", records), model=args.model)
        (out / "test_tiles.json").write_text(json.dumps(test, indent=2))
        for src, m in sorted(test["prf"].items()):
            print(f"test {src}: F1 {m['F1']:.3f} P {m['P']:.3f} R {m['R']:.3f} (n={m['n_gt']})", flush=True)

    for key, m in yaml.safe_load(MOSAICS.read_text()).items():
        if (args.mosaics and key not in args.mosaics) or not Path(m["raster"]).exists():
            continue
        det, meta = detect_mosaic(predict, m["raster"], gsd=m["gsd"])
        det.to_file(out / f"{key}_detections.gpkg", driver="GPKG")
        res = evaluate(det, read_gt(m["gt"], m.get("layer")), score_thrs=[thr])
        res.update(mosaic=key, name=m["name"], resolution=m["resolution"], model=args.model,
                   threshold_rule="val_f1_optimal", inference=meta)
        (out / f"{key}_metrics.json").write_text(json.dumps(res, indent=2))
        vs = res["at_iou50"][0]
        print(f"{key}: RF1 {vs['RF1']:.3f} RP {vs['RP']:.3f} RR {vs['RR']:.3f}  AP50 {res['AP50']:.3f}  "
              f"({meta['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
