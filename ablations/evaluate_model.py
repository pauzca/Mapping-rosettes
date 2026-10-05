"""Evaluate one detector end to end: test tiles, then the evaluation orthomosaics.

1. Operating threshold: --thr (the validation-selected value from ablations/thresholds.json), or, if not
   given, the F1-optimal confidence on the validation tiles.
2. Test tiles: P / R / F1 per source at that threshold, plus mAP.
3. Each mosaic in ablations/mosaics.yaml found under --mosaics-root: tiled inference + global NMS, then
   RP / RR / RF1 at that threshold, plus AP50 and AP50:95.

Example:
    python -m ablations.evaluate_model --model run:checkpoints/frcnn_r50 --thr 0.86517 \
        --data dataset/data.yaml --mosaics-root mosaics --out outputs/eval/mixed/frcnn_r50
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
    ap.add_argument("--data", required=True, help="data.yaml with val and test splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=None, help="fixed operating threshold (skips validation)")
    ap.add_argument("--mosaics-root", default=None, help="folder with the evaluation orthomosaics")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    predict = load_predictor(args.model, args.device)

    thr = args.thr
    if thr is None:
        val = evaluate_split(predict, args.data, "val")
        (out / "val_tiles.json").write_text(json.dumps(val, indent=2))
        thr = val["all"]["best_conf"]
    print(f"threshold {thr}", flush=True)

    records = collect(predict, args.data, "test")
    test = dict(prf=prf_at(records, thr), map=evaluate_split(predict, args.data, "test", records), model=args.model,
                thr=thr)
    (out / "test_tiles.json").write_text(json.dumps(test, indent=2))
    for src, m in sorted(test["prf"].items()):
        print(f"test {src}: F1 {m['F1']:.3f} P {m['P']:.3f} R {m['R']:.3f} (n={m['n_gt']})", flush=True)

    for key, m in yaml.safe_load(MOSAICS.read_text()).items():
        raster = Path(args.mosaics_root or ".") / m["raster"]
        if not raster.exists():
            continue
        det, meta = detect_mosaic(predict, str(raster), gsd=m["gsd"])
        res = evaluate(det, read_gt(Path(__file__).parent.parent / m["gt"], m.get("layer")), score_thrs=[thr])
        res.update(mosaic=key, name=m["name"], resolution=m["resolution"], model=args.model, inference=meta)
        (out / f"{key}_metrics.json").write_text(json.dumps(res, indent=2))
        vs = res["at_iou50"][0]
        print(f"{key}: RF1 {vs['RF1']:.3f} RP {vs['RP']:.3f} RR {vs['RR']:.3f}  AP50 {res['AP50']:.3f}", flush=True)


if __name__ == "__main__":
    main()
