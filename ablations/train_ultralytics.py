"""Fine-tune Ultralytics detectors (YOLO11, YOLOv8, RT-DETR-L) with the recipe of the paper's YOLO11s.

Early stopping is disabled (patience=0) so every model runs the same 200 epochs / 22,400 iterations.

Example:
    python -m ablations.train_ultralytics --weights /disk/guille/mapping-rosettes/models/pretrained/yolov8s.pt \
        --data /disk/guille/mapping-rosettes/data/mixed-resolution/data.yaml --out outputs/train/mixed/yolov8s_s123
"""
import argparse
import json
from pathlib import Path

from ablations.recipe import AUG, OPTIM, TRAIN, ULTRALYTICS


def plain_val_trainer():
    """YOLO trainer whose validation tiles are not padded (rect=False), like every other model.

    Ultralytics validates YOLO with rect batches that add a grey border to 512 px tiles, which raises its
    tile mAP50-95 (0.18 -> 0.31 for the paper model) and would make best.pt selection not comparable.
    """
    from ultralytics.data import build_yolo_dataset
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    class PlainValTrainer(DetectionTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)

    return PlainValTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="COCO-pretrained .pt (yolo11s.pt, yolov8s.pt, rtdetr-l.pt)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    ap.add_argument("--seed", type=int, default=TRAIN["seed"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fraction", type=float, default=1.0, help="debug: use a fraction of the train set")
    args = ap.parse_args()

    from ultralytics import RTDETR, YOLO

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    is_detr = "rtdetr" in Path(args.weights).stem
    hyp = dict(ULTRALYTICS)
    if is_detr:  # 'auto' resolves to AdamW 2e-3, which diverges to NaN with AMP; use the RT-DETR reference LR
        o = OPTIM["rtdetr_r18"]
        hyp.update(optimizer="AdamW", lr0=o["lr"], weight_decay=o["weight_decay"], momentum=0.9)
    (out / "args.json").write_text(json.dumps(dict(vars(args), train=TRAIN, aug=AUG, ultralytics=hyp), indent=2))
    model = (RTDETR if is_detr else YOLO)(args.weights)
    extra = {} if is_detr else {"trainer": plain_val_trainer()}  # RT-DETR already validates with rect=False
    model.train(**extra, data=args.data, epochs=args.epochs, batch=TRAIN["batch"], imgsz=TRAIN["imgsz"], seed=args.seed,
                close_mosaic=TRAIN["close_mosaic"], warmup_epochs=TRAIN["warmup_epochs"], lrf=TRAIN["lrf"],
                **AUG, **hyp, single_cls=True, amp=True, pretrained=True, patience=0,
                fraction=args.fraction, workers=args.workers, device=0,
                project=str(out.parent), name=out.name, exist_ok=True, plots=True)


if __name__ == "__main__":
    main()
