"""Detector registry with one interface across Ultralytics, torchvision, Hugging Face and DeepForest.

Trainable adapters (non-Ultralytics models) expose
    loss(imgs, targets) -> (loss, parts)   imgs: float (B,3,H,W) in [0,1]; targets: list of (n,4) xyxy px
    predict(imgs)       -> [(boxes xyxy px (k,4), scores (k,)), ...]
Predictors (every model, inference only) map an RGB uint8 tile (H,W,3) to numpy boxes and scores, so
the tile validator and the mosaic pipeline run all detectors through the same code.

Model specs for load_predictor():
    ultralytics:<weights.pt>   YOLO / RT-DETR checkpoints (COCO weights = zero-shot, class-agnostic)
    zeroshot:<arch>            off-the-shelf torchvision / HF / DeepForest weights, class-agnostic
    run:<run_dir>[:last]       model fine-tuned with ablations.train_torch (weights/best.pt or last.pt)
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ablations.recipe import OPTIM

MAX_DET = 300        # same detection cap for every model (Ultralytics default)
SCORE_FLOOR = 0.001  # keep low-score boxes for AP; operating thresholds are applied at evaluation
HF_RTDETR = "PekingU/rtdetr_r18vd"   # COCO-only pretraining, like the other COCO models
DEEPFOREST = "weecology/deepforest-tree"
ARCHS = tuple(OPTIM)


def _frcnn(arch, num_classes):
    import torchvision.models.detection as tvd
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    fn = {"frcnn_mnv3": tvd.fasterrcnn_mobilenet_v3_large_fpn, "frcnn_r50": tvd.fasterrcnn_resnet50_fpn}[arch]
    m = fn(weights="DEFAULT", min_size=512, max_size=512, box_detections_per_img=MAX_DET,
           box_score_thresh=SCORE_FLOOR)
    if num_classes:  # replace the COCO head (+1 for the background class); None keeps it for zero-shot
        m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,
                                                      num_classes + 1)
    return m


def _deepforest(fine_tune):
    from deepforest import main as dfm

    df = dfm.deepforest()
    df.load_model(DEEPFOREST)  # RetinaNet-R50-FPN trained on NEON tree crowns, single class
    m = df.model
    m.detections_per_img, m.score_thresh = MAX_DET, SCORE_FLOOR
    if fine_tune:  # same input size and NMS as the other torchvision detectors
        m.transform.min_size, m.transform.max_size = (512,), 512
        m.nms_thresh = 0.5
    return m  # zero-shot keeps DeepForest's own resize (800 px) and NMS (0.05)


def _rtdetr(num_classes):
    from transformers import RTDetrForObjectDetection

    if not num_classes:
        return RTDetrForObjectDetection.from_pretrained(HF_RTDETR)
    return RTDetrForObjectDetection.from_pretrained(
        HF_RTDETR, num_labels=num_classes, id2label={0: "frailejon"}, label2id={"frailejon": 0},
        ignore_mismatched_sizes=True)


class TorchvisionAdapter(nn.Module):
    """Faster R-CNN and RetinaNet. Faster R-CNN reserves label 0 for background, RetinaNet does not."""

    def __init__(self, model, label):
        super().__init__()
        self.model, self.label = model, label

    def loss(self, imgs, targets):
        tg = [{"boxes": b, "labels": torch.full((len(b),), self.label, dtype=torch.int64, device=b.device)}
              for b in targets]
        parts = self.model(list(imgs), tg)
        return sum(parts.values()), {k: v.detach() for k, v in parts.items()}

    @torch.no_grad()
    def predict(self, imgs):
        return [(o["boxes"].float(), o["scores"].float()) for o in self.model(list(imgs))]


class RTDetrAdapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def loss(self, imgs, targets):
        h, w = imgs.shape[-2:]
        scale = torch.tensor([w, h, w, h], device=imgs.device, dtype=torch.float32)
        labels = []
        for b in targets:
            b = b.float() / scale
            cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                                  b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
            labels.append({"class_labels": torch.zeros(len(b), dtype=torch.int64, device=b.device),
                           "boxes": cxcywh})
        out = self.model(pixel_values=imgs, labels=labels)
        parts = {k: v.detach() for k, v in (out.loss_dict or {}).items()
                 if k in ("loss_vfl", "loss_bbox", "loss_giou")}
        return out.loss, parts

    @torch.no_grad()
    def predict(self, imgs):
        h, w = imgs.shape[-2:]
        out = self.model(pixel_values=imgs)
        scores = out.logits.float().sigmoid().max(-1).values  # max over classes: class-agnostic for COCO
        cx, cy, bw, bh = out.pred_boxes.float().unbind(-1)
        boxes = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], -1)
        return [(b[s >= SCORE_FLOOR], s[s >= SCORE_FLOOR]) for b, s in zip(boxes, scores)]


def build_adapter(arch, num_classes=1):
    """Adapter ready to fine-tune (num_classes=1) or the off-the-shelf model (num_classes=None)."""
    if arch in ("frcnn_mnv3", "frcnn_r50"):
        return TorchvisionAdapter(_frcnn(arch, num_classes), label=1)
    if arch == "deepforest":
        return TorchvisionAdapter(_deepforest(fine_tune=num_classes is not None), label=0)
    if arch == "rtdetr_r18":
        return RTDetrAdapter(_rtdetr(num_classes))
    raise ValueError(f"unknown arch {arch}")


def make_optimizer(adapter, arch):
    o = OPTIM[arch]
    params = [(n, p) for n, p in adapter.named_parameters() if p.requires_grad]
    if o["name"] == "SGD":
        return torch.optim.SGD([p for _, p in params], lr=o["lr"], momentum=o["momentum"],
                               weight_decay=o["weight_decay"])
    # RT-DETR official groups: backbone at 1e-5, no weight decay on encoder/decoder norms
    backbone = [p for n, p in params if ".backbone." in n]
    norms = [p for n, p in params if ".backbone." not in n and ("norm" in n or ".bn." in n)]
    rest = [p for n, p in params if ".backbone." not in n and not ("norm" in n or ".bn." in n)]
    return torch.optim.AdamW([{"params": backbone, "lr": o["backbone_lr"]},
                              {"params": norms, "weight_decay": 0.0},
                              {"params": rest}],
                             lr=o["lr"], betas=(0.9, 0.999), weight_decay=o["weight_decay"])


class UltralyticsPredictor:
    def __init__(self, weights, device, imgsz=512):
        from ultralytics import RTDETR, YOLO

        ck = torch.load(weights, map_location="cpu", weights_only=False)
        is_detr = "RTDETR" in type(ck.get("ema") or ck["model"]).__name__
        self.model = (RTDETR if is_detr else YOLO)(weights)
        self.device, self.imgsz = device, imgsz

    def __call__(self, img, min_score):
        # numpy input is treated as BGR by Ultralytics (like cv2.imread on the training PNGs)
        r = self.model.predict(img[:, :, ::-1], imgsz=self.imgsz, conf=min_score, iou=0.7, max_det=MAX_DET,
                               agnostic_nms=True, device=self.device, verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy().astype(np.float64), r.boxes.conf.cpu().numpy()


class AdapterPredictor:
    def __init__(self, adapter, device):
        self.adapter, self.device = adapter.to(device).eval(), device

    @torch.no_grad()
    def __call__(self, img, min_score):
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255)
        b, s = self.adapter.predict(x[None].to(self.device))[0]
        k = s >= min_score
        return b[k].cpu().numpy().astype(np.float64), s[k].cpu().numpy()


def load_run(run_dir, device="cpu", which="best"):
    run_dir = Path(run_dir)
    arch = json.loads((run_dir / "args.json").read_text())["arch"]
    adapter = build_adapter(arch, num_classes=1)
    ck = torch.load(run_dir / "weights" / f"{which}.pt", map_location="cpu", weights_only=False)
    adapter.load_state_dict(ck["state_dict"])
    return adapter.to(device).eval()


def load_predictor(spec, device):
    kind, _, target = spec.partition(":")
    if kind == "ultralytics":
        return UltralyticsPredictor(target, device)
    if kind == "zeroshot":
        return AdapterPredictor(build_adapter(target, num_classes=None), device)
    if kind == "run":
        run_dir, _, which = target.partition(":")
        return AdapterPredictor(load_run(run_dir, device, which or "best"), device)
    raise ValueError(f"unknown model spec {spec}")
