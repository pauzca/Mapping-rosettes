"""Tile-level detection metrics computed like Ultralytics' DetectionValidator.

Same IoU matching (greedy by IoU at 0.50:0.95) and the same ap_per_class, so YOLO and
non-YOLO models get comparable mAP numbers and checkpoint selection. Also returns the
confidence that maximises F1 on the validation split, used as each model's operating
threshold on the mosaics.
"""
import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou, smooth

IOUV = torch.linspace(0.5, 0.95, 10)


def match_predictions(iou):
    """iou: (n_gt, n_pred) numpy array -> (n_pred, 10) bool, like BaseValidator.match_predictions."""
    correct = np.zeros((iou.shape[1], len(IOUV)), dtype=bool)
    for i, thr in enumerate(IOUV.tolist()):
        m = np.array(np.nonzero(iou >= thr)).T
        if len(m):
            if len(m) > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


class TileStats:
    def __init__(self):
        self.tp, self.conf, self.n_gt = [], [], 0

    def add(self, pred_boxes, pred_scores, gt_boxes):
        pb = torch.as_tensor(np.asarray(pred_boxes, dtype=np.float32)).reshape(-1, 4)
        ps = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
        gb = torch.as_tensor(np.asarray(gt_boxes, dtype=np.float32)).reshape(-1, 4)
        self.n_gt += len(gb)
        if not len(pb):
            return
        tp = match_predictions(box_iou(gb, pb).numpy()) if len(gb) else np.zeros((len(pb), len(IOUV)), bool)
        self.tp.append(tp)
        self.conf.append(ps)

    def compute(self):
        if not self.tp or not self.n_gt:
            return dict(map50=0.0, map=0.0, p=0.0, r=0.0, f1=0.0, best_conf=0.5, n_gt=self.n_gt)
        tp, conf = np.concatenate(self.tp), np.concatenate(self.conf)
        _, _, p, r, f1, ap, _, _, _, f1_curve, x, _ = ap_per_class(
            tp, conf, np.zeros(len(conf)), np.zeros(self.n_gt))
        best = float(x[smooth(f1_curve.mean(0), 0.1).argmax()])  # same rule Ultralytics uses for P/R
        return dict(map50=float(ap[:, 0].mean()), map=float(ap.mean()), p=float(p.mean()),
                    r=float(r.mean()), f1=float(f1.mean()), best_conf=best, n_gt=self.n_gt)


def targets_from_batch(batch, hw, device, min_side=0.0):
    """Ultralytics batch (normalised xywh + batch_idx) -> list of xyxy px tensors per image."""
    h, w = hw
    xywh = batch["bboxes"].to(device) * torch.tensor([w, h, w, h], device=device)
    xyxy = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
    idx = batch["batch_idx"].to(device).long()
    out = []
    for k in range(batch["img"].shape[0]):
        b = xyxy[idx == k]
        if min_side:
            b = b[((b[:, 2] - b[:, 0]) > min_side) & ((b[:, 3] - b[:, 1]) > min_side)]
        out.append(b)
    return out
