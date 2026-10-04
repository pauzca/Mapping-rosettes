"""Mosaic-level detection metrics: P/R/F1 at fixed score thresholds, AP50 and AP50:95.

Matching is greedy by descending score; each prediction takes the unmatched
ground-truth box with the highest IoU (the notebook took the first GT above the
threshold, which can steal matches from better-aligned predictions).

Precision = TP / predictions, Recall = TP / ground truth. Reported as RP / RR / RF1 (mosaic level),
the names used in the paper's tables. Score thresholds: fixed values (--score-thrs) and, optionally,
the F1-optimal confidence found on the validation tiles (--conf-from val_tiles.json).
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.strtree import STRtree

from ablations.geo import read_gt, to_metric_boxes


def iou_matrix(pred, gt):
    """Sparse IoU as dict {(i_pred, j_gt): iou} for intersecting pairs."""
    tree = STRtree(gt.geometry.values)
    p_idx, g_idx = tree.query(pred.geometry.values, predicate="intersects")
    pg, gg = pred.geometry.values[p_idx], gt.geometry.values[g_idx]
    inter = np.array([a.intersection(b).area for a, b in zip(pg, gg)])
    union = np.array([a.area for a in pg]) + np.array([b.area for b in gg]) - inter
    return {(i, j): v for i, j, v in zip(p_idx, g_idx, inter / union)}


def match(scores, ious, n_gt, thr):
    """Return per-prediction TP flags (in descending score order) and the order itself."""
    order = np.argsort(-scores, kind="stable")
    cand = {}
    for (i, j), v in ious.items():
        if v >= thr:
            cand.setdefault(i, []).append((v, j))
    taken, tp = set(), np.zeros(len(order), bool)
    for k, i in enumerate(order):
        for v, j in sorted(cand.get(i, []), reverse=True):
            if j not in taken:
                taken.add(j)
                tp[k] = True
                break
    return tp, order


def average_precision(tp, n_gt):
    """All-point interpolated AP (VOC 2010+/COCO style without 101-point sampling)."""
    if n_gt == 0:
        return float("nan")
    ctp = np.cumsum(tp)
    rec = ctp / n_gt
    prec = ctp / np.arange(1, len(tp) + 1)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def prf(tp_flags, scores_sorted, n_gt, score_thr):
    keep = scores_sorted >= score_thr
    tp = int(tp_flags[keep].sum())
    n_pred = int(keep.sum())
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(score_thr=round(float(score_thr), 4), tp=tp, fp=n_pred - tp, fn=n_gt - tp,
                RP=round(p, 4), RR=round(r, 4), RF1=round(f1, 4))


def evaluate(pred, gt, score_thrs=(0.4, 0.5), iou_thr=0.5):
    pred = to_metric_boxes(pred).reset_index(drop=True)
    gt = to_metric_boxes(gt).reset_index(drop=True)
    scores = pred["score"].to_numpy(float)
    ious = iou_matrix(pred, gt)
    tp, order = match(scores, ious, len(gt), iou_thr)
    res = dict(n_gt=len(gt), n_pred=len(pred),
               at_iou50=[prf(tp, scores[order], len(gt), s) for s in score_thrs],
               AP50=round(average_precision(tp, len(gt)), 4))
    aps = [average_precision(match(scores, ious, len(gt), t)[0], len(gt)) for t in np.arange(0.5, 0.96, 0.05)]
    res["AP50_95"] = round(float(np.mean(aps)), 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--gt-layer", default=None)
    ap.add_argument("--score-thrs", type=float, nargs="+", default=[0.5])
    ap.add_argument("--conf-from", default=None, help="val_tiles.json: adds its F1-optimal confidence")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    thrs = list(args.score_thrs)
    if args.conf_from:
        thrs.append(json.loads(Path(args.conf_from).read_text())["all"]["best_conf"])
    res = evaluate(gpd.read_file(args.pred), read_gt(args.gt, args.gt_layer), score_thrs=thrs)
    res.update(pred=args.pred, gt=args.gt)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
