"""Inference cost of each detector, measured the same way for all of them.

Parameters, GFLOPs for one 512x512 tile (torch.utils.flop_counter, 2 FLOPs per MAC) and
end-to-end latency per tile through the shared predictor interface (pre-processing, network
and the model's own post-processing/NMS), batch 1, fp32 and fp16 autocast, median of N runs.

Example:
    python -m ablations.benchmark --out outputs/eval/benchmark.json \
        yolo11s=ultralytics:outputs/train/mixed/yolo11s_s123/weights/last.pt frcnn_r50=run:outputs/train/mixed/frcnn_r50_s123
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from ablations.models import AdapterPredictor, UltralyticsPredictor, load_predictor


def network(predictor):
    if isinstance(predictor, UltralyticsPredictor):
        return predictor.model.model
    if isinstance(predictor, AdapterPredictor):
        return predictor.adapter
    raise TypeError(type(predictor))


@torch.no_grad()
def gflops(predictor):
    net = network(predictor).eval()
    x = torch.rand(1, 3, 512, 512, device=next(net.parameters()).device)  # Ultralytics moves to GPU lazily
    with FlopCounterMode(display=False) as fc:
        net(x) if isinstance(predictor, UltralyticsPredictor) else net.predict(x)
    return fc.get_total_flops() / 1e9


def latency_ms(predictor, img, n, half):
    with torch.autocast("cuda", dtype=torch.float16, enabled=half):
        for _ in range(20):
            predictor(img, 0.01)
        times = []
        for _ in range(n):
            torch.cuda.synchronize()
            t = time.perf_counter()
            predictor(img, 0.01)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="<name>=<model spec>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    img = (np.random.default_rng(0).random((512, 512, 3)) * 255).astype(np.uint8)
    res = {}
    for item in args.models:
        name, spec = item.split("=", 1)
        pred = load_predictor(spec, args.device)
        res[name] = dict(spec=spec, params_M=round(sum(p.numel() for p in network(pred).parameters()) / 1e6, 2),
                         gflops_512=round(gflops(pred), 1),
                         ms_fp32=round(latency_ms(pred, img, args.runs, half=False), 2),
                         ms_fp16=round(latency_ms(pred, img, args.runs, half=True), 2),
                         gpu=torch.cuda.get_device_name(0))
        print(name, res[name], flush=True)
        del pred
        torch.cuda.empty_cache()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
