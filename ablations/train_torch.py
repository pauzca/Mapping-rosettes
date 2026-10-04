"""Fine-tune torchvision / Hugging Face / DeepForest detectors with the Ultralytics data pipeline.

Mirrors the Ultralytics trainer used for the YOLO runs: same YOLODataset tiles and augmentations
(mosaic, rotation, shear, perspective, scale, HSV, flips, close_mosaic), batch, iterations,
warmup length, per-epoch linear LR decay, ModelEMA, AMP and checkpoint selection on validation
mAP50-95. Only the optimizer and base LR follow each architecture's reference recipe (recipe.OPTIM).

Example:
    python -m ablations.train_torch --arch frcnn_r50 \
        --data /disk/guille/mapping-rosettes/data/mixed-resolution/data.yaml --out outputs/train/mixed/frcnn_r50_s123
"""
import argparse
import csv
import json
import time
from copy import copy
from pathlib import Path

import torch
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.torch_utils import ModelEMA, init_seeds

from ablations.models import ARCHS, build_adapter, make_optimizer
from ablations.recipe import AUG, OPTIM, TRAIN
from ablations.tile_metrics import TileStats, targets_from_batch


@torch.no_grad()
def validate(adapter, loader, device, amp):
    adapter.eval()
    stats = TileStats()
    for batch in loader:
        imgs = batch["img"].to(device, non_blocking=True).float() / 255
        gts = targets_from_batch(batch, imgs.shape[-2:], device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            preds = adapter.predict(imgs)
        for (b, s), g in zip(preds, gts):
            stats.add(b.float().cpu(), s.float().cpu(), g.cpu())
    return stats.compute()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=ARCHS)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=TRAIN["epochs"])
    ap.add_argument("--batch", type=int, default=TRAIN["batch"])
    ap.add_argument("--seed", type=int, default=TRAIN["seed"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--fraction", type=float, default=1.0, help="debug: use a fraction of the train set")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "weights").mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(dict(vars(args), train=TRAIN, aug=AUG, optim=OPTIM[args.arch]), indent=2))
    init_seeds(args.seed)
    device, amp = torch.device(args.device), not args.no_amp

    cfg = get_cfg(overrides=dict(AUG, imgsz=TRAIN["imgsz"], single_cls=True, data=args.data, epochs=args.epochs,
                                 batch=args.batch, seed=args.seed, close_mosaic=TRAIN["close_mosaic"],
                                 fraction=args.fraction))
    data = check_det_dataset(args.data)
    train_ds = build_yolo_dataset(cfg, data["train"], args.batch, data, mode="train")
    val_ds = build_yolo_dataset(cfg, data["val"], args.batch, data, mode="val")
    train_loader = build_dataloader(train_ds, args.batch, args.workers, shuffle=True)
    val_loader = build_dataloader(val_ds, args.batch, args.workers, shuffle=False)

    adapter = build_adapter(args.arch, num_classes=1).to(device)
    optimizer = make_optimizer(adapter, args.arch)
    for g in optimizer.param_groups:
        g["initial_lr"] = g["lr"]
    clip = OPTIM[args.arch]["clip"]
    ema = ModelEMA(adapter)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    nb = len(train_loader)
    nw = max(round(TRAIN["warmup_epochs"] * nb), 100)  # same warmup length as Ultralytics

    def lf(epoch):  # Ultralytics linear schedule, stepped once per epoch
        return max(1 - epoch / args.epochs, 0) * (1.0 - TRAIN["lrf"]) + TRAIN["lrf"]

    log = open(out / "results.csv", "w", newline="")
    writer = None
    best, n_bad, t0 = -1.0, 0, time.time()
    for epoch in range(args.epochs):
        if epoch == args.epochs - TRAIN["close_mosaic"]:
            train_ds.mosaic = False
            train_ds.close_mosaic(hyp=copy(cfg))
            train_loader.reset()
        adapter.train()
        sums, n = {}, 0
        for i, batch in enumerate(train_loader):
            ni = i + nb * epoch
            warm = min(1.0, ni / nw)
            for g in optimizer.param_groups:
                g["lr"] = g["initial_lr"] * lf(epoch) * warm
            imgs = batch["img"].to(device, non_blocking=True).float() / 255
            targets = targets_from_batch(batch, imgs.shape[-2:], device, min_side=1.0)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss, parts = adapter.loss(imgs, targets)
            if not torch.isfinite(loss):
                n_bad += 1
                optimizer.zero_grad(set_to_none=True)
                if n_bad > 0.01 * nb * args.epochs + 20:
                    raise RuntimeError(f"too many non-finite losses ({n_bad}); try --no-amp")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(adapter)
            for k, v in dict(parts, loss=loss.detach()).items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1

        if not all(torch.isfinite(v).all() for v in ema.ema.state_dict().values() if v.is_floating_point()):
            raise RuntimeError(f"non-finite EMA weights at epoch {epoch + 1}; try --no-amp")  # fail fast
        m = validate(ema.ema, val_loader, device, amp)
        row = dict(epoch=epoch + 1, iters=nb * (epoch + 1), time_min=round((time.time() - t0) / 60, 2),
                   lr=optimizer.param_groups[-1]["lr"], n_bad=n_bad,
                   **{f"train/{k}": round(float(v) / max(n, 1), 5) for k, v in sums.items()},
                   **{f"val/{k}": round(v, 5) for k, v in m.items()})
        if writer is None:
            writer = csv.DictWriter(log, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        log.flush()
        ck = dict(arch=args.arch, epoch=epoch + 1, fitness=m["map"], val=m, state_dict=ema.ema.state_dict())
        torch.save(ck, out / "weights" / "last.pt")
        if m["map"] > best:  # fitness = mAP50-95, as Ultralytics 8.3.238
            best = m["map"]
            torch.save(ck, out / "weights" / "best.pt")
        print(f"epoch {epoch + 1}/{args.epochs} loss {row.get('train/loss')} val mAP50 {m['map50']:.4f} "
              f"mAP50-95 {m['map']:.4f} best {best:.4f} ({row['time_min']} min)", flush=True)
    log.close()


if __name__ == "__main__":
    main()
