"""Shared training recipe for the detection ablation.

Everything except the optimizer is common to all detectors and copied from the
`train_args` stored in the paper's checkpoint (models/yolov11_best.pt, Ultralytics
8.3.168): same tiles, augmentation, input size, batch, number of iterations,
warmup, LR decay shape, EMA, AMP and checkpoint selection. The optimizer and base
LR follow each architecture's reference implementation, because a single setting
(e.g. AdamW 2e-3) is known to break some of them (DETR-style models).
"""

TRAIN = dict(
    epochs=200,          # 200 x ceil(1792 / 16) = 22,400 iterations of 16 images for every model
    batch=16,
    imgsz=512,
    seed=123,
    close_mosaic=10,     # last 10 epochs without mosaic
    warmup_epochs=3.0,   # Ultralytics: max(3 * iters_per_epoch, 100) warmup iterations
    lrf=0.01,            # linear decay to 1% of the initial LR
)

AUG = dict(
    mosaic=1.0, mixup=0.0, copy_paste=0.0, cutmix=0.0,
    degrees=180.0, translate=0.0, scale=0.5, shear=5.0, perspective=0.0005,
    fliplr=0.5, flipud=0.0,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
)

# Ultralytics models keep their own recipe from the paper: optimizer='auto' resolves to
# AdamW(lr=0.002, betas=(0.9, 0.999)) for this dataset size, nominal batch 64 (accumulate 4).
# Exception: RT-DETR-L diverges to NaN with that LR under AMP, so it uses the RT-DETR reference
# optimizer (OPTIM["rtdetr_r18"]: AdamW 1e-4, wd 1e-4), like the Hugging Face RT-DETR-R18.
ULTRALYTICS = dict(optimizer="auto", lr0=0.01, momentum=0.937, weight_decay=0.0005, nbs=64,
                   box=7.5, cls=0.1, dfl=0.1)  # YOLO loss gains used in the paper

# Reference recipes for the other families (batch 16, no accumulation).
OPTIM = {
    # torchvision references/detection: SGD 0.02 (Faster R-CNN), 0.01 (RetinaNet), wd 1e-4
    "frcnn_mnv3": dict(name="SGD", lr=0.02, momentum=0.9, weight_decay=1e-4, clip=10.0),
    "frcnn_r50": dict(name="SGD", lr=0.02, momentum=0.9, weight_decay=1e-4, clip=10.0),
    "deepforest": dict(name="SGD", lr=0.01, momentum=0.9, weight_decay=1e-4, clip=10.0),
    # RT-DETR official (lyuwenyu/RT-DETR): AdamW 1e-4, backbone 1e-5, wd 1e-4, grad clip 0.1
    "rtdetr_r18": dict(name="AdamW", lr=1e-4, backbone_lr=1e-5, weight_decay=1e-4, clip=0.1),
}
