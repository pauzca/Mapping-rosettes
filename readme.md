# Mapping rosettes: detection ablation

Code to reproduce the detection ablation of the paper *Mapping post-fire mortality of paramo giant rosette
plants using artificial intelligence on multispectral drone imagery*: eight detectors (YOLOv11s, YOLOv8s,
YOLOv11m, RT-DETR-L, RT-DETR-R18, Faster R-CNN with MobileNetV3 and ResNet-50 backbones, DeepForest), each
trained on two datasets (mixed resolution, and only low resolution `*`), evaluated per tile on the test
set and per orthomosaic on Diciembre 1 cm and 3.6 cm.

## Reproduce the tables

Open [`reproduce_detection_ablation.ipynb`](reproduce_detection_ablation.ipynb) in Google Colab with a GPU and run
all cells. It installs the exact environment (`requirements.txt`), downloads the Kaggle datasets
`paulauzcategui/mixed-resolution-frailejones` and `paulauzcategui/low-resolution-frailejones-detection`
(tiles, checkpoints and orthomosaics), evaluates the 16 checkpoints with their fixed validation-selected
thresholds (`ablations/thresholds.json`) and checks every value against `ablations/expected_results.json`.

## Contents

| Path | What it is |
|---|---|
| `ablations/evaluate_model.py` | evaluates one detector on the test tiles and the orthomosaics |
| `ablations/models.py` | one interface for Ultralytics, torchvision, Hugging Face and DeepForest detectors |
| `ablations/val_tiles.py`, `tile_metrics.py` | tile-level metrics (P / R / F1, Ultralytics-equivalent mAP) |
| `ablations/detect.py`, `evaluate_detection.py`, `geo.py` | mosaic-level protocol (geodataset tiles, global NMS, RP / RR / RF1) |
| `ablations/tables.py`, `check.py` | LaTeX tables and exact comparison with the published values |
| `ablations/train_ultralytics.py`, `train_torch.py`, `recipe.py` | training with the shared recipe (same data, augmentation and iterations for every model) |
| `ablations/make_splits.py`, `export_kaggle.py`, `splits.csv` | how the train / val / test splits and the Kaggle datasets were built |
| `ablations/benchmark.py`, `benchmark_t4.json` | parameters and throughput (Tesla T4) |
| `mosaics_val/` | ground truth of the Diciembre 1 cm and 3.6 cm orthomosaics |

## Splits

Train and val come from the original training tiles: ~20% of the plants of each source orthomosaic
(Agosto, Diciembre 1 cm, Diciembre 3.6 cm) moved to val as 8 x 8-tile spatial blocks, and training tiles
overlapping val removed. Test holds tiles of the evaluation orthomosaics (Agosto restricted to its annotated
area, Diciembre 1 cm and 3.6 cm) plus Junio 2025. Checkpoints are selected on val (best mAP50-95) and the
score threshold of each model is the F1-optimal one on val.

## Retraining

Training is not part of the notebook (about 55 GPU-hours on a Tesla T4). With the datasets downloaded:

```bash
python -m ablations.train_ultralytics --weights yolo11s.pt --data <dataset>/data.yaml --out outputs/train/mixed/yolo11s
python -m ablations.train_torch --arch frcnn_r50 --data <dataset>/data.yaml --out outputs/train/mixed/frcnn_r50
```

`--weights` takes the COCO checkpoints `yolo11s.pt`, `yolov8s.pt`, `yolo11m.pt` or `rtdetr-l.pt`; `--arch` takes
`rtdetr_r18`, `frcnn_mnv3`, `frcnn_r50` or `deepforest`. Retrained models will differ slightly from the
published checkpoints (GPU non-determinism).
