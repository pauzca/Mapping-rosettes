#!/bin/bash
#SBATCH --job-name=rosettes_3cm
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/disk/guille/mapping-rosettes/outputs/logs/%x_%j.out

# Baseline pipeline at 3 cm: YOLOv11 -> metrics -> SAM2-L / box / ellipse masks.
set -euo pipefail
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /disk/envs/rosettes
export HF_HOME=/disk/cache/huggingface TORCH_HOME=/disk/cache/torch YOLO_CONFIG_DIR=/disk/guille/.config/ultralytics

cd /home/guille/Mapping-rosettes
MOSAIC=data/mosaics/rosettes_gsd_3cm_mosaic.tif
GT=mosaics_val/rosettes_gsd_3cm/rosettes_gsd_3cm_boxes.gpkg
OUT=outputs/3cm/yolov11
mkdir -p $OUT

nvidia-smi --query-gpu=name,memory.total --format=csv

# main setting: resample to 1.36 cm/px as geodataset's ground_resolution in the notebook
python -m ablations.detect --mosaic $MOSAIC --model ultralytics:models/yolov11_best.pt --gsd 0.0136 --out $OUT/detections.gpkg
python -m ablations.evaluate_detection --pred $OUT/detections.gpkg --gt $GT --out $OUT/detection_metrics.json
# reference: native 3.65 cm/px, no resampling
python -m ablations.detect --mosaic $MOSAIC --model ultralytics:models/yolov11_best.pt --gsd 0 --out $OUT/native/detections.gpkg
python -m ablations.evaluate_detection --pred $OUT/native/detections.gpkg --gt $GT --out $OUT/native/detection_metrics.json

for m in sam2_l box ellipse; do
    python -m ablations.segment --mosaic $MOSAIC --boxes $OUT/detections.gpkg --min-score 0.4 \
        --method $m --out $OUT/masks_$m.gpkg
done
