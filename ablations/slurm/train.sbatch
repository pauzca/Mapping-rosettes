#!/bin/bash
#SBATCH --job-name=rosettes_train
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=23G
#SBATCH --time=12:00:00
#SBATCH --output=/disk/guille/mapping-rosettes/outputs/logs/%x_%j.out

# Fine-tune one detector with the shared recipe (ablations/recipe.py).
# Usage: sbatch --job-name=tr_<model>_<regime> ablations/slurm/train.sbatch <model> <regime> [seed] [extra args]
#   model:  yolo11s | yolov8s | yolo11m | rtdetr_l | rtdetr_r18 | frcnn_mnv3 | frcnn_r50 | deepforest
#   regime: mixed  (original resolutions, main models of the paper)
#           lowres (agosto/dic1cm degraded to low resolution, models marked * in the paper)
# OUT_ROOT (env) overrides the output root, e.g. for smoke tests.
set -euo pipefail
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /disk/envs/rosettes
export HF_HOME=/disk/cache/huggingface TORCH_HOME=/disk/cache/torch MPLCONFIGDIR=/disk/guille/.tmp \
       YOLO_CONFIG_DIR=/disk/guille/.config/ultralytics

MODEL=$1; REGIME=$2; SEED=${3:-123}
shift 2; if [ $# -gt 0 ]; then shift; fi
DATA_ROOT=/disk/guille/mapping-rosettes/data
case $REGIME in
    mixed)  DATA=$DATA_ROOT/kaggle_export/mixed-resolution-frailejones/dataset/data.yaml ;;
    lowres) DATA=$DATA_ROOT/kaggle_export/low-resolution-frailejones-detection/dataset/data.yaml ;;
    *) echo "unknown regime $REGIME"; exit 1 ;;
esac
PRE=/disk/guille/mapping-rosettes/models/pretrained
OUT=${OUT_ROOT:-/home/guille/Mapping-rosettes/outputs/train_v2}/$REGIME/${MODEL}_s$SEED

cd /home/guille/Mapping-rosettes
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case $MODEL in
    yolo11s|yolov8s|yolo11m) python -m ablations.train_ultralytics --weights $PRE/$MODEL.pt --data $DATA --out $OUT --seed $SEED "$@" ;;
    rtdetr_l) python -m ablations.train_ultralytics --weights $PRE/rtdetr-l.pt --data $DATA --out $OUT --seed $SEED "$@" ;;
    *) python -m ablations.train_torch --arch $MODEL --data $DATA --out $OUT --seed $SEED "$@" ;;
esac
