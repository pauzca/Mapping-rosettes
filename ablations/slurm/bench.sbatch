#!/bin/bash
#SBATCH --job-name=rosettes_bench
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/disk/guille/mapping-rosettes/outputs/logs/%x_%j.out

# Inference cost (params, GFLOPs, ms/tile on the T4) of the fine-tuned detectors of a regime.
# Usage: sbatch ablations/slurm/bench.sbatch [regime]
set -euo pipefail
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /disk/envs/rosettes
export HF_HOME=/disk/cache/huggingface TORCH_HOME=/disk/cache/torch MPLCONFIGDIR=/disk/guille/.tmp \
       YOLO_CONFIG_DIR=/disk/guille/.config/ultralytics

REGIME=${1:-mixed}
T=outputs/train/$REGIME
cd /home/guille/Mapping-rosettes
python -m ablations.benchmark --out outputs/eval/$REGIME/benchmark.json \
    yolo11s=ultralytics:$T/yolo11s_s123/weights/last.pt \
    yolov8s=ultralytics:$T/yolov8s_s123/weights/last.pt \
    yolo11m=ultralytics:$T/yolo11m_s123/weights/last.pt \
    rtdetr_l=ultralytics:$T/rtdetr_l_s123/weights/last.pt \
    rtdetr_r18=run:$T/rtdetr_r18_s123:last \
    frcnn_mnv3=run:$T/frcnn_mnv3_s123:last \
    frcnn_r50=run:$T/frcnn_r50_s123:last \
    deepforest=run:$T/deepforest_s123:last
