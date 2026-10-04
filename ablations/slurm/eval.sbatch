#!/bin/bash
#SBATCH --job-name=rosettes_eval
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/disk/guille/mapping-rosettes/outputs/logs/%x_%j.out

# Evaluate one or more detectors (validation tiles + every available mosaic in ablations/mosaics.yaml).
# Usage: sbatch ablations/slurm/eval.sbatch <regime> <name>=<model spec> [<name>=<model spec> ...]
#   regime: mixed | lowres  (its val split selects the operating threshold)
#   spec:   ultralytics:<weights.pt> | zeroshot:<arch> | run:<run_dir>[:last]
set -euo pipefail
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate /disk/envs/rosettes
export HF_HOME=/disk/cache/huggingface TORCH_HOME=/disk/cache/torch MPLCONFIGDIR=/disk/guille/.tmp \
       YOLO_CONFIG_DIR=/disk/guille/.config/ultralytics

REGIME=$1; shift
DATA_ROOT=/disk/guille/mapping-rosettes/data
case $REGIME in
    mixed)  DATA=$DATA_ROOT/kaggle_export/mixed-resolution-frailejones/dataset/data.yaml ;;
    lowres) DATA=$DATA_ROOT/kaggle_export/low-resolution-frailejones-detection/dataset/data.yaml ;;
    *) echo "unknown regime $REGIME"; exit 1 ;;
esac

cd /home/guille/Mapping-rosettes
for item in "$@"; do
    NAME=${item%%=*}; SPEC=${item#*=}
    echo "=== $NAME ($SPEC)"
    python -m ablations.evaluate_model --model "$SPEC" --data $DATA --out outputs/eval_v2/$REGIME/$NAME
done
