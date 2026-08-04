#!/bin/bash
#SBATCH --job-name=instella-infer
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:mi300a:4
#SBATCH --mem=450G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$HOME/instella-demo}
DATA_ROOT=${DATA_ROOT:-$HOME/scratch/instella-demo}
module load apptainer/1.4.5
image="$DATA_ROOT/containers/rocm-megatron-lm-v25.8-py310.sif"
model="$DATA_ROOT/models/current"
model_id_file="$DATA_ROOT/models/current-model-id"
[[ -d "$model" && -r "$model_id_file" ]] || { echo "Model not found; run scripts/bootstrap.sh first" >&2; exit 1; }
MODEL_ID=$(<"$model_id_file")
export HSA_XNACK=1

apptainer exec --rocm \
  --bind "$PROJECT_ROOT:/workspace:ro,$DATA_ROOT:/demo-data" \
  --env MODEL_ID="$MODEL_ID" \
  --env MODEL_PATH=/demo-data/models/current \
  --env HF_HOME=/demo-data/huggingface \
  "$image" python /workspace/scripts/infer.py
