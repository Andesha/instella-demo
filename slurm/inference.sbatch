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
source "$PROJECT_ROOT/scripts/versions.env"
module load apptainer/1.4.5
image="$DATA_ROOT/containers/$CONTAINER_NAME"
model_name=${MODEL_ID//\//--}
[[ -d "$DATA_ROOT/models/$model_name" ]] || { echo "Model not found; run scripts/bootstrap.sh first" >&2; exit 1; }
export HSA_XNACK=1

apptainer exec --rocm \
  --bind "$PROJECT_ROOT:/workspace:ro,$DATA_ROOT:/demo-data" \
  --env MODEL_ID="$MODEL_ID" \
  --env MODEL_PATH="/demo-data/models/$model_name" \
  --env HF_HOME=/demo-data/huggingface \
  "$image" python /workspace/scripts/infer.py
