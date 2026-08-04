#!/bin/bash
#SBATCH --job-name=instella-download
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:mi300a:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$HOME/instella-demo}
DATA_ROOT=${DATA_ROOT:-$HOME/scratch/instella-demo}
source "$PROJECT_ROOT/scripts/versions.env"
module load apptainer/1.4.5
mkdir -p "$DATA_ROOT"/{huggingface,models,logs}
image="$DATA_ROOT/containers/$CONTAINER_NAME"
model_dir="$DATA_ROOT/models/${MODEL_ID//\//--}"

apptainer exec --rocm \
  --bind "$DATA_ROOT:/demo-data" \
  --env HF_HOME=/demo-data/huggingface \
  "$image" python - "$MODEL_ID" "/demo-data/models/${MODEL_ID//\//--}" <<'PY'
import sys
from huggingface_hub import snapshot_download
model, destination = sys.argv[1:]
path = snapshot_download(repo_id=model, local_dir=destination)
print(f"Downloaded {model} to {path}")
PY

echo "Model bytes: $(du -sb "$model_dir" | cut -f1)"
