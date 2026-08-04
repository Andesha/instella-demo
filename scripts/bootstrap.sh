#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$HOME/scratch/instella-demo}

# Upstream inputs for this demo. Edit these constants to update the source,
# container, or checkpoint used by bootstrap.
INSTELLA_REPO=https://github.com/AMD-AGI/Instella-MoE.git
INSTELLA_REF=main
CONTAINER_URI=docker://rocm/megatron-lm:v25.8_py310
CONTAINER_NAME=rocm-megatron-lm-v25.8-py310.sif
MODEL_ID=amd/Instella-MoE-16B-A3B-SFT

mkdir -p "$DATA_ROOT"/{containers,huggingface,models,outputs,logs} "$PROJECT_ROOT/vendor"

if [[ ! -d "$PROJECT_ROOT/vendor/Instella-MoE/.git" ]]; then
  git clone --recurse-submodules "$INSTELLA_REPO" "$PROJECT_ROOT/vendor/Instella-MoE"
fi
git -C "$PROJECT_ROOT/vendor/Instella-MoE" fetch --tags origin
git -C "$PROJECT_ROOT/vendor/Instella-MoE" checkout "$INSTELLA_REF"
git -C "$PROJECT_ROOT/vendor/Instella-MoE" submodule update --init --recursive

module load apptainer/1.4.5
image="$DATA_ROOT/containers/$CONTAINER_NAME"
[[ -f "$image" ]] || apptainer pull "$image" "$CONTAINER_URI"

# Download (or resume) the selected public checkpoint. This does not need a GPU.
model_name=${MODEL_ID//\//--}
apptainer exec --bind "$DATA_ROOT:/demo-data" "$image" \
  huggingface-cli download "$MODEL_ID" \
  --local-dir "/demo-data/models/$model_name" \
  --cache-dir /demo-data/huggingface
ln -sfn "$model_name" "$DATA_ROOT/models/current"
printf '%s\n' "$MODEL_ID" > "$DATA_ROOT/models/current-model-id"

# Software-only smoke test. GPU visibility is checked inside the inference job,
# where Slurm has actually allocated the devices.
apptainer exec "$image" python - <<'PY'
import torch, transformers
print(f"Container check: torch={torch.__version__}, transformers={transformers.__version__}")
PY

cat <<EOF
Bootstrap complete.
Container: $image
Model: $DATA_ROOT/models/$model_name
Upstream commit: $(git -C "$PROJECT_ROOT/vendor/Instella-MoE" rev-parse HEAD)

Run the GPU demonstration with:
  $PROJECT_ROOT/scripts/submit.sh
EOF
