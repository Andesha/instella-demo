#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$HOME/scratch/instella-demo}
source "$PROJECT_ROOT/scripts/versions.env"
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

# Download (or resume) the selected public checkpoint during setup. No GPU is
# needed, so this is safe to run on the login node.
model_name=${MODEL_ID//\//--}
apptainer exec \
  --bind "$DATA_ROOT:/demo-data" \
  --env HF_HOME=/demo-data/huggingface \
  "$image" python - "$MODEL_ID" "/demo-data/models/$model_name" <<'PY'
import sys
from huggingface_hub import snapshot_download
model, destination = sys.argv[1:]
print(f"Model ready at {snapshot_download(repo_id=model, local_dir=destination)}")
PY

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
