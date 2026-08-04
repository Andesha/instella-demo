#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$HOME/scratch/instella-demo}
source "$PROJECT_ROOT/scripts/versions.env"
mkdir -p "$DATA_ROOT"/{containers,huggingface,outputs,logs} "$PROJECT_ROOT/vendor"

if [[ ! -d "$PROJECT_ROOT/vendor/Instella-MoE/.git" ]]; then
  git clone --recurse-submodules "$INSTELLA_REPO" "$PROJECT_ROOT/vendor/Instella-MoE"
fi
git -C "$PROJECT_ROOT/vendor/Instella-MoE" fetch --tags origin
git -C "$PROJECT_ROOT/vendor/Instella-MoE" checkout "$INSTELLA_REF"
git -C "$PROJECT_ROOT/vendor/Instella-MoE" submodule update --init --recursive

module load apptainer/1.4.5
image="$DATA_ROOT/containers/$CONTAINER_NAME"
[[ -f "$image" ]] || apptainer pull "$image" "$CONTAINER_URI"

cat <<EOF
Bootstrap complete. Before submitting jobs, run:
  export PROJECT_ROOT=$PROJECT_ROOT
  export DATA_ROOT=$DATA_ROOT
Container: $image
Upstream commit: $(git -C "$PROJECT_ROOT/vendor/Instella-MoE" rev-parse HEAD)
EOF
