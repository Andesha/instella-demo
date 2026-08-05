#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT}
: "${SLURM_TMPDIR:?Bootstrap must run inside a Slurm allocation; use bootstrap-batch.sh or bootstrap-interactive.sh}"

# Upstream inputs for this demo. Edit these constants to update the source,
# containers, or checkpoint used by bootstrap.
INSTELLA_REPO=https://github.com/AMD-AGI/Instella-MoE.git
# Pinned because configs/mock-train.patch targets this exact upstream recipe.
INSTELLA_REF=57cc9411bca170d190acacbde62096948d18023f
TRAIN_CONTAINER_URI=docker://rocm/megatron-lm:v25.8_py310
TRAIN_CONTAINER_NAME=rocm-megatron-lm-v25.8-py310.sif
INFER_CONTAINER_URI=docker://rlsys/miles:rocm7-mi300-sglang0.5.9-te2.10.0-dev-307b5e86
INFER_CONTAINER_NAME=miles-rocm7-mi300-sglang0.5.9.sif
MODEL_ID=amd/Instella-MoE-16B-A3B-SFT

mkdir -p "$DATA_ROOT"/{containers,huggingface,models,outputs,logs} "$PROJECT_ROOT/vendor"

# OCI extraction performs huge numbers of metadata operations. Keep the cache,
# temporary rootfs, and initial SIF on fast node-local storage, then copy only
# each completed image back to scratch.
export APPTAINER_CACHEDIR="$SLURM_TMPDIR/apptainer-cache"
export APPTAINER_TMPDIR="$SLURM_TMPDIR/apptainer-tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if [[ ! -d "$PROJECT_ROOT/vendor/Instella-MoE/.git" ]]; then
  git clone --recurse-submodules "$INSTELLA_REPO" "$PROJECT_ROOT/vendor/Instella-MoE"
fi
git -C "$PROJECT_ROOT/vendor/Instella-MoE" fetch --tags origin
git -C "$PROJECT_ROOT/vendor/Instella-MoE" checkout "$INSTELLA_REF"
git -C "$PROJECT_ROOT/vendor/Instella-MoE" submodule update --init --recursive

module load apptainer/1.4.5

build_image() {
  local uri=$1 name=$2 image="$DATA_ROOT/containers/$2"
  [[ -f "$image" ]] && return

  local local_image="$SLURM_TMPDIR/$name"
  # Preserve the node-local layer cache across transient registry errors.
  for attempt in 1 2 3; do
    rm -f "$local_image"
    if apptainer pull "$local_image" "$uri"; then
      break
    fi
    ((attempt < 3)) || { echo "OCI pull failed after 3 attempts: $uri" >&2; exit 1; }
    echo "OCI pull failed (attempt $attempt/3); retrying in 30 seconds" >&2
    sleep 30
  done
  cp "$local_image" "$image.partial"
  mv "$image.partial" "$image"
}

# Fail quickly if the allocated node cannot reach Docker Hub's auth service.
python3 - <<'PY'
import json
import urllib.request

url = "https://auth.docker.io/token?service=registry.docker.io"
try:
    with urllib.request.urlopen(url, timeout=20) as response:
        if not json.load(response).get("token"):
            raise RuntimeError("Docker Hub returned no registry token")
except Exception as error:
    raise SystemExit(f"Docker Hub network check failed: {error}")
print("Docker Hub network check passed")
PY

build_image "$TRAIN_CONTAINER_URI" "$TRAIN_CONTAINER_NAME"
build_image "$INFER_CONTAINER_URI" "$INFER_CONTAINER_NAME"
train_image="$DATA_ROOT/containers/$TRAIN_CONTAINER_NAME"
infer_image="$DATA_ROOT/containers/$INFER_CONTAINER_NAME"

# Download (or resume) the selected public checkpoint with the training image.
# Nibi's host CA path does not exist in the Ubuntu-based container, so point
# HTTPS clients at the certificate bundle shipped inside it.
model_name=${MODEL_ID//\//--}
apptainer exec --bind "$DATA_ROOT:/demo-data" \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  "$train_image" hf download "$MODEL_ID" \
  --local-dir "/demo-data/models/$model_name" \
  --cache-dir /demo-data/huggingface
ln -sfn "$model_name" "$DATA_ROOT/models/current"
printf '%s\n' "$MODEL_ID" > "$DATA_ROOT/models/current-model-id"

# Software-only checks. GPU execution is checked in the inference/training jobs.
apptainer exec "$train_image" python - <<'PY'
import torch
print(f"Training container check: torch={torch.__version__}")
PY
apptainer exec "$infer_image" python - <<'PY'
import sglang
print(f"Inference container check: sglang={getattr(sglang, '__version__', 'unknown')}")
PY

cat <<EOF
Bootstrap complete.
Training container: $train_image
Inference container: $infer_image
Model: $DATA_ROOT/models/$model_name
Upstream commit: $(git -C "$PROJECT_ROOT/vendor/Instella-MoE" rev-parse HEAD)

Run the GPU demonstration with:
  $PROJECT_ROOT/scripts/submit.sh
EOF
