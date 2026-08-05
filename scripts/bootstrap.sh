#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT}
: "${SLURM_TMPDIR:?Bootstrap must run inside a Slurm allocation; use bootstrap-batch.sh or bootstrap-interactive.sh}"

# Upstream inputs for this demo. Edit these constants to update the source,
# container, or checkpoint used by bootstrap.
INSTELLA_REPO=https://github.com/AMD-AGI/Instella-MoE.git
# Pinned because configs/mock-train.patch targets this exact upstream recipe.
INSTELLA_REF=57cc9411bca170d190acacbde62096948d18023f
CONTAINER_URI=docker://rocm/megatron-lm:v25.8_py310
CONTAINER_NAME=rocm-megatron-lm-v25.8-py310.sif
MODEL_ID=amd/Instella-MoE-16B-A3B-SFT
# The checkpoint's remote model code declares this newer Transformers release.
TRANSFORMERS_VERSION=4.57.1

mkdir -p "$DATA_ROOT"/{containers,huggingface,models,outputs,logs} "$PROJECT_ROOT/vendor"

# OCI extraction performs huge numbers of metadata operations. Keep the cache,
# temporary rootfs, and initial SIF on fast node-local storage, then copy only
# the completed image back to scratch.
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
image="$DATA_ROOT/containers/$CONTAINER_NAME"
if [[ ! -f "$image" ]]; then
  # Fail quickly with a useful message instead of letting an OCI pull appear to
  # hang when the compute node cannot reach Docker Hub's registry/auth service.
  python3 - <<'PY'
import json
import urllib.request

url = (
    "https://auth.docker.io/token?service=registry.docker.io"
    "&scope=repository:rocm/megatron-lm:pull"
)
try:
    with urllib.request.urlopen(url, timeout=20) as response:
        if not json.load(response).get("token"):
            raise RuntimeError("Docker Hub returned no registry token")
except Exception as error:
    raise SystemExit(f"Docker Hub network check failed: {error}")
print("Docker Hub network check passed")
PY

  local_image="$SLURM_TMPDIR/$CONTAINER_NAME"
  # Preserve the node-local layer cache across a few transient registry errors.
  for attempt in 1 2 3; do
    rm -f "$local_image"
    if apptainer pull "$local_image" "$CONTAINER_URI"; then
      break
    fi
    ((attempt < 3)) || { echo "OCI pull failed after 3 attempts" >&2; exit 1; }
    echo "OCI pull failed (attempt $attempt/3); retrying in 30 seconds" >&2
    sleep 30
  done
  cp "$local_image" "$image.partial"
  mv "$image.partial" "$image"
fi

# Download (or resume) the selected public checkpoint. This does not need a GPU.
model_name=${MODEL_ID//\//--}
# Nibi exports a host CA path that does not exist inside this Ubuntu image.
# Point HTTPS clients at the certificate bundle shipped in the container.
apptainer exec --bind "$DATA_ROOT:/demo-data" \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  "$image" hf download "$MODEL_ID" \
  --local-dir "/demo-data/models/$model_name" \
  --cache-dir /demo-data/huggingface

# The training image ships Transformers 4.46.3, but Instella's model code uses
# APIs from 4.57.1. Install the compatible inference library beside the model
# rather than modifying the immutable SIF or AMD's pinned training environment.
apptainer exec --bind "$DATA_ROOT:/demo-data" \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  "$image" pip install --quiet --no-cache-dir --upgrade \
  --target /demo-data/python "transformers==$TRANSFORMERS_VERSION"

ln -sfn "$model_name" "$DATA_ROOT/models/current"
printf '%s\n' "$MODEL_ID" > "$DATA_ROOT/models/current-model-id"

# Software-only smoke test. GPU visibility is checked inside the inference job,
# where Slurm has actually allocated the devices.
apptainer exec --bind "$DATA_ROOT:/demo-data" \
  --env PYTHONPATH=/demo-data/python "$image" python - <<'PY'
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
