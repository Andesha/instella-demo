#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT must point to the demo checkout}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT}
image="$DATA_ROOT/containers/miles-rocm7-mi300-sglang0.5.9.sif"
model="$DATA_ROOT/models/current"
model_id_file="$DATA_ROOT/models/current-model-id"

[[ -r "$image" && -d "$model" && -r "$model_id_file" ]] || {
  echo "Inference artifacts are missing; rerun bootstrap." >&2
  exit 1
}
module load apptainer/1.4.5
MODEL_ID=$(<"$model_id_file")

# Apptainer's default writable tmpfs is intentionally small, but AITER compiles
# ROCm kernels under /app on first use. Give the ephemeral writable layer enough
# node-local space for those build products.
export APPTAINER_TMPDIR="$SLURM_TMPDIR/apptainer-tmp"
mkdir -p "$APPTAINER_TMPDIR"
overlay="$SLURM_TMPDIR/instella-inference-overlay.img"
apptainer overlay create --size 16384 "$overlay"

# These are AMD's documented Instella SGLang settings. The writable temporary
# layer lets the pinned AMD overlay patch /app/sglang without changing the SIF.
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export GPU_MAX_HW_QUEUES=8
export MODE=FARSKIP_REFERENCE
export FARSKIP_REFERENCE_DECODER_LAYER=1
export HSA_XNACK=1
for name in SGLANG_ROCM_FUSED_DECODE_MLA GPU_MAX_HW_QUEUES MODE \
            FARSKIP_REFERENCE_DECODER_LAYER HSA_XNACK; do
  export "APPTAINERENV_${name}=${!name}"
done

apptainer exec --rocm --overlay "$overlay" \
  --bind "$PROJECT_ROOT:/workspace:ro,$DATA_ROOT:/demo-data" \
  --env MODEL_ID="$MODEL_ID" \
  --env MODEL_PATH=/demo-data/models/current \
  --env HF_HOME=/demo-data/huggingface \
  "$image" bash -lc '
    unset WORLD_SIZE RANK MASTER_ADDR LOCAL_RANK MASTER_PORT
    unset PET_MASTER_ADDR PET_NPROC_PER_NODE
    bash /workspace/vendor/Instella-MoE/inference/scripts/update_sglang_workspace_files.sh \
      /workspace/vendor/Instella-MoE/inference/sglang/python /app/sglang/python
    export PYTHONPATH=/app/sglang/python:${PYTHONPATH:-}
    # The image enables AITER by default. Unset it rather than assigning "0";
    # some SGLang paths treat any non-empty value as enabled.
    unset SGLANG_USE_AITER
    python /workspace/scripts/infer.py
  '
