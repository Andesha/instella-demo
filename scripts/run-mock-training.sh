#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT must point to the demo checkout}
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT}
upstream="$PROJECT_ROOT/vendor/Instella-MoE"
image="$DATA_ROOT/containers/rocm-megatron-lm-v25.8-py310.sif"
source_config="$upstream/training/examples/megatron/configs_instella_moe/instella_moe-mock_pretrain.yaml"
config="$DATA_ROOT/run-configs/mock-${SLURM_JOB_ID}.yaml"

[[ -r "$source_config" && -r "$image" ]] || {
  echo "Bootstrap artifacts are missing; run scripts/bootstrap.sh first." >&2
  exit 1
}
mkdir -p "$DATA_ROOT"/{outputs,huggingface,run-configs}
module load apptainer/1.4.5

# Why: AMD's mock recipe is intentionally production-sized. Work from a copy
# so the vendored source remains pristine, then apply the checked-in patch that
# bounds runtime, batch size, checkpoint frequency, and output location.
cp "$source_config" "$config"
patch "$config" < "$PROJECT_ROOT/configs/mock-train.patch"

# Why: the upstream launcher defaults to eight GPUs and long sequences. These
# values describe Nibi's single four-MI300A node and keep the proof run bounded.
export HSA_XNACK=1
export GPUS_PER_NODE=4 NNODES=1 NODE_RANK=0
export PRIMUS_EP=4 PRIMUS_TP=1 PRIMUS_PP=1
export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-512}
export PRIMUS_MAX_POSITION_EMBEDDINGS=$PRIMUS_SEQ_LENGTH
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

# Why: Apptainer only guarantees explicitly prefixed host variables inside the
# container. Primus reads these values to configure torchrun and parallelism.
for name in HSA_XNACK GPUS_PER_NODE NNODES NODE_RANK PRIMUS_EP PRIMUS_TP PRIMUS_PP \
            PRIMUS_SEQ_LENGTH PRIMUS_MAX_POSITION_EMBEDDINGS MASTER_ADDR MASTER_PORT; do
  export "APPTAINERENV_${name}=${!name}"
done

# Why: --rocm exposes only the GPUs granted by Slurm; the bind keeps source,
# generated configuration, logs, and checkpoints in the scratch checkout.
apptainer exec --rocm \
  --bind "$PROJECT_ROOT:/workspace,$DATA_ROOT:/demo-data" \
  --env HF_HOME=/demo-data/huggingface \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  --env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  --env EXP="/demo-data/run-configs/mock-${SLURM_JOB_ID}.yaml" \
  --env TRAIN_LOG="/demo-data/outputs/mock-train-${SLURM_JOB_ID}.log" \
  "$image" bash -lc \
  'cd /workspace/vendor/Instella-MoE/training && bash examples/run_instella.sh --task pretrain'
