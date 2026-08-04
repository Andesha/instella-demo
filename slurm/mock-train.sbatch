#!/bin/bash
#SBATCH --job-name=instella-mock-train
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:mi300a:4
#SBATCH --mem=450G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$HOME/instella-demo}
DATA_ROOT=${DATA_ROOT:-$HOME/scratch/instella-demo}
module load apptainer/1.4.5
image="$DATA_ROOT/containers/rocm-megatron-lm-v25.8-py310.sif"
upstream="$PROJECT_ROOT/vendor/Instella-MoE"
[[ -d "$upstream/training" ]] || { echo "Run scripts/bootstrap.sh first" >&2; exit 1; }
mkdir -p "$DATA_ROOT"/{outputs,huggingface,run-configs}

# Bound AMD's official long-running mock recipe to a demo-sized smoke run.
config="$DATA_ROOT/run-configs/mock-${SLURM_JOB_ID}.yaml"
cp "$upstream/training/examples/megatron/configs_instella_moe/instella_moe-mock_pretrain.yaml" "$config"
sed -i \
  -e 's/train_iters: 443876/train_iters: 10/' \
  -e 's/micro_batch_size: 8/micro_batch_size: 1/' \
  -e 's/global_batch_size: 4096/global_batch_size: 32/' \
  -e 's/save_interval: 1000/save_interval: 5/' \
  -e 's|save: /outputs/path/to/checkpoints/instella_moe_mock_pretrain|save: /demo-data/outputs/instella_moe_mock_pretrain|' \
  -e 's/disable_last_saving: true/disable_last_saving: false/' \
  "$config"

export HSA_XNACK=1
export GPUS_PER_NODE=4 NNODES=1 NODE_RANK=0
export PRIMUS_EP=4 PRIMUS_TP=1 PRIMUS_PP=1
export PRIMUS_SEQ_LENGTH=${PRIMUS_SEQ_LENGTH:-512}
export PRIMUS_MAX_POSITION_EMBEDDINGS=$PRIMUS_SEQ_LENGTH
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
# APPTAINERENV_* is the unambiguous way to pass host values into the container.
for name in HSA_XNACK GPUS_PER_NODE NNODES NODE_RANK PRIMUS_EP PRIMUS_TP PRIMUS_PP \
            PRIMUS_SEQ_LENGTH PRIMUS_MAX_POSITION_EMBEDDINGS MASTER_ADDR MASTER_PORT; do
  export "APPTAINERENV_${name}=${!name}"
done

apptainer exec --rocm \
  --bind "$PROJECT_ROOT:/workspace,$DATA_ROOT:/demo-data" \
  --env HF_HOME=/demo-data/huggingface \
  --env EXP="/demo-data/run-configs/mock-${SLURM_JOB_ID}.yaml" \
  --env TRAIN_LOG="/demo-data/outputs/mock-train-${SLURM_JOB_ID}.log" \
  "$image" bash -lc 'cd /workspace/vendor/Instella-MoE/training && bash examples/run_instella.sh --task pretrain'
