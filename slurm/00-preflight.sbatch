#!/bin/bash
#SBATCH --job-name=instella-preflight
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:mi300a:4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out

set -euo pipefail
PROJECT_ROOT=${PROJECT_ROOT:-$HOME/instella-demo}
DATA_ROOT=${DATA_ROOT:-$HOME/scratch/instella-demo}
source "$PROJECT_ROOT/scripts/versions.env"
module load apptainer/1.4.5
image="$DATA_ROOT/containers/$CONTAINER_NAME"
[[ -r "$image" ]] || { echo "Missing $image; run scripts/bootstrap.sh" >&2; exit 1; }

export HSA_XNACK=1
apptainer exec --rocm "$image" bash -lc '
  echo "host=$(hostname) ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
  rocminfo | grep -E "^[[:space:]]*Name:[[:space:]]+gfx" || true
  python - <<"PY"
import torch
print("torch", torch.__version__, "hip", torch.version.hip)
print("visible GPUs", torch.cuda.device_count())
for i in range(torch.cuda.device_count()): print(i, torch.cuda.get_device_name(i))
assert torch.cuda.device_count() == 4
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul checksum", (x @ x).float().mean().item())
PY
'
