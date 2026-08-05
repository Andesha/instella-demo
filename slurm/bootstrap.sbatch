#!/bin/bash
#SBATCH --job-name=instella-bootstrap
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:mi300a:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
export PROJECT_ROOT=${PROJECT_ROOT:?PROJECT_ROOT must be exported by bootstrap-batch.sh}
export DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT}
exec "$PROJECT_ROOT/scripts/bootstrap.sh"
