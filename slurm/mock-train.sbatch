#!/bin/bash
#SBATCH --job-name=instella-mock-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:mi300a:4
#SBATCH --mem=450G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Keep scheduler policy here; the separate runner documents how and why AMD's
# upstream recipe is adapted for this small Nibi demonstration.
export PROJECT_ROOT=${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
export DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT}
exec "$PROJECT_ROOT/scripts/run-mock-training.sh"
