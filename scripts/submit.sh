#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/submit.sh [--reservation NAME] [--account ACCOUNT] [--data-root PATH]

Submit the complete demo as a dependency chain. A reservation is optional;
without one, jobs enter the normal MI300A queue.

Examples:
  scripts/submit.sh
  scripts/submit.sh --reservation TylerJobs
EOF
}

reservation=
account=cc-debug
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT}
while (($#)); do
  case "$1" in
    --reservation) reservation=${2:?reservation name required}; shift 2 ;;
    --account) account=${2:?account name required}; shift 2 ;;
    --data-root) DATA_ROOT=${2:?path required}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -r "$DATA_ROOT/containers/rocm-megatron-lm-v25.8-py310.sif" ]] || {
  echo "Container not found under $DATA_ROOT; run scripts/bootstrap.sh first." >&2
  exit 1
}

args=(--parsable --account="$account" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,DATA_ROOT=$DATA_ROOT")
if [[ -n "$reservation" ]]; then
  scontrol show reservation "$reservation" >/dev/null
  args+=(--reservation="$reservation")
fi

submit() {
  local dependency=$1 file=$2 job
  local extra=()
  [[ -n "$dependency" ]] && extra+=(--dependency="afterok:$dependency")
  job=$(sbatch "${args[@]}" "${extra[@]}" "$PROJECT_ROOT/slurm/$file")
  printf '%s' "$job"
}

inference=$(submit '' inference.sbatch); echo "inference: $inference"
training=$(submit "$inference" mock-train.sbatch); echo "training:  $training"
echo "Submitted successfully. Monitor with: squeue -u $USER"
