#!/usr/bin/env bash
set -euo pipefail

reservation=
if [[ ${1:-} == --reservation ]]; then
  reservation=${2:?reservation name required}
  shift 2
fi
(($# == 0)) || { echo "Usage: $0 [--reservation NAME]" >&2; exit 2; }

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT}
args=(--export="ALL,PROJECT_ROOT=$PROJECT_ROOT,DATA_ROOT=$DATA_ROOT")
if [[ -n "$reservation" ]]; then
  scontrol show reservation "$reservation" >/dev/null
  args+=(--reservation="$reservation")
fi

job=$(sbatch --parsable "${args[@]}" "$PROJECT_ROOT/slurm/bootstrap.sbatch")
echo "Bootstrap job: $job"
echo "Monitor with: tail -f instella-bootstrap-$job.out"
