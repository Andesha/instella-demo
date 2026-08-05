#!/usr/bin/env bash
set -euo pipefail

reservation=
account=cc-debug
while (($#)); do
  case "$1" in
    --reservation) reservation=${2:?reservation name required}; shift 2 ;;
    --account) account=${2:?account name required}; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--reservation NAME] [--account ACCOUNT]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${DATA_ROOT:-$PROJECT_ROOT}
args=(--account="$account" --export="ALL,PROJECT_ROOT=$PROJECT_ROOT,DATA_ROOT=$DATA_ROOT")
if [[ -n "$reservation" ]]; then
  scontrol show reservation "$reservation" >/dev/null
  args+=(--reservation="$reservation")
fi

job=$(sbatch --parsable "${args[@]}" "$PROJECT_ROOT/slurm/bootstrap.sbatch")
echo "Bootstrap job: $job"
echo "Monitor with: tail -f instella-bootstrap-$job.out"
