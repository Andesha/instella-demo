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
args=(
  --job-name=instella-bootstrap
  --account="$account"
  --partition=gpubase_bygpu_b1
  --nodes=1
  --ntasks=1
  --cpus-per-task=16
  --gres=gpu:mi300a:1
  --mem=64G
  --time=04:00:00
)
if [[ -n "$reservation" ]]; then
  scontrol show reservation "$reservation" >/dev/null
  args+=(--reservation="$reservation")
fi

export PROJECT_ROOT DATA_ROOT
salloc "${args[@]}" srun --ntasks=1 "$PROJECT_ROOT/scripts/bootstrap.sh"
