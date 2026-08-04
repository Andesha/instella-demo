# Slurm jobs

Use `scripts/submit.sh` after bootstrap to run these jobs in order. They can also be submitted individually with `sbatch`:

- `inference.sbatch` — verify the four GPUs and generate text
- `mock-train.sbatch` — run the bounded mock-training demonstration

See the main [`README.md`](../README.md) for setup, reservations, storage, and monitoring.
