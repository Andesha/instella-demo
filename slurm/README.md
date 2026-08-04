# Slurm jobs

Model and container setup is handled by `scripts/bootstrap.sh`; it does not require a GPU allocation. After bootstrap, use `scripts/submit.sh` to submit inference and mock training as an automatic dependency chain. The wrapper accepts an optional reservation:

```bash
scripts/submit.sh --reservation NAME
```

Without that option, jobs use the normal MI300A queue. No reservation or node name is hard-coded.

| Job | Purpose | Expected result |
|---|---|---|
| `inference.sbatch` | verify four GPUs and run Transformers generation | generated response in job output |
| `mock-train.sbatch` | 10 iterations of AMD's recipe | losses and checkpoints under `$DATA_ROOT/outputs` |

The inference job fails early if Slurm does not expose four GPUs, so a separate preflight allocation is unnecessary. Each job may also be submitted directly after bootstrap.

Slurm writes scheduler output in the submission directory. The training application's detailed log is written under `$DATA_ROOT/outputs`.
