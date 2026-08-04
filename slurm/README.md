# Slurm jobs

Run these jobs in numeric order, or use `scripts/submit.sh` to create an automatic dependency chain. The wrapper accepts an optional reservation:

```bash
scripts/submit.sh --reservation NAME
```

Without that option, jobs use the normal MI300A queue. No reservation or node name is hard-coded.

| Job | Purpose | Expected result |
|---|---|---|
| `00-preflight.sbatch` | ROCm and BF16 test | four GPUs and a matmul checksum |
| `10-download.sbatch` | cache released SFT weights | model under `$DATA_ROOT/models` |
| `20-inference.sbatch` | Transformers generation | generated response in job output |
| `30-mock-train.sbatch` | 10 iterations of AMD's recipe | losses and checkpoints under `$DATA_ROOT/outputs` |

Slurm writes scheduler output in the submission directory. The training application's detailed log is written under `$DATA_ROOT/outputs`.
