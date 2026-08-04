# Instella-MoE on Nibi

A reproducible demonstration that AMD's fully open **Instella-MoE 16B-A3B** model can run and train on Nibi's AMD Instinct MI300A nodes.

Instella-MoE has 16B total parameters, 2.8B active parameters/token, and 64 experts. This project provides two bounded demonstrations rather than attempting to reproduce AMD's 7.1T-token pretraining run:

1. download a released checkpoint and run an inference smoke test;
2. run 10 iterations of AMD's official mock-data training recipe on four MI300A GPUs.

## Quick start

On a Nibi login node:

```bash
git clone <this-repository-url>
cd instella-demo

# Optional: choose storage with at least 100 GB free.
export DATA_ROOT="$HOME/scratch/instella-demo"

# Fetch AMD's source, build its ROCm Apptainer SIF, and download the model.
bash scripts/bootstrap.sh

# Normal scheduler operation—no reservation required.
bash scripts/submit.sh

# Or submit the same workflow against an active reservation.
bash scripts/submit.sh --reservation TylerJobs
```

`bootstrap.sh` performs all non-GPU setup on the login node, including a container import check. Downloads are resumable. The demo checkpoint is intentionally hard-coded: change `MODEL_ID` near the top of `scripts/bootstrap.sh` to use another released checkpoint, then rerun bootstrap. `submit.sh` then passes the repository and data paths to Slurm and submits inference followed by mock training as an `afterok` dependency chain. The inference script also verifies that Slurm exposed all four GPUs, replacing the need for a separate preflight job. Therefore the checkout can live anywhere; users do not need to edit job files. Use `--data-root PATH` instead of exporting `DATA_ROOT` if preferred.

Monitor the workflow with:

```bash
squeue -u "$USER"
tail -f instella-*.out
```

Successful evidence is generated text from a process that detected four ROCm devices, followed by loss and checkpoint output from mock training. Slurm writes scheduler output in the directory where `submit.sh` is invoked; the training application also writes its detailed log and checkpoints under `$DATA_ROOT/outputs`.

## Nibi and MI300A notes

The jobs request `gpu:mi300a` from Nibi's `gpubase_bygpu_b1` partition. A full MI300A node has four GPU GRES, 96 CPU cores, and approximately 507 GB of unified CPU/GPU memory. Both inference and training request all four GPUs and 450 GB.

Nibi provides Apptainer. The bootstrap script converts AMD's documented `rocm/megatron-lm:v25.8_py310` image into a SIF, and jobs use `apptainer exec --rocm` to expose allocated AMD devices. Durable models and checkpoints are stored under `$DATA_ROOT`; do not place them in `$SLURM_TMPDIR`, which disappears after a job.

MI300A is an APU with unified memory, while AMD's original large-scale run used MI300X and MI325X GPUs. This demo establishes functional portability, not equivalent scale or performance. `HSA_XNACK=1` is enabled for retryable page faults on the supported MI300A stack.

Common issues:

- **Image pull or quota failure:** choose a larger filesystem with `DATA_ROOT` and rerun bootstrap.
- **No ROCm devices:** verify the job received its GPU GRES; do not run GPU commands directly on the login node.
- **Out of memory:** confirm no other processes occupy the node and inspect the job output/`rocm-smi`.
- **Reservation rejected:** check its spelling and dates with `scontrol show reservation NAME`. Omit `--reservation` to use the normal queue.
- **Upstream recipe changes:** preserve the upstream commit printed by bootstrap with demo results.

## Repository layout

- [`docs/MODELS.md`](docs/MODELS.md) — released checkpoint selection
- [`scripts/bootstrap.sh`](scripts/bootstrap.sh) — clone the official recipe and pull its container
- [`scripts/submit.sh`](scripts/submit.sh) — submit the workflow, optionally using a reservation
- [`scripts/infer.py`](scripts/infer.py) — minimal Transformers smoke test
- [`slurm/`](slurm/) — independently runnable Slurm jobs

Each `.sbatch` file may also be submitted directly after bootstrap. `inference.sbatch` fails early unless Slurm exposes all four GPUs, so no separate preflight allocation is needed. Export `PROJECT_ROOT` and `DATA_ROOT` when submitting directly if the checkout and data are not at their defaults. Neither job hard-codes a reservation or node name.

## Scope and cautions

- Released weights are research-only under AMD's ResearchRAIL license.
- Hugging Face model code uses `trust_remote_code=True`; pin and review revisions for production use.
- Full pretraining, SFT, DPO, and RL require curated datasets and substantially more compute.
- The upstream repository, revision, container, and model constants are documented near the top of `scripts/bootstrap.sh`.

## Upstream references

- [Instella-MoE model collection](https://huggingface.co/collections/amd/instella-moe)
- [AMD training code and recipes](https://github.com/AMD-AGI/Instella-MoE)
- [AMD technical blog](https://rocm.blogs.amd.com/artificial-intelligence/instella-moe/README.html)
