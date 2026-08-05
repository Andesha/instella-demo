# Instella-MoE on Nibi

A reproducible demonstration that AMD's fully open **Instella-MoE 16B-A3B** model can run and train on Nibi's AMD Instinct MI300A nodes.

Instella-MoE has 16B total parameters, 2.8B active parameters/token, and 64 experts. This project provides two bounded demonstrations rather than attempting to reproduce AMD's 7.1T-token pretraining run:

1. download a released checkpoint and run an inference smoke test;
2. run 10 iterations of AMD's official mock-data training recipe on four MI300A GPUs.

## Quick start

On a Nibi login node, clone the project directly onto scratch (**at least 300 GB free**):

```bash
cd "$SCRATCH"
git clone <this-repository-url> instella-demo
cd instella-demo

# Queue bootstrap, optionally against an active reservation. The command
# prints the log filename; wait for "Bootstrap complete" before continuing.
bash scripts/bootstrap-batch.sh
# bash scripts/bootstrap-batch.sh --reservation TylerJobs

# Then submit inference and mock training with the same scheduling choice.
bash scripts/submit.sh
# bash scripts/submit.sh --reservation TylerJobs
```

By default, persistent artifacts stay inside the scratch checkout: the training and SGLang SIFs under `containers/`, model under `models/`, Hugging Face cache under `huggingface/`, AMD source under `vendor/`, and results under `outputs/`. These generated paths are ignored by Git. The bootstrap allocation performs the container import check and resumable downloads without burdening the login node.

Container conversion runs inside a Slurm allocation. Apptainer's cache, extracted root filesystem, and initial SIF are built under fast node-local `$SLURM_TMPDIR`; only the completed SIF is copied to scratch. For an attended run instead of a batch job, use `bash scripts/bootstrap-interactive.sh [--reservation NAME] [--account ACCOUNT]`. Bootstrap and demo submission default to the `cc-debug` account; pass `--account ACCOUNT` to either wrapper to override it.

The demo checkpoint is intentionally hard-coded: change `MODEL_ID` near the top of `scripts/bootstrap.sh` to use another released checkpoint, then rerun bootstrap. `submit.sh` passes the checkout path to Slurm and submits inference followed by mock training as an `afterok` dependency chain. The inference script verifies its allocated MI300A GPU, replacing the need for a separate preflight job. Use `--data-root PATH` only if generated artifacts should live outside the checkout.

Monitor the workflow with:

```bash
squeue -u "$USER"
tail -f instella-*.out
```

Successful evidence is generated text through AMD's four-MI300A SGLang path, followed by loss and checkpoint output from four-GPU mock training. Slurm writes scheduler output in the directory where `submit.sh` is invoked; the training application also writes its detailed log and checkpoints under `$DATA_ROOT/outputs`.

## Nibi and MI300A notes

The jobs request Nibi's `gpu:mi300a` resources without forcing a partition, allowing Slurm or the selected reservation to route them correctly. A full MI300A node has four GPU GRES, 96 CPU cores, and approximately 507 GB of unified CPU/GPU memory. Both AMD's SGLang inference path and mock training request all four GPUs and 450 GB.

Nibi provides Apptainer. Bootstrap converts AMD's documented Megatron training image and Miles/SGLang inference image into separate SIFs. Jobs use `apptainer exec --rocm` to expose allocated AMD devices. At inference startup, AMD's pinned Instella SGLang/FarSkip files are applied through an ephemeral writable layer; the persisted SIF remains unchanged. Durable models and checkpoints are stored under `$DATA_ROOT`; do not place them in `$SLURM_TMPDIR`, which disappears after a job.

During image conversion, Apptainer may print many `ignoring (usually) harmless EPERM on setxattr "user.rootlesscontainers"` warnings. This is expected when rootless Apptainer cannot restore optional OCI extended attributes on the node-local build filesystem. The warnings can be ignored if bootstrap continues to the container check and prints `Bootstrap complete`.

MI300A is an APU with unified memory, while AMD's original large-scale run used MI300X and MI325X GPUs. This demo establishes functional portability, not equivalent scale or performance. `HSA_XNACK=1` is enabled for retryable page faults on the supported MI300A stack.

## Repository layout

- [`docs/MODELS.md`](docs/MODELS.md) — released checkpoint selection
- [`scripts/bootstrap-batch.sh`](scripts/bootstrap-batch.sh) — queue bootstrap on node-local storage
- [`scripts/bootstrap-interactive.sh`](scripts/bootstrap-interactive.sh) — run the same bootstrap through `salloc`
- [`scripts/bootstrap.sh`](scripts/bootstrap.sh) — shared bootstrap worker
- [`scripts/submit.sh`](scripts/submit.sh) — submit the workflow, optionally using a reservation
- [`scripts/run-inference.sh`](scripts/run-inference.sh) — apply AMD's SGLang overlay and launch inference
- [`scripts/infer.py`](scripts/infer.py) — minimal offline SGLang smoke test
- [`scripts/run-mock-training.sh`](scripts/run-mock-training.sh) — documented adaptation of AMD's mock recipe
- [`configs/mock-train.patch`](configs/mock-train.patch) — bounded changes applied to the upstream recipe
- [`slurm/`](slurm/) — independently runnable Slurm jobs

Each `.sbatch` file may also be submitted directly from the repository root after bootstrap. `inference.sbatch` fails early unless Slurm exposes its requested GPU, so no separate preflight allocation is needed. Neither job hard-codes a reservation or node name.

## Scope and cautions

- Released weights are research-only under AMD's ResearchRAIL license.
- Hugging Face model code uses `trust_remote_code=True`; pin and review revisions for production use.
- Full pretraining, SFT, DPO, and RL require curated datasets and substantially more compute.
- The upstream repository, revision, container, and model constants are documented near the top of `scripts/bootstrap.sh`.

## Upstream references

- [Instella-MoE model collection](https://huggingface.co/collections/amd/instella-moe)
- [AMD training code and recipes](https://github.com/AMD-AGI/Instella-MoE)
- [AMD technical blog](https://rocm.blogs.amd.com/artificial-intelligence/instella-moe/README.html)
