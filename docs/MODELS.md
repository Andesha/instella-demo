# Released checkpoints

All released models have 16B total and 2.8B active parameters/token.

| Checkpoint | Stage | Use in this demo |
|---|---|---|
| `amd/Instella-MoE-16B-A3B-Pretrain` | pretraining | inspect an early checkpoint |
| `amd/Instella-MoE-16B-A3B-Midtrain` | mid-training | continuation experiments |
| `amd/Instella-MoE-16B-A3B-Base` | final long-context base | start custom domain adaptation/SFT |
| `amd/Instella-MoE-16B-A3B-SFT` | supervised fine-tuning | **default lightweight inference demo** |
| `amd/Instella-MoE-16B-A3B-DPO` | preference optimization | aligned checkpoint experiments |
| `amd/Instella-MoE-16B-A3B-Think` | RL | strongest reasoning/chat demonstration |

To select another model, change `MODEL_ID` near the top of `scripts/bootstrap.sh` and rerun bootstrap. For actual fine-tuning, use `Base` as the conventional starting point; the included mock training starts from scratch because its purpose is only to exercise AMD's official training stack.

The weights are roughly tens of GB and the loader may create temporary/cached copies. Budget at least 100 GB. Models are research-only under ResearchRAIL. The upstream card also warns that outputs may be inaccurate, unsafe, or biased.
