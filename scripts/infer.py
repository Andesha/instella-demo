#!/usr/bin/env python3
"""Minimal offline generation through AMD's supported SGLang overlay."""

import os

import sglang as sgl
from transformers import AutoTokenizer


def main() -> None:
    model_path = os.environ.get("MODEL_PATH", "/demo-data/models/current")
    model_id = os.environ.get("MODEL_ID", "amd/Instella-MoE-16B-A3B-SFT")
    prompt = os.environ.get(
        "PROMPT",
        "Explain in three sentences why mixture-of-experts models are computationally efficient.",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )

    # AMD's reference uses eight-way EP/TP/DP. Nibi has four MI300As, and the
    # model's 64 routed experts divide evenly across four expert-parallel ranks.
    engine = sgl.Engine(
        model_path=model_path,
        dp_size=4,
        tp_size=4,
        ep_size=4,
        enable_dp_attention=True,
        dtype="bfloat16",
        cuda_graph_max_bs=1,
        # Avoid first-run graph capture and AITER JIT dependencies; this is a
        # functional portability test rather than a throughput benchmark.
        disable_cuda_graph=True,
        disable_shared_experts_fusion=True,
        disable_radix_cache=True,
        attention_backend="triton",
        mem_fraction_static=0.8,
        # SGLang otherwise fills nearly all remaining memory with a large KV
        # cache. This smoke test needs only one short prompt and response.
        max_total_tokens=1024,
        max_running_requests=4,
        trust_remote_code=True,
    )
    try:
        output = engine.generate(
            [formatted_prompt],
            sampling_params={"temperature": 0, "max_new_tokens": 32},
        )
        print(output[0]["text"])
        print(f"\nmodel={model_id} visible_gpus=4 engine=sglang")
    finally:
        engine.shutdown()


# SGLang starts scheduler workers with multiprocessing "spawn". Without this
# guard, every worker re-runs this file, tries to create another engine, and
# collides with the parent's local ports.
if __name__ == "__main__":
    main()
