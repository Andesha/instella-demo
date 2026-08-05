#!/usr/bin/env python3
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

visible_gpus = torch.cuda.device_count()
if visible_gpus != 1:
    raise RuntimeError(f"Expected 1 allocated MI300A GPU, found {visible_gpus}")

model_id = os.environ.get("MODEL_ID", "amd/Instella-MoE-16B-A3B-SFT")
model_path = os.environ.get("MODEL_PATH", model_id)
prompt = os.environ.get(
    "PROMPT",
    "Explain in three sentences why mixture-of-experts models are computationally efficient.",
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path, trust_remote_code=True, local_files_only=True
)
# The 16B BF16 checkpoint fits in one MI300A's unified memory. Keeping the
# smoke test on one device avoids the very slow cross-device path produced by
# Accelerate's automatic placement for this custom MoE architecture.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
)
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
print(f"\nmodel={model_id} visible_gpus={visible_gpus}")
