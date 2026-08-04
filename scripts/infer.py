#!/usr/bin/env python3
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

visible_gpus = torch.cuda.device_count()
if visible_gpus != 4:
    raise RuntimeError(f"Expected 4 allocated MI300A GPUs, found {visible_gpus}")

model_id = os.environ.get("MODEL_ID", "amd/Instella-MoE-16B-A3B-SFT")
model_path = os.environ.get("MODEL_PATH", model_id)
prompt = os.environ.get("PROMPT", "Explain in three sentences why mixture-of-experts models are computationally efficient.")

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
with torch.inference_mode():
    output = model.generate(inputs, max_new_tokens=192, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
print(f"\nmodel={model_id} visible_gpus={visible_gpus}")
