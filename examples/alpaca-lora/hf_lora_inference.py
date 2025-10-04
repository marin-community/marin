# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

# Simple example script to use peft to do inference with an HF model.
import os
import sys

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# from ..alpaca.alpaca import DEFAULT_PROMPT_DICT
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "..", "alpaca"))
import alpaca  # noqa: E402


PROMPT = alpaca.DEFAULT_PROMPT_DICT["prompt_no_input"]


def format_prompt(**kwargs):
    return PROMPT.format(**kwargs)


def main(peft_model_id):
    config = PeftConfig.from_pretrained(peft_model_id)
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto")
    model = PeftModel.from_pretrained(model, peft_model_id, device_map="auto")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # if on macos, set cpu b/c mps doesn't work yet w/ int64.cumsum
    if model.device.type == "mps":
        model.cpu()

    while True:
        msg = input("> ")
        msg = format_prompt(instruction=msg)
        inputs = tokenizer(msg, return_tensors="pt")
        print("... ", end="")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, min_new_tokens=1)
            print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print("Usage: python hf_lora_inference.py [model_name_or_path]")
        sys.exit(1)

    model_name_or_path = sys.argv[1] if len(sys.argv) == 2 else "dlwh/levanter-lora-test"

    main(model_name_or_path)
