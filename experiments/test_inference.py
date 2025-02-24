import os
import tempfile

import numpy as np
import ray
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer

from marin.evaluation.utils import download_from_gcs

model_name = "gs://marin-us-central2/checkpoints/scaling-law-suite-default-512-9b1182/hf/step-49999"


@ray.remote(resources={"TPU": 1, "TPU-v6e-8-head": 1})
def test_inference():
    with tempfile.TemporaryDirectory() as temp_dir:
        download_from_gcs(model_name, temp_dir)

        print(os.listdir(temp_dir))

        # Run inference
        device = xm.xla_device()
        tokenizer = AutoTokenizer.from_pretrained(temp_dir)
        model = AutoModelForCausalLM.from_pretrained(temp_dir).to(device)

        text = "Hello, how are you?"
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        outputs = model(**inputs, labels=input_ids)
        loss = outputs.loss.item()
        num_tokens = input_ids.shape[-1]
        num_bytes = len(text.encode("utf-8"))
        bits_per_byte = (num_tokens / num_bytes) * loss / np.log(2)
        print(f"Bits per byte: {bits_per_byte}")


if __name__ == "__main__":
    ray.get(test_inference.remote())
