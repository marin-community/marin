# ruff: noqa
#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple vLLM inference test script using Modal for GPU deployment.

Usage:
    # Set up a separate environment for this script
    uv venv ~/.venvs/modal-test

    # Activate it
    source ~/.venvs/modal-test/bin/activate

    # Install modal
    uv pip install modal

    # Setup Modal API key
    modal setup

    # Upload your HuggingFace token to access the gated Llama model
    modal secret create huggingface HF_TOKEN=<your-huggingface-token>

    # Run on Modal cloud
    modal run experiments/exp_vllm_inference_modal.py

Reference: https://modal.com/blog/how-to-deploy-vllm
"""

import modal

# Constants
MINUTES = 60
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = None  # Use latest version, or pin a specific commit hash
N_GPU = 1  # Number of GPUs for tensor parallelism
FAST_BOOT = True  # Set False for full JIT compilation + CUDA graphs (slower start, faster inference)
TEMPERATURE = 1.0
MAX_OUTPUT_TOKENS = 1024
GPU_TYPE = "H100"


# Create Modal Image with vLLM and dependencies
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install(
        "vllm==0.6.6.post1",
        "huggingface_hub[hf_transfer]==0.27.0",
        "torch==2.5.1",
        "transformers>=4.45.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

# Create Modal Volumes for caching
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# Create Modal App
app = modal.App("marin-vllm-inference")


def get_stop_tokens(model_type: str) -> list[str]:
    """Return stop tokens based on model type."""
    if model_type == "llama":
        return ["<|eot_id|>"]
    elif model_type == "qwen":
        return ["<|im_end|>"]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


@app.function(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:1",
    scaledown_window=600,  # Keep warm for 10 minutes
    timeout=30 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],  # HF_TOKEN for gated models
)
def run_inference(prompts: list[str]) -> list[list[str]]:
    """
    Run vLLM inference on GPU via Modal.

    Args:
        prompts: List of prompts to generate completions for.

    Returns:
        List of [prompt, response] pairs.
    """
    from vllm import LLM, SamplingParams
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    model_type = "llama"
    max_input_tokens = 4096

    logger.info(f"Initializing vLLM with model: {MODEL_NAME}")

    llm_kwargs = {
        "model": MODEL_NAME,
        "max_model_len": max_input_tokens + MAX_OUTPUT_TOKENS,
        "tensor_parallel_size": N_GPU,
        "gpu_memory_utilization": 0.90,
    }

    if MODEL_REVISION:
        llm_kwargs["revision"] = MODEL_REVISION

    if FAST_BOOT:
        llm_kwargs["enforce_eager"] = True

    llm = LLM(**llm_kwargs)

    logger.info(f"Running inference on {len(prompts)} prompts...")

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        n=1,
        max_tokens=MAX_OUTPUT_TOKENS,
        stop=get_stop_tokens(model_type),
        include_stop_str_in_output=True,
        logprobs=1,
    )

    logger.info(f"{sampling_params.temperature=}")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        generated_text = output.outputs[0].text
        results.append([output.prompt, generated_text])

    return results


@app.local_entrypoint()
def main():
    """Main entrypoint when running with `modal run`."""
    import json
    from math500_prompts import PROMPTS  # Import locally

    print("Starting Modal vLLM inference...")
    print(f"Model: {MODEL_NAME}")
    print(f"Number of prompts: {len(PROMPTS)}")
    print()

    # Run inference on Modal
    results = run_inference.remote(PROMPTS)

    # Format output and save to file
    output_data = {"columns": ["prompt", "response"], "data": results}
    output_path = f"temp_{TEMPERATURE}_{MAX_OUTPUT_TOKENS}_{GPU_TYPE}.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n=== Inference Complete ===")
    print(f"Generated {len(results)} responses")
    print(f"Results saved to: {output_path}")
