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

"""Simple vLLM inference test script."""

import logging

import os
import json
import ray
from fray.cluster import ResourceConfig
from fray.cluster.ray import as_remote_kwargs
from fray.cluster.ray.tpu import run_on_pod_ray
from transformers import AutoTokenizer

from marin.training.training import _add_run_env_variables
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)

from math500_prompts import PROMPTS


def get_stop_tokens(model_type: str) -> list[str]:
    """Get model-specific stop tokens."""
    if model_type == "llama":
        return ["<|eot_id|>"]
    elif model_type == "qwen":
        return ["<|im_end|>"]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_inference(prompts: list[str]):
    """Run inference inside a TPU-allocated task."""
    from vllm import LLM, SamplingParams

    # Model configuration
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_type = "llama"
    max_input_tokens = 4096
    max_output_tokens = 1024

    logger.info(f"Initializing vLLM with model: {model_name}")

    # Initialize vLLM LLM
    llm = LLM(
        model=model_name,
        max_model_len=max_input_tokens + max_output_tokens,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.90,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info(f"Running inference on {len(prompts)} prompts...")

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=0.0,
        n=1,
        max_tokens=max_output_tokens,
        stop=get_stop_tokens(model_type),
        include_stop_str_in_output=True,
        logprobs=1,
    )

    logger.info(f"{sampling_params.temperature=}")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        # Assuming n=1
        generated_text = output.outputs[0].text
        results.append([output.prompt, generated_text])

    return results


def main():
    # Setup environment
    env = {}
    env = _add_run_env_variables(env)
    env["EQX_ON_ERROR"] = "nan"
    # Disable multiprocessing to have direct access to the model weights
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # Skip jax precompile to speed up bootstrap time
    env["SKIP_JAX_PRECOMPILE"] = "1"

    # Configure TPU resources (v5p-8 to match tensor_parallel_size=8)
    tpu_type = "v5p-8"
    inference_resources = ResourceConfig.with_tpu(tpu_type)
    inference_kwargs = dict(max_calls=1, **as_remote_kwargs(inference_resources, env_vars=env))

    # Define remote task with TPU allocation
    @ray.remote(**inference_kwargs)
    def inference_task():
        with remove_tpu_lockfile_on_exit():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
            return run_inference(PROMPTS)

    # Run on TPU pod

    task = run_on_pod_ray.remote(
        inference_task,
        tpu_type,
        num_slices=1,
        max_retries_failure=3,
        max_retries_preemption=100,
    )

    # Wait for completion
    results = ray.get(task)

    output_data = {"columns": ["prompt", "response"], "data": results}
    print(json.dumps(output_data))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
