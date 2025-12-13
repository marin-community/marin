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
import ray
from fray.cluster import ResourceConfig
from fray.cluster.ray import as_remote_kwargs
from fray.cluster.ray.tpu import run_on_pod_ray
from transformers import AutoTokenizer

from marin.training.training import _add_run_env_variables
from marin.utils import remove_tpu_lockfile_on_exit

# Disable multiprocessing to have direct access to the model weights
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# Skip jax precompile to speed up bootstrap time
os.environ["SKIP_JAX_PRECOMPILE"] = "1"

try:
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
except ImportError:
    LLM = None
    SamplingParams = None
    TokensPrompt = None
    raise ImportError("vLLM is not installed. Please install it to run this test.")

logger = logging.getLogger(__name__)


def get_stop_tokens(model_type: str) -> list[str]:
    """Get model-specific stop tokens."""
    if model_type == "llama":
        return ["<|eot_id|>"]
    elif model_type == "qwen":
        return ["<|im_end|>"]
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def run_inference():
    """Run inference inside a TPU-allocated task."""
    # Model configuration
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    model_type = "llama"
    max_input_tokens = 4096
    max_output_tokens = 512

    # Test prompt (already formatted with chat template)
    prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

How many r's are in strawberry? Write your answer in \\boxed{} format.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}<|eot_id|><|start_header_id|>user<|end_header_id|>

Find the modulo $7$ remainder of the sum $1+3+5+7+9+\\dots+195+197+199.$ Write your answer in \\boxed{} format.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

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

    logger.info("Running inference...")
    logger.info(f"Prompt:\n{prompt}")
    logger.info("-" * 80)

    # Tokenize the prompt
    prompt_token_ids = tokenizer.encode(prompt)
    logger.info(f"Tokenized prompt length: {len(prompt_token_ids)} tokens")

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        n=1,
        max_tokens=max_output_tokens,
        stop=get_stop_tokens(model_type),
        include_stop_str_in_output=True,
        logprobs=1,
    )

    # Create TokensPrompt and generate
    prompts_for_vllm = [TokensPrompt(prompt_token_ids=prompt_token_ids)]
    outputs = llm.generate(prompts_for_vllm, sampling_params)

    # Display results
    logger.info(f"Generated {len(outputs)} output objects:")
    for i, output in enumerate(outputs):
        logger.info(f"\n--- Output {i+1} ---")
        logger.info(f"Prompt: {output.prompt}")
        logger.info(f"Number of completions: {len(output.outputs)}")
        for j, completion in enumerate(output.outputs):
            logger.info(f"\n  Completion {j+1}:")
            logger.info(f"  Text: {completion.text}")
        logger.info("-" * 80)


def main():
    # Setup environment
    env = {}
    env = _add_run_env_variables(env)
    env["EQX_ON_ERROR"] = "nan"

    # Configure TPU resources (v5p-8 to match tensor_parallel_size=8)
    tpu_type = "v5p-8"
    inference_resources = ResourceConfig.with_tpu(tpu_type)
    inference_kwargs = dict(max_calls=1, **as_remote_kwargs(inference_resources, env_vars=env))

    # Define remote task with TPU allocation
    @ray.remote(**inference_kwargs)
    def inference_task():
        with remove_tpu_lockfile_on_exit():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
            run_inference()

    # Run on TPU pod
    task = run_on_pod_ray.remote(
        inference_task,
        tpu_type,
        num_slices=1,
        max_retries_failure=3,
        max_retries_preemption=100,
    )

    # Wait for completion
    ray.get(task)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
