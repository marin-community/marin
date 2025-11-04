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
Inference-only experiment to test vLLM throughput in isolation.
This experiment runs vLLM inference to measure pure inference throughput
without RL training overhead.

Supports two modes:
1. Synthetic: Random input tokens with fixed output length (ignore_eos=True)
2. Realistic: Sample prompts from a HuggingFace dataset with natural generation
"""

import logging

import ray
from levanter.infra.ray_tpu import run_on_pod_ray

from marin.rl.environments.math_env import MathEnv
from marin.rl.inference_only_worker import InferenceOnlyConfig, InferenceOnlyWorker
from marin.utils import remove_tpu_lockfile_on_exit

logger = logging.getLogger(__name__)


def main():
    """Run inference-only benchmark on Llama 1B with v4-8."""
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        logger.info("Initializing Ray...")
        ray.init(address="auto", ignore_reinit_error=True)

    # Option 1: Synthetic benchmark with random tokens (default)
    # config = InferenceOnlyConfig(
    #     # Model configuration
    #     model_name="meta-llama/Llama-3.2-1B-Instruct",
    #     max_model_len=4096,
    #     # v4-8 configuration (8-way tensor parallelism)
    #     tensor_parallel_size=8,
    #     gpu_memory_utilization=0.90,
    #     # Benchmark configuration
    #     input_length=128,  # Random input tokens
    #     output_length=1024,  # Always generate 1024 tokens (ignore_eos=True)
    #     batch_size=64,  # Number of prompts per batch
    #     n_generations_per_prompt=8,  # Match RL experiment sampling
    #     num_batches=100,  # Total batches to run
    #     log_freq=1,
    #     # WandB configuration
    #     wandb_project="vllm-inference-benchmark",
    #     wandb_run_name="llama-3.2-1b-v4-8",
    #     wandb_tags=["vllm", "inference", "benchmark", "llama-3.2-1b", "v4-8"],
    # )
    
    # Option 2: Realistic benchmark with env prompts
    # Create MathEnv with train dataset
    math_env = MathEnv(
        train_source="di-zhang-fdu/MATH12000",  # or "HuggingFaceH4/MATH-500"
        max_train_examples=10000,  # Optional limit
        seed=42,
    )
    
    config = InferenceOnlyConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        max_model_len=4096,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.90,
        batch_size=32,
        n_generations_per_prompt=8,
        num_batches=100,
        log_freq=1,
        # Env configuration (input_length/output_length ignored when env is set)
        env=math_env,
        env_mode="train",  # Use 'train' or 'eval' mode
        wandb_project="vllm-inference-benchmark",
        wandb_run_name="llama-3.2-1b-v4-8-realistic",
        wandb_tags=["vllm", "inference", "benchmark", "llama-3.2-1b", "v4-8", "realistic"],
    )

    logger.info("Starting vLLM inference-only benchmark on v4-8")
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Tensor parallel size: {config.tensor_parallel_size}")
    if config.env:
        logger.info(f"Env type: {type(config.env).__name__}")
        logger.info(f"Env mode: {config.env_mode}")
    else:
        logger.info(f"Input length: {config.input_length}")
        logger.info(f"Output length: {config.output_length}")
    logger.info(f"Batch size: {config.batch_size}")

    # Define the worker task to run on TPU
    def inference_worker_task():
        with remove_tpu_lockfile_on_exit():
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
            worker = InferenceOnlyWorker(config)
            return worker.run()

    # Launch on TPU pod using run_on_pod_ray
    logger.info("Launching inference worker on v4-8...")
    task = run_on_pod_ray.remote(
        inference_worker_task,
        "v4-8",
        num_slices=1,
        max_retries_failure=3,
        max_retries_preemption=10,
    )

    # Wait for completion and get results
    logger.info("Waiting for inference worker to complete...")
    results = ray.get(task)

    logger.info("\nBenchmark completed successfully!")
    logger.info(f"Results: {results}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
