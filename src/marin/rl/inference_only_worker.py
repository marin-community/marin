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
Inference-only worker for testing vLLM throughput in isolation.

This worker initializes vLLM with a model and generates completions from random
input tokens to measure pure inference throughput without RL training overhead.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
import wandb
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class InferenceOnlyConfig:
    """Configuration for inference-only worker."""

    model_name: str
    """HuggingFace model name or path."""

    max_model_len: int
    """Maximum sequence length the model can handle."""

    tensor_parallel_size: int
    """Number of GPUs/TPUs to use for tensor parallelism."""

    gpu_memory_utilization: float
    """Fraction of GPU memory to use."""

    input_length: int = 128
    """Length of random input tokens."""

    output_length: int = 1024
    """Number of tokens to generate."""

    batch_size: int = 32
    """Number of prompts per batch."""

    n_generations_per_prompt: int = 1
    """Number of generations per prompt (vLLM 'n' parameter)."""

    num_batches: int = 100
    """Total number of batches to run."""

    log_freq: int = 10
    """Log metrics every N batches."""

    wandb_project: str | None = None
    """WandB project name for logging. If None, wandb logging is disabled."""

    wandb_run_name: str | None = None
    """WandB run name. If None, a default name will be generated."""

    wandb_tags: list[str] | None = None
    """WandB tags for the run."""


class InferenceOnlyWorker:
    """Worker that runs inference-only to measure vLLM throughput."""

    def __init__(self, config: InferenceOnlyConfig):
        self.config = config
        logger.info(f"Initializing vLLM with model: {config.model_name}")

        # Initialize wandb if configured
        if config.wandb_project:
            try:
                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    tags=config.wandb_tags or [],
                    reinit=True,  # Required for Ray workers to avoid conflicts
                    config={
                        "model_name": config.model_name,
                        "max_model_len": config.max_model_len,
                        "tensor_parallel_size": config.tensor_parallel_size,
                        "gpu_memory_utilization": config.gpu_memory_utilization,
                        "input_length": config.input_length,
                        "output_length": config.output_length,
                        "batch_size": config.batch_size,
                        "n_generations_per_prompt": config.n_generations_per_prompt,
                        "num_batches": config.num_batches,
                    },
                )
                logger.info(f"Successfully initialized wandb logging to project: {config.wandb_project}")
            except Exception as e:
                logger.error(f"Failed to initialize wandb: {e}")
                logger.warning("Continuing without wandb logging")

        # Initialize vLLM engine
        self.llm = LLM(
            model=config.model_name,
            max_model_len=config.max_model_len,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        logger.info("vLLM initialization complete")

    def _generate_random_prompts(self, batch_size: int) -> list[str]:
        """Generate random token sequences as prompts."""
        prompts = []
        for _ in range(batch_size):
            # Generate random token IDs
            random_tokens = np.random.randint(
                100,  # Skip special tokens
                self.tokenizer.vocab_size,
                size=self.config.input_length,
            )
            # Decode to text
            prompt = self.tokenizer.decode(random_tokens, skip_special_tokens=False)
            prompts.append(prompt)
        return prompts

    def run(self):
        """Run inference-only benchmark."""
        logger.info("Starting inference-only benchmark...")

        # Create sampling params with ignore_eos=True to always generate output_length tokens
        sampling_params = SamplingParams(
            temperature=1.0,
            n=self.config.n_generations_per_prompt,
            max_tokens=self.config.output_length,
            ignore_eos=True,  # Always generate exactly output_length tokens
            logprobs=1,  # Request logprobs for more realistic scenario
        )

        total_tokens_generated = 0
        total_time = 0.0
        batch_times = []

        for batch_idx in range(self.config.num_batches):
            # Generate random prompts
            prompts = self._generate_random_prompts(self.config.batch_size)

            # Time the batch generation
            start_time = time.time()
            outputs = self.llm.generate(prompts, sampling_params)
            batch_time = time.time() - start_time

            # Calculate tokens generated in this batch (sum across all outputs for all prompts)
            tokens_in_batch = sum(len(out.token_ids) for output in outputs for out in output.outputs)
            total_tokens_generated += tokens_in_batch
            total_time += batch_time
            batch_times.append(batch_time)

            # Log periodically
            if (batch_idx + 1) % self.config.log_freq == 0:
                avg_throughput = total_tokens_generated / total_time
                batch_throughput = tokens_in_batch / batch_time
                logger.info(
                    f"Batch {batch_idx + 1}/{self.config.num_batches}: "
                    f"batch_time={batch_time:.2f}s, "
                    f"batch_throughput={batch_throughput:.0f} tok/s, "
                    f"avg_throughput={avg_throughput:.0f} tok/s"
                )

                # Log to wandb if enabled
                if self.config.wandb_project:
                    try:
                        wandb.log(
                            {
                                "batch_idx": batch_idx + 1,
                                "batch_time": batch_time,
                                "batch_throughput": batch_throughput,
                                "avg_throughput": avg_throughput,
                                "tokens_in_batch": tokens_in_batch,
                                "total_tokens_generated": total_tokens_generated,
                            },
                            step=batch_idx + 1,
                        )
                        logger.info(f"Successfully logged batch {batch_idx + 1} to wandb")
                    except Exception as e:
                        logger.error(f"Failed to log to wandb: {e}")

        # Final statistics
        avg_throughput = total_tokens_generated / total_time
        avg_batch_time = np.mean(batch_times)
        logger.info("\n" + "=" * 80)
        logger.info("Inference Benchmark Results:")
        logger.info(f"  Model: {self.config.model_name}")
        logger.info(f"  Input length: {self.config.input_length}")
        logger.info(f"  Output length: {self.config.output_length}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Num batches: {self.config.num_batches}")
        logger.info(f"  Total tokens generated: {total_tokens_generated:,}")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Average throughput: {avg_throughput:.0f} tokens/sec")
        logger.info(f"  Average batch time: {avg_batch_time:.2f}s")
        logger.info("=" * 80)

        # Log final summary to wandb
        if self.config.wandb_project:
            try:
                wandb.log(
                    {
                        "final/total_tokens": total_tokens_generated,
                        "final/total_time": total_time,
                        "final/avg_throughput": avg_throughput,
                        "final/avg_batch_time": avg_batch_time,
                    }
                )
                wandb.finish()
                logger.info("Successfully finished wandb logging")
            except Exception as e:
                logger.error(f"Failed to log final metrics to wandb: {e}")

        return {
            "model_name": self.config.model_name,
            "input_length": self.config.input_length,
            "output_length": self.config.output_length,
            "batch_size": self.config.batch_size,
            "num_batches": self.config.num_batches,
            "total_tokens": total_tokens_generated,
            "total_time": total_time,
            "avg_throughput": avg_throughput,
            "avg_batch_time": avg_batch_time,
        }
