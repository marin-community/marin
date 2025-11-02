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

import datasets
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
    """Length of random input tokens (ignored if dataset is specified)."""

    output_length: int = 1024
    """Number of tokens to generate (ignored if dataset is specified)."""

    batch_size: int = 32
    """Number of prompts per batch."""

    n_generations_per_prompt: int = 1
    """Number of generations per prompt (vLLM 'n' parameter)."""

    num_batches: int = 100
    """Total number of batches to run."""

    log_freq: int = 10
    """Log metrics every N batches."""

    dataset: str | None = None
    """HuggingFace dataset path for realistic prompt sampling. If specified, input_length/output_length are ignored and ignore_eos is disabled."""

    dataset_split: str = "train"
    """Dataset split to use (e.g., 'train', 'test')."""

    dataset_prompt_field: str = "problem"
    """Field name in dataset containing the prompt text."""

    max_dataset_examples: int | None = None
    """Optional limit on number of examples to load from dataset."""

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

        # Load dataset if specified
        self.dataset_prompts = None
        if config.dataset:
            logger.info(f"Loading dataset: {config.dataset}")
            dataset = datasets.load_dataset(config.dataset, trust_remote_code=True)
            if isinstance(dataset, dict):
                dataset_split = dataset.get(config.dataset_split) or dataset.get("train")
            else:
                dataset_split = dataset
            
            # Extract prompts from dataset
            prompts = []
            for idx, item in enumerate(dataset_split):
                if config.dataset_prompt_field not in item:
                    raise ValueError(f"Field '{config.dataset_prompt_field}' not found in dataset")
                prompts.append(item[config.dataset_prompt_field])
                if config.max_dataset_examples is not None and len(prompts) >= config.max_dataset_examples:
                    break
            
            self.dataset_prompts = prompts
            logger.info(f"Loaded {len(self.dataset_prompts)} prompts from dataset")

        # Initialize wandb if configured
        if config.wandb_project:
            try:
                wandb_config = {
                    "model_name": config.model_name,
                    "max_model_len": config.max_model_len,
                    "tensor_parallel_size": config.tensor_parallel_size,
                    "gpu_memory_utilization": config.gpu_memory_utilization,
                    "batch_size": config.batch_size,
                    "n_generations_per_prompt": config.n_generations_per_prompt,
                    "num_batches": config.num_batches,
                }
                if config.dataset:
                    wandb_config["dataset"] = config.dataset
                    wandb_config["dataset_split"] = config.dataset_split
                else:
                    wandb_config["input_length"] = config.input_length
                    wandb_config["output_length"] = config.output_length
                
                wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    tags=config.wandb_tags or [],
                    reinit=True,  # Required for Ray workers to avoid conflicts
                    config=wandb_config,
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

    def _generate_prompts(self, batch_size: int) -> list[str]:
        """Generate prompts either from dataset or random tokens."""
        if self.dataset_prompts:
            # Sample from dataset
            indices = np.random.choice(len(self.dataset_prompts), size=batch_size, replace=True)
            return [self.dataset_prompts[int(idx)] for idx in indices]
        else:
            # Generate random token sequences
            prompts = []
            for _ in range(batch_size):
                random_tokens = np.random.randint(
                    100,  # Skip special tokens
                    self.tokenizer.vocab_size,
                    size=self.config.input_length,
                )
                prompt = self.tokenizer.decode(random_tokens, skip_special_tokens=False)
                prompts.append(prompt)
            return prompts

    def run(self):
        """Run inference-only benchmark."""
        logger.info("Starting inference-only benchmark...")

        # Create sampling params - adjust based on dataset usage
        if self.dataset_prompts:
            # Realistic generation: let model stop at EOS
            sampling_params = SamplingParams(
                temperature=1.0,
                n=self.config.n_generations_per_prompt,
                logprobs=1,  # Request logprobs for more realistic scenario
            )
            logger.info("Using dataset prompts with realistic generation (ignore_eos=False)")
        else:
            # Synthetic benchmark: always generate output_length tokens
            sampling_params = SamplingParams(
                temperature=1.0,
                n=self.config.n_generations_per_prompt,
                max_tokens=self.config.output_length,
                ignore_eos=True,  # Always generate exactly output_length tokens
                logprobs=1,  # Request logprobs for more realistic scenario
            )
            logger.info(f"Using random prompts with fixed output length: {self.config.output_length}")

        total_tokens_generated = 0
        total_time = 0.0
        batch_times = []

        for batch_idx in range(self.config.num_batches):
            # Generate prompts
            prompts = self._generate_prompts(self.config.batch_size)

            # Time the batch generation
            start_time = time.time()
            outputs = self.llm.generate(prompts, sampling_params)
            # Reset prefix cache to simulate syncing weights
            self.llm.llm_engine.reset_prefix_cache()
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
        if self.dataset_prompts:
            logger.info(f"  Dataset: {self.config.dataset}")
            logger.info(f"  Dataset size: {len(self.dataset_prompts)}")
        else:
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

        results = {
            "model_name": self.config.model_name,
            "batch_size": self.config.batch_size,
            "num_batches": self.config.num_batches,
            "total_tokens": total_tokens_generated,
            "total_time": total_time,
            "avg_throughput": avg_throughput,
            "avg_batch_time": avg_batch_time,
        }
        if self.dataset_prompts:
            results["dataset"] = self.config.dataset
            results["dataset_size"] = len(self.dataset_prompts)
        else:
            results["input_length"] = self.config.input_length
            results["output_length"] = self.config.output_length
        return results
