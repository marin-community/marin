#!/usr/bin/env python
# Copyright 2025 Marin Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark script that starts an EasyDel vSurge inference server using Ray runtime isolation
and runs vLLM benchmark client against it.

This script uses Ray's runtime_env to isolate EasyDel's dependencies (flax==0.10.4)
from the main project's dependencies, avoiding version conflicts.

Usage:
    python experiments/bench/easydel_bench.py --model_path <path> --num_prompts 100 --request_rate 10
"""

import argparse
import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add bench directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""

    # Model configuration
    model_path: str

    # Server configuration
    host: str
    port: int
    max_seq_len: int
    max_prefill_len: int
    max_concurrent_prefill: int
    max_concurrent_decodes: int

    # Benchmark configuration
    num_prompts: int
    request_rate: float
    dataset_name: str
    seed: int
    num_fewshot: int

    tokenizer_path: str | None = None
    dataset_path: str | None = None

    # Server parameters
    max_workers: int = 64

    # Additional flags
    force_run_failed: bool = False


def run_easydel_server(config_dict: dict):
    """
    Run EasyDel vSurge server. This function will be executed in a Ray actor
    with its own isolated runtime environment containing EasyDel dependencies.
    """
    import logging
    from transformers import AutoTokenizer, AutoProcessor

    logger = logging.getLogger(__name__)
    logger.info("Starting EasyDel vSurge inference server in isolated environment...")

    try:
        import easydel as ed
        import jax
        import jax.numpy as jnp
    except ImportError as e:
        logger.error(
            "Failed to import EasyDel or JAX. Please install with: pip install easydel"
        )
        raise

    model_path = config_dict["model_path"]
    host = config_dict["host"]
    port = config_dict["port"]
    max_seq_len = config_dict["max_seq_len"]
    max_prefill_len = config_dict["max_prefill_len"]
    max_concurrent_prefill = config_dict["max_concurrent_prefill"]
    max_concurrent_decodes = config_dict["max_concurrent_decodes"]
    max_workers = config_dict["max_workers"]
    seed = config_dict["seed"]

    # Load the model and processor
    logger.info(f"Loading model from {model_path}")

    try:
        # Load processor/tokenizer
        processor = AutoProcessor.from_pretrained(model_path)
    except Exception:
        # Fall back to tokenizer if processor doesn't exist
        processor = AutoTokenizer.from_pretrained(model_path)

    # Configure and load the model with EasyDel
    logger.info("Loading pretrained model...")
    model = ed.AutoEasyDeLModelForCausalLM.from_pretrained(
        model_path,
        auto_shard_model=True,
        sharding_axis_dims=(1, 1, 1, -1, 1),
        config_kwargs=ed.EasyDeLBaseConfigDict(
            freq_max_position_embeddings=max_seq_len,
            mask_max_position_embeddings=max_seq_len,
            kv_cache_quantization_method=ed.EasyDeLQuantizationMethods.NONE,
            gradient_checkpointing=ed.EasyDeLGradientCheckPointers.NONE,
            attn_mechanism=ed.AttentionMechanisms.RAGGED_PAGE_ATTENTION,
            decode_attn_mechanism=ed.AttentionMechanisms.REGRESSIVE_DECODE,
            kvdtype=jnp.bfloat16,
        ),
        quantization_method=ed.EasyDeLQuantizationMethods.NONE,
        param_dtype=jnp.bfloat16,
        dtype=jnp.bfloat16,
        partition_axis=ed.PartitionAxis(),
        precision=jax.lax.Precision.DEFAULT,
    )

    # Create vSurge instance
    logger.info("Creating vSurge instance...")
    surge = ed.vSurge.from_model(
        model=model,
        processor=processor,
        max_prefill_length=max_prefill_len,
        max_concurrent_prefill=max_concurrent_prefill,
        max_concurrent_decodes=max_concurrent_decodes,
        seed=seed,
    )

    # Start the API server
    logger.info(f"Starting API server on {host}:{port}")
    api_server = ed.vSurgeApiServer(
        surge,
        max_workers=max_workers,
        host=host,
        port=port,
    )

    logger.info(f"Server initialized, listening on {host}:{port}")

    # Start serving (blocking call)
    api_server.fire()


async def wait_for_server(host: str, port: int, timeout: int = 60):
    """Wait for the server to be ready."""
    import aiohttp

    url = f"http://{host}:{port}/health"
    start_time = time.time()

    logger.info(f"Waiting for server to be ready at {url}")

    while time.time() - start_time < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                    if response.status == 200:
                        logger.info("Server is ready!")
                        return True
        except Exception as e:
            logger.debug(f"Server not ready yet: {e}")

        await asyncio.sleep(1)

    raise TimeoutError(f"Server did not become ready within {timeout} seconds")


async def run_benchmark_client(config: BenchmarkConfig):
    """Run the vLLM benchmark client."""
    from transformers import AutoTokenizer, AutoProcessor

    logger.info("Starting benchmark client...")

    # Import the vLLM serve benchmark module
    from vllm_serve import benchmark, TaskType

    # Prepare benchmark arguments
    api_url = f"http://{config.host}:{config.port}/v1/completions"
    base_url = f"http://{config.host}:{config.port}"

    # Load tokenizer for the benchmark
    tokenizer_path = config.tokenizer_path or config.model_path
    try:
        tokenizer = AutoProcessor.from_pretrained(tokenizer_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Create sample requests
    from datasets import get_samples

    # Create an args object for get_samples
    class SampleArgs:
        def __init__(self):
            self.dataset_name = config.dataset_name
            self.dataset_path = config.dataset_path
            self.num_prompts = config.num_prompts
            self.seed = config.seed
            self.request_id_prefix = "benchmark-serving"
            # Set defaults for optional args that get_samples might check
            self.disable_shuffle = False
            self.backend = "openai"
            self.skip_chat_template = False
            self.no_oversample = False
            # Random dataset specific args
            self.random_prefix_len = 0
            self.random_input_len = 1024
            self.random_output_len = 1024
            self.random_range_ratio = 0.0
            self.random_batch_size = 1

    args = SampleArgs()
    input_requests = get_samples(args, tokenizer)

    logger.info(f"Running benchmark with {len(input_requests)} requests")

    # Run the benchmark
    await benchmark(
        task_type=TaskType.GENERATION,
        endpoint_type="openai-nonstreaming",  # Use non-streaming handler for EasyDel
        api_url=api_url,
        base_url=base_url,
        model_id=config.model_path,
        model_name=config.model_path,
        tokenizer=tokenizer,
        input_requests=input_requests,
        logprobs=None,
        request_rate=config.request_rate,
        burstiness=1.0,
        disable_tqdm=False,
        profile=False,
        selected_percentile_metrics=["e2el"],  # Only e2el makes sense for non-streaming
        selected_percentiles=[25, 50, 75, 90, 95, 99],
        ignore_eos=True,
        goodput_config_dict={},
        max_concurrency=None,
        extra_headers=None,
        extra_body=None,
        ready_check_timeout_sec=0,  # We already waited for the server
    )

    logger.info("Benchmark completed!")


async def main_async(config: BenchmarkConfig):
    """Main async entry point."""
    import ray
    from marin.resources import TpuPodConfig, RuntimeEnv

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    TPU_EXECUTION_ENV_VARS = {
        "EASYDEL_AUTO": "1",
        "HF_DATASETS_CACHE": "/dev/shm/huggingface-dataset",
        "HF_HOME": "/dev/shm/huggingface",
        "HF_DATASETS_OFFLINE": "0",
    }

    tpu_cfg = TpuPodConfig(
        tpu_type="v4-8",
        runtime_env=RuntimeEnv(
            env_vars=TPU_EXECUTION_ENV_VARS,
            pip=["easydel[tpu]"],
        ),
    )

    remote_kwargs = tpu_cfg.as_remote_kwargs()
    # On Marin clusters you typically include the explicit TPU tags:
    remote_kwargs.setdefault("resources", {"TPU": 4, "TPU-v4-8-head": 1})
    remote_kwargs.setdefault("num_cpus", 8)

    @ray.remote(**remote_kwargs)
    class EasyDeLServerActor:
        def __init__(self, config_dict):
            self.config_dict = config_dict

        def start_server(self):
            """Start the EasyDel server - this runs in isolated environment."""
            run_easydel_server(self.config_dict)

        def ping(self):
            """Health check method."""
            return "pong"

    # Convert config to dict for passing to actor
    config_dict = {
        "model_path": config.model_path,
        "host": config.host,
        "port": config.port,
        "max_prefill_len": config.max_prefill_len,
        "max_seq_len": config.max_seq_len,
        "max_concurrent_prefill": config.max_concurrent_prefill,
        "max_concurrent_decodes": config.max_concurrent_decodes,
        "max_workers": config.max_workers,
        "seed": config.seed,
    }

    logger.info("Starting EasyDel server actor with isolated runtime environment...")
    server_actor = EasyDeLServerActor.remote(config_dict)

    # Start the server in the background (non-blocking)
    server_task = server_actor.start_server.remote()

    try:
        # Wait for server to be ready
        await wait_for_server(config.host, config.port, timeout=600)

        # Run the benchmark
        await run_benchmark_client(config)

    finally:
        # Cleanup: kill the actor
        logger.info("Shutting down server actor...")
        ray.kill(server_actor)

        # Cancel the server task if still running
        try:
            ray.cancel(server_task, force=True)
        except Exception as e:
            logger.debug(f"Error canceling server task: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark EasyDel vSurge inference server with vLLM client using Ray runtime isolation"
    )

    # Model arguments
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B", help="Model path or HF repo")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path (defaults to model_path)")

    # Server arguments
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=11556, help="Server port (default: 11556)")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max_prefill_len", type=int, default=1024, help="Maximum input prompt length")
    parser.add_argument("--max_concurrent_prefill", type=int, default=1, help="Maximum concurrent prefill operations")
    parser.add_argument("--max_concurrent_decodes", type=int, default=64, help="Maximum concurrent decode operations")
    parser.add_argument("--max_workers", type=int, default=64, help="Maximum worker threads for API server")

    # Benchmark arguments
    parser.add_argument("--num_prompts", type=int, default=256, help="Number of prompts to benchmark")
    parser.add_argument("--request_rate", type=float, default=float("inf"), help="Request rate (requests/sec)")
    parser.add_argument("--dataset_name", type=str, default="random", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path")
    parser.add_argument("--seed", type=int, default=877, help="Random seed")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")

    # Additional flags
    parser.add_argument("--force_run_failed", action="store_true", help="Force run even if previous run failed")

    args = parser.parse_args()

    # Create config
    config = BenchmarkConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        host=args.host,
        port=args.port,
        max_seq_len=args.max_seq_len,
        max_prefill_len=args.max_prefill_len,
        max_concurrent_prefill=args.max_concurrent_prefill,
        max_concurrent_decodes=args.max_concurrent_decodes,
        max_workers=args.max_workers,
        num_prompts=args.num_prompts,
        request_rate=args.request_rate,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        seed=args.seed,
        num_fewshot=args.num_fewshot,
        force_run_failed=args.force_run_failed,
    )

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run the benchmark
    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
