#!/usr/bin/env python
# Copyright 2025 Marin Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark script that starts a Levanter inference server and runs vLLM benchmark client against it.

Usage:
    python experiments/bench/vllm_bench.py --model_path <path> --num_prompts 100 --request_rate 10
"""

import argparse
import asyncio
import logging
import multiprocessing
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax.random as jrandom
import jmp
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

# Add bench directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent))

# Add levanter to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "submodules" / "levanter" / "src"))

from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, load_tokenizer
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.models.lm_model import LmConfig
from levanter.models.llama import LlamaConfig
from levanter.trainer import TrainerConfig
from transformers import AutoConfig

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
    max_seqs: int
    page_size: int
    max_pages: int

    # Benchmark configuration
    num_prompts: int
    request_rate: float
    dataset_name: str
    seed: int
    num_fewshot: int

    tokenizer_path: str | None = None
    model_config: LmConfig = None
    dataset_path: str | None = None
    
    # Trainer configuration
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(
            model_axis_size=4,
            tensor_parallel_axes=["mlp", "heads", "kv_head", "vocab"],
            mp=jmp.get_policy("p=f32,c=bfloat16"),
        )
    )

    # Additional flags
    force_run_failed: bool = False

    def __post_init__(self):
        if self.model_config is None:
            self.model_config = LlamaConfig.from_hf_config(
                AutoConfig.from_pretrained(self.model_path)
            )


def load_model(config: BenchmarkConfig):
    """Load a model from HuggingFace checkpoint or local path."""
    tokenizer_path = config.tokenizer_path or config.model_path
    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = len(tokenizer)

    mp = config.trainer.mp
    key = jrandom.PRNGKey(config.seed)

    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), config.trainer.compute_axis_mapping)

        # Try loading as HF checkpoint first
        logger.info(f"Loading model from {config.model_path}")
        logger.info(f"Model type: {config.model_config.model_type}")
        logger.info(f"Vocab size: {vocab_size}")
        logger.info(f"Compute dtype: {mp.compute_dtype}")

        try:
            # Check if it's a local Levanter checkpoint
            checkpoint_path = Path(config.model_path)
            if checkpoint_path.exists() and (checkpoint_path / "model").exists():
                logger.info("Loading from Levanter checkpoint")
                model = eqx.filter_eval_shape(config.model_config.build, Vocab, key=key)
                model = load_checkpoint(model, config.model_path, subpath="model")
                model = mp.cast_to_compute(model)
                return model, tokenizer
        except Exception as e:
            logger.debug(f"Not a Levanter checkpoint: {e}")

        # Load from HuggingFace
        logger.info(f"Loading from HuggingFace checkpoint: {config.model_path}")
        converter = HFCheckpointConverter(
            type(config.model_config),
            reference_checkpoint=config.model_path,
            tokenizer=tokenizer,
        )

        model = converter.load_pretrained(
            config.model_config.model_type,
            ref=config.model_path,
            dtype=mp.compute_dtype,
            axis_mapping=config.trainer.parameter_axis_mapping,
        )

        return model, tokenizer


def start_server_process(config: BenchmarkConfig):
    """Start the Levanter inference server in a separate process."""
    logger.info("Starting Levanter inference server...")

    # Load model and create server within the device mesh and axis mapping context
    model, tokenizer = load_model(config)

    server_config = InferenceServerConfig(
        service=InferenceEngineConfig(
            max_seq_len=config.max_seq_len,
            max_seqs=config.max_seqs,
            page_size=config.page_size,
            max_pages=config.max_pages,
        ),
        host=config.host,
        port=config.port,
        temperature=0.7,
        seed=config.seed,
        trainer=config.trainer,
    )

    # Create server within the device mesh and axis mapping context
    with config.trainer.use_device_mesh(), hax.axis_mapping(config.trainer.compute_axis_mapping):
        server = InferenceServer.create(server_config, model, tokenizer)

    logger.info(f"Server initialized, listening on {config.host}:{config.port}")

    # Start serving (blocking call)
    server.serve()


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
    logger.info("Starting benchmark client...")

    # Import the vLLM serve benchmark module
    from vllm_serve import benchmark

    # Prepare benchmark arguments
    api_url = f"http://{config.host}:{config.port}/v1/completions"
    base_url = f"http://{config.host}:{config.port}"

    # Load tokenizer for the benchmark
    tokenizer = load_tokenizer(config.tokenizer_path or config.model_path)

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
    from vllm_serve import TaskType

    await benchmark(
        task_type=TaskType.GENERATION,
        endpoint_type="openai-nonstreaming",  # Use non-streaming handler for Levanter
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
    # Start the server in a separate process
    server_process = multiprocessing.Process(target=start_server_process, args=(config,))
    server_process.start()

    try:
        # Wait for server to be ready
        await wait_for_server(config.host, config.port)

        # Run the benchmark
        await run_benchmark_client(config)

    finally:
        # Cleanup
        logger.info("Shutting down server...")
        server_process.terminate()
        server_process.join(timeout=10)
        if server_process.is_alive():
            logger.warning("Server did not terminate gracefully, killing...")
            server_process.kill()
            server_process.join()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark Levanter inference server with vLLM client")

    # Model arguments
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.1-8B", help="Model path or HF repo")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer path (defaults to model_path)")

    # Server arguments
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max_seqs", type=int, default=256, help="Maximum concurrent sequences")
    parser.add_argument("--page_size", type=int, default=128, help="Page size for KV cache")
    parser.add_argument("--max_pages", type=int, default=None, help="Maximum number of pages")

    # Benchmark arguments
    parser.add_argument("--num_prompts", type=int, default=256, help="Number of prompts to benchmark")
    parser.add_argument("--request_rate", type=float, default=float("inf"), help="Request rate (requests/sec)")
    parser.add_argument("--dataset_name", type=str, default="random", help="Dataset name")
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")

    # Additional flags
    parser.add_argument("--force_run_failed", type=bool, default=False, help="Force run even if previous run failed")

    args = parser.parse_args()

    # Create config
    config = BenchmarkConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        host=args.host,
        port=args.port,
        max_seq_len=args.max_seq_len,
        max_seqs=args.max_seqs,
        page_size=args.page_size,
        max_pages=args.max_pages,
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
