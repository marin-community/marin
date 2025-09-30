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

"""Replay OpenAI chat completion requests through a Levanter inference server.

This tool loads a JSON file of chat completion requests (as exported by
export_env_prompts.py) and replays them through a Levanter inference server.

If a bad batch is detected (e.g., due to timeout), the failing batch is written
to /tmp/failing_batch.json for further analysis.

Usage:
    uv run src/marin/rl/scripts/replay_completions.py \
        --requests math_prompts.json \
        --checkpoint meta-llama/Llama-3.2-1B-Instruct \
        --timeout 30 \
        --batch-size 8
"""

import argparse
import asyncio
import json
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any

from levanter.compat.hf_checkpoints import RepoRef
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.models.llama import LlamaConfig
from levanter.trainer import TrainerConfig
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def find_open_port() -> int:
    """Find an open port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@dataclass
class ReplayConfig:
    """Configuration for replaying completions."""

    requests_file: str
    checkpoint: str | None = None
    tokenizer: str | None = None
    timeout: float = 30.0
    batch_size: int = 8
    verbose: bool = False

    # Model and training configuration
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LlamaConfig = field(default_factory=LlamaConfig)
    service: InferenceEngineConfig = field(
        default_factory=lambda: InferenceEngineConfig(
            max_seqs=16,
            page_size=128,
            max_pages_per_seq=8,
            max_seqs_in_prefill=16,
        )
    )


def preprocess_request(req: dict[str, Any]) -> dict[str, Any]:
    """Remove metadata fields from request."""
    return {k: v for k, v in req.items() if not k.startswith("_")}


def load_requests(requests_file: str) -> list[dict[str, Any]]:
    """Load requests from JSON file."""
    logger.info(f"Loading requests from {requests_file}")
    with open(requests_file, "r") as f:
        requests = json.load(f)
    logger.info(f"Loaded {len(requests)} chat completion requests")
    return requests


def create_inference_server(config: ReplayConfig) -> tuple[InferenceServer, AsyncOpenAI, int]:
    """Create and start inference server with client."""
    logger.info(f"Starting inference server with checkpoint: {config.checkpoint}")

    # Determine tokenizer
    tokenizer_path = config.tokenizer or config.checkpoint
    if not tokenizer_path:
        raise ValueError("Must specify either --checkpoint or --tokenizer")

    # Configure server
    port = find_open_port()
    server_config = InferenceServerConfig(
        host="localhost",
        port=port,
        tokenizer=tokenizer_path,
        model=config.model,
        trainer=config.trainer,
        service=config.service,
    )

    # Set checkpoint path
    if config.checkpoint:
        if "/" in config.checkpoint and not config.checkpoint.startswith(("/", "./")):
            # Looks like an HF model
            server_config.hf_checkpoint = RepoRef.from_string(config.checkpoint)
        else:
            server_config.checkpoint_path = config.checkpoint

    # Create and start server
    server = InferenceServer.create(server_config)

    # run serving in background
    import threading

    threading.Thread(target=server.serve, daemon=True).start()

    # Create client
    base_url = f"http://localhost:{port}/v1"
    logger.info(f"Creating OpenAI client for {base_url}")
    client = AsyncOpenAI(base_url=base_url, api_key="replay")

    return server, client, port


async def send_batch_requests(
    client: AsyncOpenAI,
    requests: list[dict[str, Any]],
    batch_idx: int,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Send a batch of requests to the server with optional timeout."""
    try:
        # Create completion requests
        tasks = []
        for req in requests:
            clean_req = preprocess_request(req)
            task = client.chat.completions.create(**clean_req, timeout=timeout)
            tasks.append(task)

        # Wait for all with timeout
        timeout_msg = f" (timeout: {timeout}s)" if timeout else " (no timeout)"
        logger.info(f"Sending batch {batch_idx} with {len(tasks)} requests{timeout_msg}...")
        start_time = time.time()

        if timeout is not None:
            # Use asyncio.wait_for for overall timeout
            responses = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
        else:
            # No timeout for warm-up batch
            responses = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start_time
        logger.info(f"Batch {batch_idx} completed in {elapsed:.2f}s")

        # Check for errors
        for i, resp in enumerate(responses):
            if isinstance(resp, Exception):
                logger.error(f"Request {i} in batch {batch_idx} failed with error: {resp}")
                raise resp

        return {
            "success": True,
            "responses": responses,
            "elapsed_time": elapsed,
            "error": None,
            "failing_requests": [],
        }

    except (asyncio.TimeoutError, Exception):
        logger.error(f"Failing batch contained {len(requests)} requests, writing to /tmp/failing_batch.json")
        # write bad batch to disk
        with open("/tmp/failing_batch.json", "w") as f:
            json.dump(requests, f, indent=2)
        raise


async def replay_all_with_warmup(
    client: AsyncOpenAI,
    requests: list[dict[str, Any]],
    config: ReplayConfig,
) -> dict[str, Any]:
    """Replay all requests in batches with warm-up."""
    if not requests:
        return {
            "total_requests": 0,
            "total_batches": 0,
            "completed_batches": 0,
            "failed_batches": [],
            "total_time": 0,
            "results": [],
        }

    total_batches = (len(requests) + config.batch_size - 1) // config.batch_size
    logger.info(f"Processing {len(requests)} requests in {total_batches} batches of " f"size {config.batch_size}")

    current_start = 0

    # First batch without timeout for warm-up
    if requests:
        warmup_end = min(config.batch_size, len(requests))
        warmup_batch = requests[current_start:warmup_end]
        await send_batch_requests(client, warmup_batch, 0, timeout=None)

    # Process remaining batches with timeout
    batch_idx = 1
    while current_start < len(requests):
        end_idx = min(current_start + config.batch_size, len(requests))
        batch = requests[current_start:end_idx]
        await send_batch_requests(client, batch, batch_idx, timeout=config.timeout)
        current_start = end_idx
        batch_idx += 1


class CompletionReplayer:
    """Replays chat completion requests through an inference server."""

    def __init__(self, config: ReplayConfig):
        self.config = config

    async def run_async(self) -> dict[str, Any]:
        """Main async entry point for replaying completions."""
        requests = load_requests(self.config.requests_file)
        server, client, _ = create_inference_server(self.config)

        return await replay_all_with_warmup(client, requests, self.config)

    def run(self) -> dict[str, Any]:
        """Main entry point for replaying completions."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(self.run_async())
        finally:
            loop.close()


def main():
    parser = argparse.ArgumentParser(description="Replay chat completion requests through inference server")
    parser.add_argument("--requests", required=True, help="JSON file with chat completion requests")
    parser.add_argument("--checkpoint", help="Model checkpoint (HF model or local path)")
    parser.add_argument("--tokenizer", help="Tokenizer name or path (if different from checkpoint)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout per batch in seconds")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of requests per batch")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = ReplayConfig(
        requests_file=args.requests,
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        timeout=args.timeout,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    replayer = CompletionReplayer(config)
    result = replayer.run()

    logger.info(f"Completed {result['completed_batches']}/{result['total_batches']} batches")
    if result["failed_batches"]:
        logger.error(f"Failed {len(result['failed_batches'])} batches")
    else:
        logger.info("All batches completed successfully")


if __name__ == "__main__":
    main()
