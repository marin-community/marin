# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Example inference worker that demonstrates how to embed an InferenceServer
in the context of a larger application.

This worker simply monitors for new checkpoints in a specified directory
and reloads the model when a new checkpoint is found.

Usage:
    uv run src/levanter/main/inference_worker.py --checkpoint_path /path/to/checkpoints \
        --hf_checkpoint timinar/baby-llama-58m --tokenizer timinar/baby-llama-58m
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional

import equinox as eqx
import haliax as hax
import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.inference.openai import InferenceServer, InferenceServerConfig
from levanter.utils.jax_utils import use_cpu_device

logger = logging.getLogger(__name__)


class InferenceWorker:
    """Example worker class that demonstrates embedding InferenceServer in a larger application."""

    checkpoint_path: Path | None
    config: InferenceServerConfig
    server: InferenceServer
    check_interval: int
    latest_checkpoint: str | None

    def __init__(self, config: InferenceServerConfig, checkpoint_path: str | None, check_interval: int = 60):
        """Initialize the inference worker.

        Args:
            config: Configuration for the inference server
            checkpoint_path: Directory to monitor for new checkpoints (optional)
            check_interval: Interval in seconds between checkpoint checks
        """
        self.config = config
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            self.checkpoint_path = None
        self.check_interval = check_interval
        self.server = InferenceServer.create(config)
        self.latest_checkpoint = None
        self.shutdown_event = threading.Event()

    async def run(self):
        logger.info("Starting InferenceWorker...")

        try:
            server_task = asyncio.create_task(self.server.serve_async(host="0.0.0.0", port=8000))
            monitor_task = asyncio.create_task(self._checkpoint_monitor_loop())
            await asyncio.gather(server_task, monitor_task)
        except asyncio.CancelledError:
            logger.info("InferenceWorker shutting down...")
        finally:
            self.server.shutdown()

    async def _checkpoint_monitor_loop(self):
        """Monitor checkpoint directory for new checkpoints."""
        logger.info(f"Monitoring checkpoint directory: {self.checkpoint_path}")

        while not self.shutdown_event.is_set():
            try:
                new_checkpoint = self._find_latest_checkpoint()
                if new_checkpoint and new_checkpoint != self.latest_checkpoint:
                    logger.info(f"Found new checkpoint: {new_checkpoint}")
                    await self._reload_checkpoint(new_checkpoint)
                    self.latest_checkpoint = new_checkpoint

            except Exception as e:
                logger.error(f"Error checking for checkpoints: {e}", exc_info=True)

            # Wait for next check
            await asyncio.sleep(self.check_interval)

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the checkpoint directory."""
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return None

        # Look for checkpoint directories (e.g., checkpoint-1000, checkpoint-2000)
        checkpoint_paths = []
        for path in self.checkpoint_path.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint-"):
                step = path.name.split("-")[-1]
                checkpoint_paths.append((step, str(path)))

        if not checkpoint_paths:
            return None

        return max(checkpoint_paths, key=lambda x: int(x[0]))[1]

    async def _reload_checkpoint(self, checkpoint_path: str):
        """Reload the model from a checkpoint."""
        try:
            logger.info(f"Reloading model from checkpoint: {checkpoint_path}")
            # Run the reload in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: self.reload_from_checkpoint(checkpoint_path=checkpoint_path))
            logger.info("Model reload completed successfully")
        except Exception as e:
            logger.error(f"Failed to reload checkpoint {checkpoint_path}: {e}", exc_info=True)

    def reload_from_checkpoint(self, checkpoint_path: str):
        """Reload the model from a specific checkpoint path.

        Args:
            checkpoint_path: Path to the checkpoint directory
        """

        def weight_loader(current_model):
            """Load weights from checkpoint and return new model."""
            with (
                self.config.trainer.device_mesh,
                hax.axis_mapping(self.config.trainer.compute_axis_mapping),
            ):
                with use_cpu_device():
                    # Create eval shape of the model first
                    key = jrandom.PRNGKey(self.config.seed)
                    vocab_size = len(self.server.inference_context.tokenizer)
                    Vocab = round_axis_for_partitioning(
                        Axis("vocab", vocab_size), self.config.trainer.compute_axis_mapping
                    )
                    model = eqx.filter_eval_shape(self.config.model.build, Vocab, key=key)

                    # Load from checkpoint
                    model = load_checkpoint(model, checkpoint_path, subpath="model")

                    # Cast to compute precision
                    model = self.config.trainer.mp.cast_to_compute(model)

                return model

        # Use the server's reload method with our weight loader
        self.server.reload(weight_loader)

    def shutdown(self):
        """Shutdown the worker."""
        self.shutdown_event.set()
        self.server.shutdown()


def main(config: InferenceServerConfig):
    """Example main function showing how to use InferenceWorker."""
    checkpoint_path = config.checkpoint_path
    if checkpoint_path is None:
        logger.warning("No checkpoint_path specified in config; InferenceWorker will not monitor for checkpoints")
    worker = InferenceWorker(config=config, checkpoint_path=checkpoint_path, check_interval=60)  # Check every minute

    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
        worker.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    levanter.config.main(main)()
