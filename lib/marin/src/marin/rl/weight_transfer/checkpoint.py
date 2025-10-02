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
GCS checkpoint-based weight transfer implementation.

This module provides weight transfer using Levanter's checkpoint system for
saving to and loading from GCS (or other filesystems supported by fsspec).
"""

import dataclasses
import logging
import os
import threading
from collections import deque

import fsspec
import jax
import levanter.checkpoint as levanter_checkpoint
from haliax.partitioning import ResourceMapping
from jax.sharding import Mesh
from jaxtyping import PyTree

from .base import (
    WeightTransferClient,
    WeightTransferClientMetrics,
    WeightTransferConfig,
    WeightTransferServer,
    WeightTransferServerMetrics,
    WeightUpdate,
)

logger = logging.getLogger(__name__)


def _rm_thread(path: str) -> None:
    try:
        fs, _ = fsspec.core.url_to_fs(path)
        fs.rm(path, recursive=True)
    except Exception as e:
        logger.error(f"Failed to delete old checkpoint at {path}: {e}", exc_info=True)


class GCSCheckpointServer(WeightTransferServer):
    """GCS checkpoint-based weight transfer server using Levanter checkpointing."""

    def __init__(
        self,
        config: WeightTransferConfig,
        axis_mapping: ResourceMapping | None = None,
        mesh: Mesh | None = None,
    ):
        self.config = config
        self.checkpoint_queue = deque()
        self.axis_mapping = axis_mapping
        self.mesh = mesh
        self.metrics = WeightTransferServerMetrics()

    def serve_weights(self, weight_id: int, model: PyTree) -> None:
        """Save checkpoint using Levanter's checkpoint system."""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"step_{weight_id}")

        self.metrics.total_transfers += 1

        try:
            # Manage checkpoint queue
            if self.config.max_checkpoints is not None and len(self.checkpoint_queue) >= self.config.max_checkpoints:
                old_weight_id = self.checkpoint_queue.popleft()
                old_path = os.path.join(self.config.checkpoint_dir, f"step_{old_weight_id}")
                if jax.process_index() == 0:  # Only delete from coordinator
                    logger.info(f"Cleaning up old checkpoint at weight_id {old_weight_id} ({old_path})...")
                    fs, _ = fsspec.core.url_to_fs(old_path)
                    # Dispatch deletion to a separate thread to avoid blocking
                    if fs.exists(old_path):
                        threading.Thread(target=_rm_thread, args=(old_path,), daemon=True).start()

            logger.info(f"Saving checkpoint at weight_id {weight_id}...")

            levanter_checkpoint.save_checkpoint(
                tree=model,
                step=weight_id,
                checkpoint_path=checkpoint_path,
            )

            self.checkpoint_queue.append(weight_id)
            self.metrics.successful_transfers += 1
            logger.info(f"Checkpoint saved at {checkpoint_path}")

        except Exception as e:
            self.metrics.failed_transfers += 1
            logger.error(f"Failed to save checkpoint at weight_id {weight_id}: {e}")
            raise

    def cleanup(self) -> None:
        """No cleanup needed for GCS checkpoints."""
        pass

    def get_metrics(self) -> dict:
        return dataclasses.asdict(self.metrics)


class GCSCheckpointClient(WeightTransferClient):
    """GCS checkpoint-based weight transfer client using Levanter checkpointing."""

    def __init__(
        self,
        config: WeightTransferConfig,
        axis_mapping: ResourceMapping | None = None,
        mesh: Mesh | None = None,
    ):
        self.config = config
        self.axis_mapping = axis_mapping
        self.mesh = mesh
        self.latest_checkpoint_path = None
        self.metrics = WeightTransferClientMetrics()

    def receive_weights(self, old_model: PyTree) -> WeightUpdate | None:
        """Load latest checkpoint using Levanter's checkpoint system."""
        self.metrics.total_polls += 1
        result = self._find_latest_checkpoint()

        if result is None:
            logger.info("No new checkpoint found.")
            return None

        latest_checkpoint, weight_step = result

        try:
            if latest_checkpoint == self.latest_checkpoint_path:
                logger.info("No new checkpoint found.")
                return None

            logger.info(f"Loading checkpoint from {latest_checkpoint}")
            params = levanter_checkpoint.load_checkpoint(
                tree=old_model,
                checkpoint_path=latest_checkpoint,
                axis_mapping=self.axis_mapping,
                mesh=self.mesh,
            )
        except Exception as e:
            # might get stuck if checkpoint is being written
            self.metrics.failed_receives += 1
            logger.warning(f"Failed to load checkpoint {latest_checkpoint}: {e}")
            return None

        self.latest_checkpoint_path = latest_checkpoint
        self.metrics.successful_receives += 1

        return WeightUpdate(model=params, weight_id=weight_step)

    def _find_latest_checkpoint(self) -> tuple[str, int] | None:
        """Find the latest checkpoint in the checkpoint directory."""
        logger.info(f"Search for new checkpoints in {self.config.checkpoint_dir}...")
        fs, path_in_fs = fsspec.core.url_to_fs(self.config.checkpoint_dir, use_listings_cache=False)
        if not fs.exists(path_in_fs):
            return None

        # Checkpoint format is {checkpoint_dir}/step_{xyz}
        # Make sure we expire before our poll-interval.
        # We disable caching above, but it's unclear if fsspec adheres to this.
        dirs = fs.ls(path_in_fs, listings_expiry_time=self.config.poll_interval_seconds)
        checkpoint_dirs = []
        for d in dirs:
            # Handle trailing slashes in directory names
            step_name = d.rstrip("/").split("/")[-1]
            if step_name.startswith("step_"):
                step_num = int(step_name.split("_")[-1])
                logger.info(f"Found checkpoint directory: {d} with step number {step_num}")
                checkpoint_dirs.append((step_num, d))
        if not checkpoint_dirs:
            return None

        step_num, latest_dir = max(checkpoint_dirs, key=lambda x: x[0])
        # Reconstruct full URL with original scheme
        if "://" in str(self.config.checkpoint_dir) and "://" not in latest_dir:
            scheme = self.config.checkpoint_dir.split("://")[0]
            return f"{scheme}://{latest_dir}", step_num
        return latest_dir, step_num

    def cleanup(self) -> None:
        """No cleanup needed for GCS checkpoints."""
        pass

    def get_metrics(self) -> dict:
        return dataclasses.asdict(self.metrics)
