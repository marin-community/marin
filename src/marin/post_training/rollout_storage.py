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
Queues for communicating rollout information between inference and training workers.

A rollout is a set of ndarrays containing all of the information required for training a model.
This include:

* input_ids: The input token IDs.
* attention_mask: The attention mask for the input.
* position_ids: The position IDs for the input.
* target_ids: The target token IDs for computing loss.
* loss_weights: Weights for each token in the loss computation.
* loss_masks: Masks indicating which tokens contribute to the loss.
* reference_logprobs: Log probabilities from the reference model.
* metadata: Additional metadata about the rollout.
"""

import logging
import pickle
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import fsspec
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RolloutBatch:
    """Single batch of rollout data from rollout workers."""

    input_ids: np.ndarray
    attention_mask: np.ndarray
    position_ids: np.ndarray
    target_ids: np.ndarray
    loss_weights: np.ndarray
    loss_masks: np.ndarray
    reference_logprobs: np.ndarray
    metadata: dict


class RolloutReader(ABC):
    """Abstract interface for reading rollout batches."""

    @abstractmethod
    def read_batch(self, timeout: float | None = None) -> RolloutBatch | None:
        """Read a single batch.

        Args:
            timeout: Timeout in seconds. If None, blocks indefinitely.

        Returns:
            RolloutBatch if available, None if timeout or no data.
        """
        pass


class RolloutWriter(ABC):
    """Abstract interface for writing rollout batches."""

    @abstractmethod
    def write_batch(self, batch: RolloutBatch) -> None:
        """Write a batch.

        Args:
            batch: RolloutBatch to write.
        """
        pass


class FileRolloutReader(RolloutReader):
    """File-based rollout reader using fsspec for various storage backends."""

    path: str
    batch_prefix: str
    poll_interval: float
    _pending_files: list[str]

    def __init__(
        self,
        path: str,
        batch_prefix: str = "batch_",
        poll_interval: float = 1.0,
    ):
        """Initialize file-based rollout reader.

        Args:
            path: Storage directory or GCS bucket
            batch_prefix: Prefix for batch files.
            poll_interval: Interval in seconds between polls when waiting.
        """
        self.path = path.rstrip("/")
        self.batch_prefix = batch_prefix
        self.poll_interval = poll_interval
        self._pending_files = []

        # Create filesystem instance
        storage_options = fsspec.utils.infer_storage_options(path)  # type: ignore[attr-defined]
        self.fs = fsspec.filesystem(storage_options["protocol"] or "file")

        logger.info(f"Initialized file rollout reader at {path}")

    def _get_available_files(self) -> list[str]:
        """Get list of available batch files sorted by timestamp."""
        pattern = f"{self.path}/{self.batch_prefix}*.pkl"
        files = self.fs.glob(pattern)

        # Parse and sort files by timestamp
        parsed_files = []
        for file_path in files:
            filename = file_path.split("/")[-1]
            if filename.startswith(self.batch_prefix) and filename.endswith(".pkl"):
                # Extract timestamp from filename: batch_TIMESTAMP_hostname_counter.pkl
                parts = filename[len(self.batch_prefix) : -4].split("_")
                if len(parts) >= 3:
                    timestamp_int = int(parts[0])
                    timestamp = timestamp_int / 1000000.0  # Convert back from microseconds
                    parsed_files.append((timestamp, file_path))

        # Sort by timestamp and return file paths
        parsed_files.sort(key=lambda x: x[0])
        return [file_path for _, file_path in parsed_files]

    def read_batch(self, timeout: float | None = None) -> RolloutBatch | None:
        """Read next batch from storage."""
        start_time = time.time()
        while (time.time() - start_time < timeout) or (not timeout):
            if not self._pending_files:
                self._pending_files = self._get_available_files()
            if not self._pending_files:
                time.sleep(self.poll_interval)
                continue

            batch_path = self._pending_files[0]
            self._pending_files.pop(0)
            with self.fs.open(batch_path, "rb") as f:
                batch = pickle.load(f)
            return batch


class FileRolloutWriter(RolloutWriter):
    """File-based rollout writer using fsspec for various storage backends."""

    def __init__(
        self,
        path: str,
        batch_prefix: str = "batch_",
    ):
        """Initialize file-based rollout writer.

        Args:
            path: Storage path (supports local filesystem, GCS, S3, etc.).
            batch_prefix: Prefix for batch files.
        """
        self.path = path.rstrip("/")
        self.batch_prefix = batch_prefix
        self.hostname = socket.gethostname()

        # Create filesystem instance
        storage_options = fsspec.utils.infer_storage_options(path)  # type: ignore[attr-defined]
        self.fs = fsspec.filesystem(storage_options["protocol"] or "file")

        # Create output directory structure
        self._ensure_directories()

        self._batch_counter = 0

        logger.info(f"Initialized file rollout writer at {path} (hostname: {self.hostname})")

    def _ensure_directories(self):
        """Ensure output directory structure exists."""
        self.fs.makedirs(self.path, exist_ok=True)

    def _get_batch_path(self, timestamp: float, counter: int) -> str:
        """Get path for batch with timestamp and hostname."""
        timestamp_int = int(timestamp * 1000000)  # microseconds for ordering
        return f"{self.path}/{self.batch_prefix}{timestamp_int:020d}_{self.hostname}_{counter:06d}.pkl"

    def write_batch(self, batch: RolloutBatch) -> None:
        """Write batch to storage."""
        timestamp = time.time()
        batch_path = self._get_batch_path(timestamp, self._batch_counter)
        with self.fs.open(batch_path, "wb") as f:
            pickle.dump(batch, f)

        logger.debug(f"Wrote batch {batch_path}")
        self._batch_counter += 1
        return

    def clear_queue(self) -> None:
        """Clear all batches from the queue (for testing/debugging)."""
        pattern = f"{self.path}/*"
        files = self.fs.glob(pattern)

        for file_path in files:
            self.fs.delete(file_path)

        self._batch_counter = 0

        logger.info(f"Cleared queue at {self.path}")


class InMemoryRolloutQueue:
    """In-memory rollout queue for testing and development."""

    def __init__(self):
        self._queue: list[RolloutBatch] = []
        self._read_index = 0

    def reader(self) -> "InMemoryRolloutReader":
        """Create a reader for this queue."""
        return InMemoryRolloutReader(self)

    def writer(self) -> "InMemoryRolloutWriter":
        """Create a writer for this queue."""
        return InMemoryRolloutWriter(self)

    def clear_queue(self) -> None:
        """Clear all batches from the queue."""
        self._queue.clear()
        self._read_index = 0


class InMemoryRolloutReader(RolloutReader):
    """In-memory rollout reader for testing."""

    def __init__(self, queue: InMemoryRolloutQueue):
        self._queue = queue

    def read_batch(self, timeout: float | None = None) -> RolloutBatch | None:
        """Read next batch from memory queue."""
        import time

        start_time = time.time()
        poll_interval = 0.1  # Poll every 100ms

        while True:
            if self._queue._read_index < len(self._queue._queue):
                batch = self._queue._queue[self._queue._read_index]
                self._queue._read_index += 1
                return batch

            # Check timeout
            if timeout is not None and (time.time() - start_time) >= timeout:
                return None

            # Sleep and try again
            time.sleep(poll_interval)


class InMemoryRolloutWriter(RolloutWriter):
    """In-memory rollout writer for testing."""

    def __init__(self, queue: InMemoryRolloutQueue):
        self._queue = queue
        self._metadata = {}

    def write_batch(self, batch: RolloutBatch) -> None:
        """Write batch to memory queue."""
        self._queue._queue.append(batch)

    def get_metadata(self) -> dict[str, Any]:
        """Get stored metadata."""
        return self._metadata.copy()
