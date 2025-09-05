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

import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
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

    @abstractmethod
    def read_batches(self, n_batches: int, timeout: float | None = None) -> Iterator[RolloutBatch]:
        """Read multiple batches.

        Args:
            n_batches: Number of batches to read.
            timeout: Timeout in seconds for each batch read.

        Yields:
            RolloutBatch instances.
        """
        pass

    @abstractmethod
    def get_queue_size(self) -> int:
        """Get current queue size.

        Returns:
            Number of batches currently available.
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

    @abstractmethod
    def write_metadata(self, metadata: dict[str, Any]) -> None:
        """Write metadata about the rollout generation.

        Args:
            metadata: Dictionary of metadata to write.
        """
        pass

    @abstractmethod
    def get_queue_size(self) -> int:
        """Get current queue size.

        Returns:
            Number of batches currently in queue.
        """
        pass


class FileRolloutReader(RolloutReader):
    """File-based rollout reader using fsspec for various storage backends."""

    def __init__(
        self,
        path: str,
        batch_prefix: str = "batch_",
        poll_interval: float = 1.0,
    ):
        """Initialize file-based rollout reader.

        Args:
            path: Storage path (supports local filesystem, GCS, S3, etc.).
            batch_prefix: Prefix for batch files.
            poll_interval: Interval in seconds between polls when waiting.
        """
        self.path = path.rstrip("/")
        self.batch_prefix = batch_prefix
        self.poll_interval = poll_interval

        # Create filesystem instance
        self.fs = fsspec.filesystem(fsspec.utils.infer_storage_options(path)["protocol"] or "file")

        self._read_index = self._get_latest_read_index()
        self._write_index = self._get_latest_write_index()

        logger.info(f"Initialized file rollout reader at {path}")
        logger.info(f"Read index: {self._read_index}, Write index: {self._write_index}")

    def _get_batch_path(self, index: int) -> str:
        """Get path for batch at given index."""
        return f"{self.path}/{self.batch_prefix}{index:010d}.pkl"

    def _get_latest_read_index(self) -> int:
        """Get the latest read index from metadata or start from 0."""
        try:
            metadata_path = f"{self.path}/read_index.json"
            if self.fs.exists(metadata_path):
                with self.fs.open(metadata_path, "r") as f:
                    metadata = json.load(f)
                return metadata.get("read_index", 0)
        except Exception as e:
            logger.warning(f"Failed to read read_index metadata: {e}")
        return 0

    def _get_latest_write_index(self) -> int:
        """Get the latest write index by scanning existing batch files."""
        try:
            pattern = f"{self.path}/{self.batch_prefix}*.pkl"
            files = self.fs.glob(pattern)

            if not files:
                return 0

            indices = []
            for file_path in files:
                try:
                    filename = file_path.split("/")[-1]
                    if filename.startswith(self.batch_prefix) and filename.endswith(".pkl"):
                        index_str = filename[len(self.batch_prefix) : -4]
                        indices.append(int(index_str))
                except (ValueError, IndexError):
                    continue

            return max(indices) + 1 if indices else 0

        except Exception as e:
            logger.warning(f"Failed to scan write indices: {e}")
            return 0

    def _save_read_index(self) -> None:
        """Save current read index."""
        try:
            metadata_path = f"{self.path}/read_index.json"
            metadata = {"read_index": self._read_index, "timestamp": time.time()}
            with self.fs.open(metadata_path, "w") as f:
                json.dump(metadata, f)
        except Exception as e:
            logger.warning(f"Failed to save read_index metadata: {e}")

    def read_batch(self, timeout: float | None = None) -> RolloutBatch | None:
        """Read next batch from storage."""
        start_time = time.time()

        while True:
            batch_path = self._get_batch_path(self._read_index)

            if self.fs.exists(batch_path):
                with self.fs.open(batch_path, "rb") as f:
                    batch = pickle.load(f)

                self._read_index += 1
                self._save_read_index()

                logger.debug(f"Read batch {self._read_index - 1}")
                return batch

            time.sleep(self.poll_interval)

            if timeout is not None and (time.time() - start_time) >= timeout:
                return None

    def read_batches(self, n_batches: int, timeout: float | None = None) -> Iterator[RolloutBatch]:
        """Read multiple batches from storage."""
        for _ in range(n_batches):
            batch = self.read_batch(timeout=timeout)
            if batch is None:
                break
            yield batch

    def get_queue_size(self) -> int:
        """Get current queue size (approximate)."""
        try:
            return max(0, self._write_index - self._read_index)
        except Exception:
            return 0


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

        # Create filesystem instance
        self.fs = fsspec.filesystem(fsspec.utils.infer_storage_options(path)["protocol"] or "file")

        # Create output directory structure
        self._ensure_directories()

        self._write_index = self._get_latest_write_index()

        logger.info(f"Initialized file rollout writer at {path}")
        logger.info(f"Write index: {self._write_index}")

    def _ensure_directories(self):
        """Ensure output directory structure exists."""
        self.fs.makedirs(self.path, exist_ok=True)

    def _get_batch_path(self, index: int) -> str:
        """Get path for batch at given index."""
        return f"{self.path}/{self.batch_prefix}{index:010d}.pkl"

    def _get_latest_write_index(self) -> int:
        """Get the latest write index by scanning existing batch files."""
        pattern = f"{self.path}/{self.batch_prefix}*.pkl"
        files = self.fs.glob(pattern)

        if not files:
            return 0

        indices = []
        for file_path in files:
            filename = file_path.split("/")[-1]
            if filename.startswith(self.batch_prefix) and filename.endswith(".pkl"):
                index_str = filename[len(self.batch_prefix) : -4]
                indices.append(int(index_str))
        return max(indices) + 1 if indices else 0

    def write_batch(self, batch: RolloutBatch) -> None:
        """Write batch to storage."""
        batch_path = self._get_batch_path(self._write_index)
        with self.fs.open(batch_path, "wb") as f:
            pickle.dump(batch, f)

        logger.debug(f"Wrote batch {self._write_index}")
        self._write_index += 1
        return

    def get_queue_size(self) -> int:
        """Get current queue size (approximate)."""
        try:
            read_index = self._get_latest_read_index()
            return max(0, self._write_index - read_index)
        except Exception:
            return 0

    def _get_latest_read_index(self) -> int:
        """Get the latest read index from metadata or start from 0."""
        metadata_path = f"{self.path}/read_index.json"
        if self.fs.exists(metadata_path):
            with self.fs.open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata.get("read_index", 0)
        logger.debug("No read index metadata found.")
        return 0

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        """Write metadata about the rollout generation."""
        metadata_path = f"{self.path}/metadata.json"

        metadata = {
            **metadata,
            "last_updated": time.time(),
        }

        with self.fs.open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Updated metadata at {metadata_path}")

    def clear_queue(self) -> None:
        """Clear all batches from the queue (for testing/debugging)."""
        pattern = f"{self.path}/*"
        files = self.fs.glob(pattern)

        for file_path in files:
            self.fs.delete(file_path)

        self._write_index = 0

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

    def read_batches(self, n_batches: int, timeout: float | None = None) -> Iterator[RolloutBatch]:
        """Read multiple batches from memory queue."""
        for _ in range(n_batches):
            batch = self.read_batch()
            if batch is None:
                break
            yield batch

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return max(0, len(self._queue._queue) - self._queue._read_index)


class InMemoryRolloutWriter(RolloutWriter):
    """In-memory rollout writer for testing."""

    def __init__(self, queue: InMemoryRolloutQueue):
        self._queue = queue
        self._metadata = {}

    def write_batch(self, batch: RolloutBatch) -> None:
        """Write batch to memory queue."""
        self._queue._queue.append(batch)

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return max(0, len(self._queue._queue) - self._queue._read_index)

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        """Write metadata (stored in memory for testing)."""
        self._metadata.update(metadata)

    def get_metadata(self) -> dict[str, Any]:
        """Get stored metadata."""
        return self._metadata.copy()
