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
This includes:

* input_ids: The input token IDs.
* attention_mask: The attention mask for the input.
* position_ids: The position IDs for the input.
* target_ids: The target token IDs for computing loss.
* loss_weights: Weights for each token in the loss computation.
* loss_masks: Masks indicating which tokens contribute to the loss.

A RolloutBatch consists of the model information in addition to:

* Environment name
* Worker ID
* Creation timestamp
* Rollout ID

During training, rollout batches are read from the file queue and compacted into
training batches continously. Training data prioritizes recent rollouts and an
even mix of data across environments.

Training workers read all files, and use their process_index() to slice out the
appropriate subset of data for training.

"""

import logging
import pickle
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from enum import Enum

import fsspec

from .types import RolloutBatch

logger = logging.getLogger(__name__)


class StorageType(Enum):
    """Type of rollout storage backend."""

    FILE = "file"
    IN_MEMORY = "in_memory"


# Global registry for named in-memory queues
_MEMORY_QUEUES: dict[str, "InMemoryRolloutQueue"] = {}


def _get_or_create_queue(queue_name: str) -> "InMemoryRolloutQueue":
    """Get or create a named in-memory queue."""
    if queue_name not in _MEMORY_QUEUES:
        _MEMORY_QUEUES[queue_name] = InMemoryRolloutQueue()
    return _MEMORY_QUEUES[queue_name]


@dataclass
class RolloutStorageConfig:
    """Configuration for rollout storage backend."""

    storage_type: StorageType
    # For file storage
    path: str | None = None
    poll_interval: float = 1.0
    max_rollout_files: int = 32
    # For in-memory storage
    queue_name: str | None = None

    def create_reader(self) -> "RolloutReader":
        if self.storage_type == StorageType.FILE:
            if self.path is None:
                raise ValueError("path must be specified for FILE storage type")
            return FileRolloutReader(self.path, self.poll_interval)
        else:
            if self.queue_name is None:
                raise ValueError("queue_name must be specified for IN_MEMORY storage type")
            return _get_or_create_queue(self.queue_name).reader()

    def create_writer(self) -> "RolloutWriter":
        if self.storage_type == StorageType.FILE:
            if self.path is None:
                raise ValueError("path must be specified for FILE storage type")
            return FileRolloutWriter(self.path, self.max_rollout_files)
        else:
            if self.queue_name is None:
                raise ValueError("queue_name must be specified for IN_MEMORY storage type")
            return _get_or_create_queue(self.queue_name).writer()


class RolloutReader(ABC):
    """Abstract interface for reading rollout batches."""

    @abstractmethod
    def read_batch(self, timeout: float | None = None) -> RolloutBatch | None:
        """Read a single batch with optional timeout.

        Args:
            timeout: Maximum time to wait in seconds. None means no wait.

        Returns:
            RolloutBatch if available, None if timeout or no batches.
        """
        pass

    @abstractmethod
    def read_all_available(self) -> list[RolloutBatch]:
        """Read all currently available batches without blocking.

        Returns:
            List of all available batches (may be empty).
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

    def __init__(
        self,
        path: str,
        poll_interval: float = 1.0,
    ):
        """Initialize file-based rollout reader.

        Args:
            path: Storage directory or GCS bucket
            poll_interval: Interval in seconds between polls when waiting.
        """
        self.path = path.rstrip("/")
        self.poll_interval = poll_interval

        # Create filesystem instance
        storage_options = fsspec.utils.infer_storage_options(path)  # type: ignore[attr-defined]
        self.fs = fsspec.filesystem(storage_options["protocol"] or "file")

        # Track already-read files to avoid re-reading
        self._read_files: set[str] = set()

        logger.info(f"Initialized file rollout reader at {path}")

    def _get_available_files(self) -> list[str]:
        """Get list of available batch files sorted by timestamp."""
        pattern = f"{self.path}/*.pkl"
        files = self.fs.glob(pattern)

        # Parse and sort files by timestamp
        parsed_files = []
        for file_path in files:
            filename = file_path.split("/")[-1]
            if filename.endswith(".pkl"):
                # Extract timestamp from filename: timestamp_hostname_counter.pkl
                parts = filename.split("_")
                if len(parts) == 3:
                    timestamp_int = int(parts[0])
                    timestamp = timestamp_int / 1000000.0  # Convert back from microseconds
                    parsed_files.append((timestamp, file_path))
                else:
                    logger.warning(f"Unexpected filename format: {filename}")

        # Sort by timestamp and return file paths
        parsed_files.sort(key=lambda x: x[0])
        return [file_path for _, file_path in parsed_files]

    def read_batch(self, timeout: float | None = None) -> RolloutBatch | None:
        """Read a single batch with optional timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            available_files = self._get_available_files()
            for file_path in available_files:
                if file_path not in self._read_files:
                    self._read_files.add(file_path)
                    try:
                        with self.fs.open(file_path, "rb") as f:
                            return pickle.load(f)
                    except Exception as e:
                        # a file might be deleted while we're trying to read it, or corrupted
                        logger.error(f"Failed to read rollout file {file_path}: {e}")
                        continue

            time.sleep(self.poll_interval)

    def read_all_available(self) -> list[RolloutBatch]:
        """Read all currently available batches without blocking."""
        batches = []
        available_files = self._get_available_files()

        for file_path in available_files:
            if file_path not in self._read_files:
                self._read_files.add(file_path)
                with self.fs.open(file_path, "rb") as f:
                    batches.append(pickle.load(f))

        return batches


class FileRolloutWriter(RolloutWriter):
    """File-based rollout writer using fsspec for various storage backends."""

    def __init__(self, path: str, max_rollout_files: int = 32):
        """Initialize file-based rollout writer.

        Args:
            path: Storage path (supports local filesystem, GCS, S3, etc.).
        """
        self.path = path.rstrip("/")
        self.hostname = socket.gethostname()
        self.max_rollout_files = max_rollout_files

        # Create filesystem instance
        storage_options = fsspec.utils.infer_storage_options(path)  # type: ignore[attr-defined]
        self.fs = fsspec.filesystem(storage_options["protocol"] or "file")

        # Create output directory structure
        self._ensure_directories()
        self._file_queue: deque[str] = deque()

        self._batch_counter = 0

        logger.info(f"Initialized file rollout writer at {path} (hostname: {self.hostname})")

    def _ensure_directories(self):
        """Ensure output directory structure exists."""
        self.fs.makedirs(self.path, exist_ok=True)

    def _get_batch_path(self, timestamp: float, counter: int) -> str:
        """Get path for batch with timestamp and hostname."""
        timestamp_int = int(timestamp * 1000000)  # microseconds for ordering
        return f"{self.path}/{timestamp_int:020d}_{self.hostname}_{counter:06d}.pkl"

    def write_batch(self, batch: RolloutBatch) -> None:
        """Write batch to storage with all required fields."""
        timestamp = time.time()

        batch_path = self._get_batch_path(timestamp, self._batch_counter)
        with self.fs.open(batch_path, "wb") as f:
            pickle.dump(batch, f)

        self._file_queue.append(batch_path)
        if len(self._file_queue) > self.max_rollout_files:
            stale_file = self._file_queue.popleft()
            try:
                self.fs.delete(stale_file)
            except Exception as e:
                logger.warning(f"Failed to delete stale rollout file {stale_file}: {e}")

        logger.debug(f"Wrote batch {batch_path}")
        self._batch_counter += 1

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
        self._lock = threading.Lock()

    def reader(self) -> "InMemoryRolloutReader":
        """Create a reader for this queue."""
        return InMemoryRolloutReader(self)

    def writer(self) -> "InMemoryRolloutWriter":
        """Create a writer for this queue."""
        return InMemoryRolloutWriter(self)

    def clear_queue(self) -> None:
        """Clear all batches from the queue."""
        with self._lock:
            self._queue.clear()


class InMemoryRolloutReader(RolloutReader):
    """In-memory rollout reader for testing."""

    def __init__(self, queue: InMemoryRolloutQueue):
        self._queue = queue
        self._read_index = 0

    def read_batch(self, timeout: float | None = None) -> RolloutBatch | None:
        """Read a single batch with optional timeout."""
        start_time = time.time()

        while True:
            with self._queue._lock:
                if self._read_index < len(self._queue._queue):
                    batch = self._queue._queue[self._read_index]
                    self._read_index += 1
                    return batch

            # No new batches available
            if timeout is None or timeout <= 0:
                return None

            # Check timeout
            if time.time() - start_time >= timeout:
                return None

            # Wait a bit before checking again
            time.sleep(min(0.01, timeout - (time.time() - start_time)))

    def read_all_available(self) -> list[RolloutBatch]:
        """Read all currently available batches without blocking."""
        with self._queue._lock:
            batches = self._queue._queue[self._read_index :]
            self._read_index = len(self._queue._queue)
            return batches


class InMemoryRolloutWriter(RolloutWriter):
    """In-memory rollout writer for testing."""

    def __init__(self, queue: InMemoryRolloutQueue):
        self._queue = queue

    def write_batch(self, batch: RolloutBatch) -> None:
        """Write batch to memory queue."""
        with self._queue._lock:
            self._queue._queue.append(batch)
