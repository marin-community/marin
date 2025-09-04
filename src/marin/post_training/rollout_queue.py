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

"""RolloutQueue interface and implementations for training worker communication."""

import json
import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
from google.cloud import storage

from ..utilities.gcs_utils import split_gcs_path

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


class RolloutQueue(ABC):
    """Abstract interface for rollout queue implementations."""

    @abstractmethod
    def read_batch(self, timeout: Optional[float] = None) -> Optional[RolloutBatch]:
        """Read a single batch from the queue.
        
        Args:
            timeout: Timeout in seconds. If None, blocks indefinitely.
            
        Returns:
            RolloutBatch if available, None if timeout or no data.
        """
        pass

    @abstractmethod
    def read_batches(self, n_batches: int, timeout: Optional[float] = None) -> Iterator[RolloutBatch]:
        """Read multiple batches from the queue.
        
        Args:
            n_batches: Number of batches to read.
            timeout: Timeout in seconds for each batch read.
            
        Yields:
            RolloutBatch instances.
        """
        pass

    @abstractmethod
    def write_batch(self, batch: RolloutBatch) -> None:
        """Write a batch to the queue.
        
        Args:
            batch: RolloutBatch to write to the queue.
        """
        pass

    @abstractmethod
    def get_queue_size(self) -> int:
        """Get current queue size.
        
        Returns:
            Number of batches currently in the queue.
        """
        pass


class GCSRolloutQueue(RolloutQueue):
    """GCS-based implementation of RolloutQueue.
    
    Uses GCS bucket as a queue by writing numbered batch files.
    Implements a simple queue using file naming with sequential indices.
    """

    def __init__(
        self,
        bucket_name: str,
        queue_path: str = "rollout_queue",
        batch_prefix: str = "batch_",
        poll_interval: float = 1.0,
        max_retries: int = 3,
    ):
        """Initialize GCS rollout queue.
        
        Args:
            bucket_name: GCS bucket name.
            queue_path: Path within bucket for queue files.
            batch_prefix: Prefix for batch files.
            poll_interval: Interval in seconds between polls when waiting.
            max_retries: Maximum retries for GCS operations.
        """
        self.bucket_name = bucket_name
        self.queue_path = queue_path
        self.batch_prefix = batch_prefix
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        self._read_index = self._get_latest_read_index()
        self._write_index = self._get_latest_write_index()
        
        logger.info(f"Initialized GCS queue at gs://{bucket_name}/{queue_path}")
        logger.info(f"Read index: {self._read_index}, Write index: {self._write_index}")

    def _get_batch_blob_name(self, index: int) -> str:
        """Get blob name for batch at given index."""
        return f"{self.queue_path}/{self.batch_prefix}{index:010d}.pkl"

    def _get_latest_read_index(self) -> int:
        """Get the latest read index from GCS metadata or start from 0."""
        try:
            metadata_blob = self.bucket.blob(f"{self.queue_path}/read_index.json")
            if metadata_blob.exists():
                metadata = json.loads(metadata_blob.download_as_text())
                return metadata.get("read_index", 0)
        except Exception as e:
            logger.warning(f"Failed to read read_index metadata: {e}")
        return 0

    def _get_latest_write_index(self) -> int:
        """Get the latest write index by scanning existing batch files."""
        try:
            prefix = f"{self.queue_path}/{self.batch_prefix}"
            blobs = list(self.client.list_blobs(self.bucket, prefix=prefix))
            
            if not blobs:
                return 0
                
            indices = []
            for blob in blobs:
                try:
                    # Extract index from filename like "batch_0000000042.pkl"
                    filename = blob.name.split("/")[-1]
                    if filename.startswith(self.batch_prefix) and filename.endswith(".pkl"):
                        index_str = filename[len(self.batch_prefix):-4]
                        indices.append(int(index_str))
                except (ValueError, IndexError):
                    continue
            
            return max(indices) + 1 if indices else 0
            
        except Exception as e:
            logger.warning(f"Failed to scan write indices: {e}")
            return 0

    def _save_read_index(self) -> None:
        """Save current read index to GCS."""
        try:
            metadata_blob = self.bucket.blob(f"{self.queue_path}/read_index.json")
            metadata = {"read_index": self._read_index, "timestamp": time.time()}
            metadata_blob.upload_from_string(json.dumps(metadata))
        except Exception as e:
            logger.warning(f"Failed to save read_index metadata: {e}")

    def read_batch(self, timeout: Optional[float] = None) -> Optional[RolloutBatch]:
        """Read next batch from GCS queue."""
        start_time = time.time()
        
        while True:
            blob_name = self._get_batch_blob_name(self._read_index)
            blob = self.bucket.blob(blob_name)
            
            for attempt in range(self.max_retries):
                try:
                    if blob.exists():
                        # Download and deserialize batch
                        batch_data = blob.download_as_bytes()
                        batch = pickle.loads(batch_data)
                        
                        # Increment read index and save
                        self._read_index += 1
                        self._save_read_index()
                        
                        logger.debug(f"Read batch {self._read_index - 1} from GCS")
                        return batch
                    else:
                        break  # Blob doesn't exist, wait or timeout
                        
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed to read batch {self._read_index}: {e}")
                    if attempt == self.max_retries - 1:
                        logger.error(f"Failed to read batch {self._read_index} after {self.max_retries} attempts")
                        return None
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
            
            # Check timeout
            if timeout is not None and (time.time() - start_time) >= timeout:
                logger.debug(f"Timeout reached while waiting for batch {self._read_index}")
                return None
            
            # Wait before polling again
            time.sleep(self.poll_interval)

    def read_batches(self, n_batches: int, timeout: Optional[float] = None) -> Iterator[RolloutBatch]:
        """Read multiple batches from GCS queue."""
        for _ in range(n_batches):
            batch = self.read_batch(timeout=timeout)
            if batch is None:
                break
            yield batch

    def write_batch(self, batch: RolloutBatch) -> None:
        """Write batch to GCS queue."""
        blob_name = self._get_batch_blob_name(self._write_index)
        blob = self.bucket.blob(blob_name)
        
        for attempt in range(self.max_retries):
            try:
                # Serialize and upload batch
                batch_data = pickle.dumps(batch)
                blob.upload_from_string(batch_data)
                
                logger.debug(f"Wrote batch {self._write_index} to GCS")
                self._write_index += 1
                return
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed to write batch {self._write_index}: {e}")
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to write batch {self._write_index} after {self.max_retries} attempts")
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff

    def get_queue_size(self) -> int:
        """Get current queue size (approximate)."""
        try:
            return max(0, self._write_index - self._read_index)
        except Exception:
            return 0

    def clear_queue(self) -> None:
        """Clear all batches from the queue (for testing/debugging)."""
        try:
            prefix = f"{self.queue_path}/"
            blobs = list(self.client.list_blobs(self.bucket, prefix=prefix))
            
            for blob in blobs:
                blob.delete()
                
            self._read_index = 0
            self._write_index = 0
            
            logger.info(f"Cleared queue at gs://{self.bucket_name}/{self.queue_path}")
            
        except Exception as e:
            logger.error(f"Failed to clear queue: {e}")
            raise


class InMemoryRolloutQueue(RolloutQueue):
    """In-memory implementation for testing and development."""

    def __init__(self):
        self._queue: list[RolloutBatch] = []
        self._read_index = 0

    def read_batch(self, timeout: Optional[float] = None) -> Optional[RolloutBatch]:
        """Read next batch from memory queue."""
        if self._read_index < len(self._queue):
            batch = self._queue[self._read_index]
            self._read_index += 1
            return batch
        return None

    def read_batches(self, n_batches: int, timeout: Optional[float] = None) -> Iterator[RolloutBatch]:
        """Read multiple batches from memory queue."""
        for _ in range(n_batches):
            batch = self.read_batch()
            if batch is None:
                break
            yield batch

    def write_batch(self, batch: RolloutBatch) -> None:
        """Write batch to memory queue."""
        self._queue.append(batch)

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return max(0, len(self._queue) - self._read_index)

    def clear_queue(self) -> None:
        """Clear all batches from the queue."""
        self._queue.clear()
        self._read_index = 0