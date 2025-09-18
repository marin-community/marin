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
Weight transfer management.

This module provides abstractions for communicating weights between training and inference workers.

Currently GCS, Ray remoting, and JAX transfer server methods are supported.
"""

import asyncio
import logging
import os
import queue
import time
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import haliax as hax
import jax
import levanter.checkpoint as levanter_checkpoint
import numpy as np
import ray
from haliax.partitioning import ResourceMapping
from haliax.util import is_named_array
from jax.sharding import Mesh
from jaxtyping import PyTree

from .flax.utils import delete_with_bucket, jax_distributed_barrier


class WeightTransferMode(Enum):
    GCS_CHECKPOINT = "gcs_checkpoint"
    JAX_TRANSFER_SERVER = "jax_transfer_server"
    RAY_REMOTING = "ray_remoting"


@dataclass
class WeightTransferServerMetrics:
    """Metrics for weight transfer servers."""

    total_transfers: int = 0
    successful_transfers: int = 0
    failed_transfers: int = 0
    start_time: float = 0.0

    @property
    def transfer_rate(self) -> float:
        """Transfers per second."""
        elapsed = time.time() - self.start_time
        return self.successful_transfers / max(elapsed, 1.0)


@dataclass
class WeightTransferClientMetrics:
    """Metrics for weight transfer clients."""

    total_polls: int = 0
    successful_receives: int = 0
    failed_receives: int = 0
    start_time: float = 0.0

    @property
    def poll_rate(self) -> float:
        """Polls per second."""
        elapsed = time.time() - self.start_time
        return self.total_polls / max(elapsed, 1.0)

    @property
    def success_rate(self) -> float:
        """Ratio of successful receives to total polls."""
        return self.successful_receives / max(self.total_polls, 1)


@dataclass
class WeightTransferConfig:
    mode: WeightTransferMode = WeightTransferMode.GCS_CHECKPOINT
    # Common settings
    sync_interval_steps: int = 100
    poll_interval_seconds: float = 30.0

    # RAY_REMOTING and JAX_TRANSFER_SERVER specific
    transfer_timeout: float = 120.0

    # GCS Checkpoint specific
    checkpoint_dir: str = ""
    max_checkpoints: int | None = 5


# Check if JAX transfer is available
try:
    from .jax_weight_transfer import (
        WeightTransferCoordinator,
        instantiate_coordinator,
        process_weight_transfers,
        receive_weight_transfers,
        start_transfer_server,
    )

    JAX_TRANSFER_AVAILABLE = True
except (ImportError, AttributeError):
    JAX_TRANSFER_AVAILABLE = False
    WeightTransferCoordinator = None
    instantiate_coordinator = None
    process_weight_transfers = None
    receive_weight_transfers = None
    start_transfer_server = None


logger = logging.getLogger(__name__)


def create_coordinator(mode: WeightTransferMode, name: str):
    """Create coordinator based on transfer mode.

    Args:
        mode: The weight transfer mode
        name: Unique name for the coordinator

    Returns:
        Ray actor handle for coordinator, or None if not needed
    """
    if mode == WeightTransferMode.RAY_REMOTING:
        return RayWeightCoordinator.options(name=name).remote()
    elif mode == WeightTransferMode.JAX_TRANSFER_SERVER:
        if not JAX_TRANSFER_AVAILABLE:
            raise RuntimeError("JAX transfer server not available")
        transfer_server = start_transfer_server()
        return instantiate_coordinator(transfer_server, name=name)
    else:
        return None  # GCS_CHECKPOINT doesn't need coordinator


@ray.remote(num_cpus=0)
class RayWeightCoordinator:
    """Ray actor for coordinating weight transfers using Ray's object store."""

    def __init__(self):
        self.latest_weight_refs = None
        self.weight_id = None

    def put_weight_refs(self, weight_id: int, weight_refs: dict) -> None:
        """Store weight references from Ray object store."""
        if self.weight_id is None or weight_id > self.weight_id:
            del self.latest_weight_refs
            self.latest_weight_refs = weight_refs
            self.weight_id = weight_id
            logger.info(f"Stored weight refs for weight_id {weight_id}")

    def get_latest_weight_refs(self) -> tuple[dict | None, int | None]:
        """Get latest weight references and ID."""
        return self.latest_weight_refs, self.weight_id


class WeightTransferServer(ABC):
    """Abstract base class for weight transfer servers (training worker side)."""

    @abstractmethod
    def serve_weights(self, weight_id: int, model) -> None:
        """Serve weights to clients.

        Args:
            weight_id: Unique identifier for this weight update
            model: Levanter model parameters (PyTree of NamedArrays)
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    @abstractmethod
    def get_metrics(self) -> WeightTransferServerMetrics:
        """Get transfer metrics."""
        pass


class WeightTransferClient(ABC):
    """Abstract base class for weight transfer clients (inference worker side)."""

    @abstractmethod
    def receive_weights(self, old_model: PyTree) -> Any:
        """Receive weights from server.

        Args:
            old_model: Previous model for memory optimization (optional)

        Returns:
            new_model or None if no update available.
            new_model will be Levanter model parameters (PyTree of NamedArrays).
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def get_metrics(self) -> WeightTransferClientMetrics:
        """Get transfer metrics."""
        return WeightTransferClientMetrics(start_time=time.time())


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

        # Metrics tracking
        self.metrics = WeightTransferServerMetrics(start_time=time.time())

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
                    delete_with_bucket(old_path, recursive=True)

            logger.info(f"Saving checkpoint at weight_id {weight_id}...")

            # Save checkpoint using Levanter's native checkpoint system
            levanter_checkpoint.save_checkpoint(
                tree=model,
                step=weight_id,
                checkpoint_path=checkpoint_path,
            )

            self.checkpoint_queue.append(weight_id)

            # Update metrics
            self.metrics.successful_transfers += 1

            logger.info(f"Checkpoint saved at {checkpoint_path}")

        except Exception as e:
            self.metrics.failed_transfers += 1
            logger.error(f"Failed to save checkpoint at weight_id {weight_id}: {e}")
            raise

    def cleanup(self) -> None:
        """No cleanup needed for GCS checkpoints."""
        pass

    def get_metrics(self) -> WeightTransferServerMetrics:
        """Get transfer metrics."""
        return self.metrics


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

        # Metrics tracking
        self.metrics = WeightTransferClientMetrics(start_time=time.time())

    def receive_weights(self, old_model: PyTree) -> Any:
        """Load latest checkpoint using Levanter's checkpoint system."""
        self.metrics.total_polls += 1

        latest_checkpoint = self._find_latest_checkpoint()

        if latest_checkpoint is None or latest_checkpoint == self.latest_checkpoint_path:
            return None

        logger.info(f"Loading checkpoint from {latest_checkpoint}")

        try:
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

        return params

    def _find_latest_checkpoint(self) -> str | None:
        """Find the latest checkpoint in the checkpoint directory."""
        checkpoint_dir = Path(self.config.checkpoint_dir)

        if not checkpoint_dir.exists():
            return None

        # Look for checkpoint directories
        checkpoint_dirs = []
        for item in checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("step_"):
                try:
                    step_num = int(item.name.split("_")[1])
                    checkpoint_dirs.append((step_num, item))
                except (ValueError, IndexError):
                    continue

        if not checkpoint_dirs:
            return None

        # Return path to the latest checkpoint directory (Levanter checkpoints are directories)
        _, latest_dir = max(checkpoint_dirs)
        return str(latest_dir)

    def cleanup(self) -> None:
        """No cleanup needed for GCS checkpoints."""
        pass

    def get_metrics(self) -> WeightTransferClientMetrics:
        """Get transfer metrics."""
        return self.metrics


class JAXTransferServer(WeightTransferServer):
    """JAX transfer server-based weight transfer server."""

    coordinator: "WeightTransferCoordinator"

    def __init__(self, config: WeightTransferConfig, mesh, coordinator, params_sharding_rules=None):
        if not JAX_TRANSFER_AVAILABLE:
            raise RuntimeError("JAX transfer server, cannot use JAX_TRANSFER_SERVER mode")

        self.config = config
        self.mesh = mesh
        self.params_sharding_rules = params_sharding_rules
        self.coordinator = coordinator

        # Start transfer server
        self.transfer_server = start_transfer_server()

        # Setup CPU transfer (always enabled)
        self._setup_cpu_transfer()

        # Single-item queue for polling
        self.poll_queue = queue.Queue(maxsize=1)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="weight_transfer")

        # Metrics tracking
        self.metrics = WeightTransferServerMetrics(start_time=time.time())

    def _setup_cpu_transfer(self):
        """Setup CPU mesh for transfers."""
        try:
            cpu_devices = jax.devices("cpu")
            if cpu_devices:
                self.cpu_mesh = Mesh(np.array(cpu_devices[:1]), axis_names=("cpu",))
                logger.info(f"Setup CPU mesh with {len(cpu_devices)} devices")
            else:
                raise RuntimeError("No CPU devices found, cannot perform weight transfer")
        except Exception as e:
            raise RuntimeError(f"Failed to setup CPU mesh: {e}") from e

    def _transfer_to_cpu(self, model) -> PyTree:
        """Transfer params to CPU devices."""
        try:
            with self.cpu_mesh:
                cpu_devices = jax.devices("cpu")
                return jax.device_put(model, cpu_devices[0])
        except Exception as e:
            logger.warning(f"Failed to transfer to CPU: {e}, using original params")
            return model

    def serve_weights(self, weight_id: int, model) -> None:
        """Serve weights with CPU transfer and threading."""
        self.metrics.total_transfers += 1

        def _serve_in_thread():
            try:
                cpu_params = self._transfer_to_cpu(model)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        process_weight_transfers(self.transfer_server, self.coordinator, weight_id, cpu_params)
                    )
                    logger.info(f"Processed {result} weight transfers for weight_id {weight_id}")

                    # Update metrics
                    self.metrics.successful_transfers += 1

                    return result
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Error serving weights {weight_id}: {e}")
                self.metrics.failed_transfers += 1
                return 0

        # Use single-item queue - drop old requests if training runs ahead
        try:
            self.poll_queue.put_nowait((weight_id, model))
        except queue.Full:
            old_item = self.poll_queue.get_nowait()
            logger.info(f"Dropping old weight transfer {old_item[0]} for new {weight_id}")
            self.poll_queue.put_nowait((weight_id, model))

        _ = self.executor.submit(_serve_in_thread)

    def cleanup(self) -> None:
        """Cleanup transfer server and thread pool."""
        logger.info("Cleaning up JAX transfer server")
        self.executor.shutdown(wait=True)
        if hasattr(self, "transfer_server"):
            self.transfer_server = None

    def get_metrics(self) -> WeightTransferServerMetrics:
        """Get transfer metrics."""
        return self.metrics


class JAXTransferClient(WeightTransferClient):
    """JAX transfer server-based weight transfer client."""

    def __init__(self, config: WeightTransferConfig, mesh, coordinator, params_sharding_rules=None):
        if not JAX_TRANSFER_AVAILABLE:
            raise RuntimeError("JAX transfer server, cannot use JAX_TRANSFER_SERVER mode")

        self.config = config
        self.mesh = mesh
        self.params_sharding_rules = params_sharding_rules
        self.coordinator = coordinator

        # Start transfer server
        self.transfer_server = start_transfer_server()
        self._setup_cpu_transfer()

        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="weight_transfer")
        self.current_params_placeholder = None

        # Metrics tracking
        self.metrics = WeightTransferClientMetrics(start_time=time.time())

    def _setup_cpu_transfer(self):
        """Setup CPU mesh for transfers."""
        try:
            cpu_devices = jax.devices("cpu")
            if cpu_devices:
                self.cpu_mesh = Mesh(np.array(cpu_devices[:1]), axis_names=("cpu",))
                logger.info(f"Setup CPU mesh with {len(cpu_devices)} devices")
            else:
                raise RuntimeError("No CPU devices found, cannot perform weight transfer")
        except Exception as e:
            raise RuntimeError(f"Failed to setup CPU mesh: {e}") from e

    def _transfer_from_cpu(self, model) -> PyTree:
        """Transfer params from CPU back to TPU."""
        return self.mesh.shard(model, self.params_sharding_rules)

    def receive_weights(self, old_model: PyTree) -> Any:
        """Receive weights with CPU transfer."""
        self.metrics.total_polls += 1

        def _receive_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                cpu_params, metadata = loop.run_until_complete(
                    receive_weight_transfers(self.coordinator, self.transfer_server, self.current_params_placeholder)
                )

                # Transfer back from CPU
                tpu_params = self._transfer_from_cpu(cpu_params)

                return tpu_params, metadata
            finally:
                loop.close()

        try:
            future = self.executor.submit(_receive_in_thread)
            params, metadata = future.result(timeout=self.config.transfer_timeout)

            if params is not None and metadata is not None:
                # Update metrics
                self.metrics.successful_receives += 1
                return params

            return None

        except Exception as e:
            self.metrics.failed_receives += 1
            logger.error(f"Failed to receive weights: {e}")
            raise

    def set_params_placeholder(self, model):
        """Set the placeholder params for transfers."""
        self.current_params_placeholder = model

    def cleanup(self) -> None:
        """Cleanup transfer server and thread pool."""
        logger.info("Cleaning up JAX transfer client")
        self.executor.shutdown(wait=True)
        if hasattr(self, "transfer_server"):
            self.transfer_server = None

    def get_metrics(self) -> WeightTransferClientMetrics:
        """Get transfer metrics."""
        return self.metrics


class RayRemotingServer(WeightTransferServer):
    """Ray remoting-based weight transfer server for Levanter models."""

    def __init__(self, config: WeightTransferConfig, coordinator):
        self.config = config
        self.coordinator = coordinator

        self.metrics = WeightTransferServerMetrics(start_time=time.time())

    def serve_weights(self, weight_id: int, model) -> None:
        self.metrics.total_transfers += 1

        try:
            logger.info(f"Serving weights for weight_id {weight_id} via Ray remoting...")
            jax_distributed_barrier()

            # Convert Levanter model params (NamedArrays) to numpy arrays for Ray
            def convert_to_numpy(x):
                if is_named_array(x):
                    return np.array(x.array)
                elif hasattr(x, "shape"):
                    return np.array(x)
                else:
                    return x

            numpy_pytree = jax.tree.map(convert_to_numpy, model)
            jax_distributed_barrier()

            if jax.process_index() == 0:
                logger.info(f"Updating Ray object store with new weights at weight_id {weight_id}...")
                leaves, treedef = jax.tree.flatten(numpy_pytree)
                weight_refs = {
                    "leaves": [ray.put(leaf) for leaf in leaves],
                    "treedef": ray.put(treedef),
                }
                ray.get(self.coordinator.put_weight_refs.remote(weight_id, weight_refs))
                del weight_refs
                self.metrics.successful_transfers += 1

                logger.info(
                    f"Served weights for weight_id {weight_id} via Ray remoting " f"(process {jax.process_index()})"
                )

            jax_distributed_barrier()
        except Exception as e:
            self.metrics.failed_transfers += 1
            logger.error(f"Failed to serve weights {weight_id} via Ray remoting: {e}")
            raise

    def cleanup(self) -> None:
        pass

    def get_metrics(self) -> WeightTransferServerMetrics:
        """Get transfer metrics."""
        return self.metrics


class RayRemotingClient(WeightTransferClient):
    """Ray remoting-based weight transfer client for Levanter models."""

    def __init__(
        self,
        config: WeightTransferConfig,
        coordinator,
    ):
        self.config = config
        self.coordinator = coordinator
        self.last_weight_id = None

        # Metrics tracking
        self.metrics = WeightTransferClientMetrics(start_time=time.time())

    def receive_weights(self, old_model: PyTree = None) -> PyTree | None:
        """Receive weights from Ray object store and reconstruct as Levanter model."""
        self.metrics.total_polls += 1

        try:
            # Poll coordinator for latest weight references
            weight_refs, weight_id = ray.get(self.coordinator.get_latest_weight_refs.remote())

            if weight_refs is None or weight_id == self.last_weight_id:
                return None

            # Get individual leaf arrays and treedef from object store
            leaf_refs = weight_refs["leaves"]
            treedef_ref = weight_refs["treedef"]

            # Get all leaves and treedef
            leaves = [ray.get(ref) for ref in leaf_refs]
            treedef = ray.get(treedef_ref)

            # Reconstruct the pytree from numpy arrays
            numpy_pytree = jax.tree.unflatten(treedef, leaves)

            # Convert back to JAX arrays and reconstruct NamedArrays structure
            if old_model is not None:

                def convert_from_numpy(numpy_val, target_val=None):
                    if target_val is not None and is_named_array(target_val):
                        # Reconstruct NamedArray with same axes as target
                        return hax.named(jax.numpy.array(numpy_val), target_val.axes)
                    else:
                        return jax.numpy.array(numpy_val)

                def _convert_tree(numpy_tree, target_tree):
                    return jax.tree.map(convert_from_numpy, numpy_tree, target_tree)

                levanter_pytree = _convert_tree(numpy_pytree, old_model)
            else:
                # Without old_model, just convert to JAX arrays (lose NamedArray structure)
                levanter_pytree = jax.tree.map(lambda x: jax.numpy.array(x), numpy_pytree)

            self.metrics.successful_receives += 1
            self.last_weight_id = weight_id

            logger.info(f"Received weights for weight_id {weight_id} via Ray remoting")

            return levanter_pytree

        except Exception as e:
            self.metrics.failed_receives += 1
            logger.error(f"Failed to receive weights via Ray remoting: {e}")
            raise

    def cleanup(self) -> None:
        """No cleanup needed for Ray remoting client."""
        pass

    def get_metrics(self) -> WeightTransferClientMetrics:
        """Get transfer metrics."""
        return self.metrics


def create_weight_transfer_server(
    config: WeightTransferConfig,
    mesh: Mesh | None = None,
    axis_mapping: ResourceMapping | None = None,
    coordinator=None,
) -> WeightTransferServer:
    """Factory function to create appropriate transfer server for Levanter models.

    Args:
        config: Weight transfer configuration
        mesh: JAX mesh for distributed computation (optional)
        axis_mapping: Levanter axis mapping for sharding (optional)
        coordinator: Ray coordinator for distributed modes (required for RAY_REMOTING)

    Returns:
        WeightTransferServer instance
    """
    if config.mode == WeightTransferMode.JAX_TRANSFER_SERVER:
        # JAX transfer server is currently not supported with Levanter models
        logger.warning("JAX transfer server not yet supported with Levanter models, " "falling back to GCS checkpoints")
        config.mode = WeightTransferMode.GCS_CHECKPOINT

    elif config.mode == WeightTransferMode.RAY_REMOTING:
        if coordinator is None:
            raise ValueError("Coordinator required for Ray remoting")
        return RayRemotingServer(config, coordinator)

    # Default to GCS checkpoint mode
    return GCSCheckpointServer(
        config,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )


def create_weight_transfer_client(
    config: WeightTransferConfig,
    mesh: Mesh | None = None,
    axis_mapping: ResourceMapping | None = None,
    coordinator=None,
) -> WeightTransferClient:
    """Factory function to create appropriate transfer client for Levanter models.

    Args:
        config: Weight transfer configuration
        mesh: JAX mesh for distributed computation (optional)
        axis_mapping: Levanter axis mapping for sharding (optional)
        target_model: Target parameter structure for reconstructing NamedArrays (optional)
        coordinator: Ray coordinator for distributed modes (required for RAY_REMOTING)

    Returns:
        WeightTransferClient instance
    """
    if config.mode == WeightTransferMode.JAX_TRANSFER_SERVER:
        # JAX transfer server is currently not supported with Levanter models
        logger.warning("JAX transfer server not supported, falling back to GCS checkpoints")
        config.mode = WeightTransferMode.GCS_CHECKPOINT

    elif config.mode == WeightTransferMode.RAY_REMOTING:
        if coordinator is None:
            raise ValueError("Coordinator required for Ray remoting")
        return RayRemotingClient(
            config,
            coordinator,
        )

    # Default to GCS checkpoint mode
    return GCSCheckpointClient(
        config,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )
