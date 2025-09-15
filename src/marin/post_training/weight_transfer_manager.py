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
from pathlib import Path
from typing import Any

import jax
import numpy as np
import ray
from flax.serialization import to_state_dict
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import Mesh
from jaxtyping import PyTree

from .training_config import WeightTransferConfig, WeightTransferMode
from .utils import checkpointer, delete_with_bucket, jax_distributed_barrier

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


def _reshard_with_donation(old_params: PyTree, new_params: PyTree, shard_fns: PyTree) -> PyTree:
    """Reshard new_params while donating old_params' buffers for memory optimization."""
    return jax.tree.map(lambda x, fn: fn(x), new_params, shard_fns)


_reshard_with_donation = jax.jit(_reshard_with_donation, donate_argnums=(0,))


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
    def serve_weights(self, weight_id: int, params: PyTree) -> None:
        """Serve weights to clients."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    @abstractmethod
    def get_metrics(self) -> dict[str, Any]:
        """Get transfer metrics."""
        pass


class WeightTransferClient:
    """Abstract base class for weight transfer clients (inference worker side)."""

    def receive_weights(self, old_params: PyTree | None = None) -> tuple[PyTree | None, dict[str, Any]]:
        """Receive weights from server. Returns (params, metadata)."""
        pass

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def set_params_placeholder(self, params: PyTree):
        """Set the placeholder params for transfers (only for JAX transfer client)."""
        pass

    def get_metrics(self) -> dict[str, Any]:
        """Get transfer metrics."""
        return {}


class GCSCheckpointServer(WeightTransferServer):
    """GCS checkpoint-based weight transfer server."""

    config: WeightTransferConfig
    checkpoint_queue: deque[int]
    gather_fns: Any
    model_config: Any  # PretrainedConfig from HF

    def __init__(self, config: WeightTransferConfig, gather_fns: Any, model_config: Any):
        self.config = config
        self.checkpoint_queue = deque()
        self.gather_fns = gather_fns
        self.model_config = model_config

        # Metrics tracking
        self.metrics = {
            "mode": "gcs_checkpoint",
            "total_transfers": 0,
            "successful_transfers": 0,
            "failed_transfers": 0,
            "total_bytes": 0,
            "checkpoints_saved": 0,
            "checkpoints_deleted": 0,
            "last_weight_id": None,
            "start_time": time.time(),
        }

    def serve_weights(self, weight_id: int, params: PyTree) -> None:
        """Save checkpoint to disk."""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"step_{weight_id}")

        self.metrics["total_transfers"] += 1

        try:
            # Manage checkpoint queue
            if self.config.max_checkpoints is not None and len(self.checkpoint_queue) >= self.config.max_checkpoints:
                old_weight_id = self.checkpoint_queue.popleft()
                old_path = os.path.join(self.config.checkpoint_dir, f"step_{old_weight_id}")
                if jax.process_index() == 0:  # Only delete from coordinator
                    delete_with_bucket(old_path, recursive=True)
                    self.metrics["checkpoints_deleted"] += 1

            logger.info(f"Saving checkpoint at weight_id {weight_id}...")

            # Save checkpoint
            metadata = {"weight_id": weight_id}

            checkpointer(
                path=checkpoint_path,
                params=params,
                config=self.model_config.to_dict() if self.model_config else {},
                gather_fns=self.gather_fns,
                metadata=metadata,
                active=jax.process_index() == 0,
                save_float_dtype="bf16",
            )

            self.checkpoint_queue.append(weight_id)

            # Update metrics
            self.metrics["successful_transfers"] += 1
            self.metrics["checkpoints_saved"] += 1
            self.metrics["last_weight_id"] = weight_id

            # Estimate checkpoint size (rough approximation)
            param_count = sum(np.prod(x.shape) for x in jax.tree.leaves(params) if hasattr(x, "shape"))
            bytes_per_param = 2  # bf16 = 2 bytes per parameter
            estimated_bytes = param_count * bytes_per_param
            self.metrics["total_bytes"] += estimated_bytes

            logger.info(f"Checkpoint saved at {checkpoint_path}")

        except Exception as e:
            self.metrics["failed_transfers"] += 1
            logger.error(f"Failed to save checkpoint at weight_id {weight_id}: {e}")
            raise

    def cleanup(self) -> None:
        """No cleanup needed for GCS checkpoints."""
        pass

    def get_metrics(self) -> dict[str, Any]:
        """Get transfer metrics."""
        current_time = time.time()
        elapsed_time = current_time - self.metrics["start_time"]

        metrics = self.metrics.copy()
        metrics["transfer_rate"] = metrics["successful_transfers"] / max(elapsed_time, 1.0)  # transfers per second
        metrics["avg_checkpoint_size_mb"] = (metrics["total_bytes"] / (1024 * 1024)) / max(
            metrics["checkpoints_saved"], 1
        )

        return {"weight_transfer." + k: v for k, v in metrics.items()}


class GCSCheckpointClient(WeightTransferClient):
    """GCS checkpoint-based weight transfer client."""

    def __init__(
        self,
        config: WeightTransferConfig,
        shard_fns: Any,
        load_checkpoint_fn: callable,
    ):
        self.config = config
        self.shard_fns = shard_fns
        self.load_checkpoint_fn = load_checkpoint_fn
        self.latest_checkpoint_path = None

        # Metrics tracking
        self.metrics = {
            "mode": "gcs_checkpoint",
            "total_polls": 0,
            "successful_loads": 0,
            "failed_loads": 0,
            "total_bytes_loaded": 0,
            "last_weight_id": None,
            "start_time": time.time(),
        }

    def receive_weights(self, old_params: PyTree | None = None) -> tuple[PyTree | None, dict[str, Any]]:
        """Load latest checkpoint from disk."""
        self.metrics["total_polls"] += 1

        latest_checkpoint = self._find_latest_checkpoint()

        if latest_checkpoint is None or latest_checkpoint == self.latest_checkpoint_path:
            return old_params, {}

        del old_params
        logger.info(f"Loading checkpoint from {latest_checkpoint}")
        params = self.load_checkpoint_fn(latest_checkpoint)

        self.latest_checkpoint_path = latest_checkpoint
        weight_id = int(Path(latest_checkpoint).parent.name.split("_")[1])
        self.metrics["successful_loads"] += 1
        self.metrics["last_weight_id"] = weight_id

        param_count = sum(np.prod(x.shape) for x in jax.tree.leaves(params) if hasattr(x, "shape"))
        bytes_per_param = 2  # bf16 = 2 bytes per parameter
        estimated_bytes = param_count * bytes_per_param
        self.metrics["total_bytes_loaded"] += estimated_bytes

        return params, {
            "weight_id": weight_id,
            "source": "gcs_checkpoint",
            "path": latest_checkpoint,
        }

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

        # Return path to the latest checkpoint's params.msgpack
        latest_step, latest_dir = max(checkpoint_dirs)
        params_path = latest_dir / "params.msgpack"

        if params_path.exists():
            return str(params_path)

        return None

    def cleanup(self) -> None:
        """No cleanup needed for GCS checkpoints."""
        pass

    def get_metrics(self) -> dict[str, Any]:
        """Get transfer metrics."""
        current_time = time.time()
        elapsed_time = current_time - self.metrics["start_time"]

        metrics = self.metrics.copy()
        metrics["poll_rate"] = metrics["total_polls"] / max(elapsed_time, 1.0)  # polls per second
        metrics["load_success_rate"] = metrics["successful_loads"] / max(metrics["total_polls"], 1)
        metrics["avg_load_size_mb"] = (metrics["total_bytes_loaded"] / (1024 * 1024)) / max(
            metrics["successful_loads"], 1
        )

        return {"weight_transfer." + k: v for k, v in metrics.items()}


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
        self.metrics = {
            "mode": "jax_transfer",
            "total_transfers": 0,
            "successful_transfers": 0,
            "failed_transfers": 0,
            "queue_drops": 0,
            "total_transfer_time_ms": 0,
            "last_weight_id": None,
            "start_time": time.time(),
        }

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

    def _transfer_to_cpu(self, params: PyTree) -> PyTree:
        """Transfer params to CPU devices."""
        try:
            with self.cpu_mesh:
                cpu_devices = jax.devices("cpu")
                return jax.device_put(params, cpu_devices[0])
        except Exception as e:
            logger.warning(f"Failed to transfer to CPU: {e}, using original params")
            return params

    def serve_weights(self, weight_id: int, params: PyTree) -> None:
        """Serve weights with CPU transfer and threading."""
        self.metrics["total_transfers"] += 1
        start_time = time.time()

        def _serve_in_thread():
            try:
                cpu_params = self._transfer_to_cpu(params)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        process_weight_transfers(self.transfer_server, self.coordinator, weight_id, cpu_params)
                    )
                    logger.info(f"Processed {result} weight transfers for weight_id {weight_id}")

                    # Update metrics
                    self.metrics["successful_transfers"] += 1
                    self.metrics["last_weight_id"] = weight_id
                    elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
                    self.metrics["total_transfer_time_ms"] += elapsed_time

                    return result
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Error serving weights {weight_id}: {e}")
                self.metrics["failed_transfers"] += 1
                return 0

        # Use single-item queue - drop old requests if training runs ahead
        try:
            self.poll_queue.put_nowait((weight_id, params))
        except queue.Full:
            old_item = self.poll_queue.get_nowait()
            logger.info(f"Dropping old weight transfer {old_item[0]} for new {weight_id}")
            self.metrics["queue_drops"] += 1
            self.poll_queue.put_nowait((weight_id, params))

        _ = self.executor.submit(_serve_in_thread)

    def cleanup(self) -> None:
        """Cleanup transfer server and thread pool."""
        logger.info("Cleaning up JAX transfer server")
        self.executor.shutdown(wait=True)
        if hasattr(self, "transfer_server"):
            self.transfer_server = None

    def get_metrics(self) -> dict[str, Any]:
        """Get transfer metrics."""
        current_time = time.time()
        elapsed_time = current_time - self.metrics["start_time"]

        metrics = self.metrics.copy()
        metrics["transfer_rate"] = metrics["successful_transfers"] / max(elapsed_time, 1.0)
        if metrics["successful_transfers"] > 0:
            metrics["avg_transfer_time_ms"] = metrics["total_transfer_time_ms"] / metrics["successful_transfers"]
        else:
            metrics["avg_transfer_time_ms"] = 0.0

        return {"weight_transfer." + k: v for k, v in metrics.items()}


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
        self.metrics = {
            "mode": "jax_transfer",
            "total_polls": 0,
            "successful_receives": 0,
            "failed_receives": 0,
            "total_bytes_received": 0,
            "total_transfer_time_ms": 0,
            "last_weight_id": None,
            "start_time": time.time(),
        }

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

    def _transfer_from_cpu(self, params: PyTree) -> PyTree:
        """Transfer params from CPU back to TPU."""
        return self.mesh.shard(params, self.params_sharding_rules)

    def receive_weights(self, old_params: PyTree | None = None) -> tuple[PyTree | None, dict[str, Any]]:
        """Receive weights with CPU transfer."""
        self.metrics["total_polls"] += 1
        start_time = time.time()

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
                self.metrics["successful_receives"] += 1
                self.metrics["last_weight_id"] = metadata.weight_id
                self.metrics["total_bytes_received"] += metadata.weight_bytes
                elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
                self.metrics["total_transfer_time_ms"] += elapsed_time

                return params, {
                    "weight_id": metadata.weight_id,
                    "transfer_time": metadata.time_elapsed,
                    "transfer_bytes": metadata.weight_bytes,
                    "source": "jax_transfer",
                }

            return None, {}

        except Exception as e:
            self.metrics["failed_receives"] += 1
            logger.error(f"Failed to receive weights: {e}")
            raise

    def set_params_placeholder(self, params: PyTree):
        """Set the placeholder params for transfers."""
        self.current_params_placeholder = params

    def cleanup(self) -> None:
        """Cleanup transfer server and thread pool."""
        logger.info("Cleaning up JAX transfer client")
        self.executor.shutdown(wait=True)
        if hasattr(self, "transfer_server"):
            self.transfer_server = None

    def get_metrics(self) -> dict[str, Any]:
        """Get transfer metrics."""
        current_time = time.time()
        elapsed_time = current_time - self.metrics["start_time"]

        metrics = self.metrics.copy()
        metrics["poll_rate"] = metrics["total_polls"] / max(elapsed_time, 1.0)
        metrics["receive_success_rate"] = metrics["successful_receives"] / max(metrics["total_polls"], 1)
        if metrics["successful_receives"] > 0:
            metrics["avg_transfer_time_ms"] = metrics["total_transfer_time_ms"] / metrics["successful_receives"]
            metrics["avg_transfer_size_mb"] = (
                metrics["total_bytes_received"] / (1024 * 1024) / metrics["successful_receives"]
            )
        else:
            metrics["avg_transfer_time_ms"] = 0.0
            metrics["avg_transfer_size_mb"] = 0.0

        return {"weight_transfer." + k: v for k, v in metrics.items()}


class RayRemotingServer(WeightTransferServer):
    """Ray remoting-based weight transfer server."""

    def __init__(self, config: WeightTransferConfig, gather_fns, coordinator):
        self.config = config
        self.gather_fns = gather_fns
        self.coordinator = coordinator

        self.metrics = {
            "mode": "ray_remoting",
            "total_transfers": 0,
            "successful_transfers": 0,
            "failed_transfers": 0,
            "total_bytes": 0,
            "last_weight_id": None,
            "start_time": time.time(),
        }

    def serve_weights(self, weight_id: int, params: PyTree) -> None:
        self.metrics["total_transfers"] += 1

        try:
            logger.info(f"Serving weights for weight_id {weight_id} via Ray remoting...")
            jax_distributed_barrier()

            train_state = to_state_dict(params)
            flattened_train_state = flatten_dict(train_state)
            gather_fns = flatten_dict(to_state_dict(self.gather_fns.params))

            gathered_params = {}
            for key, value in flattened_train_state.items():
                gathered_params[key] = gather_fns[key](value)

            gathered_train_state = unflatten_dict(gathered_params)
            jax_distributed_barrier()

            if jax.process_index() == 0:
                logger.info("Updating Ray object store with new weights at weight_id {weight_id}...")
                numpy_pytree = jax.tree.map(lambda x: np.array(x) if hasattr(x, "shape") else x, gathered_train_state)
                leaves, treedef = jax.tree.flatten(numpy_pytree)
                weight_refs = {
                    "leaves": [ray.put(leaf) for leaf in leaves],
                    "treedef": ray.put(treedef),
                }
                ray.get(self.coordinator.put_weight_refs.remote(weight_id, weight_refs))
                del weight_refs
                self.metrics["successful_transfers"] += 1
                self.metrics["last_weight_id"] = weight_id
                param_count = sum(np.prod(x.shape) for x in jax.tree.leaves(gathered_train_state) if hasattr(x, "shape"))
                bytes_per_param = 4
                estimated_bytes = param_count * bytes_per_param
                self.metrics["total_bytes"] += estimated_bytes

                logger.info(f"Served weights for weight_id {weight_id} via Ray remoting (process {jax.process_index()})")

            jax_distributed_barrier()
        except Exception as e:
            self.metrics["failed_transfers"] += 1
            logger.error(f"Failed to serve weights {weight_id} via Ray remoting: {e}")
            raise

    def cleanup(self) -> None:
        pass

    def get_metrics(self) -> dict[str, Any]:
        """Get transfer metrics."""
        current_time = time.time()
        elapsed_time = current_time - self.metrics["start_time"]

        metrics = self.metrics.copy()
        metrics["transfer_rate"] = metrics["successful_transfers"] / max(elapsed_time, 1.0)
        metrics["avg_transfer_size_mb"] = (
            metrics["total_bytes"] / (1024 * 1024) / max(metrics["successful_transfers"], 1)
        )

        return {"weight_transfer." + k: v for k, v in metrics.items()}


class RayRemotingClient(WeightTransferClient):
    """Ray remoting-based weight transfer client."""

    def __init__(
        self,
        config: WeightTransferConfig,
        coordinator,
        shard_fns=None,
        remove_dict_prefix=None,
        convert_to_dtypes=None,
    ):
        self.config = config
        self.coordinator = coordinator
        self.shard_fns = shard_fns
        self.remove_dict_prefix = remove_dict_prefix
        self.convert_to_dtypes = convert_to_dtypes
        self.last_weight_id = None

        # Metrics tracking
        self.metrics = {
            "mode": "ray_remoting",
            "total_polls": 0,
            "successful_receives": 0,
            "failed_receives": 0,
            "total_bytes_received": 0,
            "last_weight_id": None,
            "start_time": time.time(),
        }

    def receive_weights(self, old_params: PyTree | None = None) -> tuple[PyTree | None, dict[str, Any]]:
        """Receive weights from Ray object store."""
        self.metrics["total_polls"] += 1

        try:
            # Poll coordinator for latest weight references
            weight_refs, weight_id = ray.get(self.coordinator.get_latest_weight_refs.remote())

            if weight_refs is None or weight_id == self.last_weight_id:
                return None, {}

            # Get individual leaf arrays and treedef from object store
            leaf_refs = weight_refs["leaves"]
            treedef_ref = weight_refs["treedef"]

            # Get all leaves and treedef
            leaves = [ray.get(ref) for ref in leaf_refs]
            treedef = ray.get(treedef_ref)

            # Reconstruct the pytree
            numpy_pytree = jax.tree.unflatten(treedef, leaves)

            # Convert back to JAX arrays
            jax_pytree = jax.tree.map(lambda x: jax.numpy.array(x), numpy_pytree)
            if self.shard_fns is not None:
                if old_params is not None:
                    jax_pytree = _reshard_with_donation(old_params, jax_pytree, self.shard_fns)
                else:
                    jax_pytree = jax.tree.map(lambda x, fn: fn(x), jax_pytree, self.shard_fns)

            self.metrics["successful_receives"] += 1
            self.metrics["last_weight_id"] = weight_id
            self.last_weight_id = weight_id

            # Estimate data size
            param_count = sum(np.prod(x.shape) for x in jax.tree.leaves(jax_pytree) if hasattr(x, "shape"))
            bytes_per_param = 4  # float32
            estimated_bytes = param_count * bytes_per_param
            self.metrics["total_bytes_received"] += estimated_bytes

            logger.info(f"Received weights for weight_id {weight_id} via Ray remoting")

            return jax_pytree, {
                "weight_id": weight_id,
                "source": "ray_remoting",
            }

        except Exception as e:
            self.metrics["failed_receives"] += 1
            logger.error(f"Failed to receive weights via Ray remoting: {e}")
            raise

    def cleanup(self) -> None:
        """No cleanup needed for Ray remoting client."""
        pass

    def get_metrics(self) -> dict[str, Any]:
        """Get transfer metrics."""
        current_time = time.time()
        elapsed_time = current_time - self.metrics["start_time"]

        metrics = self.metrics.copy()
        metrics["poll_rate"] = metrics["total_polls"] / max(elapsed_time, 1.0)
        metrics["receive_success_rate"] = metrics["successful_receives"] / max(metrics["total_polls"], 1)
        metrics["avg_transfer_size_mb"] = (
            metrics["total_bytes_received"] / (1024 * 1024) / max(metrics["successful_receives"], 1)
        )

        return {"weight_transfer." + k: v for k, v in metrics.items()}


def create_weight_transfer_server(
    config: WeightTransferConfig,
    mesh=None,
    params_sharding_rules=None,
    gather_fns=None,
    model_config=None,
    coordinator=None,
) -> WeightTransferServer:
    """Factory function to create appropriate transfer server."""
    if config.mode == WeightTransferMode.JAX_TRANSFER_SERVER:
        if not JAX_TRANSFER_AVAILABLE:
            logger.warning("JAX transfer server not available, falling back to GCS checkpoints")
            config.mode = WeightTransferMode.GCS_CHECKPOINT
        elif mesh is None:
            raise ValueError("Mesh required for JAX transfer server")
        elif coordinator is None:
            raise ValueError("Coordinator required for JAX transfer server")
        else:
            return JAXTransferServer(config, mesh, coordinator, params_sharding_rules)

    elif config.mode == WeightTransferMode.RAY_REMOTING:
        if coordinator is None:
            raise ValueError("Coordinator required for Ray remoting")
        return RayRemotingServer(config, gather_fns, coordinator)

    return GCSCheckpointServer(
        config,
        gather_fns=gather_fns,
        model_config=model_config,
    )


def create_weight_transfer_client(
    config: WeightTransferConfig,
    mesh=None,
    params_sharding_rules=None,
    shard_fns=None,
    load_checkpoint_fn=None,
    coordinator=None,
) -> WeightTransferClient:
    """Factory function to create appropriate transfer client."""
    if config.mode == WeightTransferMode.JAX_TRANSFER_SERVER:
        if not JAX_TRANSFER_AVAILABLE:
            logger.warning("JAX transfer server not available, falling back to GCS checkpoints")
            config.mode = WeightTransferMode.GCS_CHECKPOINT
        elif mesh is None:
            raise ValueError("Mesh required for JAX transfer server")
        elif coordinator is None:
            raise ValueError("Coordinator required for JAX transfer server")
        else:
            return JAXTransferClient(config, mesh, coordinator, params_sharding_rules)

    elif config.mode == WeightTransferMode.RAY_REMOTING:
        if coordinator is None:
            raise ValueError("Coordinator required for Ray remoting")
        return RayRemotingClient(
            config,
            coordinator,
            shard_fns=shard_fns,
        )

    if load_checkpoint_fn is None:
        raise ValueError("load_checkpoint_fn must be provided for GCSCheckpointClient")
    return GCSCheckpointClient(
        config,
        shard_fns=shard_fns,
        load_checkpoint_fn=load_checkpoint_fn,
    )
