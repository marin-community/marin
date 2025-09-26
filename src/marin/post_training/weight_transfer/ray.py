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
Ray remoting-based weight transfer implementation.

This module provides weight transfer using Ray's object store for
high-performance distributed communication between training and inference workers.
"""

import logging
import time

import haliax as hax
import jax
import numpy as np
import ray
from haliax.util import is_named_array
from jaxtyping import PyTree
from levanter.utils.jax_utils import barrier_sync

from .base import (
    WeightTransferClient,
    WeightTransferClientMetrics,
    WeightTransferConfig,
    WeightTransferServer,
    WeightTransferServerMetrics,
    get_or_create_actor,
)

logger = logging.getLogger(__name__)


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


class RayRemotingServer(WeightTransferServer):
    """Ray remoting-based weight transfer server for Levanter models."""

    def __init__(self, config: WeightTransferConfig):
        self.config = config

        self.coordinator = get_or_create_actor(RayWeightCoordinator, config.coordinator_name)

        self.metrics = WeightTransferServerMetrics(start_time=time.time())

    def serve_weights(self, weight_id: int, model) -> None:
        self.metrics.total_transfers += 1

        try:
            logger.info(f"Serving weights for weight_id {weight_id} via Ray remoting...")
            barrier_sync()

            # Convert Levanter model params (NamedArrays) to numpy arrays for Ray
            def convert_to_numpy(x):
                if is_named_array(x):
                    return np.array(x.array)
                elif hasattr(x, "shape"):
                    return np.array(x)
                else:
                    return x

            numpy_pytree = jax.tree.map(convert_to_numpy, model)
            barrier_sync()

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

                logger.info(f"Served weights for weight_id {weight_id} via Ray remoting (process {jax.process_index()})")

            barrier_sync()
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
    ):
        self.config = config

        self.coordinator = get_or_create_actor(RayWeightCoordinator, config.coordinator_name)

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
