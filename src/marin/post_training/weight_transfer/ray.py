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

import dataclasses
import logging
import time

import jax
import ray
from haliax import state_dict as hsd
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

            state_dict = hsd.to_numpy_state_dict(model)

            barrier_sync()

            if jax.process_index() == 0:
                logger.info(f"Updating Ray object store with new weights at weight_id {weight_id}...")
                ref_dict = {}
                for k, v in state_dict.items():
                    ref_dict[k] = ray.put(v)

                ray.get(self.coordinator.put_weight_refs.remote(weight_id, ref_dict))
                self.metrics.successful_transfers += 1

                logger.info(f"Served weights for weight_id {weight_id} via Ray remoting (process {jax.process_index()})")

            barrier_sync()
        except Exception as e:
            self.metrics.failed_transfers += 1
            logger.error(f"Failed to serve weights {weight_id} via Ray remoting: {e}")
            raise

    def cleanup(self) -> None:
        pass

    def get_metrics(self) -> dict:
        return dataclasses.asdict(self.metrics)


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

            state_dict = {}
            for k, ref in weight_refs.items():
                array = ray.get(ref)
                state_dict[k] = array

            model = hsd.from_state_dict(old_model, state_dict)

            self.metrics.successful_receives += 1
            self.last_weight_id = weight_id

            logger.info(f"Received weights for weight_id {weight_id} via Ray remoting")

            return model

        except Exception as e:
            self.metrics.failed_receives += 1
            logger.error(f"Failed to receive weights via Ray remoting: {e}")
            raise

    def cleanup(self) -> None:
        """No cleanup needed for Ray remoting client."""
        pass

    def get_metrics(self) -> dict:
        return dataclasses.asdict(self.metrics)
