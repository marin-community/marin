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

"""Core types for the fluster actor layer.

This module contains actor-specific types that depend on the cluster layer.
"""

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NewType

if TYPE_CHECKING:
    from fluster.actor.resolver import Resolver
    from fluster.cluster.client import Cluster
    from fluster.cluster.types import JobId, Namespace


# Type aliases
ActorId = NewType("ActorId", str)


@dataclass
class ActorEndpoint:
    """Actor endpoint for discovery and RPC.

    Wraps a cluster Endpoint with actor-specific semantics.

    Args:
        actor_id: Unique actor identifier
        name: Actor name for discovery
        address: Network address (host:port)
        job_id: Job hosting this actor
        namespace: Namespace for scoping
        metadata: Optional key-value metadata
    """

    actor_id: ActorId
    name: str
    address: str
    job_id: "JobId"
    namespace: "Namespace"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ActorContext:
    """Context passed to actor methods as first argument.

    Enables actors to call other actors and access cluster services.

    Args:
        cluster: Cluster client for job management
        resolver: Resolver for actor discovery
        job_id: Current job ID
        namespace: Current namespace
    """

    cluster: "Cluster"
    resolver: "Resolver"
    job_id: "JobId"
    namespace: "Namespace"

    @classmethod
    def from_environment(cls) -> "ActorContext":
        """Create context from FLUSTER_* environment variables.

        Environment variables:
        - FLUSTER_CLUSTER_ADDRESS: Controller address
        - FLUSTER_JOB_ID: Current job ID
        - FLUSTER_NAMESPACE: Actor namespace

        Returns:
            ActorContext constructed from environment

        Raises:
            ValueError: If FLUSTER_CLUSTER_ADDRESS is not set
        """
        from fluster.actor.resolver import ClusterResolver
        from fluster.cluster.client import Cluster
        from fluster.cluster.types import JobId, Namespace

        address = os.environ.get("FLUSTER_CLUSTER_ADDRESS")
        if not address:
            raise ValueError("FLUSTER_CLUSTER_ADDRESS not set")

        cluster = Cluster(address)
        resolver = ClusterResolver(cluster)
        job_id = JobId(os.environ.get("FLUSTER_JOB_ID", ""))
        namespace = Namespace(os.environ.get("FLUSTER_NAMESPACE", "default"))

        return cls(
            cluster=cluster,
            resolver=resolver,
            job_id=job_id,
            namespace=namespace,
        )
