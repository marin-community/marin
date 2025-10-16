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

import os
from pathlib import Path

from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from ray.util.state import list_nodes


def is_local_ray_cluster():
    address = os.environ.get("RAY_ADDRESS")
    if isinstance(address, str):
        address = address.strip()
    cluster_file = Path("/tmp/ray/ray_current_cluster")

    # Check if it's a local cluster
    if address is None and not cluster_file.exists():
        return True
    if address == "local":
        return True

    # Set when we're running with the unittest helper.
    if os.environ.get("RAY_LOCAL_CLUSTER", None) == "1":
        return True

    return False


def get_head_ip() -> str:
    # detail=True yields NodeState objects with is_head_node & node_ip fields
    nodes = list_nodes(detail=True)
    try:
        head = next(n for n in nodes if n.is_head_node)
        return head.node_ip
    except StopIteration:
        raise RuntimeError("No head node found in the Ray cluster. Ensure the cluster is running.") from None


def get_head_node_id() -> str:
    """Get the node ID of the Ray head node."""
    try:
        head = list_nodes(filters=[("is_head_node", "=", True)])[0]
        return head.node_id
    except StopIteration:
        raise RuntimeError("No head node found in the Ray cluster. Ensure the cluster is running.") from None


def schedule_on_head_node_strategy(soft=False) -> NodeAffinitySchedulingStrategy:
    """
    Create a scheduling strategy that targets the Ray head node.

    We do this in Marin because usually only the head node is non-preemptible,
    and some actors (e.g. StatusActor) should not be preempted.
    """

    node_id = get_head_node_id()
    return NodeAffinitySchedulingStrategy(node_id=node_id, soft=soft)
