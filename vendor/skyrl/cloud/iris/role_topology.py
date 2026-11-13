"""Connect Iris's rank-contiguous NVLink rack slices to Ray role resources."""

from collections.abc import Mapping, Sequence
from typing import Any
import xml.etree.ElementTree as ET

from iris.cluster.platforms.k8s.coreweave_topology import (
    COSCHEDULE_NVLINK_DOMAIN_SLICED,
    gpu_gang_coscheduling_level,
    gpu_gang_rack_slice_size,
)

from cloud.iris.role_plan import ModelRoleKind, RoleExecution, derive_role_plan
from marinskyrl.role_placement import POLICY_ROLE, ROLE_RESOURCE_PREFIX, inference_role

NVLINK_ROLE_TOPOLOGY = "nvlink.domain"
FABRIC_RESOURCE_PREFIX = "skyrl_nvlink:"
RANK_RESOURCE_PREFIX = "skyrl_rank:"
HOST_RESOURCE_PREFIX = "skyrl_host:"
GPU_RESOURCE_PREFIX = "skyrl_gpu:"


def role_topology_groups(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Require one complete native rack slice for the policy and each inference engine."""
    topology = config["iris"].get("role_topology")
    if topology is None:
        return ()
    if topology != NVLINK_ROLE_TOPOLOGY:
        raise ValueError(f"Unsupported Iris role topology: {topology}")
    allocation = config["iris"]["allocation"]
    nodes, width = allocation["num_nodes"], allocation["gpus_per_node"]
    if gpu_gang_coscheduling_level(allocation["gpu_variant"], width, nodes) != COSCHEDULE_NVLINK_DOMAIN_SLICED:
        raise ValueError("NVLink role placement requires Iris's multi-rack sliced topology")
    slice_nodes = gpu_gang_rack_slice_size(width, nodes)
    plan = derive_role_plan(config["skyrl"])
    local = {claim.kind for claim in plan.claims if claim.execution is RoleExecution.LOCAL}
    if local != {ModelRoleKind.POLICY, ModelRoleKind.ROLLOUT} or plan.colocate_all:
        raise ValueError("NVLink role placement supports a separate policy and local inference engines")
    policy, rollout = plan.claim("policy"), plan.claim("rollout")
    if policy.num_nodes != slice_nodes or policy.gpus_per_node != width:
        raise ValueError("The policy must fill exactly one Iris rack slice")
    if (
        rollout.backend != "vllm"
        or rollout.tensor_parallel_size != 1
        or rollout.pipeline_parallel_size != 1
        or rollout.expert_parallel_size != rollout.data_parallel_size
        or rollout.data_parallel_size != slice_nodes * width
        or nodes != (rollout.replicas + 1) * slice_nodes
    ):
        raise ValueError("Each TP1/PP1 EP=DP inference engine must fill its own Iris rack slice")
    return (POLICY_ROLE, *(inference_role(index) for index in range(rollout.replicas)))


def node_role_resources(
    *, groups: Sequence[str], rank: int, num_nodes: int, gpus_per_node: int, host: str, fabric_xml: str
) -> dict[str, float]:
    """Advertise the actual GPU fabric and the role of this native Iris task slice."""
    devices = ET.fromstring(fabric_xml).findall("gpu")
    uuids = {device.findtext("uuid") or "" for device in devices}
    fabrics = set()
    for device in devices:
        fabric = device.find("fabric")
        if fabric is None or fabric.findtext("state") != "Completed" or fabric.findtext("status") != "Success":
            raise ValueError("Every GPU must report a completed, healthy NVLink fabric")
        identity = (fabric.findtext("clusterUuid") or "", fabric.findtext("cliqueId") or "")
        if any(not value or value == "N/A" for value in identity):
            raise ValueError("Missing NVLink fabric identity")
        fabrics.add(identity)
    if len(devices) != gpus_per_node or len(uuids) != gpus_per_node or "" in uuids:
        raise ValueError("Physical GPU identities disagree with the allocated node width")
    if len(fabrics) != 1 or not host or not 0 <= rank < num_nodes or num_nodes % len(groups):
        raise ValueError("Invalid Iris rank, host or node NVLink domain")
    cluster, clique = fabrics.pop()
    role = groups[rank // (num_nodes // len(groups))]
    return {
        ROLE_RESOURCE_PREFIX + role: float(gpus_per_node),
        FABRIC_RESOURCE_PREFIX + f"{cluster}:{clique}": float(gpus_per_node),
        RANK_RESOURCE_PREFIX + str(rank): 1.0,
        HOST_RESOURCE_PREFIX + host: 1.0,
        **{GPU_RESOURCE_PREFIX + gpu_uuid: 1.0 for gpu_uuid in sorted(uuids)},
    }


def validate_role_nodes(
    nodes: Sequence[Mapping[str, Any]], *, groups: Sequence[str], num_nodes: int, gpus_per_node: int
) -> list[dict[str, Any]]:
    """Reject split or overlapping role fabrics before starting any model actors."""
    alive = [node for node in nodes if node["Alive"]]
    if len(alive) != num_nodes:
        raise ValueError(f"Expected exactly {num_nodes} Ray nodes, observed {len(alive)}")
    rows = []
    fabrics: dict[str, set[str]] = {role: set() for role in groups}
    for node in alive:
        resources = node["Resources"]
        selected = {}
        for field, prefix in (
            ("role", ROLE_RESOURCE_PREFIX), ("fabric", FABRIC_RESOURCE_PREFIX),
            ("rank", RANK_RESOURCE_PREFIX), ("host", HOST_RESOURCE_PREFIX),
        ):
            matches = [name.removeprefix(prefix) for name in resources if name.startswith(prefix)]
            if len(matches) != 1:
                raise ValueError(f"Ray node {node['NodeID']} has ambiguous {field} identity")
            selected[field] = matches[0]
        rank = int(selected["rank"])
        role = selected["role"]
        if not 0 <= rank < num_nodes or role != groups[rank // (num_nodes // len(groups))]:
            raise ValueError("Ray roles disagree with the native Iris rack slices")
        if any(resources.get(key) != gpus_per_node for key in (
            "GPU", ROLE_RESOURCE_PREFIX + role, FABRIC_RESOURCE_PREFIX + selected["fabric"]
        )):
            raise ValueError("Ray GPU and role/fabric capacity disagree")
        fabrics[role].add(selected["fabric"])
        gpu_uuids = sorted(name.removeprefix(GPU_RESOURCE_PREFIX) for name in resources if name.startswith(GPU_RESOURCE_PREFIX))
        if len(gpu_uuids) != gpus_per_node:
            raise ValueError("Ray node has incomplete physical GPU identities")
        rows.append({"node_id": node["NodeID"], **selected, "rank": rank, "gpu_uuids": gpu_uuids})
    if {row["rank"] for row in rows} != set(range(num_nodes)) or len({row["host"] for row in rows}) != num_nodes:
        raise ValueError("Ray nodes must cover distinct Iris ranks and physical hosts")
    if len({gpu_uuid for row in rows for gpu_uuid in row["gpu_uuids"]}) != num_nodes * gpus_per_node:
        raise ValueError("Ray nodes report duplicate physical GPUs")
    if any(len(value) != 1 for value in fabrics.values()) or len(set.union(*fabrics.values())) != len(groups):
        raise ValueError("Each role must occupy one distinct physical NVLink domain")
    return sorted(rows, key=lambda row: row["rank"])
