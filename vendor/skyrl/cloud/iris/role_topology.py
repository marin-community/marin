"""Bind SkyRL groups to native Iris domains and verify their physical identity."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import xml.etree.ElementTree as ET

from iris.cluster.platforms.k8s.coreweave_topology import (
    COSCHEDULE_NVLINK_DOMAIN_SLICED,
    NVL72_GPUS_PER_NODE,
    gpu_gang_coscheduling_level,
    gpu_gang_rack_slice_size,
    is_rack_based,
)

from cloud.iris.role_plan import ModelRoleKind, RoleExecution, derive_policy_export_role_plan, derive_role_plan
from marinskyrl.role_placement import FABRIC_RESOURCE_PREFIX, POLICY_ROLE, ROLE_RESOURCE_PREFIX, inference_role

NVLINK_ROLE_TOPOLOGY = "nvlink.domain"
RANK_RESOURCE_PREFIX = "skyrl_rank:"
HOST_RESOURCE_PREFIX = "skyrl_host:"
GPU_RESOURCE_PREFIX = "skyrl_gpu:"


@dataclass(frozen=True)
class RoleAllocation:
    role: str
    gpu_start: int
    num_gpus: int


@dataclass(frozen=True)
class RoleTopology:
    groups: tuple[RoleAllocation, ...]
    num_nodes: int
    gpus_per_node: int
    nodes_per_domain: int
    rack_based: bool


def derive_role_topology(config: Mapping[str, Any]) -> RoleTopology | None:
    """Pack each requested group within one native Iris NVLink domain."""
    requested = config["iris"].get("role_topology")
    if requested is None:
        return None
    if requested != NVLINK_ROLE_TOPOLOGY:
        raise ValueError(f"Unsupported Iris role topology: {requested}")
    allocation = config["iris"]["allocation"]
    nodes, width = allocation["num_nodes"], allocation["gpus_per_node"]
    variant = allocation["gpu_variant"]
    rack_based = is_rack_based(variant)
    mode = gpu_gang_coscheduling_level(variant, width, nodes)
    if rack_based:
        if width != NVL72_GPUS_PER_NODE:
            raise ValueError(f"NVLink role placement requires whole NVL72 hosts ({NVL72_GPUS_PER_NODE} GPUs each)")
        domain_nodes = gpu_gang_rack_slice_size(width, nodes) if mode == COSCHEDULE_NVLINK_DOMAIN_SLICED else nodes
    elif variant.upper().startswith("H100") and width == 8:
        domain_nodes = 1
    else:
        raise ValueError("NVLink role placement supports native Iris NVL72 domains or eight-GPU H100 hosts")

    exporting = config["run"]["mode"] == "checkpoint_export"
    plan = derive_policy_export_role_plan(config["skyrl"]) if exporting else derive_role_plan(config["skyrl"])
    policy = plan.claim("policy")
    requirements = [(POLICY_ROLE, policy.num_nodes * policy.gpus_per_node)]
    for claim in plan.claims:
        if claim.execution is RoleExecution.REMOTE or claim.kind in (ModelRoleKind.POLICY, ModelRoleKind.ROLLOUT):
            continue
        if claim.kind is not ModelRoleKind.REFERENCE or claim.colocation_group != policy.colocation_group:
            raise ValueError(
                "NVLink role placement supports only policy, a shared reference, and local inference engines"
            )
    if not exporting:
        rollout = plan.claim("rollout")
        if plan.colocate_all or rollout.execution is not RoleExecution.LOCAL:
            raise ValueError("NVLink role placement requires separate policy and local inference GPUs")
        if (
            rollout.backend != "vllm"
            or rollout.tensor_parallel_size != 1
            or rollout.pipeline_parallel_size != 1
            or rollout.expert_parallel_size != rollout.data_parallel_size
        ):
            raise ValueError("NVLink role placement requires vLLM TP1/PP1 EP=DP inference engines")
        requirements.extend((inference_role(index), rollout.data_parallel_size) for index in range(rollout.replicas))
    domain_gpus = domain_nodes * width
    occupied = [0] * (nodes // domain_nodes)
    groups = {}
    # Reserve whole trainer hosts first. Inference engines all have the same
    # configured size and can share the remaining hosts within a domain.
    for role, count in requirements:
        if count > domain_gpus:
            raise ValueError(f"NVLink group {role} needs {count} GPUs but one allocated domain holds {domain_gpus}")
        domain = next((index for index, used in enumerate(occupied) if used + count <= domain_gpus), None)
        if domain is None:
            raise ValueError(
                f"NVLink groups do not fit the {len(occupied)} allocated domains of {domain_gpus} GPUs; "
                f"cannot place {role} ({count} GPUs) without crossing a domain"
            )
        groups[role] = RoleAllocation(role, domain * domain_gpus + occupied[domain], count)
        occupied[domain] += count
    return RoleTopology(tuple(groups[role] for role, _ in requirements), nodes, width, domain_nodes, rack_based)


def _node_roles(topology: RoleTopology, rank: int) -> dict[str, float]:
    start = rank * topology.gpus_per_node
    end = start + topology.gpus_per_node
    return {
        group.role: float(overlap)
        for group in topology.groups
        if (overlap := min(end, group.gpu_start + group.num_gpus) - max(start, group.gpu_start)) > 0
    }


def node_role_resources(*, topology: RoleTopology, rank: int, host: str, fabric_xml: str) -> dict[str, float]:
    """Advertise assigned role capacity and independently observed physical GPU identities."""
    devices = ET.fromstring(fabric_xml).findall("gpu")
    uuids = {device.findtext("uuid") or "" for device in devices}
    width = topology.gpus_per_node
    if len(devices) != width or len(uuids) != width or "" in uuids:
        raise ValueError("Physical GPU identities disagree with the allocated node width")
    if not host or not 0 <= rank < topology.num_nodes:
        raise ValueError("Invalid Iris rank or physical host")
    fabric_identity = f"host:{host}"
    if topology.rack_based:
        fabrics = set()
        for device in devices:
            fabric = device.find("fabric")
            if fabric is None or fabric.findtext("state") != "Completed" or fabric.findtext("status") != "Success":
                raise ValueError("Every GPU must report a completed, healthy NVLink fabric")
            identity = (fabric.findtext("clusterUuid") or "", fabric.findtext("cliqueId") or "")
            if any(not value or value == "N/A" for value in identity):
                raise ValueError("Missing NVLink fabric identity")
            fabrics.add(identity)
        if len(fabrics) != 1:
            raise ValueError("GPUs on one host report different physical NVLink domains")
        cluster, clique = fabrics.pop()
        fabric_identity = f"{cluster}:{clique}"
    return {
        **{ROLE_RESOURCE_PREFIX + role: count for role, count in _node_roles(topology, rank).items()},
        FABRIC_RESOURCE_PREFIX + fabric_identity: float(width),
        RANK_RESOURCE_PREFIX + str(rank): 1.0,
        HOST_RESOURCE_PREFIX + host: 1.0,
        **{GPU_RESOURCE_PREFIX + gpu_uuid: 1.0 for gpu_uuid in sorted(uuids)},
    }


def validate_role_nodes(nodes: Sequence[Mapping[str, Any]], *, topology: RoleTopology) -> list[dict[str, Any]]:
    """Verify native slices and every group's real domain before model creation."""
    alive = [node for node in nodes if node["Alive"]]
    if len(alive) != topology.num_nodes:
        raise ValueError(f"Expected exactly {topology.num_nodes} Ray nodes, observed {len(alive)}")
    rows = []
    domains: dict[int, set[str]] = {}
    fabrics: dict[str, set[str]] = {group.role: set() for group in topology.groups}
    for node in alive:
        resources = node["Resources"]
        selected = {}
        for field, prefix in (
            ("fabric", FABRIC_RESOURCE_PREFIX),
            ("rank", RANK_RESOURCE_PREFIX),
            ("host", HOST_RESOURCE_PREFIX),
        ):
            matches = [name.removeprefix(prefix) for name in resources if name.startswith(prefix)]
            if len(matches) != 1:
                raise ValueError(f"Ray node {node['NodeID']} has ambiguous {field} identity")
            selected[field] = matches[0]
        rank = int(selected["rank"])
        if not 0 <= rank < topology.num_nodes:
            raise ValueError(f"Ray node reports an unexpected Iris rank: {rank}")
        roles = {
            name.removeprefix(ROLE_RESOURCE_PREFIX): count
            for name, count in resources.items()
            if name.startswith(ROLE_RESOURCE_PREFIX)
        }
        if roles != _node_roles(topology, rank):
            raise ValueError(f"Ray role capacity on Iris rank {rank} disagrees with the requested groups")
        if any(
            resources.get(key) != topology.gpus_per_node for key in ("GPU", FABRIC_RESOURCE_PREFIX + selected["fabric"])
        ):
            raise ValueError("Ray GPU and physical fabric capacity disagree")
        domains.setdefault(rank // topology.nodes_per_domain, set()).add(selected["fabric"])
        for role in roles:
            fabrics[role].add(selected["fabric"])
        gpu_uuids = sorted(
            name.removeprefix(GPU_RESOURCE_PREFIX) for name in resources if name.startswith(GPU_RESOURCE_PREFIX)
        )
        if len(gpu_uuids) != topology.gpus_per_node:
            raise ValueError("Ray node has incomplete physical GPU identities")
        rows.append({"node_id": node["NodeID"], **selected, "rank": rank, "roles": roles, "gpu_uuids": gpu_uuids})
    if {row["rank"] for row in rows} != set(range(topology.num_nodes)) or len(
        {row["host"] for row in rows}
    ) != topology.num_nodes:
        raise ValueError("Ray nodes must cover distinct Iris ranks and physical hosts")
    if len({gpu for row in rows for gpu in row["gpu_uuids"]}) != topology.num_nodes * topology.gpus_per_node:
        raise ValueError("Ray nodes report duplicate physical GPUs")
    if any(len(values) != 1 for values in domains.values()) or len(set.union(*domains.values())) != len(domains):
        observed = {index: sorted(values) for index, values in domains.items()}
        raise ValueError(f"Physical NVLink domains disagree with Iris's native allocation slices: {observed}")
    if any(len(values) != 1 for values in fabrics.values()):
        observed = {role: sorted(values) for role, values in fabrics.items()}
        raise ValueError(f"A requested group spans physical NVLink domains: {observed}")
    return sorted(rows, key=lambda row: row["rank"])
