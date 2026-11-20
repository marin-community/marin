"""Assign each model to an Iris group confined to one NVLink domain."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any
import xml.etree.ElementTree as ET

from iris.cluster.platforms.k8s.coreweave_topology import NVL72_GPUS_PER_NODE, SCHEDULABLE_RACK_NODES, is_rack_based
from skyrlrecipe.locality import LocalityPlan, RoleTopologyMode, plan_nvlink_roles

from cloud.iris.role_plan import ModelRoleKind, RoleExecution, SkyRLRolePlan

LOCALITY_GROUP_LABEL = "iris.locality_group"
RANK_LABEL = "iris.task_rank"
HOST_LABEL = "iris.physical_host"
FABRIC_LABEL = "iris.nvlink_fabric"
CLIQUE_LABEL = "iris.nvlink_clique"
GPU_LABEL_PREFIX = "iris.physical_gpu."


@dataclass(frozen=True)
class RoleTopology:
    roles: tuple[str, ...]
    locality: LocalityPlan

    @property
    def num_nodes(self) -> int:
        return self.locality.num_nodes

    def selector(self, role: str) -> dict[str, str]:
        return {LOCALITY_GROUP_LABEL: str(self.locality.role_groups[self.roles.index(role)])}

    def node_labels(self, rank: int) -> dict[str, str]:
        return {LOCALITY_GROUP_LABEL: str(self.locality.group_for_rank(rank))}


def derive_role_topology(config: Mapping[str, Any], plan: SkyRLRolePlan) -> RoleTopology | None:
    requested = config["iris"].get("role_topology")
    if requested is None:
        return None
    if requested != RoleTopologyMode.NVLINK_DOMAIN:
        raise ValueError(f"Unsupported Iris role topology: {requested}")
    allocation = config["iris"]["allocation"]
    width, variant = allocation["gpus_per_node"], allocation["gpu_variant"]
    if not is_rack_based(variant) or width != NVL72_GPUS_PER_NODE:
        raise ValueError("NVLink role placement requires four-GPU NVL72 hosts")

    policy = plan.claim("policy")
    roles = ["policy"]
    shared_reference = True
    for claim in plan.claims:
        if claim.execution is RoleExecution.REMOTE or claim.kind in (ModelRoleKind.POLICY, ModelRoleKind.ROLLOUT):
            continue
        if claim.kind is not ModelRoleKind.REFERENCE:
            raise ValueError(
                "NVLink role placement supports only policy, a shared reference, and local inference engines"
            )
        shared_reference = claim.colocation_group == policy.colocation_group
    rollout = next((claim for claim in plan.claims if claim.kind is ModelRoleKind.ROLLOUT), None)
    if rollout is not None:
        if rollout.execution is not RoleExecution.LOCAL:
            raise ValueError("NVLink role placement requires separate policy and local inference GPUs")
        roles.extend(f"inference_{index}" for index in range(rollout.replicas))
    locality = plan_nvlink_roles(
        policy_gpus=policy.num_nodes * policy.gpus_per_node,
        num_inference_engines=rollout.replicas if rollout is not None else 0,
        tensor_parallel_size=rollout.tensor_parallel_size if rollout is not None else 1,
        pipeline_parallel_size=rollout.pipeline_parallel_size if rollout is not None else 1,
        data_parallel_size=rollout.data_parallel_size if rollout is not None else 1,
        expert_parallel_size=rollout.expert_parallel_size if rollout is not None else 1,
        inference_backend=rollout.backend if rollout is not None else "",
        colocate_all=plan.colocate_all,
        colocate_policy_ref=shared_reference,
        gpus_per_node=width,
        max_nodes_per_domain=SCHEDULABLE_RACK_NODES,
    )
    return RoleTopology(tuple(roles), locality)


def node_physical_labels(*, rank: int, gpus_per_node: int, host: str, fabric_xml: str) -> dict[str, str]:
    """Read physical GPU identities independently of the planned locality labels."""
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
    if len(fabrics) != 1 or not host or rank < 0:
        raise ValueError("Invalid Iris rank, host or node NVLink domain")
    cluster, clique = fabrics.pop()
    return {
        RANK_LABEL: str(rank),
        HOST_LABEL: host,
        FABRIC_LABEL: cluster,
        CLIQUE_LABEL: clique,
        **{GPU_LABEL_PREFIX + str(index): gpu_uuid for index, gpu_uuid in enumerate(sorted(uuids))},
    }


@dataclass(frozen=True)
class PhysicalRoleNode:
    node_id: str
    rank: int
    host: str
    locality_group: int
    roles: list[str]
    fabric: str
    gpu_uuids: list[str]


def validate_role_nodes(
    nodes: Sequence[Mapping[str, Any]],
    *,
    topology: RoleTopology,
    gpus_per_node: int,
) -> list[dict[str, Any]]:
    """Reject split or overlapping physical locality groups before model actors start."""
    alive = [node for node in nodes if node["Alive"]]
    if len(alive) != topology.num_nodes:
        raise ValueError(f"Expected exactly {topology.num_nodes} Ray nodes, observed {len(alive)}")
    rows = []
    fabrics: dict[int, set[str]] = {group: set() for group in range(len(topology.locality.group_sizes))}
    for node in alive:
        labels = node.get("Labels", {})
        try:
            rank = int(labels[RANK_LABEL])
            host = labels[HOST_LABEL]
            cluster, clique = labels[FABRIC_LABEL], labels[CLIQUE_LABEL]
        except (KeyError, ValueError) as error:
            raise ValueError("Ray node has incomplete physical identity labels") from error
        group = topology.locality.group_for_rank(rank)
        if labels.get(LOCALITY_GROUP_LABEL) != str(group):
            raise ValueError("Ray locality labels disagree with the native Iris task ranks")
        uuids = sorted(value for key, value in labels.items() if key.startswith(GPU_LABEL_PREFIX))
        if not host or not cluster or not clique or len(uuids) != gpus_per_node or "" in uuids:
            raise ValueError("Ray node has incomplete physical GPU identities")
        fabric = f"{cluster}:{clique}"
        if node["Resources"].get("GPU") != gpus_per_node:
            raise ValueError("Ray GPU capacity disagrees with the physical node width")
        fabrics[group].add(fabric)
        rows.append(
            PhysicalRoleNode(
                node_id=node["NodeID"], rank=rank, host=host, locality_group=group,
                roles=[role for role, role_group in zip(topology.roles, topology.locality.role_groups, strict=True) if role_group == group],
                fabric=fabric, gpu_uuids=uuids,
            )
        )
    if {row.rank for row in rows} != set(range(topology.num_nodes)) or len(
        {row.host for row in rows}
    ) != topology.num_nodes:
        raise ValueError("Ray nodes must cover distinct Iris ranks and physical hosts")
    if len({gpu for row in rows for gpu in row.gpu_uuids}) != topology.num_nodes * gpus_per_node:
        raise ValueError("Ray nodes report duplicate physical GPUs")
    if any(len(value) != 1 for value in fabrics.values()) or len(set.union(*fabrics.values())) != len(fabrics):
        raise ValueError("Each locality group must occupy one distinct physical NVLink domain")
    return [asdict(row) for row in sorted(rows, key=lambda row: row.rank)]
