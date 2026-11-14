"""Assign each model to an Iris group confined to one NVLink domain."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from iris.cluster.platforms.k8s.coreweave_topology import NVL72_GPUS_PER_NODE, SCHEDULABLE_RACK_NODES, is_rack_based
from skyrlrecipe.locality import LocalityPlan, plan_locality_groups

from cloud.iris.role_plan import ModelRoleKind, RoleExecution, derive_policy_export_role_plan, derive_role_plan

NVLINK_ROLE_TOPOLOGY = "nvlink.domain"
LOCALITY_GROUP_LABEL = "iris.locality_group"


@dataclass(frozen=True)
class RoleTopology:
    roles: tuple[str, ...]
    locality: LocalityPlan
    rack_based: bool

    @property
    def num_nodes(self) -> int:
        return self.locality.num_nodes

    def selector(self, role: str) -> dict[str, str]:
        return {LOCALITY_GROUP_LABEL: str(self.locality.role_groups[self.roles.index(role)])}

    def node_labels(self, rank: int) -> dict[str, str]:
        return {LOCALITY_GROUP_LABEL: str(self.locality.group_for_rank(rank))}


def derive_role_topology(config: Mapping[str, Any]) -> RoleTopology | None:
    requested = config["iris"].get("role_topology")
    if requested is None:
        return None
    if requested != NVLINK_ROLE_TOPOLOGY:
        raise ValueError(f"Unsupported Iris role topology: {requested}")
    allocation = config["iris"]["allocation"]
    width, variant = allocation["gpus_per_node"], allocation["gpu_variant"]
    rack_based = is_rack_based(variant)
    if rack_based and width == NVL72_GPUS_PER_NODE:
        domain_nodes = SCHEDULABLE_RACK_NODES
    elif variant.upper().startswith("H100") and width == 8:
        domain_nodes = 1
    else:
        raise ValueError("NVLink role placement requires four-GPU NVL72 hosts or eight-GPU H100 hosts")

    exporting = config["run"]["mode"] == "checkpoint_export"
    plan = derive_policy_export_role_plan(config["skyrl"]) if exporting else derive_role_plan(config["skyrl"])
    policy = plan.claim("policy")
    roles = ["policy"]
    requirements = [policy.num_nodes * policy.gpus_per_node]
    for claim in plan.claims:
        if claim.execution is RoleExecution.REMOTE or claim.kind in (ModelRoleKind.POLICY, ModelRoleKind.ROLLOUT):
            continue
        if claim.kind is not ModelRoleKind.REFERENCE or claim.colocation_group != policy.colocation_group:
            raise ValueError("NVLink role placement supports only policy, a shared reference, and local inference engines")
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
        roles.extend(f"inference_{index}" for index in range(rollout.replicas))
        requirements.extend([rollout.data_parallel_size] * rollout.replicas)
    locality = plan_locality_groups(requirements, gpus_per_node=width, max_nodes_per_domain=domain_nodes)
    return RoleTopology(tuple(roles), locality, rack_based)
