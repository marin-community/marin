"""Assign each model to an Iris group confined to one NVLink domain."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from iris.cluster.platforms.k8s.coreweave_topology import NVL72_GPUS_PER_NODE, SCHEDULABLE_RACK_NODES, is_rack_based
from skyrlrecipe.locality import LocalityPlan, RoleTopologyMode, plan_nvlink_roles

from cloud.iris.role_plan import ModelRoleKind, RoleExecution, SkyRLRolePlan

LOCALITY_GROUP_LABEL = "iris.locality_group"


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
            raise ValueError("NVLink role placement supports only policy, a shared reference, and local inference engines")
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
