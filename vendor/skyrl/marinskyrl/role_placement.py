"""Keep the policy and each inference engine within their assigned NVLink domain."""

from collections.abc import Mapping, Sequence

import ray
from ray.util.placement_group import placement_group_table

ROLE_RESOURCE_PREFIX = "skyrl_role:"
FABRIC_RESOURCE_PREFIX = "skyrl_nvlink:"
POLICY_ROLE = "policy"


def inference_role(replica: int) -> str:
    return f"inference_{replica}"


def role_resources(role: str, cluster_resources: Mapping[str, float]) -> dict[str, float]:
    """Return a per-GPU constraint when the Iris cluster advertises role placement."""
    if not any(name.startswith(ROLE_RESOURCE_PREFIX) for name in cluster_resources):
        return {}
    key = ROLE_RESOURCE_PREFIX + role
    if cluster_resources.get(key, 0) <= 0:
        raise ValueError(f"The Ray cluster has no capacity for required role {role}")
    return {key: 1.0}


def role_bundles(
    role: str, bundles: Sequence[Mapping[str, float]], cluster_resources: Mapping[str, float]
) -> list[dict[str, float]]:
    """Reserve one unit of the assigned role for each GPU in a bundle."""
    resources = role_resources(role, cluster_resources)
    required = sum(bundle.get("GPU", 0) for bundle in bundles)
    for resource in resources:
        if cluster_resources[resource] < required:
            raise ValueError(f"Role {role} needs {required} GPUs but advertises only {cluster_resources[resource]}")
    return [
        {**bundle, **{resource: amount * bundle["GPU"] for resource, amount in resources.items()}} for bundle in bundles
    ]


def verify_role_placement(pg, role: str) -> dict[int, str]:
    """Check ready bundle assignments against physical domains before model actors start."""
    nodes = {node["NodeID"]: node["Resources"] for node in ray.nodes() if node["Alive"]}
    table = placement_group_table(pg)
    assignments = table["bundles_to_node_id"]
    fabrics = set()
    resource = ROLE_RESOURCE_PREFIX + role
    for index, bundle in table["bundles"].items():
        node_id = assignments.get(index)
        resources = nodes.get(node_id, {})
        if bundle.get(resource) != bundle.get("GPU") or resources.get(resource, 0) < bundle["GPU"]:
            raise ValueError(f"Placement bundle {index} for {role} is outside its assigned GPU capacity")
        domains = {name for name in resources if name.startswith(FABRIC_RESOURCE_PREFIX)}
        if len(domains) != 1:
            raise ValueError(f"Placement bundle {index} for {role} has no unambiguous physical NVLink domain")
        fabrics.update(domains)
    if len(assignments) != len(table["bundles"]) or len(fabrics) != 1:
        raise ValueError(f"Placement group for {role} is incomplete or spans physical NVLink domains")
    return assignments
