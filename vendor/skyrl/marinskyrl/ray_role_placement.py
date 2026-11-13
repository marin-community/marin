"""Verify the Ray reservations for each NVLink-local model group."""

import ray
from ray.util.placement_group import placement_group_table

from marinskyrl.role_placement import FABRIC_RESOURCE_PREFIX, ROLE_RESOURCE_PREFIX


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
