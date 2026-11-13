"""Keep the policy and each inference engine within their assigned NVLink domain."""

from collections.abc import Mapping, Sequence

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
