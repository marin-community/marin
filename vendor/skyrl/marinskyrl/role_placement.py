"""Ray resources that keep a policy and each inference replica in its Iris rack slice."""

from collections.abc import Mapping

ROLE_RESOURCE_PREFIX = "skyrl_role:"
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
