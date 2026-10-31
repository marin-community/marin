from pathlib import Path

from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import ResourceMode


def validate_local_resources(
    task: EnvironmentConfig,
    *,
    cpu_enforcement_policy: ResourceMode,
    memory_enforcement_policy: ResourceMode,
    cgroup_parent: Path | None = None,
) -> None:
    """Require delegated cgroups or explicit ignore policies for local resource declarations."""
    validate_resource_capabilities(
        environment_label="local",
        resource_capabilities=EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True),
        cpu_enforcement_policy=cpu_enforcement_policy,
        memory_enforcement_policy=memory_enforcement_policy,
    )
    if task.storage_mb is not None or task.gpus or task.gpu_types or task.tpu is not None:
        raise ValueError("Local prepared sandboxes do not enforce storage, GPU, or TPU allocations")
    for name, value, mode in (
        ("CPU", task.cpus, cpu_enforcement_policy),
        ("memory", task.memory_mb, memory_enforcement_policy),
    ):
        if value is not None and mode is not ResourceMode.IGNORE and cgroup_parent is None:
            raise ValueError(f"Local {name} limits require cgroup_parent or explicit enforcement_policy=ignore")


def validate_resource_capabilities(
    *,
    environment_label: str,
    resource_capabilities: EnvironmentResourceCapabilities,
    cpu_enforcement_policy: ResourceMode,
    memory_enforcement_policy: ResourceMode,
) -> None:
    checks = (
        (
            "CPU",
            cpu_enforcement_policy,
            resource_capabilities.cpu_limit,
            resource_capabilities.cpu_request,
        ),
        (
            "memory",
            memory_enforcement_policy,
            resource_capabilities.memory_limit,
            resource_capabilities.memory_request,
        ),
    )
    for label, mode, supports_limit, supports_request in checks:
        if mode in (ResourceMode.AUTO, ResourceMode.IGNORE):
            continue
        if mode in (ResourceMode.LIMIT, ResourceMode.GUARANTEE) and not supports_limit:
            raise ValueError(f"{environment_label} environment does not support {label} resource limits.")
        if mode in (ResourceMode.REQUEST, ResourceMode.GUARANTEE) and not supports_request:
            raise ValueError(f"{environment_label} environment does not support {label} resource requests.")


def validate_resource_values(
    *,
    cpu_enforcement_policy: ResourceMode,
    memory_enforcement_policy: ResourceMode,
    cpus: int | None,
    memory_mb: int | None,
) -> None:
    checks = (
        ("CPU", cpu_enforcement_policy, cpus),
        ("memory", memory_enforcement_policy, memory_mb),
    )
    for label, mode, value in checks:
        if mode in (ResourceMode.AUTO, ResourceMode.IGNORE):
            continue
        if value is None:
            raise ValueError(f"{label} resource mode '{mode.value}' requires a task value or numeric override.")
