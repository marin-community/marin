"""Configure Harbor-owned environments for original-package trials."""

from enum import StrEnum

from harbor_config.models.task.config import MCPServerConfig
from harbor_config.models.trial.config import EnvironmentConfig, TrialConfig
from pydantic import BaseModel, ConfigDict, Field, model_validator


class HarborEnvironmentProvisioning(StrEnum):
    """How an original-package trial obtains its Harbor environment."""

    NATIVE = "native"
    APPROVED_SNAPSHOTS = "approved_snapshots"


class EnvironmentResourceOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    override_cpus: int | None = None
    override_memory_mb: int | None = None
    override_storage_mb: int | None = None
    override_gpus: int | None = None


class DeferredVerifierEnvironment(BaseModel):
    """Keep host variable names unresolved until Harbor starts the verifier."""

    model_config = ConfigDict(extra="forbid")

    values: dict[str, str] = Field(default_factory=dict)
    variables: dict[str, str] = Field(default_factory=dict)

    def harbor_environment(self) -> dict[str, str]:
        return {**self.values, **{name: "${" + variable + "}" for name, variable in self.variables.items()}}


class HarborTaskBinding(BaseModel):
    """Runtime policy applied to one canonical task ID."""

    model_config = ConfigDict(extra="forbid")

    environment_resources: EnvironmentResourceOverrides = Field(default_factory=EnvironmentResourceOverrides)
    agent_mcp_servers: list[MCPServerConfig] | None = None
    verifier_environment: DeferredVerifierEnvironment | None = None


class HarborEnvironmentSettings(BaseModel):
    """Explicit provisioning and task-specific policy for original trials."""

    model_config = ConfigDict(extra="forbid")

    provisioning: HarborEnvironmentProvisioning
    approved_snapshots: dict[str, str] = Field(default_factory=dict)
    task_bindings: dict[str, HarborTaskBinding] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_provisioning(self) -> "HarborEnvironmentSettings":
        if self.provisioning is HarborEnvironmentProvisioning.NATIVE and self.approved_snapshots:
            raise ValueError("Native Harbor provisioning does not accept approved snapshots")
        if self.provisioning is HarborEnvironmentProvisioning.APPROVED_SNAPSHOTS and not self.approved_snapshots:
            raise ValueError("Approved-snapshot provisioning requires at least one snapshot")
        return self


def original_trial_environment(
    environment: EnvironmentConfig,
    source_key: str,
    settings: HarborEnvironmentSettings,
    resources: EnvironmentResourceOverrides,
) -> EnvironmentConfig:
    """Apply task resources and the explicitly selected provisioning policy."""
    effective = environment.model_copy(update=resources.model_dump(exclude_none=True))
    if settings.provisioning is HarborEnvironmentProvisioning.NATIVE:
        return effective
    if source_key not in settings.approved_snapshots:
        raise ValueError(f"Harbor environment has no approved snapshot: {source_key}")
    return effective.model_copy(
        update={
            "force_build": False,
            "kwargs": {
                **effective.kwargs,
                "snapshot_template_name": settings.approved_snapshots[source_key],
                "auto_snapshot": False,
                "assume_global_snapshot": True,
            },
        }
    )


def apply_task_binding(config: TrialConfig, binding: HarborTaskBinding | None) -> TrialConfig:
    """Overlay task-scoped agent and verifier policy on a trial config."""
    if binding is None:
        return config
    configured = config
    if binding.agent_mcp_servers is not None:
        configured = configured.model_copy(
            update={"agent": configured.agent.model_copy(update={"mcp_servers": binding.agent_mcp_servers})}
        )
    if binding.verifier_environment is not None:
        verifier_environment = binding.verifier_environment.harbor_environment()
        configured = configured.model_copy(
            update={
                "verifier": configured.verifier.model_copy(
                    update={"env": {**configured.verifier.env, **verifier_environment}}
                )
            }
        )
    return configured
