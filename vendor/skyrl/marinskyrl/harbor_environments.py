"""Configure Harbor-owned environments for original-package trials."""

from harbor_config.models.task.config import MCPServerConfig
from harbor_config.models.trial.config import TrialConfig
from pydantic import BaseModel, ConfigDict, Field


class EnvironmentResourceOverrides(BaseModel):
    model_config = ConfigDict(extra="forbid")

    override_cpus: int | None = None
    override_memory_mb: int | None = None
    override_storage_mb: int | None = None
    override_gpus: int | None = None


class HarborTaskBinding(BaseModel):
    """Runtime policy applied to one canonical task ID."""

    model_config = ConfigDict(extra="forbid")

    environment_resources: EnvironmentResourceOverrides = Field(default_factory=EnvironmentResourceOverrides)
    agent_mcp_servers: list[MCPServerConfig] | None = None
    verifier_environment: dict[str, str] | None = None


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
        configured = configured.model_copy(
            update={
                "verifier": configured.verifier.model_copy(
                    update={"env": {**configured.verifier.env, **binding.verifier_environment}}
                )
            }
        )
    return configured
