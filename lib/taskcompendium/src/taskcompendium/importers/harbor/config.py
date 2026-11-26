# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor's ``task.toml`` (schemas 1.0 to 1.2), with Harbor's defaults, and the subset the importer lowers.

Harbor ignores keys it does not model, and so does this model. Fields that change how a task
runs are modelled so that :func:`read_task_config` can reject the ones the importer cannot lower.
"""

import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TASK_CONFIG = "task.toml"
SUPPORTED_SCHEMAS = frozenset({"1.0", "1.0.0", "1.1", "1.2"})
"""Harbor does not interpret the version; some 1.0 packages write it as ``1.0.0``."""
ROOT_USERS = (None, "root", 0)


class UnsupportedHarborFeature(Exception):
    """A package uses a Harbor feature the importer cannot lower; publication records ``reason`` as a rejection."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class HarborModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class TaskOS(StrEnum):
    LINUX = "linux"
    WINDOWS = "windows"


class VerifierMode(StrEnum):
    """``shared`` grades in the agent's container after the agent stops; ``separate`` in a fresh one."""

    SHARED = "shared"
    SEPARATE = "separate"


class McpTransport(StrEnum):
    SSE = "sse"
    STREAMABLE_HTTP = "streamable-http"
    STDIO = "stdio"


class Healthcheck(HarborModel):
    """Docker HEALTHCHECK semantics: failures during the start period do not count toward retries."""

    command: str
    interval_sec: float = 5.0
    timeout_sec: float = 30.0
    start_period_sec: float = 0.0
    start_interval_sec: float = 5.0
    retries: int = 3


class McpServer(HarborModel):
    name: str
    transport: McpTransport = McpTransport.SSE
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()

    @field_validator("transport", mode="before")
    @classmethod
    def normalize_transport(cls, value: Any) -> Any:
        return McpTransport.STREAMABLE_HTTP if value == "http" else value


def _size_mb(size: str) -> int:
    size = size.strip().upper()
    units = {"G": 1024, "M": 1, "K": 1 / 1024}
    if not size or size[-1] not in units:
        raise ValueError(f"Invalid size {size!r}; expected a value like '1G' or '512M'")
    return int(float(size[:-1]) * units[size[-1]])


class EnvironmentSetup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str = Field(min_length=1)
    timeout_sec: float = Field(gt=0)


class EnvironmentConfig(HarborModel):
    build_timeout_sec: float = 600.0
    docker_image: str | None = None
    os: TaskOS = TaskOS.LINUX
    cpus: int | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int | None = None
    gpu_types: tuple[str, ...] | None = None
    tpu: dict[str, Any] | None = None
    allow_internet: bool = True
    mcp_servers: tuple[McpServer, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)
    skills_dir: str | None = None
    healthcheck: Healthcheck | None = None
    workdir: str | None = None
    setup: EnvironmentSetup | None = None

    @field_validator("os", mode="before")
    @classmethod
    def normalize_os(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    @model_validator(mode="before")
    @classmethod
    def migrate_sizes(cls, data: Any) -> Any:
        """Read schema 1.0's ``memory`` and ``storage`` size strings as Harbor does."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for legacy, field in (("memory", "memory_mb"), ("storage", "storage_mb")):
            value = data.pop(legacy, None)
            if not isinstance(value, str):
                continue
            size = _size_mb(value)
            if data.get(field, size) != size:
                raise ValueError(f"Conflicting {legacy}={value!r} and {field}={data[field]!r}")
            data[field] = size
        return data


class CollectHook(HarborModel):
    command: str
    service: Literal["main"] = "main"
    timeout_sec: float = Field(default=60.0, gt=0)
    user: str | int | None = None


class VerifierConfig(HarborModel):
    timeout_sec: float = 600.0
    env: dict[str, str] = Field(default_factory=dict)
    user: str | int | None = None
    environment_mode: VerifierMode | None = None
    environment: EnvironmentConfig | None = None
    collect: tuple[CollectHook, ...] = ()

    @model_validator(mode="after")
    def check_mode(self) -> "VerifierConfig":
        if self.environment_mode == VerifierMode.SHARED and self.environment is not None:
            raise ValueError("[verifier].environment_mode='shared' is incompatible with [verifier.environment]")
        return self

    @property
    def mode(self) -> VerifierMode:
        """Harbor's resolution: explicit mode, else separate when an environment is declared, else shared."""
        if self.environment_mode is not None:
            return self.environment_mode
        return VerifierMode.SHARED if self.environment is None else VerifierMode.SEPARATE


class AgentConfig(HarborModel):
    timeout_sec: float | None = None
    env: dict[str, str] = Field(default_factory=dict)
    user: str | int | None = None


class ArtifactConfig(HarborModel):
    source: str
    destination: str | None = None
    exclude: tuple[str, ...] = ()


class HarborTaskConfig(HarborModel):
    schema_version: str = "1.2"
    metadata: dict[str, Any] = Field(default_factory=dict)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    source: str | None = None
    multi_step_reward_strategy: str | None = None
    steps: tuple[dict[str, Any], ...] | None = None
    artifacts: tuple[ArtifactConfig, ...] = ()

    @field_validator("artifacts", mode="before")
    @classmethod
    def expand_artifact_paths(cls, value: Any) -> Any:
        """Harbor also writes an artifact as its bare source path."""
        return tuple({"source": item} if isinstance(item, str) else item for item in value)

    @model_validator(mode="before")
    @classmethod
    def rename_version(cls, data: Any) -> Any:
        if isinstance(data, dict) and "version" in data:
            data = dict(data)
            data.setdefault("schema_version", data.pop("version"))
        return data

    @property
    def verifier_environment(self) -> EnvironmentConfig:
        """The separate verifier's environment: its own, or a copy of the task's."""
        return self.verifier.environment or self.environment


def _require_environment(environment: EnvironmentConfig, section: str) -> None:
    if environment.os != TaskOS.LINUX:
        raise UnsupportedHarborFeature(f"[{section}] targets {environment.os}; only Linux tasks are supported")
    if environment.gpus or environment.gpu_types:
        raise UnsupportedHarborFeature(f"[{section}] requests GPUs")
    if environment.tpu is not None:
        raise UnsupportedHarborFeature(f"[{section}] requests a TPU")
    if environment.skills_dir is not None:
        raise UnsupportedHarborFeature(f"[{section}] installs agent skills from skills_dir")


def require_supported(
    config: HarborTaskConfig, *, agent_user: str | None = None, setup_timeout: float | None = None
) -> None:
    """Reject task-level features the importer cannot lower; environment files are checked while lowering."""
    if config.schema_version not in SUPPORTED_SCHEMAS:
        raise UnsupportedHarborFeature(f"task.toml schema {config.schema_version} is not supported")
    if config.steps or config.multi_step_reward_strategy is not None:
        raise UnsupportedHarborFeature("multi-step tasks are not supported")
    _require_environment(config.environment, "environment")
    if agent_user is None and config.agent.user not in ROOT_USERS:
        raise UnsupportedHarborFeature(f"agent runs as user {config.agent.user!r}; only root is supported")
    if agent_user is not None and str(config.agent.user) != agent_user:
        raise UnsupportedHarborFeature(f"agent runs as user {config.agent.user!r}; deployment expects {agent_user}")
    setup = config.environment.setup
    if (setup.timeout_sec if setup is not None else None) != setup_timeout:
        raise UnsupportedHarborFeature("environment setup timeout does not match the deployment")
    if config.verifier.user not in ROOT_USERS:
        raise UnsupportedHarborFeature(f"verifier runs as user {config.verifier.user!r}; only root is supported")
    if any(hook.user is not None for hook in config.verifier.collect):
        raise UnsupportedHarborFeature("verifier collect hooks with their own users are not supported")
    # Harbor runs healthchecks and MCP servers only for the agent, even in a copied verifier environment.
    verifier = config.verifier.environment
    if verifier is not None:
        _require_environment(verifier, "verifier.environment")
        if verifier.healthcheck is not None or verifier.mcp_servers or verifier.setup is not None:
            raise UnsupportedHarborFeature("verifier environments with healthchecks, MCP servers, or setup commands")


def read_task_config(
    directory: Path, *, agent_user: str | None = None, setup_timeout: float | None = None
) -> HarborTaskConfig:
    """Parse a package's ``task.toml`` and reject task-level features the importer cannot lower."""
    config = HarborTaskConfig.model_validate(tomllib.loads((directory / TASK_CONFIG).read_text()))
    require_supported(config, agent_user=agent_user, setup_timeout=setup_timeout)
    return config
