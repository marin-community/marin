# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serialized environment inputs for an isolated rollout."""

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class EnvironmentKind(StrEnum):
    NULL = "null"
    SHELLSIM = "shellsim"
    DOCKER = "docker"


class EnvironmentFile(BaseModel):
    """A file installed in the task machine, with base64 content in JSON."""

    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_bytes="base64", val_json_bytes="base64")

    path: str
    content: bytes
    mode: int = Field(default=0o644, ge=0, le=0o777)

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Environment files require absolute paths without parent traversal")
        return value


class EnvironmentCommand(BaseModel):
    """A command that prepares a machine or collects grading inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    argv: tuple[str, ...] = Field(min_length=1)
    timeout: float = Field(gt=0)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    user: str | None = None


class LocalImage(BaseModel):
    """An image already installed in the machine backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["local"] = "local"
    reference: str = Field(min_length=1)


class RegistryImage(BaseModel):
    """An image that the machine backend resolves from a container registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["registry"] = "registry"
    reference: str = Field(min_length=1)


class DockerBuild(BaseModel):
    """A portable Docker build context. File paths start at the context root."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["build"] = "build"
    files: tuple[EnvironmentFile, ...]
    dockerfile: str = "/Dockerfile"

    @model_validator(mode="after")
    def validate_files(self) -> "DockerBuild":
        paths = {file.path for file in self.files}
        if len(paths) != len(self.files) or self.dockerfile not in paths:
            raise ValueError("A build context requires unique file paths and its Dockerfile")
        return self


ImageSpec = Annotated[LocalImage | RegistryImage | DockerBuild, Field(discriminator="kind")]


class HealthcheckSpec(BaseModel):
    """A readiness command with a startup grace period and retry limit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: EnvironmentCommand
    interval: float = Field(ge=0)
    start_period: float = Field(ge=0)
    start_interval: float = Field(ge=0)
    retries: int = Field(gt=0)


class EnvironmentSpec(BaseModel):
    """Machine inputs visible to the agent before execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EnvironmentKind
    image: ImageSpec | None = None
    workdir: str = "/workspace"
    files: tuple[EnvironmentFile, ...] = ()
    env: dict[str, str] = Field(default_factory=dict)
    setup: tuple[EnvironmentCommand, ...] = ()
    healthcheck: HealthcheckSpec | None = None
    startup_timeout: float | None = Field(default=None, gt=0)
    network: bool = False
    memory_mb: int | None = Field(default=None, gt=0)
    cpus: int | None = Field(default=None, gt=0)
    storage_mb: int | None = Field(default=None, gt=0)
    gpus: int = Field(default=0, ge=0)
    interaction: str | None = None

    @model_validator(mode="after")
    def validate_machine(self) -> "EnvironmentSpec":
        if (self.kind == EnvironmentKind.DOCKER) != bool(self.image):
            raise ValueError("Only Docker environments require an image")
        if self.kind == EnvironmentKind.NULL and (
            self.files
            or self.env
            or self.setup
            or self.healthcheck is not None
            or self.network
            or self.memory_mb is not None
            or self.cpus is not None
            or self.storage_mb is not None
            or self.gpus
        ):
            raise ValueError("Null environments cannot contain machine resources")
        if len({file.path for file in self.files}) != len(self.files):
            raise ValueError("Environment file paths must be unique")
        return self


class StdoutReward(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["stdout"] = "stdout"


class ExitCodeReward(BaseModel):
    """Score a completed command as one on success and zero on failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["exit_code"] = "exit_code"


class ArtifactKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"
    AUTO = "auto"


class MissingArtifactPolicy(StrEnum):
    ERROR = "error"
    SKIP = "skip"


class VerifierArtifact(BaseModel):
    """An agent file or directory copied into a fresh grading environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    target: str
    kind: ArtifactKind
    exclude: tuple[str, ...] = ()
    missing: MissingArtifactPolicy = MissingArtifactPolicy.ERROR

    @field_validator("source", "target")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        return EnvironmentFile.absolute_path(value)


class RewardFileFormat(StrEnum):
    NUMBER = "number"
    JSON = "json"


class RewardFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    format: RewardFileFormat
    key: str = "reward"

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value: str) -> str:
        return EnvironmentFile.absolute_path(value)


class FileReward(BaseModel):
    """Reward files in priority order. The first existing file supplies the score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["file"] = "file"
    files: tuple[RewardFile, ...] = Field(min_length=1)
    pass_above: float | None = Field(default=None, allow_inf_nan=False)


class ShellVerifierSpec(BaseModel):
    """Private files, a verifier command, and its reward source.

    The engine installs private files after the final model response. For file
    rewards, a valid reward file takes priority over the command exit code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    argv: tuple[str, ...] = Field(min_length=1)
    files: tuple[EnvironmentFile, ...] = ()
    timeout: float = Field(gt=0)
    env: dict[str, str] = Field(default_factory=dict)
    user: str | None = None
    reward: Annotated[StdoutReward | FileReward | ExitCodeReward, Field(discriminator="kind")] = StdoutReward()
    environment: EnvironmentSpec | None = None
    collect: tuple[EnvironmentCommand, ...] = ()
    artifacts: tuple[VerifierArtifact, ...] = ()

    @model_validator(mode="after")
    def validate_environment(self) -> "ShellVerifierSpec":
        if self.environment is None:
            if self.artifacts:
                raise ValueError("Verifier artifacts require a separate grading environment")
            return self
        if self.environment.kind == EnvironmentKind.NULL or self.environment.interaction is not None:
            raise ValueError("A private grading environment requires a machine without a task session")
        return self


class ExternalVerifierSpec(BaseModel):
    """Private inputs for a verifier supplied by the execution application."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    parameters: dict[str, JsonValue]
