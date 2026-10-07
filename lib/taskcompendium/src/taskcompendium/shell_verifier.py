# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private shell verification commands and reward files."""

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rigging.filesystem.path_validation import validate_relative_file_path


def _absolute_path(value: str) -> str:
    if not PurePosixPath(value).is_absolute():
        raise ValueError("Verifier paths must be absolute")
    validate_relative_file_path(value.removeprefix("/"))
    return value


class VerifierCommand(BaseModel):
    """A trusted command that collects grading inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)


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
    """An agent file or directory copied into a private grading machine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    target: str
    kind: ArtifactKind
    exclude: tuple[str, ...] = ()
    missing: MissingArtifactPolicy = MissingArtifactPolicy.ERROR

    _validate_paths = field_validator("source", "target")(_absolute_path)


class RewardFileFormat(StrEnum):
    NUMBER = "number"
    JSON = "json"


class RewardFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    format: RewardFileFormat
    key: str = "reward"

    _validate_path = field_validator("path")(_absolute_path)


class FileReward(BaseModel):
    """The first existing file supplies the score. Invalid content is a failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["file"] = "file"
    files: tuple[RewardFile, ...] = Field(min_length=1)
    pass_above: float | None = Field(default=None, allow_inf_nan=False)


class ShellVerifierSpec(BaseModel):
    """A verifier command and its reward source, without deployment settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    argv: tuple[str, ...] = Field(min_length=1)
    reward: Annotated[StdoutReward | FileReward | ExitCodeReward, Field(discriminator="kind")] = StdoutReward()
    collect: tuple[VerifierCommand, ...] = ()
    artifacts: tuple[VerifierArtifact, ...] = ()
