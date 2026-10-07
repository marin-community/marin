# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source grader command and result contract for isolated native execution."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from taskcompendium.models import validate_workspace_path

NATIVE_COMMAND_KIND = "native_command"


class NativeCommandSpec(BaseModel):
    """Execute a source command and read its private reward file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    argv: tuple[str, ...] = Field(min_length=1)
    cwd: str
    env: dict[str, str] = Field(default_factory=dict)
    result_format: Literal["reward_file", "reward_json", "score_json"]
    result_path: str
    timeout: float = Field(gt=0)

    @field_validator("cwd", "result_path")
    @classmethod
    def validate_absolute_path(cls, value: str) -> str:
        path = validate_workspace_path(value)
        if ".." in path.parts or path.as_posix() != value:
            raise ValueError("Native grader paths must be normalized")
        return value
