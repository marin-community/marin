# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

JSONDict = dict[str, Any]
SecretValue = str | tuple[str, list[str]]
SecretsMapping = Mapping[str, SecretValue]


class _StrictModel(BaseModel):
    """Immutable pydantic base with strict field handling."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class OpenRewardPromptBlockType(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class OpenRewardPromptBlock(_StrictModel):
    """Serializable prompt block snapshot from an OpenReward session."""

    type: OpenRewardPromptBlockType
    text: str | None = None
    data: str | None = None
    mime_type: str | None = None
    detail: JSONDict | None = None

    @model_validator(mode="after")
    def _validate_block_shape(self) -> "OpenRewardPromptBlock":
        if self.type == OpenRewardPromptBlockType.TEXT:
            if self.text is None:
                raise ValueError("text blocks must include text")
            if self.data is not None or self.mime_type is not None:
                raise ValueError("text blocks cannot include image payload fields")
            return self

        if self.data is None or self.mime_type is None:
            raise ValueError("image blocks must include data and mime_type")
        if self.text is not None:
            raise ValueError("image blocks cannot include text")
        return self


class OpenRewardToolSpec(_StrictModel):
    """Serializable snapshot of a tool exposed by an OpenReward session."""

    name: str
    description: str
    input_schema: JSONDict | None = None


class OpenRewardTaskManifestEntry(_StrictModel):
    """A single task snapshot with the prompt and tools seen by the agent."""

    task_index: int
    task_spec: JSONDict
    prompt_blocks: list[OpenRewardPromptBlock]
    tools: list[OpenRewardToolSpec]


class OpenRewardTaskManifest(_StrictModel):
    """Snapshot of a deterministic task subset for one OpenReward split."""

    deployment_name: str
    environment_name: str
    split: str
    tasks: list[OpenRewardTaskManifestEntry]

    @property
    def task_count(self) -> int:
        return len(self.tasks)
