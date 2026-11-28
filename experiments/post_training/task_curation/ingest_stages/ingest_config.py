# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reviewed source-ingest policy; counts and artifact locations are measured later."""

import re
from enum import StrEnum
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, field_validator
from taskcompendium.identity import canonical_sha256
from taskcompendium.pipeline.models import IntendedUse

from infra.marina.apps.rl_data_catalog.server.source_identity import SourceId

GIT_REVISION = r"^[0-9a-f]{40}$"


class IngestMode(StrEnum):
    INGEST = "ingest"
    REPAIR = "repair"


class UpstreamPin(BaseModel):
    """An immutable HF repository revision and its source file or folder."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repository: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    revision: str = Field(pattern=GIT_REVISION)
    path: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value or any(c in value for c in "*?[]"):
            raise ValueError("upstream.path must name one relative file or folder without globs")
        return str(path)


class IngestConfig(BaseModel):
    """Policy choices for one source; supplying measured state is a schema error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: SourceId = Field(min_length=1)
    upstream: UpstreamPin
    intended_use: tuple[IntendedUse, ...] = Field(min_length=1)
    converter_module: str
    cluster: str = Field(min_length=1)
    mode: IngestMode

    @field_validator("intended_use")
    @classmethod
    def canonical_use(cls, value: tuple[IntendedUse, ...]) -> tuple[IntendedUse, ...]:
        if len(value) != len(set(value)):
            raise ValueError("intended_use contains duplicate purposes")
        return tuple(sorted(value))

    @field_validator("converter_module")
    @classmethod
    def maintained_converter(cls, value: str) -> str:
        if re.fullmatch(r"(?:experiments|taskcompendium)(?:\.[A-Za-z_][A-Za-z0-9_]*)+", value) is None:
            raise ValueError("converter_module must name a maintained repository module")
        return value

    def digest(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


def load_ingest_config(path: Path) -> IngestConfig:
    return IngestConfig.model_validate_json(path.read_text(encoding="utf-8"))
