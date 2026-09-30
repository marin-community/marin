# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tool services composed into a trial environment."""

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class ProviderIdentity(BaseModel):
    """Immutable action interface, initial state, and implementation identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action_interface: str = Field(min_length=1)
    seed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider_revision: str = Field(min_length=1)


def tool_schema_sha256(definitions: Sequence[dict[str, Any]]) -> str:
    """Hash the ordered OpenAI-compatible schemas in their canonical JSON form."""
    return hashlib.sha256(
        json.dumps(list(definitions), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


@runtime_checkable
class ToolProvider(Protocol):
    """A service that advertises tools and executes calls against trial-local state."""

    async def native_tool_definitions(self) -> list[dict[str, Any]]: ...

    async def dispatch_action(self, name: str, arguments: str, call_id: str) -> str:
        """Execute one call and return its model-visible observation string."""
        ...


@runtime_checkable
class ManagedToolProvider(Protocol):
    """Optional startup and cleanup for a tool service."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class ToolProviderFactory(Protocol):
    """A provider implementation with pinned identity and a per-trial constructor."""

    __module__: str
    ACTION_INTERFACE: str
    SEED_SHA256: str
    PROVIDER_REVISION: str
    TOOL_DEFINITIONS: Sequence[dict[str, Any]]

    def __call__(self, *, seed_sha256: str, action_interface: str) -> ToolProvider: ...
