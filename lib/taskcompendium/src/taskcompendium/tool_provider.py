# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tool services composed into a trial environment."""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ToolProvider(Protocol):
    """A service that advertises tools and executes calls against trial-local state."""

    async def native_tool_definitions(self) -> list[dict[str, Any]]: ...

    async def dispatch_action(self, name: str, arguments: str, call_id: str) -> str: ...


@runtime_checkable
class ManagedToolProvider(Protocol):
    """Optional startup and cleanup for a tool service."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...


class ToolProviderFactory(Protocol):
    """A provider implementation with pinned identity and a per-trial constructor."""

    ACTION_INTERFACE: str
    SEED_SHA256: str
    PROVIDER_REVISION: str
    TOOL_DEFINITIONS: Sequence[dict[str, Any]]

    def __call__(self, *, seed_sha256: str, action_interface: str) -> ToolProvider: ...
