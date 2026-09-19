# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provider adapter interface for stateful domain actions in Harbor."""

from typing import Any, Protocol, runtime_checkable

from taskcompendium.models import ActionInterface, GradingResult


@runtime_checkable
class ProviderActionEnvironment(Protocol):
    """A source adapter exposes native tools and authoritative state through one interface."""

    interface: ActionInterface

    async def native_tool_definitions(self) -> list[dict[str, Any]]:
        """Return the provider's advertised native function definitions."""
        ...

    async def dispatch_action(self, name: str, arguments: str, call_id: str) -> str:
        """Apply a provider-native action and return its native observation payload."""
        ...

    async def authoritative_state(self) -> dict[str, Any]:
        """Return state for private source-specific verification."""
        ...

    async def grade_provider_state(self, adapter: str, parameters: dict[str, Any]) -> GradingResult:
        """Run the selected source-specific verifier against authoritative provider state."""
        ...
