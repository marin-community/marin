# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolver implementations for Fray v2 actor discovery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from fray.v2.cluster import ActorPool


@runtime_checkable
class Resolver(Protocol):
    """Protocol for actor discovery.

    Maps actor names to ActorPool instances.
    """

    def lookup(self, name: str) -> ActorPool:
        """Look up actors by name and return a pool.

        Args:
            name: Actor name to look up

        Returns:
            ActorPool for the named actor(s)
        """
        ...


class FixedResolver:
    """Resolver with fixed actor addresses.

    Useful for testing or when connecting to known endpoints.

    Example:
        resolver = FixedResolver({
            "inference": "localhost:8080",
            "workers": ["localhost:8081", "localhost:8082"],
        })
        pool = resolver.lookup("inference")
    """

    def __init__(self, addresses: dict[str, str | list[str]]):
        """Create resolver with fixed addresses.

        Args:
            addresses: Mapping of actor names to addresses.
                       Values can be a single address or list of addresses.
        """
        self._addresses: dict[str, list[str]] = {}
        for name, addr in addresses.items():
            if isinstance(addr, str):
                self._addresses[name] = [addr]
            else:
                self._addresses[name] = list(addr)

    def lookup(self, name: str) -> Any:
        """Look up actors by name.

        Note: Returns a FixedActorPool which provides a basic ActorPool-like
        interface for connecting to fixed addresses.
        """
        from fray.v2.actor.pool import FixedActorPool

        addresses = self._addresses.get(name, [])
        return FixedActorPool(name, addresses)
