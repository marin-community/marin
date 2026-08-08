# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact lookup for configured backend resource bindings."""

from collections.abc import Mapping

from iris.cluster.backends.protocol import BackendBinding
from iris.cluster.resources.errors import BackendIdentityUnknown


class BackendResolver:
    """Resolve backend identities without a default or capability fallback."""

    def __init__(self, bindings: Mapping[str, BackendBinding]) -> None:
        self._bindings = dict(bindings)
        for backend_id, binding in self._bindings.items():
            if not backend_id.strip():
                raise ValueError("backend identity must be non-empty")
            if binding.tasks.backend_id != backend_id:
                raise ValueError(
                    f"backend binding key {backend_id!r} does not match task backend {binding.tasks.backend_id!r}"
                )

    def require(self, backend_id: str) -> BackendBinding:
        """Return the exact configured binding or reject an unknown identity."""
        try:
            return self._bindings[backend_id]
        except KeyError:
            raise BackendIdentityUnknown(f"unknown backend identity: {backend_id!r}") from None
