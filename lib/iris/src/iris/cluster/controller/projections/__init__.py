# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Projection registry: write-through caches owning specific tables.

Each projection instance registers itself in :data:`PROJECTIONS` at
construction. Stage 12 will iterate this registry to enforce the
``@writes_to`` invariant; until then it serves as a discoverable list of
in-memory caches that need rehydrating after ``ControllerDB.replace_from``.
"""

from __future__ import annotations

from typing import Any

# Module-level registry of every projection instance. Typed as ``Any`` because
# the projections themselves are defined in submodules that import from here;
# referring to the concrete classes would create an import cycle.
PROJECTIONS: list[Any] = []
