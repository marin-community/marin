# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Projection registry: write-through caches owning specific tables.

Each projection instance registers itself in :data:`PROJECTIONS` at
construction. :func:`assert_owned_tables_not_externally_written` is wired
into ``ControllerDB.__init__`` to enforce the ``@writes_to`` invariant:
no Projection-owned table may be mutated (directly or via FK cascade)
from outside its owning Projection.

Re-exporting the entity submodules at import time ensures that every
projection instance — and every Projection class used by the startup
check's ``__qualname__`` exemption — is materialized before the check
runs. Without this, the check would silently pass on a half-loaded
registry.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Table

# Module-level registry of every projection instance. Typed as ``Any`` because
# the projections themselves are defined in submodules that import from here;
# referring to the concrete classes would create an import cycle.
PROJECTIONS: list[Any] = []


class ConfigurationError(RuntimeError):
    """Raised by :func:`assert_owned_tables_not_externally_written` on a violation.

    Signals a programming error: a write function declared
    ``@writes_to(<projection-owned table>)`` from outside the owning
    Projection class, or its ``cascades_into`` fans out into a
    Projection-owned table without an explicit invalidation hook.
    """


def assert_owned_tables_not_externally_written() -> None:
    """Startup-time invariant: no ``@writes_to`` function writes (or cascades)
    into a Projection-owned table from outside the owning Projection.

    For projection-owned tables, all SQL mutations must flow through the
    Projection so the in-memory dict can be updated atomically. A write
    function that bypasses the Projection (or whose FK cascade silently
    mutates the table) leaves the dict stale.

    The exemption is by ``fn.__qualname__``: a method whose qualified
    name starts with ``<OwningProjection>.`` is allowed to mutate the
    table directly. Free functions that need to cascade into a
    Projection-owned table must call the Projection's invalidation method
    inline; they should then drop the Projection-owned table from their
    ``cascades_into`` declaration so the linkage is documented at the
    call site rather than buried in the decorator metadata.

    Raises:
        ConfigurationError: when a violation is detected.
    """
    # Imported here to keep the projections package import-time-clean —
    # writes/__init__.py imports from projections during its own setup
    # for the cascade hooks, so importing it at module top-level would
    # cycle.
    from iris.cluster.controller.writes import REGISTERED_WRITE_FUNCTIONS  # noqa: PLC0415

    owned: dict[Table, type] = {}
    for projection in PROJECTIONS:
        for table in projection.sources:
            owned[table] = type(projection)

    violations: list[str] = []
    for fn in REGISTERED_WRITE_FUNCTIONS:
        for table in (*fn.writes_to, *fn.cascades_into):
            if table not in owned:
                continue
            # Methods of the owning Projection class are allowed.
            if fn.__qualname__.startswith(owned[table].__name__ + "."):
                continue
            violations.append(
                f"  - {fn.__qualname__} writes (or cascades) into " f"{table.name!r} owned by {owned[table].__name__}"
            )

    if violations:
        raise ConfigurationError(
            "Projection-owned tables externally written:\n"
            + "\n".join(violations)
            + "\n\nFix: either move this write onto the Projection, or have "
            "the write function call the Projection's invalidation method "
            "(e.g. projection.invalidate_for_worker(tx, ...)) and document "
            "the linkage at the call site."
        )


# Re-export entity modules so importing ``iris.cluster.controller.projections``
# materializes every Projection class (and its registry entry). The startup
# check relies on PROJECTIONS being fully populated.
from iris.cluster.controller.projections import endpoints, worker_attrs
