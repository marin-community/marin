# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Projection registry: write-through caches owning specific tables.

Each projection instance registers itself in :data:`PROJECTIONS` at
construction. The ``@writes_to`` invariant — that no Projection-owned
table may be mutated from outside its owning Projection — is enforced at
controller startup by
:func:`iris.cluster.controller.writes_validation.assert_owned_tables_not_externally_written`,
which lives downstream of both this package and ``writes`` to keep this
module's import graph cycle-free.

Re-exporting the entity submodules at import time ensures that every
projection instance is materialized before the check runs. Without this,
the check would silently pass on a half-loaded registry.
"""

from __future__ import annotations

from typing import Any

# Module-level registry of every projection instance. Typed as ``Any`` because
# the projections themselves are defined in submodules that import from here;
# referring to the concrete classes would create an import cycle.
PROJECTIONS: list[Any] = []


# Re-export entity modules so importing ``iris.cluster.controller.projections``
# materializes every Projection class (and its registry entry). The startup
# check relies on PROJECTIONS being fully populated.
from iris.cluster.controller.projections import endpoints, worker_attrs
