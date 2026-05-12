# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write-side helpers: module-level functions decorated with ``@writes_to``.

Stage 11 of the SA Core migration moves every write that today lives on
the per-entity ``*Store`` classes into module-level functions under
``writes/<entity>.py``. Each function takes a :class:`iris.cluster.
controller.db_v2.Tx` and uses SA Core ``text(...)`` with named binds.

The :func:`writes_to` decorator is pure metadata: it records the table
set on the function as ``fn.writes_to`` / ``fn.cascades_into`` and
appends the function to :data:`REGISTERED_WRITE_FUNCTIONS`. Stage 12's
startup check walks that registry and the ``PROJECTIONS`` list to
verify no Projection-owned table is written outside its Projection.
"""

from collections.abc import Callable

from sqlalchemy import Table

REGISTERED_WRITE_FUNCTIONS: list[Callable] = []


def writes_to(
    *tables: Table,
    cascades_into: tuple[Table, ...] = (),
) -> Callable:
    """Mark a write function with the tables it mutates.

    Pure metadata. The startup-time owned-table check in
    ``projections/__init__.py`` (Stage 12) reads ``fn.writes_to`` and
    ``fn.cascades_into`` to verify no Projection-owned table is written
    outside its Projection.

    ``cascades_into`` lists tables mutated via FK ``ON DELETE CASCADE``
    by writes to ``tables``; the check treats them identically to direct
    writes.
    """

    def deco(fn: Callable) -> Callable:
        fn.writes_to = tables  # type: ignore[attr-defined]
        fn.cascades_into = cascades_into  # type: ignore[attr-defined]
        REGISTERED_WRITE_FUNCTIONS.append(fn)
        return fn

    return deco
