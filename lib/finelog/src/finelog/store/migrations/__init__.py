# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned migrations for the finelog registry DuckDB sidecar.

Each ``NNNN_name.py`` file defines a ``migrate(conn)`` callable. The
:func:`apply_migrations` runner in :mod:`._runner` walks the directory in
filename order and applies any not yet recorded in the ``schema_migrations``
table. Each migration runs inside a transaction that also inserts the
``schema_migrations`` row, so partial application can't leave the catalog
in a half-migrated state.

Migration files must be idempotent against any pre-migrations on-disk
state: pre-existing deployments are bootstrapped through 0001 even though
their tables already exist.
"""

from finelog.store.migrations._runner import apply_migrations, transactional

__all__ = ["apply_migrations", "transactional"]
