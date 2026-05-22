# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned migrations for the finelog registry DuckDB sidecar.

Each ``NNNN_name.py`` file defines a ``migrate(conn, *, data_dir)``
callable. The runner applies them in filename order and records each
in ``schema_migrations`` only on success. Migrations run without an
enclosing transaction, so they must be idempotent; multi-statement
atomicity is opt-in via :func:`transactional`.
"""

from finelog.store.migrations._runner import apply_migrations, transactional

__all__ = ["apply_migrations", "transactional"]
