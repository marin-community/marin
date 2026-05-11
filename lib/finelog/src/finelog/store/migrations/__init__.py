# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Versioned migrations for the finelog registry DuckDB sidecar.

Each ``NNNN_name.py`` file defines a ``migrate(conn, *, data_dir)``
callable. The :func:`apply_migrations` runner in :mod:`._runner` walks
the directory in filename order and applies any not yet recorded in the
``schema_migrations`` table. After a successful ``migrate`` the runner
inserts a row in ``schema_migrations``; a raised exception leaves the
row out so the next open re-runs the migration.

The runner does not wrap migrations in an outer transaction. DuckDB
rejects several useful statement sequences (notably multiple
schema-altering DDLs + DML on the same table) inside a single
transaction, so each migration is responsible for its own atomicity and
must be idempotent under partial application. Migrations that *can*
benefit from a multi-statement transaction can call
:func:`transactional` themselves.

Migration files must be idempotent against any pre-migrations on-disk
state: pre-existing deployments are bootstrapped through 0001 even though
their tables already exist.
"""

from finelog.store.migrations._runner import apply_migrations, transactional

__all__ = ["apply_migrations", "transactional"]
