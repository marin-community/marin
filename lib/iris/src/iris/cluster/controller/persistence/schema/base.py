# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared metadata and bookkeeping tables for resource schema v2."""

from sqlalchemy import Column, Integer, MetaData, Table, Text

metadata = MetaData()


schema_migrations_table = Table(
    "schema_migrations",
    metadata,
    Column("name", Text, primary_key=True),
    Column("source_fingerprint", Text, nullable=False),
    Column("applied_at_ms", Integer, nullable=False),
)


meta_table = Table(
    "meta",
    metadata,
    Column("key", Text, primary_key=True),
    Column("value", Text, nullable=False),
)
