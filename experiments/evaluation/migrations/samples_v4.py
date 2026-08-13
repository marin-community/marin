# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bring a run's ``samples`` table up to schema v4, which added ``filter`` to the primary key.

Rows written under the narrower key cannot collapse against rows written under the wider one. The
old table is copied to region-local 30-day storage, then removed from the active manifest. Its
immutable objects remain available to read views that pinned the earlier commit.

An lm-eval export can only reproduce lm-eval rows, so a table holding agentic samples is refused
here; ``migrate_archive`` replaces those from the legacy parquets that do reproduce them.
"""

from __future__ import annotations

import logging

from finestore.admin import drop_table as drop_manifest_table
from finestore.reader import ReadView
from marin.evaluation.archive import ARCHIVE_SAMPLES_TABLE, SCHEMA_VERSION, SampleKind
from rigging.filesystem import StoragePath, factory, prefix_join

from experiments.evaluation.migrations.archive_backup import superseded_samples_prefix

logger = logging.getLogger(__name__)


def copy_table(root: str, table: str, destination: str) -> int:
    """Copy a table's pinned metadata and shards to ``destination``; return objects copied.

    Each shard copy's size is checked against its source before this returns. An object already at
    the destination is left alone: the earliest snapshot is the pristine one, and a resumed
    migration must not write over it.
    """
    reader = ReadView(root)
    shards = reader.list_shards(table)
    if not shards:
        return 0
    destination_fs, _ = factory.url_to_fs(destination)
    destination_path = StoragePath(destination)
    destination_path.mkdirs()
    copied = 0
    metadata_target = destination_path / "_schema.json"
    metadata_bytes = reader.table_metadata(table).model_dump_json(indent=2).encode()
    if metadata_target.exists():
        if metadata_target.read_bytes() != metadata_bytes:
            raise ValueError(f"existing table metadata backup does not match {table!r} at {root}")
    else:
        metadata_target.write_bytes(metadata_bytes)
        copied += 1
    for shard in shards:
        source_fs, source = factory.url_to_fs(shard.path)
        _, target = factory.url_to_fs(prefix_join(destination, StoragePath(shard.path).name))
        if destination_fs.exists(target):
            written = destination_fs.info(target)["size"]
            expected = source_fs.info(source)["size"]
            if written != expected:
                raise OSError(f"existing copy of {source} is {written} bytes, expected {expected}")
            continue
        with source_fs.open(source, "rb") as source_handle, destination_fs.open(target, "wb") as writer:
            writer.write(source_handle.read())
        written = destination_fs.info(target)["size"]
        expected = source_fs.info(source)["size"]
        if written != expected:
            raise OSError(f"copy of {source} is {written} bytes, expected {expected}")
        copied += 1
    logger.info("copied %d object(s) of table %s to %s", copied, table, destination)
    return copied


def drop_table(root: str, table: str) -> int:
    """Remove a table from the next manifest and return its active shard count."""
    shards = ReadView(root).list_shards(table)
    if not shards:
        return 0
    drop_manifest_table(root, table)
    logger.info("removed %d shard(s) of table %s from the active manifest under %s", len(shards), table, root)
    return len(shards)


def replace_table(root: str, table: str, destination: str) -> None:
    """Snapshot ``table`` to ``destination``, then drop it.

    The snapshot is complete and size-verified before the manifest removes the table, so a failure
    leaves either the original logical table or a full copy outside the run.
    """
    copy_table(root, table, destination)
    drop_table(root, table)


def preserve_and_replace_samples(results_path: str, stored_version: int) -> None:
    """Snapshot one run's samples table to its 30-day prefix, keyed by the version being replaced."""
    destination = superseded_samples_prefix(results_path, stored_version)
    logger.info(
        "replacing %s samples written under schema v%s at v%s, preserved at %s",
        results_path,
        stored_version,
        SCHEMA_VERSION,
        destination,
    )
    replace_table(results_path, ARCHIVE_SAMPLES_TABLE, destination)


def replace_stale_samples(results_path: str) -> int | None:
    """Replace a stale samples table so a caller can rewrite it; return the version replaced.

    ``None`` means the table was already current and nothing was touched. Refuses a table holding
    agentic samples, which come from Harbor and no lm-eval source can regenerate.
    """
    reader = ReadView(results_path)
    stored_version = reader.schema_version(ARCHIVE_SAMPLES_TABLE)
    if stored_version is None or stored_version == SCHEMA_VERSION:
        return None
    stored = reader.scan(ARCHIVE_SAMPLES_TABLE, columns=["kind"])
    if stored is not None:
        agentic = sum(1 for kind in stored["kind"].to_pylist() if kind == SampleKind.AGENTIC)
        if agentic:
            raise ValueError(
                f"{results_path} holds {agentic} agentic sample(s) that an lm-eval export cannot "
                "regenerate; migrate the run with migrate_archive instead"
            )
    preserve_and_replace_samples(results_path, stored_version)
    return stored_version
