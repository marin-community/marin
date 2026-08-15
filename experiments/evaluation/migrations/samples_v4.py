# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bring a run's ``samples`` table up to schema v4, which added ``filter`` to the primary key.

Rows written under the narrower key cannot collapse against rows written under the wider one — the
added column reads as null on their shards — so they have to be removed rather than merged into. That
makes this the one place in the eval pipeline that deletes, and every deletion is preceded by a
verified copy to region-local 30-day storage.

An lm-eval export can only reproduce lm-eval rows, so a table holding agentic samples is refused
here; ``migrate_archive`` replaces those from the legacy parquets that do reproduce them.
"""

from __future__ import annotations

import logging

import rigging.filesystem.factory as factory
from finestore.eval import ARCHIVE_SAMPLES_TABLE, SCHEMA_VERSION, SampleKind
from finestore.reader import CompositeReader
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.migrations.archive_backup import superseded_samples_prefix

logger = logging.getLogger(__name__)


def copy_table(root: str, table: str, destination: str) -> int:
    """Copy every object of ``table`` under ``root`` to ``destination``; return the number copied.

    Each copy's size is checked against its source before this returns, so a caller may treat success
    as licence to delete the originals. An object already at the destination is left alone: the
    earliest snapshot is the pristine one, and a resumed migration must not write over it.
    """
    source_fs, source_key = factory.url_to_fs(prefix_join(root, table))
    destination_fs, _ = factory.url_to_fs(destination)
    if not source_fs.exists(source_key):
        return 0
    copied = 0
    for source in source_fs.find(source_key):
        relative = source[len(source_key) :].strip("/")
        _, target = factory.url_to_fs(prefix_join(destination, relative))
        if destination_fs.exists(target):
            continue
        destination_fs.makedirs(target.rsplit("/", 1)[0], exist_ok=True)
        with source_fs.open(source, "rb") as reader, destination_fs.open(target, "wb") as writer:
            writer.write(reader.read())
        written = destination_fs.info(target)["size"]
        expected = source_fs.info(source)["size"]
        if written != expected:
            raise OSError(f"copy of {source} is {written} bytes, expected {expected}")
        copied += 1
    logger.info("copied %d object(s) of table %s to %s", copied, table, destination)
    return copied


def drop_table(root: str, table: str) -> int:
    """Delete every shard of ``table`` under ``root``; return the number removed.

    Leaves ``_schema.json`` for the next writer to overwrite.
    """
    shards = CompositeReader(root).list_shards(table)
    if not shards:
        return 0
    fs, _ = factory.url_to_fs(root)
    for shard in shards:
        fs.rm(shard.path)
    logger.info("dropped %d shard(s) of table %s under %s", len(shards), table, root)
    return len(shards)


def replace_table(root: str, table: str, destination: str) -> None:
    """Snapshot ``table`` to ``destination``, then drop it.

    The snapshot is complete and size-verified before anything is deleted, so a failure at any point
    leaves either the original table or a full copy of it outside the run.
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
    reader = CompositeReader(results_path)
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
