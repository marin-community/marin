# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit: composable pipeline stages with a standard Parquet format.

The standard format pins three mandatory columns on every normalized record:
``id`` (deterministic content hash), ``text`` (UTF-8 primary content), and
``partition_id`` (int, the output shard the row was written to at normalize
time). The shard count itself lives on the artifact, not the row.

Downstream stages preserve ``partition_id`` and use it as the ``group_by`` key
when a global shuffle (e.g. cross-document dedup) needs to land output back
co-partitioned with the source.
"""


def partition_filename(partition_id: int, num_partitions: int) -> str:
    """Return the standard datakit partition filename for the given index.

    Datakit shards follow ``part-NNNNN-of-MMMMM.parquet`` naming. Routing
    output through this helper keeps shuffler-written attribute files
    discoverable by consolidate's filename-based join.
    """
    return f"part-{partition_id:05d}-of-{num_partitions:05d}.parquet"
