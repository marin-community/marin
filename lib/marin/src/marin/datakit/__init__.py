# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Datakit: composable pipeline stages with a standard Parquet format.

The standard format pins two mandatory columns on every normalized record:
``id`` (deterministic content hash) and ``text`` (UTF-8 primary content).
The partition index lives in the filename (``part-NNNNN-of-MMMMM.parquet``)
and is derived at reader time from sorted file order, not stamped per row.
"""


def partition_filename(partition_id: int, num_partitions: int) -> str:
    """Return the standard datakit partition filename for the given index.

    Datakit shards follow ``part-NNNNN-of-MMMMM.parquet`` naming. Routing
    output through this helper keeps shuffler-written attribute files
    discoverable by consolidate's filename-based join.
    """
    return f"part-{partition_id:05d}-of-{num_partitions:05d}.parquet"
