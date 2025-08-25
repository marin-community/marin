"""Rollout sinks for Marin RL.

This module provides utilities to create sinks that consume ``RolloutGroup``
batches and persist them. The primary sink writes groups to a Parquet dataset.
"""

from __future__ import annotations

import ray

from .datatypes import RolloutGroup, RolloutSink
from .parquet_store import write_rollout_groups


def make_parquet_sink(root_path: str, *, compression: str = "zstd") -> RolloutSink:
    """Create a ``RolloutSink`` that appends rollout groups to a Parquet dataset.

    Args:
        root_path: Filesystem or URI path to the dataset root (e.g., "/tmp/rl_ds" or "gs://bucket/rl_ds").
        compression: Parquet compression codec.

    Returns:
        A callable suitable to pass into environment ``build`` methods.
    """

    def sink(groups: list[RolloutGroup]) -> None:
        if not groups:
            return
        write_rollout_groups(groups, root_path, compression=compression)

    # ``sink`` will be serialized and executed inside Ray actors.
    return sink


def tee_sinks(*sinks: RolloutSink) -> RolloutSink:
    """Create a sink that dispatches to multiple sinks (fan-out)."""

    def sink(groups: list[RolloutGroup]) -> None:
        for s in sinks:
            s(groups)

    return sink


@ray.remote
class ParquetWriter:
    """Ray actor that serializes Parquet writes to avoid concurrent writer issues."""

    def __init__(self, root_path: str, compression: str = "zstd"):
        self._root_path = root_path
        self._compression = compression

    def write(self, groups: list[RolloutGroup]) -> None:
        if groups:
            write_rollout_groups(groups, self._root_path, compression=self._compression)

    def ping(self) -> bool:
        return True


def make_parquet_actor_sink(root_path: str, *, compression: str = "zstd") -> tuple[RolloutSink, ray.actor.ActorHandle]:
    """Create a sink backed by a single ParquetWriter actor.

    Returns the sink function and the actor handle so callers can keep the
    actor alive and optionally "ping" it to ensure it's started.
    """

    writer = ParquetWriter.remote(root_path, compression)

    def sink(groups: list[RolloutGroup]) -> None:
        if groups:
            writer.write.remote(groups)

    return sink, writer
