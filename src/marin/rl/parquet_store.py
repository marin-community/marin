"""Lightweight Parquet helpers for replay buffer rollouts."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .datatypes import RolloutGroup, RolloutRecord


def _groups_to_table(groups: list[RolloutGroup]) -> pa.Table:
    rows = []
    for g in groups:
        g_meta = json.dumps(g.metadata, separators=(",", ":"))
        for r in g.rollouts:
            rows.append(
                {
                    "group_id": g.id,
                    "environment": g.environment,
                    "example_id": g.example_id,
                    "policy_version": g.policy_version,
                    "segment_idx": g.segment_idx,
                    "sealed_ts": g.sealed_ts,
                    "group_metadata_json": g_meta,
                    "replica_id": r.replica_id,
                    "rollout_uid": r.rollout_uid,
                    "token_count": r.token_count,
                    "reward": r.reward,
                    "logprobs": r.logprobs,
                    "output_tokens": r.output_tokens,
                    "is_last_segment": r.is_last_segment,
                    "rr_metadata_json": json.dumps(r.metadata or {}, separators=(",", ":")),
                    "created_ts": r.created_ts,
                }
            )
    return pa.Table.from_pylist(rows)


def write_rollout_groups(groups: list[RolloutGroup], root_path: str, *, compression: str = "zstd") -> None:
    fs, dataset_root = pafs.FileSystem.from_uri(root_path)
    try:
        fs.create_dir(dataset_root, recursive=True)
    except FileExistsError:
        pass
    table = _groups_to_table(groups)
    filename = f"{dataset_root.rstrip('/')}/part-{uuid.uuid4().hex}.parquet"
    pq.write_table(table, filename, compression=compression, filesystem=fs)


def iter_rollout_groups(root_path: str) -> Iterator[RolloutGroup]:
    fs, dataset_root = pafs.FileSystem.from_uri(root_path)
    dataset = ds.dataset(dataset_root, format="parquet", filesystem=fs)

    pending: dict[str, RolloutGroup] = {}
    for batch in dataset.to_batches():
        for record in batch.to_pylist():
            gid = record["group_id"]
            g = pending.get(gid)
            if g is None:
                g = RolloutGroup(
                    id=gid,
                    environment=record["environment"],
                    example_id=record["example_id"],
                    policy_version=record["policy_version"],
                    segment_idx=int(record["segment_idx"]),
                    rollouts=[],
                    sealed_ts=record["sealed_ts"],
                    metadata=json.loads(record["group_metadata_json"]),
                )
            r = RolloutRecord(
                environment=record["environment"],
                example_id=record["example_id"],
                policy_version=record["policy_version"],
                segment_idx=int(record["segment_idx"]),
                is_last_segment=record["is_last_segment"],
                replica_id=record["replica_id"],
                rollout_uid=record["rollout_uid"],
                token_count=int(record["token_count"]),
                reward=record["reward"],
                logprobs=record["logprobs"],
                output_tokens=record["output_tokens"],
                metadata=json.loads(record["rr_metadata_json"]),
                created_ts=record["created_ts"],
            )
            g.rollouts.append(r)
            pending[gid] = g
    yield from pending.values()
