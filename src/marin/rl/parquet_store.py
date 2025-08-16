"""Lightweight Parquet helpers for replay buffer rollouts."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .datatypes import RolloutGroup, RolloutRecord, Turn


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
                    "reward": r.reward,
                    "turns_json": json.dumps(
                        [
                            {
                                "message": t.message,
                                "logprobs": t.logprobs.tolist() if t.logprobs is not None else None,
                                "role": t.role,
                                "reward": t.reward,
                                "inference_metadata": t.inference_metadata,
                            }
                            for t in r.turns
                        ],
                        separators=(",", ":"),
                    ),
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
                reward=record["reward"],
                turns=[
                    Turn(
                        message=t["message"],
                        logprobs=np.array(t["logprobs"], dtype=float) if t["logprobs"] is not None else None,
                        role=t["role"],
                        reward=t["reward"],
                        inference_metadata=t["inference_metadata"],
                    )
                    for t in json.loads(record["turns_json"])
                ],
                metadata=json.loads(record["rr_metadata_json"]),
                created_ts=record["created_ts"],
            )
            g.rollouts.append(r)
            pending[gid] = g
    yield from pending.values()
