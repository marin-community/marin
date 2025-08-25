"""Lightweight Parquet helpers for replay buffer rollouts."""

from __future__ import annotations

import dataclasses
import json
import uuid
from collections.abc import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .datatypes import InferenceMetadata, RolloutGroup, RolloutRecord, Turn


def _maybe_meta_to_dict(meta: InferenceMetadata | dict | None) -> dict | None:
    if meta is None:
        return None
    if isinstance(meta, InferenceMetadata):
        return dataclasses.asdict(meta)
    if isinstance(meta, dict):
        return meta
    # Fallback: try to JSON-ify unknown object via __dict__
    try:
        return dict(meta)  # type: ignore[arg-type]
    except Exception:
        return None


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
                    "sealed_ts": g.sealed_ts,
                    "group_metadata_json": g_meta,
                    "replica_id": r.replica_id,
                    "rollout_uid": r.rollout_uid,
                    "rollout_reward": r.reward,
                    "turns_json": json.dumps(
                        [
                            {
                                "message": t.message,
                                "tokens": t.tokens,
                                "logprobs": t.logprobs.tolist() if t.logprobs is not None else None,
                                "role": t.role,
                                "reward": t.reward,
                                "inference_metadata": _maybe_meta_to_dict(t.inference_metadata),
                                "timestamp": t.timestamp,
                            }
                            for t in r.turns
                        ],
                        separators=(",", ":"),
                    ),
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


def _meta_from_obj(obj: dict | None) -> dict | None:
    """Return a JSON-serializable dict for inference metadata.

    - If the stored value was a dict, return it as-is.
    - If it was None, return None.
    - If it was an InferenceMetadata, convert to dict.
    This keeps backward-compatibility for round-tripping tests that expect dicts.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, InferenceMetadata):
        return dataclasses.asdict(obj)
    return None


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
                    rollouts=[],
                    sealed_ts=record["sealed_ts"],
                    metadata=json.loads(record["group_metadata_json"]),
                )
            r = RolloutRecord(
                environment=record["environment"],
                example_id=record["example_id"],
                policy_version=record["policy_version"],
                replica_id=record["replica_id"],
                rollout_uid=record["rollout_uid"],
                reward=record.get("rollout_reward"),
                turns=[
                    Turn(
                        message=t.get("message", ""),
                        tokens=t.get("tokens"),
                        logprobs=(np.array(t.get("logprobs"), dtype=float) if t.get("logprobs") is not None else None),
                        role=t.get("role", "assistant"),
                        reward=t.get("reward"),
                        inference_metadata=_meta_from_obj(t.get("inference_metadata")),
                        timestamp=t.get("timestamp"),
                    )
                    for t in json.loads(record["turns_json"])
                ],
                metadata=json.loads(record["rr_metadata_json"]),
                created_ts=record["created_ts"],
            )
            g.rollouts.append(r)
            pending[gid] = g
    yield from pending.values()
