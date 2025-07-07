"""Parquet storage utilities for :pymod:`marin.rl` rollouts.

The helpers defined here keep the writer/reader logic *very* thin—primarily
bridging between the in-memory dataclass objects and Apache Arrow structures so
we can leverage Arrow/Parquet's excellent performance and Ray integration.

Design goals
------------
1. Zero third-party dependencies beyond ``pyarrow`` (already required by Ray).
2. Immutable dataclasses stay immutable; conversion happens *outside* them.
3. Files are written as independent parts (UUID filenames) so multiple actors
   can append concurrently without coordination.  The target directory therefore
   represents a Parquet *dataset*.
"""

import json
import uuid
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .types import Rollout, RolloutGroup, Turn

# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _turn_to_pyobj(turn: Turn) -> dict:
    """Convert a :class:`Turn` into a Python mapping understood by Arrow."""

    return {
        "message": turn.message,
        "role": turn.role,
        "logprobs": list(turn.logprobs) if turn.logprobs is not None else None,
        "reward": turn.reward,
        "inference_metadata_json": json.dumps(turn.inference_metadata, separators=(",", ":")),
    }


def _rollout_to_pyobj(rollout: Rollout) -> dict:
    """Convert a :class:`Rollout` into a flat Python mapping suitable for Arrow."""

    return {
        "turns": [_turn_to_pyobj(t) for t in rollout.turns],
        "rollout_metadata_json": json.dumps(rollout.metadata, separators=(",", ":")),
    }


def _groups_to_table(groups: list[RolloutGroup]) -> pa.Table:
    rows = []
    for g in groups:
        for r in g.rollouts:
            row = _rollout_to_pyobj(r)
            row["id"] = g.id
            row["source"] = g.source
            row["created"] = g.created
            row["group_metadata_json"] = json.dumps(g.metadata, separators=(",", ":"))
            rows.append(row)

    return pa.Table.from_pylist(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_rollout_groups(
    groups: list[RolloutGroup],
    root_path: str,
    *,
    compression: str = "zstd",
) -> None:
    """Append *groups* to a Parquet dataset located at *root_path*.

    Each call writes a new part file named ``part-<uuid>.parquet`` so that
    concurrent writers (e.g. many Ray env actors) can operate without locking.
    """

    # Resolve path to a pyarrow filesystem (handles "gs://", "s3://", etc.).
    fs, dataset_root = pafs.FileSystem.from_uri(root_path)

    # Ensure directory exists (noop if already present).  Some remote FS may
    # raise EEXIST—ignore it.
    try:
        fs.create_dir(dataset_root, recursive=True)
    except FileExistsError:
        pass

    table = _groups_to_table(groups)

    filename = f"{dataset_root.rstrip('/')}/part-{uuid.uuid4().hex}.parquet"
    pq.write_table(table, filename, compression=compression, filesystem=fs)


def iter_rollout_groups(root_path: str) -> Iterator[RolloutGroup]:
    """Yield :class:`RolloutGroup` objects stored under *root_path*.

    Groups are reconstructed on a *best-effort* basis using the serialized group
    metadata.  If multiple rollouts share identical ``group_metadata_json`` they
    will be packed into the same :class:`RolloutGroup`.
    """

    fs, dataset_root = pafs.FileSystem.from_uri(root_path)

    dataset = ds.dataset(dataset_root, format="parquet", filesystem=fs)

    # We'll accumulate rows with identical group metadata together.
    pending: dict[str, RolloutGroup] = {}

    # Iterate over record batches to avoid loading the full dataset in memory.
    for batch in dataset.to_batches():
        for record in batch.to_pylist():
            gid: str = record["id"]
            source: str = record["source"]
            created: float = record["created"]
            group_meta_json: str = record["group_metadata_json"]
            rollout_meta_json: str = record["rollout_metadata_json"]

            turns = []
            for t in record["turns"]:
                turns.append(
                    Turn(
                        message=t["message"],
                        role=t["role"],
                        logprobs=t.get("logprobs"),
                        reward=t.get("reward"),
                        inference_metadata=json.loads(t["inference_metadata_json"]),
                    )
                )

            rollout = Rollout(
                turns=turns,
                metadata=json.loads(rollout_meta_json),
            )

            if gid not in pending:
                pending[gid] = RolloutGroup(
                    id=gid,
                    source=source,
                    created=created,
                    rollouts=[rollout],
                    metadata=json.loads(group_meta_json),
                )
            else:
                grp = pending[gid]
                pending[gid] = RolloutGroup(
                    id=gid,
                    source=source,
                    created=created,
                    rollouts=[*grp.rollouts, rollout],
                    metadata=grp.metadata,
                )

    yield from pending.values()
