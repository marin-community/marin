# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Drop repeated instructions within a source and apply the optional per-source cap.

Only converted rows are candidates; unconverted rows pass through untouched so the ledger keeps
their reasons. Two tasks are duplicates when their ``instruction_key`` columns match (the
normalized instruction hash the converted step wrote). The first occurrence in shard order wins;
with a cap, survivors are sampled by a seeded hash of the path so reruns pick the same rows.

The decision pass reads only the small columns of every converted shard; the rewrite pass then
streams one shard at a time, so memory stays bounded by one shard however large the source.
"""

import hashlib
import logging
from collections import Counter
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.tasktrove.convert import CONVERTED_GLOB
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus

logger = logging.getLogger(__name__)

DEDUPED_GLOB = "deduped/*.parquet"
_CAP_SEED = b"tasktrove-clean-cap-v1"
_DECISION_COLUMNS = ["source", "path", "status", "instruction_key"]


class DedupStatus:
    DUPLICATE = "duplicate"
    CAPPED = "capped"


def iter_rows(glob: StoragePath) -> Iterator[dict]:
    """Every row of every parquet the glob matches, one row group in memory at a time."""
    for shard in sorted(glob.glob(), key=str):
        with shard.open("rb") as handle:
            pf = pq.ParquetFile(handle)
            for rg in range(pf.num_row_groups):
                yield from pf.read_row_group(rg).to_pylist()


def cap_rank(path: str) -> str:
    return hashlib.sha256(_CAP_SEED + path.encode()).hexdigest()


def converted_shards(converted_path: str) -> list[StoragePath]:
    return sorted((StoragePath(converted_path) / CONVERTED_GLOB).glob(), key=str)


def dedup_decisions(shards: list[StoragePath], max_tasks_per_source: int | None) -> dict[tuple[str, str], str]:
    """``(source, path) -> DedupStatus`` for every converted row that does not survive."""
    seen: dict[str, set[str]] = {}
    kept: dict[str, list[str]] = {}
    decisions: dict[tuple[str, str], str] = {}
    for shard in shards:
        with shard.open("rb") as handle:
            table = pq.read_table(handle, columns=_DECISION_COLUMNS)
        for row in table.to_pylist():
            if row["status"] != ConvertStatus.CONVERTED:
                continue
            source = row["source"]
            if row["instruction_key"] in seen.setdefault(source, set()):
                decisions[(source, row["path"])] = DedupStatus.DUPLICATE
                continue
            seen[source].add(row["instruction_key"])
            kept.setdefault(source, []).append(row["path"])
    if max_tasks_per_source is not None:
        for source, paths in kept.items():
            for path in sorted(paths, key=cap_rank)[max_tasks_per_source:]:
                decisions[(source, path)] = DedupStatus.CAPPED
    return decisions


def apply_decisions(shard: StoragePath, target: StoragePath, decisions: dict[tuple[str, str], str]) -> Counter:
    """Rewrite one converted shard with the non-surviving rows marked and their binaries dropped."""
    counts: Counter = Counter()
    with shard.open("rb") as handle:
        rows = pq.read_table(handle).to_pylist()
    for row in rows:
        status = decisions.get((row["source"], row["path"]))
        if status is None:
            continue
        row["status"] = status
        row["task_binary"] = None
        row["solution_binary"] = None
        counts[status] += 1
    target.parent.mkdirs()
    with target.open("wb") as handle:
        pq.write_table(pa.Table.from_pylist(rows), handle)
    return counts


def dedup_tasks(converted_path: str, output_path: str, max_tasks_per_source: int | None) -> None:
    """Rewrite the converted shards with duplicate and capped rows marked, keeping the shard layout."""
    shards = converted_shards(converted_path)
    decisions = dedup_decisions(shards, max_tasks_per_source)
    counts: Counter = Counter()
    for shard in shards:
        counts.update(apply_decisions(shard, StoragePath(output_path) / "deduped" / shard.name, decisions))
    logger.info("dedup over %d shards: %s", len(shards), dict(counts))
