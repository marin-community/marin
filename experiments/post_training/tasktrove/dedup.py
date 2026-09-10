# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Group-by keys and reducers that drop repeated instructions within a source and apply the per-source cap.

Only converted rows are candidates; unconverted rows pass through untouched so the ledger keeps
their reasons. Two tasks are duplicates when their ``instruction_key`` columns match (the
normalized instruction hash the converted step wrote). The lowest path wins; with a cap,
survivors are the first ``max_tasks`` by a seeded hash of the path so reruns pick the same rows.
"""

import hashlib
from collections.abc import Iterator

from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus

_CAP_SEED = b"tasktrove-clean-cap-v1"


class DedupStatus:
    DUPLICATE = "duplicate"
    CAPPED = "capped"


def cap_rank(path: str) -> str:
    return hashlib.sha256(_CAP_SEED + path.encode()).hexdigest()


def dedup_key(row: dict) -> tuple[str, str, str]:
    """Converted rows group by instruction within their source; every other row is its own group."""
    if row["status"] == ConvertStatus.CONVERTED:
        return ("instruction", row["source"], row["instruction_key"])
    return ("path", row["source"], row["path"])


def dropped(row: dict, status: str, detail: str) -> dict:
    """The row with its binaries removed and the reason it left the dataset in ``status``/``error``."""
    return {**row, "status": status, "error": detail, "task_binary": None, "solution_binary": None}


def keep_first(_key: tuple, rows: Iterator[dict]) -> Iterator[dict]:
    first = next(rows)
    yield first
    for row in rows:
        yield dropped(row, DedupStatus.DUPLICATE, f"same instruction as {first['path']}")


def cap_source(rows: Iterator[dict], max_tasks: int) -> Iterator[dict]:
    kept = 0
    for row in rows:
        if row["status"] != ConvertStatus.CONVERTED:
            yield row
            continue
        kept += 1
        yield row if kept <= max_tasks else dropped(row, DedupStatus.CAPPED, f"beyond the {max_tasks}-task cap")
