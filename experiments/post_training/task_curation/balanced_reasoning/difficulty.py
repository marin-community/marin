# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-task difficulty estimates from the Atlas difficulty evidence ledger.

The reconciler's evidence ledger (``s3://marin-us-east-02a/marin/rl-data/atlas/difficulty-ledger``,
``rows/`` partitions) records one immutable row per scored attempt, keyed by source, task bytes and
solver. This module folds those rows into a per-task solve rate and places each task into its
difficulty band. The row schema here covers only the fields the lane consumes; the authoritative
schema lives with the reconciler on its own branch.
"""

import json
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import PurePosixPath

import fsspec
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.task_curation.balanced_reasoning.bands import Band, band_for

logger = logging.getLogger(__name__)

DEFAULT_LEDGER = "s3://marin-us-east-02a/marin/rl-data/atlas/difficulty-ledger"
SCORED_OUTCOME = "scored"

SOLVER_HOSTED = "hosted"
"""The ``atlas-difficulty-v3`` solver whose family the generator belongs to (GLM-5.3)."""

SOLVER_ALL = "all"
"""Pool every pinned solver's attempts per task (the fallback ladder's sparsity basis)."""


def url_protocol(url: str) -> str:
    """The fsspec protocol of a storage URL; local paths are ``file``."""
    return url.split("://", 1)[0] if "://" in url else "file"


def url_parent(url: str) -> str:
    """The parent directory of a storage URL."""
    return str(PurePosixPath(url).parent)


@dataclass(frozen=True, slots=True)
class LedgerAttempt:
    """The fields of one reconciler evidence row the lane consumes.

    Source-level receipts (measurement and cost rows) carry no task identity; their
    ``task_sha256`` is ``None`` and the tally never folds them into per-task rates.
    """

    source_id: str
    task_sha256: str | None
    solver: str
    outcome: str
    reward: float | None
    attempt_id: str
    task_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class TaskDifficulty:
    """One task's measured difficulty and the basis it was estimated on."""

    source_id: str
    task_sha256: str
    task_id: str
    solve_rate: float
    scored: int
    basis: str

    @property
    def band(self) -> Band:
        return band_for(self.solve_rate)


def parse_ledger_row(row: dict) -> LedgerAttempt:
    """Validate and project one evidence-ledger JSON row."""
    for field in ("source_id", "solver", "outcome"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"Evidence row is missing a usable {field!r}: {row!r}")
    reward = row.get("reward")
    if reward is not None and not isinstance(reward, (int, float)):
        raise ValueError(f"Evidence row carries a non-numeric reward: {row!r}")
    attempt_id = row.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError(f"Evidence row is missing a usable 'attempt_id': {row!r}")
    task_sha256 = row.get("task_sha256")
    if task_sha256 is not None and not isinstance(task_sha256, str):
        raise ValueError(f"Evidence row carries a non-string 'task_sha256': {row!r}")
    task_id = row.get("task_id")
    if task_id is not None and not isinstance(task_id, str):
        raise ValueError(f"Evidence row carries a non-string 'task_id': {row!r}")
    return LedgerAttempt(
        source_id=row["source_id"],
        task_sha256=task_sha256,
        solver=row["solver"],
        outcome=row["outcome"],
        reward=float(reward) if reward is not None else None,
        attempt_id=attempt_id,
        task_id=task_id,
        reason=row.get("reason"),
    )


def read_ledger_attempts(ledger_root: str = DEFAULT_LEDGER) -> Iterator[LedgerAttempt]:
    """Stream scored and unscored attempt rows from the ledger's ``rows/`` partitions."""
    rows_prefix = prefix_join(ledger_root, "rows")
    logger.info("Reading difficulty evidence from %s", rows_prefix)
    filesystem = fsspec.filesystem(url_protocol(ledger_root))
    # s3fs globs return bucket-key paths without the scheme; open them on the same
    # filesystem so keys stay remote instead of falling through to local paths.
    for path in sorted(filesystem.glob(f"{rows_prefix}/**/*.jsonl")):
        with filesystem.open(path, "rt") as source:
            for line in source:
                if line.strip():
                    yield parse_ledger_row(json.loads(line))


def tally_attempts(
    attempts: Iterable[LedgerAttempt],
    *,
    solver: str = SOLVER_HOSTED,
    reward_pass: float = 1.0,
) -> dict[tuple[str, str], TaskDifficulty]:
    """Fold scored attempts into per-task solve rates, keyed by ``(source_id, task_sha256)``.

    A task's rate is ``solved / scored`` over its distinct scored attempts under ``solver``;
    an attempt solves the task when its reward reaches ``reward_pass``. Rows of other outcomes
    (infrastructure, cost, escalations) never count as scored. ``solver="all"`` pools every
    pinned solver's attempts per task — the fallback ladder's basis where the hosted solver's
    own attempts are too sparse to band with (one wave attempt per task is a coin flip).
    """
    pooled = solver == SOLVER_ALL
    scored: dict[tuple[str, str], dict[str, float | None]] = {}
    identities: dict[tuple[str, str], tuple[str, str | None]] = {}
    for attempt in attempts:
        if not pooled and attempt.solver != solver:
            continue
        if attempt.outcome != SCORED_OUTCOME or attempt.task_sha256 is None:
            continue
        key = (attempt.source_id, attempt.task_sha256)
        by_attempt = scored.setdefault(key, {})
        if attempt.attempt_id in by_attempt:
            continue
        by_attempt[attempt.attempt_id] = attempt.reward
        if attempt.task_id is not None:
            identities.setdefault(key, (attempt.task_id, None))
    tasks: dict[tuple[str, str], TaskDifficulty] = {}
    for key, by_attempt in scored.items():
        source_id, task_sha256 = key
        rewards = [reward for reward in by_attempt.values() if reward is not None]
        solved = sum(1 for reward in rewards if reward >= reward_pass)
        task_id, _ = identities.get(key, (task_sha256, None))
        tasks[key] = TaskDifficulty(
            source_id=source_id,
            task_sha256=task_sha256,
            task_id=task_id or task_sha256,
            solve_rate=solved / len(by_attempt),
            scored=len(by_attempt),
            basis="ledger:pooled" if pooled else f"ledger:{solver}",
        )
    return tasks
