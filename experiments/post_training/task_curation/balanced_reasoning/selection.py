# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn verified tasks plus per-task difficulty into the lane's selection plan.

The plan is the lane's contract with the generator: which tasks run, in which band, under
which budget, in which order. Rows are written in the canonical difficulty-sorted order
(``(band, solve_rate, domain, task_id)``) so a curriculum consumer can re-derive the order
from the published metadata.
"""

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256

import fsspec

from experiments.post_training.task_curation.balanced_reasoning.bands import (
    BAND_BUDGETS,
    BAND_ORDER,
    Band,
    BandBudget,
    budget_for,
)
from experiments.post_training.task_curation.balanced_reasoning.difficulty import (
    TaskDifficulty,
    url_protocol,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CandidateTask:
    """One verified Atlas task offered to the lane."""

    task_id: str
    source_id: str
    task_sha256: str
    domain: str
    task_path: str
    """Where the task's ``TaskSpec`` JSON lives (any fsspec URL or local path)."""


@dataclass(frozen=True, slots=True)
class SelectedTask:
    """One task admitted into a generation round, with its band and budget."""

    task: CandidateTask
    difficulty: TaskDifficulty
    band: Band
    budget: BandBudget

    @property
    def sort_key(self) -> tuple[int, float, str, str]:
        """The canonical difficulty-sorted order: band, then solve rate, then domain, then id."""
        return (BAND_ORDER[self.band], self.difficulty.solve_rate, self.task.domain, self.task.task_id)


@dataclass(frozen=True, slots=True)
class Quota:
    """Target row counts per difficulty band, optionally per domain."""

    per_band: Mapping[Band, int]

    def target_for(self, band: Band, domain: str, domain_targets: Mapping[str, Mapping[Band, int]]) -> int:
        """The row target for one (band, domain) cell; zero when the band has no quota."""
        base = self.per_band.get(band, 0)
        overrides = domain_targets.get(domain)
        if overrides is None:
            return base
        return overrides.get(band, 0)


DEFAULT_QUOTA = Quota({Band.EASY: 4_000, Band.MEDIUM: 4_000, Band.HARD: 2_500, Band.UNSOLVED: 1_500})
"""The v1 store's band targets (12k rows), mirroring the pricing table in DESIGN.md §6."""


def select_tasks(
    candidates: Iterable[CandidateTask],
    difficulty_by_key: Mapping[tuple[str, str], TaskDifficulty],
    quota: Quota,
    *,
    domain_targets: Mapping[str, Mapping[Band, int]] | None = None,
) -> tuple[list[SelectedTask], dict[str, int]]:
    """Admit tasks into bands until each (band, domain) cell meets its quota.

    Candidates without a difficulty estimate are skipped and counted, never banded by
    guesswork. The census reports achieved vs target per band and the overall shortfall, so
    shortfalls are visible instead of silently rebalanced.
    """
    domain_targets = domain_targets or {}
    cells: dict[tuple[Band, str], int] = {}
    selected: list[SelectedTask] = []
    skipped_without_estimate = 0
    for candidate in candidates:
        difficulty = difficulty_by_key.get((candidate.source_id, candidate.task_sha256))
        if difficulty is None:
            skipped_without_estimate += 1
            continue
        # The per-band quota is global across domains unless the plan overrides one domain.
        cell_domain = candidate.domain if candidate.domain in domain_targets else "*"
        key = (difficulty.band, cell_domain)
        if cells.get(key, 0) >= quota.target_for(difficulty.band, candidate.domain, domain_targets):
            continue
        cells[key] = cells.get(key, 0) + 1
        selected.append(SelectedTask(candidate, difficulty, difficulty.band, budget_for(difficulty.band)))
    selected.sort(key=lambda entry: entry.sort_key)
    census = cells_census(selected, quota)
    census["skipped_without_estimate"] = skipped_without_estimate
    return selected, census


def cells_census(
    selected: Iterable[SelectedTask],
    quota: Quota,
) -> dict[str, int]:
    """Per-band achieved vs target counts, plus the overall shortfall."""
    achieved: dict[Band, int] = {}
    for entry in selected:
        achieved[entry.band] = achieved.get(entry.band, 0) + 1
    census: dict[str, int] = {}
    shortfall = 0
    for budget in BAND_BUDGETS:
        band = budget.band
        target = quota.per_band.get(band, 0)
        count = achieved.get(band, 0)
        census[band.value] = count
        census[f"{band.value}_target"] = target
        shortfall += max(0, target - count)
    census["shortfall"] = shortfall
    return census


def selection_digest(selected: Iterable[SelectedTask]) -> str:
    """The content digest of a selection, pinning the plan into receipts and manifests."""
    payload = [
        {
            "task_id": entry.task.task_id,
            "source_id": entry.task.source_id,
            "task_sha256": entry.task.task_sha256,
            "domain": entry.task.domain,
            "task_path": entry.task.task_path,
            "band": entry.band.value,
            "solve_rate": entry.difficulty.solve_rate,
            "basis": entry.difficulty.basis,
            "cap_tokens": entry.budget.cap_tokens,
            "attempts": entry.budget.attempts,
        }
        for entry in selected
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _entry_document(entry: SelectedTask) -> dict:
    return {
        "task_id": entry.task.task_id,
        "source_id": entry.task.source_id,
        "task_sha256": entry.task.task_sha256,
        "domain": entry.task.domain,
        "task_path": entry.task.task_path,
        "band": entry.band.value,
        "solve_rate": entry.difficulty.solve_rate,
        "scored": entry.difficulty.scored,
        "difficulty_basis": entry.difficulty.basis,
        "budget_target_tokens": list(entry.budget.target_tokens),
        "budget_cap_tokens": entry.budget.cap_tokens,
        "attempts": entry.budget.attempts,
    }


def write_selection(path: str, selected: Iterable[SelectedTask], census: Mapping[str, int]) -> str:
    """Write ``selection.json`` at ``path`` and return its content digest."""
    entries = list(selected)
    document = {
        "digest": selection_digest(entries),
        "census": dict(census),
        "tasks": [_entry_document(entry) for entry in entries],
    }
    filesystem = fsspec.filesystem(url_protocol(path))
    # Split on "/" rather than PurePosixPath: the URL's scheme prefix is not a path segment.
    filesystem.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with filesystem.open(path, "wt") as sink:
        json.dump(document, sink, indent=2, sort_keys=True)
    logger.info("Wrote selection %s with %d tasks (digest %s)", path, len(entries), document["digest"])
    return document["digest"]


def read_selection(path: str) -> tuple[str, list[SelectedTask], dict[str, int]]:
    """Read back a ``selection.json`` written by :func:`write_selection`."""
    with fsspec.open(path, "rt") as source:
        document = json.load(source)
    selected = []
    for entry in document["tasks"]:
        candidate = CandidateTask(
            task_id=entry["task_id"],
            source_id=entry["source_id"],
            task_sha256=entry["task_sha256"],
            domain=entry["domain"],
            task_path=entry["task_path"],
        )
        difficulty = TaskDifficulty(
            source_id=entry["source_id"],
            task_sha256=entry["task_sha256"],
            task_id=entry["task_id"],
            solve_rate=entry["solve_rate"],
            scored=entry["scored"],
            basis=entry["difficulty_basis"],
        )
        budget = budget_for(Band(entry["band"]))
        selected.append(SelectedTask(candidate, difficulty, budget.band, budget))
    return document["digest"], selected, dict(document["census"])
