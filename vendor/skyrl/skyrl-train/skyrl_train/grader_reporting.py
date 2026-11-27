# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-subset grader health and task success for a SkyRL run."""

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from statistics import mean
from typing import TypedDict

import numpy as np
from finestore.rl.rl_progress import (
    GRADER_REPORT_FILENAME,
    GraderReport,
    SubsetSummary,
    SuccessSummary,
    TaskSummary,
)
from loguru import logger
from marinskyrl.resource_locator import join_resource_path
from rolloutengine.models import RolloutOperation
from skyrl_gym.verification import VerificationStatus
from taskcompendium.grading_result import GRADING_DURATION_SECONDS, GRADING_FAILURE

from skyrl_train.io import io
from skyrl_train.task_inputs import TrajectoryMetadata
from skyrl_train.trajectory_runners.trajectory_processing import rollout_successes
from skyrl_train.trajectory_runners.types import TrajectoryBatch


@dataclass
class _SuccessCounts:
    attempted: int = 0
    verified: int = 0
    passed: int = 0

    def record(self, verified: bool, passed: bool) -> None:
        self.attempted += 1
        self.verified += verified
        self.passed += passed

    def merge(self, other: "_SuccessCounts") -> None:
        self.attempted += other.attempted
        self.verified += other.verified
        self.passed += other.passed

    def summary(self) -> "SuccessSummary":
        return {
            "attempted": self.attempted,
            "verified": self.verified,
            "passed": self.passed,
            "verification_coverage": self.verified / self.attempted,
            "success_rate": self.passed / self.verified if self.verified else None,
        }


@dataclass
class _SubsetStats:
    counts: _SuccessCounts = field(default_factory=_SuccessCounts)
    data_sources: set[str] = field(default_factory=set)
    statuses: Counter[str] = field(default_factory=Counter)
    grading_failures: Counter[str] = field(default_factory=Counter)
    rollout_failures: Counter[str] = field(default_factory=Counter)
    durations: list[float] = field(default_factory=list)


@dataclass
class _TaskStats:
    counts: _SuccessCounts = field(default_factory=_SuccessCounts)
    subset_ids: set[str] = field(default_factory=set)


class GraderSummary(TypedDict):
    subsets: list[SubsetSummary]
    tasks: list[TaskSummary]


class GraderStats:
    """Collect final verdicts by task and subset."""

    def __init__(self) -> None:
        self._subsets: dict[str, _SubsetStats] = {}
        self._tasks: dict[str, _TaskStats] = {}

    def add(self, batch: TrajectoryBatch, metadata: Iterable[TrajectoryMetadata]) -> None:
        """Record one observation per completed rollout, using only its final row."""
        results = batch.get("verification_results") or [None] * len(batch["rewards"])
        successes = rollout_successes(batch["rewards"], results)
        final_rows = batch.get("is_last_step") or [True] * len(results)
        for row, result, passed, final in zip(metadata, results, successes, final_rows, strict=True):
            if not final:
                continue
            subset_id = row.subset_id
            assert subset_id is not None, "Reporting metadata must include a resolved subset"
            verified = result is not None and result.status is VerificationStatus.VERIFIED
            passed = passed and verified
            stats = self._subsets.setdefault(subset_id, _SubsetStats())
            stats.counts.record(verified, passed)
            if row.data_source is not None:
                stats.data_sources.add(row.data_source)
            if row.task_id is not None:
                task = self._tasks.setdefault(row.task_id, _TaskStats())
                task.counts.record(verified, passed)
                task.subset_ids.add(subset_id)
            if result is None:
                stats.statuses["missing"] += 1
                continue
            stats.statuses[result.status.value] += 1
            if (duration := result.diagnostics.get(GRADING_DURATION_SECONDS)) is not None:
                stats.durations.append(duration)
            if failure := result.diagnostics.get(GRADING_FAILURE):
                stats.grading_failures[failure] += 1
            elif result.diagnostics.get("operation") == RolloutOperation.GRADE:
                stats.grading_failures["execution"] += 1
            elif operation := result.diagnostics.get("operation"):
                stats.rollout_failures[operation] += 1

    def merge(self, other: "GraderStats") -> None:
        """Accumulate observations from a completed group."""
        for key, other_stats in other._subsets.items():
            stats = self._subsets.setdefault(key, _SubsetStats())
            stats.counts.merge(other_stats.counts)
            stats.data_sources.update(other_stats.data_sources)
            stats.statuses.update(other_stats.statuses)
            stats.grading_failures.update(other_stats.grading_failures)
            stats.rollout_failures.update(other_stats.rollout_failures)
            stats.durations.extend(other_stats.durations)
        for task_id, other_stats in other._tasks.items():
            stats = self._tasks.setdefault(task_id, _TaskStats())
            stats.counts.merge(other_stats.counts)
            stats.subset_ids.update(other_stats.subset_ids)

    def summary(self) -> GraderSummary:
        """Return JSON-ready counts, rates, and latency statistics."""
        subsets: list[SubsetSummary] = []
        for subset_id, stats in sorted(self._subsets.items()):
            grading_count = len(stats.durations)
            subsets.append(
                {
                    **stats.counts.summary(),
                    "subset_id": subset_id,
                    "data_sources": sorted(stats.data_sources),
                    "verification_statuses": dict(sorted(stats.statuses.items())),
                    "grading_failures": dict(sorted(stats.grading_failures.items())),
                    "rollout_failures": dict(sorted(stats.rollout_failures.items())),
                    "grading_latency_seconds": {
                        "count": grading_count,
                        "mean": mean(stats.durations) if grading_count else None,
                        "p95": float(np.percentile(stats.durations, 95)) if grading_count else None,
                        "max": max(stats.durations, default=None),
                    },
                }
            )
        return {
            "subsets": subsets,
            "tasks": [
                {**stats.counts.summary(), "task_id": task_id, "subset_ids": sorted(stats.subset_ids)}
                for task_id, stats in sorted(self._tasks.items())
            ],
        }


class GraderReportSections(TypedDict, total=False):
    training: list[SubsetSummary]
    training_tasks: list[TaskSummary]
    evaluation: list[SubsetSummary]
    evaluation_tasks: list[TaskSummary]
    evaluation_duration_seconds: float


def write_grader_report(report_dir: str, key: str, sections: GraderReportSections) -> None:
    """Update the attempt's diagnostics for a training or evaluation step."""
    path = join_resource_path(report_dir, GRADER_REPORT_FILENAME)
    try:
        try:
            report = json.loads(io.read_bytes(path))
        except FileNotFoundError:
            report = {}
        for section, value in sections.items():
            report.setdefault(section, {})[key] = value
        payload = GraderReport.model_validate(report).model_dump_json()
        io.write_bytes_atomic(path, (payload + "\n").encode())
    except Exception as error:
        logger.warning("Could not write grader report for {} at {} ({})", key, path, type(error).__name__)
