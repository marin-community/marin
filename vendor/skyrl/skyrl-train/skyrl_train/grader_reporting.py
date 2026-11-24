# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-environment grader health and task success for a SkyRL run."""

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from statistics import mean
from typing import Literal, TypedDict

import numpy as np
from loguru import logger
from finestore.rl.rl_progress import GRADER_REPORT_FILENAME
from marinskyrl.resource_locator import join_resource_path
from rolloutengine.models import RolloutOperation
from skyrl_gym.verification import VerificationStatus
from taskcompendium.grading_result import GRADING_DURATION_SECONDS, GRADING_FAILURE

from skyrl_train.io import io
from skyrl_train.trajectory_runners.trajectory_processing import rollout_successes
from skyrl_train.trajectory_runners.types import TrajectoryBatch


@dataclass
class _EnvironmentStats:
    statuses: Counter[str] = field(default_factory=Counter)
    grading_failures: Counter[str] = field(default_factory=Counter)
    rollout_failures: Counter[str] = field(default_factory=Counter)
    durations: list[float] = field(default_factory=list)
    passed: int = 0


class GradingLatency(TypedDict):
    count: int
    mean: float | None
    p95: float | None
    max: float | None


class EnvironmentSummary(TypedDict):
    environment: str
    data_source: str | None
    attempted: int
    verified: int
    verification_coverage: float
    passed: int
    success_rate: float | None
    verification_statuses: dict[str, int]
    grading_failures: dict[str, int]
    rollout_failures: dict[str, int]
    grading_latency_seconds: GradingLatency


class GraderStats:
    """Collect final verdicts and timings by environment class and task source."""

    def __init__(self) -> None:
        self._environments: dict[tuple[str, str | None], _EnvironmentStats] = {}

    def add(self, batch: TrajectoryBatch, environments: Iterable[tuple[str, str | None]]) -> None:
        """Record one observation per completed rollout, using only its final row."""
        results = batch.get("verification_results") or [None] * len(batch["rewards"])
        successes = rollout_successes(batch["rewards"], results)
        final_rows = batch.get("is_last_step") or [True] * len(results)
        for key, result, passed, final in zip(environments, results, successes, final_rows, strict=True):
            if not final:
                continue
            stats = self._environments.setdefault(key, _EnvironmentStats())
            if result is None:
                stats.statuses["missing"] += 1
                continue
            stats.statuses[result.status.value] += 1
            stats.passed += passed and result.status is VerificationStatus.VERIFIED
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
        for key, other_stats in other._environments.items():
            stats = self._environments.setdefault(key, _EnvironmentStats())
            stats.statuses.update(other_stats.statuses)
            stats.grading_failures.update(other_stats.grading_failures)
            stats.rollout_failures.update(other_stats.rollout_failures)
            stats.durations.extend(other_stats.durations)
            stats.passed += other_stats.passed

    def summary(self) -> list[EnvironmentSummary]:
        """Return JSON-ready counts, rates, and latency statistics."""
        rows: list[EnvironmentSummary] = []
        for (environment, source), stats in sorted(
            self._environments.items(), key=lambda item: (item[0][0], item[0][1] or "", item[0][1] is not None)
        ):
            attempted = sum(stats.statuses.values())
            verified = stats.statuses[VerificationStatus.VERIFIED]
            grading_count = len(stats.durations)
            rows.append(
                {
                    "environment": environment,
                    "data_source": source,
                    "attempted": attempted,
                    "verified": verified,
                    "verification_coverage": verified / attempted,
                    "passed": stats.passed,
                    "success_rate": stats.passed / verified if verified else None,
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
        return rows


type GraderReportSection = Literal["training", "evaluation", "evaluation_duration_seconds"]


def write_grader_report(export_path: str, key: str, sections: Mapping[GraderReportSection, object]) -> None:
    """Update report sections for a training execution or evaluation, logging failures."""
    path = join_resource_path(export_path, GRADER_REPORT_FILENAME)
    try:
        report = {"training": {}, "evaluation": {}}
        with suppress(FileNotFoundError):
            report = json.loads(io.read_bytes(path))
        for section, value in sections.items():
            report.setdefault(section, {})[key] = value
        io.write_bytes_atomic(path, (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    except Exception:
        logger.exception("Failed to update grader report at {} for {}", path, key)
