# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn eval run records into measurements the statistics engine can work with.

This is the only place that knows the shape of a harness's output: how lm-eval names a task's stderr
and item count, how a group task's subtask rows roll up, how evalchemy can write the same task twice,
and where a mechanism records the items it attempted. :mod:`marin.evaluation.eval_stats` holds the
statistics and the selection rules and knows nothing about any of it, so a new harness is a change
here alone.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from marin.evaluation.archive import base_metric, primary_metric
from marin.evaluation.eval_stats import (
    BINARY_METRICS,
    SAMPLE_COUNT_METRIC,
    TOTAL_METRICS,
    Coverage,
    Measurement,
    MetricKind,
    ResultFlag,
)
from marin.evaluation.records import EVALCHEMY_INFRASTRUCTURE_ERROR, EvalRunRecord

# A value derived from n items is integral in k to within this tolerance when it really is k/n.
_INTEGRALITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class _TaskScore:
    """One task's contribution to a record's benchmark measurement."""

    leaf: str
    value: float
    metric: str
    stderr: float | None
    n_scored: int | None


def stderr_for(metrics: Mapping[str, float], metric_key: str) -> float | None:
    """The standard error paired with ``metric_key``: its ``<base>_stderr,<filter>`` value, or None.

    lm-eval names the stderr for ``acc,none`` as ``acc_stderr,none``; a filterless ``acc`` pairs with
    ``acc_stderr``.
    """
    base, _, metric_filter = metric_key.partition(",")
    key = f"{base}_stderr,{metric_filter}" if metric_filter else f"{base}_stderr"
    value = metrics.get(key)
    return float(value) if value is not None else None


def _task_item_count(metrics: Mapping[str, float]) -> int | None:
    """The graded-item count a task's metric dict reports, or None when it reports none."""
    for key in (SAMPLE_COUNT_METRIC, *TOTAL_METRICS):
        value = metrics.get(key)
        if value is not None:
            return int(value)
    return None


def _task_scores(record: EvalRunRecord) -> list[_TaskScore]:
    """Each task entry's primary metric, deduplicated by leaf task name.

    A record can carry the same task twice under different evalchemy task directories (a real record
    holds the whole 62-entry mmlu panel under both ``mmlu_5shot`` and a ``tmp...`` directory, scoring
    0.63502 and 0.63488). Those entries measure the same items, so keeping both would double the item
    count and average a benchmark against itself; the first wins and the rest are dropped.
    """
    scores: dict[str, _TaskScore] = {}
    for task_key, metrics in (record.metrics or {}).items():
        picked = primary_metric(metrics)
        if picked is None:
            continue
        name, value = picked
        leaf = task_key.rsplit("/", 1)[-1]
        if leaf in scores:
            continue
        scores[leaf] = _TaskScore(
            leaf=leaf,
            value=value,
            metric=name,
            stderr=stderr_for(metrics, name),
            n_scored=_task_item_count(metrics),
        )
    return list(scores.values())


def _rollup_scores(scores: list[_TaskScore]) -> list[_TaskScore]:
    """Collapse a group task's subtask rows onto the aggregate row that already summarizes them.

    lm-eval writes a group's document-weighted aggregate as a subtask whose name prefixes every other
    subtask (``mmlu`` beside ``mmlu_anatomy``); scoring from that row alone is not the same as
    re-averaging it with the per-subject rows it summarizes.
    """
    aggregates = [score for score in scores if all(other.leaf.startswith(score.leaf) for other in scores)]
    if len(aggregates) == 1 and len(scores) > 1:
        return aggregates
    return scores


def _mechanism_coverage(record: EvalRunRecord, n_scored: int | None) -> Coverage:
    """The record's coverage for its benchmark, from the typed field when the producer wrote one.

    A producer that records no attempted count leaves coverage unreported rather than complete: the
    record cannot establish that nothing was lost upstream of it, so readers widen instead. One task
    with an unknown attempted count makes the whole benchmark's count unknown -- a partial sum would
    understate what the run set out to grade -- and the same holds for the pass count.
    """
    reported = record.coverage or {}
    if not reported:
        return Coverage(n_scored=n_scored or 0)
    attempted = [entry.n_attempted for entry in reported.values()]
    correct = [entry.n_correct for entry in reported.values()]
    errors: dict[str, int] = {}
    for entry in reported.values():
        for name, count in entry.errors.items():
            errors[name] = errors.get(name, 0) + count
    return Coverage(
        n_scored=sum(entry.n_scored for entry in reported.values()),
        n_attempted=None if any(c is None for c in attempted) else sum(c for c in attempted if c is not None),
        n_correct=None if any(c is None for c in correct) else sum(c for c in correct if c is not None),
        n_unanswered=sum(entry.n_unanswered for entry in reported.values()),
        errors=errors,
    )


def _item_cap(record: EvalRunRecord) -> int | None:
    """The per-run item cap the launcher declared, if any."""
    if record.evaluation.evalchemy is not None:
        return record.evaluation.evalchemy.max_eval_instances
    if record.evaluation.harbor is not None:
        return record.evaluation.harbor.task_limit
    return None


def measurement_from_record(record: EvalRunRecord) -> Measurement | None:
    """One record's benchmark measurement, or None when it produced no primary metric.

    The benchmark is the registry eval name (the leaderboard column); a record's task entries roll up
    to it exactly as the dashboard has always rolled them up, with the group-aggregate rule preserved.
    """
    scores = _rollup_scores(_task_scores(record))
    if not scores:
        return None
    value = sum(score.value for score in scores) / len(scores)
    labels = {score.metric for score in scores}
    metric = next(iter(labels)) if len(labels) == 1 else "mean"
    counts = [score.n_scored for score in scores if score.n_scored is not None]
    n_scored = sum(counts) if len(counts) == len(scores) else None

    coverage = _mechanism_coverage(record, n_scored)
    kind = MetricKind.BINARY if base_metric(metric) in BINARY_METRICS else MetricKind.CONTINUOUS
    stderr = _combined_stderr([score.stderr for score in scores])
    n_correct = _successes(value, coverage) if kind is MetricKind.BINARY else None
    if n_correct is None and kind is MetricKind.BINARY:
        # A binary metric whose value is not k/n (an unweighted rollup across subtasks) has no
        # Bernoulli count behind it, so it takes the recorded-dispersion path instead.
        kind = MetricKind.CONTINUOUS

    item_cap = _item_cap(record)
    return Measurement(
        benchmark=record.evaluation.name,
        metric=metric,
        kind=kind,
        value=value,
        coverage=coverage,
        n_correct=n_correct,
        recorded_stderr=stderr,
        item_cap=item_cap,
        flags=_flags(coverage, kind, stderr, item_cap),
        run_id=record.run_id,
        created_at=record.created_at,
        version=record.version,
        model=record.model.name,
        git_sha=record.provenance.git_sha,
        eval_runtime=record.provenance.eval_runtime,
        status=record.status,
    )


def measurements_from_records(records: Iterable[EvalRunRecord]) -> list[Measurement]:
    """Every record's benchmark measurement, skipping records that produced no primary metric."""
    return [measurement for record in records if (measurement := measurement_from_record(record)) is not None]


def _successes(value: float, coverage: Coverage) -> int | None:
    """The Bernoulli numerator behind ``value``, or None when the value is not a count over items.

    A producer that tallied its own passes supplies the numerator directly, which is exact where
    recovering it from the reported rate is not. It also settles the question the rate cannot: a
    benchmark whose headline is an unweighted mean across differently-sized subtasks is not the
    pooled ``k/n`` those tallies sum to, and a tally that disagrees with the reported value is proof
    that no Bernoulli count stands behind it -- inverting the rate anyway would find a spurious one
    whenever the mean happened to land on a whole number of items.
    """
    if coverage.n_scored <= 0:
        return None
    scaled = value * coverage.n_scored
    tolerance = _INTEGRALITY_TOLERANCE * max(1.0, coverage.n_scored)
    if coverage.n_correct is not None:
        return coverage.n_correct if abs(scaled - coverage.n_correct) <= tolerance else None
    nearest = round(scaled)
    return int(nearest) if abs(scaled - nearest) <= tolerance else None


def _combined_stderr(stderrs: Sequence[float | None]) -> float | None:
    """Standard error of an unweighted mean of independent means: ``sqrt(sum se^2)/n``.

    None when any component is missing, since the aggregate is then unknown.
    """
    if not stderrs or any(stderr is None for stderr in stderrs):
        return None
    values = [stderr for stderr in stderrs if stderr is not None]
    return math.sqrt(sum(value * value for value in values)) / len(values)


def _flags(coverage: Coverage, kind: MetricKind, stderr: float | None, item_cap: int | None) -> frozenset[ResultFlag]:
    flags: set[ResultFlag] = set()
    if coverage.n_scored <= 0:
        flags.add(ResultFlag.NO_ITEMS)
    if not coverage.reported:
        flags.add(ResultFlag.ATTRITION_UNREPORTED)
    elif coverage.n_missing:
        flags.add(ResultFlag.ATTRITION)
    if item_cap is not None:
        flags.add(ResultFlag.CAPPED)
    unanswered_scored = coverage.n_unanswered - coverage.errors.get(EVALCHEMY_INFRASTRUCTURE_ERROR, 0)
    if coverage.n_scored > 0 and unanswered_scored >= coverage.n_scored:
        flags.add(ResultFlag.NO_ANSWERS)
    if kind is MetricKind.CONTINUOUS:
        if stderr is None:
            flags.add(ResultFlag.NO_DISPERSION)
        elif stderr == 0.0 and coverage.n_scored > 1:
            flags.add(ResultFlag.DEGENERATE_STDERR)
    return frozenset(flags)
