# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn eval run records into measurements the statistics engine can work with.

This module and :mod:`marin.evaluation.metric_selection` are the only places that know the shape of
a harness's output: how lm-eval and Evalchemy name a task's score and stderr, how a task reports its
item count, how a group task's subtask rows roll up, how evalchemy can write the same task twice, and
where a mechanism records the items it attempted. :mod:`marin.evaluation.eval_stats` holds the
statistics and the selection rules and knows nothing about any of it, so a new harness is a change
in these two modules alone.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from marin.evaluation.eval_stats import (
    BINARY_METRICS,
    SAMPLE_COUNT_METRIC,
    TOTAL_METRICS,
    Coverage,
    Measurement,
    ResultFlag,
)
from marin.evaluation.evaluation_config import eval_task_directory
from marin.evaluation.metric_selection import (
    LM_EVAL_STDERR_SUFFIX,
    REPEAT_MEAN_SUFFIX,
    REPEAT_STDERR_SUFFIX,
    base_metric,
    declared_metric,
)
from marin.evaluation.records import EvalRunRecord, EvalTaskRef, MetricKind
from marin.evaluation.records import TaskCoverage as RecordTaskCoverage

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
    kind: MetricKind | None = None
    declared: bool = False
    protocol_metric: str | None = None


def stderr_for(metrics: Mapping[str, float], metric_key: str) -> float | None:
    """The standard error paired with ``metric_key``, or None when the task recorded none.

    lm-eval names the stderr for ``acc,none`` as ``acc_stderr,none``; a filterless ``acc`` pairs with
    ``acc_stderr``. Evalchemy's repeated-sample tasks pair ``accuracy_avg`` with ``accuracy_std_err``.
    """
    base, _, metric_filter = metric_key.partition(",")
    if base.endswith(REPEAT_MEAN_SUFFIX):
        key = base.removesuffix(REPEAT_MEAN_SUFFIX) + REPEAT_STDERR_SUFFIX
    else:
        key = base + LM_EVAL_STDERR_SUFFIX
    if metric_filter:
        key = f"{key},{metric_filter}"
    value = metrics.get(key)
    return float(value) if value is not None else None


def _task_item_count(metrics: Mapping[str, float]) -> int | None:
    """The graded-item count a task's metric dict reports, or None when it reports none."""
    for key in (SAMPLE_COUNT_METRIC, *TOTAL_METRICS):
        value = metrics.get(key)
        if value is not None:
            return int(value)
    return None


def _task_ref(record: EvalRunRecord, task_key: str) -> EvalTaskRef | None:
    """Match a metrics row to its recorded task declaration."""
    if len(record.evaluation.tasks) == 1:
        return record.evaluation.tasks[0]
    leaf = task_key.rsplit("/", 1)[-1]
    for task in record.evaluation.tasks:
        if task.benchmark is not None and task.benchmark.task == leaf:
            return task
    directory = task_key.split("/", 1)[0]
    for task in record.evaluation.tasks:
        if directory == eval_task_directory(task.name, task.num_fewshot, task.task_alias):
            return task
    return None


def _canonical_task_scores(record: EvalRunRecord) -> tuple[list[_TaskScore], bool]:
    """Read evaluator-canonical scores under their recorded benchmark protocols."""
    scores: dict[str, _TaskScore] = {}
    missing_primary = False
    task_keys = dict.fromkeys((*record.metrics, *record.canonical_metrics))
    for task_key in task_keys:
        task = _task_ref(record, task_key)
        benchmark = task.benchmark if task is not None else None
        if benchmark is None:
            continue
        metrics = record.canonical_metrics.get(task_key, {})
        value = metrics.get(benchmark.primary_metric)
        if value is None:
            missing_primary |= bool(metrics) or bool(record.metrics.get(task_key))
            continue
        leaf = task_key.rsplit("/", 1)[-1]
        scores.setdefault(
            leaf,
            _TaskScore(
                leaf=leaf,
                value=value,
                metric=benchmark.primary_metric,
                stderr=metrics.get(f"{benchmark.primary_metric}_stderr"),
                n_scored=None,
                kind=benchmark.metric_kind,
                declared=True,
                protocol_metric=benchmark.primary_metric,
            ),
        )
    return list(scores.values()), missing_primary


_LEGACY_METRIC_ALIASES = {
    "acc": "accuracy",
    "accuracy": "accuracy",
    "accuracy_avg": "accuracy",
    "em": "accuracy",
    "exact_match": "accuracy",
    "exact-match": "accuracy",
    "acc_norm": "normalized_accuracy",
    "acc_norm_nospace": "normalized_accuracy",
    "pass@1": "pass_at_1",
    "mean_reward": "reward",
}


def _legacy_canonical_metric_name(name: str) -> str:
    """Canonicalize old record metric names so they remain comparable with evaluator metadata."""
    return _LEGACY_METRIC_ALIASES.get(name, name)


def _legacy_task_scores(record: EvalRunRecord, *, undeclared_only: bool = False) -> tuple[list[_TaskScore], bool]:
    """Each task entry's primary metric, deduplicated by leaf task name.

    A record can carry the same task twice under different evalchemy task directories (a real record
    holds the whole 62-entry mmlu panel under both ``mmlu_5shot`` and a ``tmp...`` directory, scoring
    0.63502 and 0.63488). Those entries measure the same items, so keeping both would double the item
    count and average a benchmark against itself; the first wins and the rest are dropped.
    """
    scores: dict[str, _TaskScore] = {}
    for task_key, metrics in (record.metrics or {}).items():
        task = _task_ref(record, task_key)
        if undeclared_only and task is not None and task.benchmark is not None:
            continue
        picked = declared_metric(metrics, None)
        if picked is None:
            continue
        name, value = picked
        leaf = task_key.rsplit("/", 1)[-1]
        if leaf in scores:
            continue
        scores[leaf] = _TaskScore(
            leaf=leaf,
            value=value,
            metric=_legacy_canonical_metric_name(base_metric(name)),
            stderr=stderr_for(metrics, name),
            n_scored=_task_item_count(metrics),
        )
    return list(scores.values()), False


def _task_scores(record: EvalRunRecord) -> tuple[list[_TaskScore], bool]:
    if any(task.benchmark is not None for task in record.evaluation.tasks):
        canonical, missing_primary = _canonical_task_scores(record)
        legacy, _ = _legacy_task_scores(record, undeclared_only=True)
        return canonical + legacy, missing_primary
    return _legacy_task_scores(record)


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
    reported_by_leaf: dict[str, RecordTaskCoverage] = {}
    for task_key, entry in (record.coverage or {}).items():
        reported_by_leaf.setdefault(task_key.rsplit("/", 1)[-1], entry)
    reported = list(reported_by_leaf.values())
    if not reported:
        return Coverage(n_scored=n_scored or 0)
    benchmark = [entry.n_benchmark for entry in reported]
    attempted = [entry.n_attempted for entry in reported]
    correct = [entry.n_correct for entry in reported]
    errors: dict[str, int] = {}
    for entry in reported:
        for name, count in entry.errors.items():
            errors[name] = errors.get(name, 0) + count
    return Coverage(
        n_scored=sum(entry.n_scored for entry in reported),
        n_benchmark=None if any(c is None for c in benchmark) else sum(c for c in benchmark if c is not None),
        n_attempted=None if any(c is None for c in attempted) else sum(c for c in attempted if c is not None),
        n_correct=None if any(c is None for c in correct) else sum(c for c in correct if c is not None),
        n_unanswered=sum(entry.n_unanswered for entry in reported),
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
    task_scores, missing_declared_metric = _task_scores(record)
    scores = _rollup_scores(task_scores)
    if not scores or missing_declared_metric:
        return None
    value = sum(score.value for score in scores) / len(scores)
    labels = {score.metric for score in scores}
    metric = next(iter(labels)) if len(labels) == 1 else "mean"
    counts = [score.n_scored for score in scores if score.n_scored is not None]
    n_scored = sum(counts) if len(counts) == len(scores) else None

    coverage = _mechanism_coverage(record, n_scored)
    declared = all(score.declared for score in scores)
    declared_kinds = {score.kind for score in scores if score.kind is not None}
    declared_metrics = {score.protocol_metric for score in scores if score.protocol_metric is not None}
    kind = (
        next(iter(declared_kinds))
        if declared and len(declared_kinds) == 1
        else MetricKind.BINARY if base_metric(metric) in BINARY_METRICS else MetricKind.CONTINUOUS
    )
    stderr = _combined_stderr([score.stderr for score in scores])
    n_correct = _successes(value, coverage) if kind is MetricKind.BINARY else None
    if n_correct is None and kind is MetricKind.BINARY:
        # A binary metric whose value is not k/n (an unweighted rollup across subtasks) has no
        # Bernoulli count behind it, so it takes the recorded-dispersion path instead.
        kind = MetricKind.CONTINUOUS

    item_cap = _item_cap(record)
    fewshot_values = {task.num_fewshot for task in record.evaluation.tasks}
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
        num_fewshot=next(iter(fewshot_values)) if len(fewshot_values) == 1 else None,
        run_id=record.run_id,
        created_at=record.created_at,
        version=record.version,
        model=record.model.name,
        git_sha=record.provenance.git_sha,
        eval_runtime=record.provenance.eval_runtime,
        status=record.status,
        declared=declared,
        protocol_metric=next(iter(declared_metrics)) if declared and len(declared_metrics) == 1 else None,
        protocol_kind=next(iter(declared_kinds)) if declared and len(declared_kinds) == 1 else None,
    )


def measurements_from_records(records: Iterable[EvalRunRecord]) -> list[Measurement]:
    """Every record's benchmark measurement, skipping records that produced no primary metric."""
    return [measurement for record in records if (measurement := measurement_from_record(record)) is not None]


def declared_metric_gap(record: EvalRunRecord) -> str | None:
    """Explain a result row that lacks its declared headline metric."""
    for task_key, source_metrics in record.metrics.items():
        task = _task_ref(record, task_key)
        benchmark = task.benchmark if task is not None else None
        if benchmark is None or not source_metrics:
            continue
        metrics = record.canonical_metrics.get(task_key, {})
        if benchmark.primary_metric not in metrics:
            return f"declared metric {benchmark.primary_metric} not in canonical results"
    return None


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
    if (
        coverage.n_scored < 0
        or (coverage.n_attempted is not None and coverage.n_scored > coverage.n_attempted)
        or (coverage.n_benchmark is not None and coverage.n_benchmark <= 0)
        or (
            coverage.n_benchmark is not None
            and coverage.n_attempted is not None
            and coverage.n_attempted > coverage.n_benchmark
        )
    ):
        flags.add(ResultFlag.INCONSISTENT_COVERAGE)
    if coverage.n_scored > 0 and coverage.n_unanswered >= coverage.n_scored:
        flags.add(ResultFlag.NO_ANSWERS)
    if kind is MetricKind.CONTINUOUS:
        if stderr is None:
            flags.add(ResultFlag.NO_DISPERSION)
        elif stderr == 0.0 and coverage.n_scored > 1:
            flags.add(ResultFlag.DEGENERATE_STDERR)
    return frozenset(flags)
