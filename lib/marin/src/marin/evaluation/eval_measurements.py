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
from dataclasses import dataclass, replace

from marin.evaluation.eval_stats import (
    BINARY_METRICS,
    SAMPLE_COUNT_METRIC,
    TOTAL_METRICS,
    Coverage,
    Measurement,
    ResultFlag,
)
from marin.evaluation.evaluation_config import eval_task_directory
from marin.evaluation.lm_eval_samples import is_scratch_artifact
from marin.evaluation.metric_selection import (
    LM_EVAL_STDERR_SUFFIX,
    PUBLICATION_METRICS,
    REPEAT_MEAN_SUFFIX,
    REPEAT_STDERR_SUFFIX,
    base_metric,
    canonical_metric_name,
    declared_metric,
)
from marin.evaluation.records import BenchmarkMetricRef, EvalRunRecord, EvalTaskRef, MetricKind
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
    protocol_identity: str | None = None


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


def task_item_count(metrics: Mapping[str, float]) -> int | None:
    """The graded-item count a task's metric dict reports, or None when it reports none.

    A reported count that is not a whole number is not an item count, so a fractional
    value also reads as None rather than being truncated.
    """
    for key in (SAMPLE_COUNT_METRIC, *TOTAL_METRICS):
        value = metrics.get(key)
        if value is not None:
            numeric = float(value)
            return int(numeric) if numeric.is_integer() else None
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


def _is_scratch_task_key(task_key: str) -> bool:
    return is_scratch_artifact(f"{task_key}/")


def _canonical_task_scores(record: EvalRunRecord) -> tuple[list[_TaskScore], bool]:
    """Read evaluator-canonical scores under their recorded benchmark protocols."""
    scores: dict[str, _TaskScore] = {}
    missing_primary = False
    selected = PUBLICATION_METRICS.get(record.evaluation.name) if record.evaluation.curated is None else None
    task_keys = dict.fromkeys((*record.metrics, *record.canonical_metrics))
    for task_key in task_keys:
        if _is_scratch_task_key(task_key):
            continue
        task = _task_ref(record, task_key)
        benchmark = task.benchmark if task is not None else None
        if benchmark is None:
            continue
        metrics = record.canonical_metrics.get(task_key, {})
        name = selected.name if selected is not None else benchmark.primary_metric
        kind = selected.kind if selected is not None else benchmark.metric_kind
        value = metrics.get(name)
        if value is None:
            missing_primary |= bool(metrics) or bool(record.metrics.get(task_key))
            continue
        leaf = task_key.rsplit("/", 1)[-1]
        scores.setdefault(
            leaf,
            _TaskScore(
                leaf=leaf,
                value=value,
                metric=name,
                stderr=metrics.get(f"{name}_stderr"),
                n_scored=None,
                kind=kind,
                declared=True,
                protocol_metric=canonical_metric_name(name, record.evaluation.name),
                protocol_identity=benchmark.identity,
            ),
        )
    return list(scores.values()), missing_primary


def _legacy_task_scores(record: EvalRunRecord, *, undeclared_only: bool = False) -> tuple[list[_TaskScore], bool]:
    """Each task entry's primary metric, deduplicated by leaf task name.

    A record can carry the same task twice under different evalchemy task directories. Scratch
    ``tmp...`` entries are excluded; the first remaining score wins for duplicate task leaves.
    """
    scores: dict[str, _TaskScore] = {}
    selected = PUBLICATION_METRICS.get(record.evaluation.name) if record.evaluation.curated is None else None
    missing = False
    for task_key, metrics in (record.metrics or {}).items():
        if _is_scratch_task_key(task_key):
            continue
        task = _task_ref(record, task_key)
        if undeclared_only and task is not None and task.benchmark is not None:
            continue
        picked = declared_metric(metrics, selected.source_name if selected is not None else None)
        if picked is None:
            missing |= selected is not None and bool(metrics)
            continue
        name, value = picked
        leaf = task_key.rsplit("/", 1)[-1]
        if leaf in scores:
            continue
        scores[leaf] = _TaskScore(
            leaf=leaf,
            value=value,
            metric=selected.name if selected is not None else canonical_metric_name(name, record.evaluation.name),
            stderr=stderr_for(metrics, name),
            n_scored=task_item_count(metrics),
            kind=selected.kind if selected is not None else None,
            declared=selected is not None,
            protocol_metric=selected.name if selected is not None else None,
        )
    return list(scores.values()), missing


def _task_scores(record: EvalRunRecord) -> tuple[list[_TaskScore], bool]:
    if any(task.benchmark is not None for task in record.evaluation.tasks):
        canonical, missing_primary = _canonical_task_scores(record)
        legacy, missing_legacy = _legacy_task_scores(record, undeclared_only=True)
        return canonical + legacy, missing_primary or missing_legacy
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
        if _is_scratch_task_key(task_key):
            continue
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
    if record.evaluation.curated is not None:
        return record.evaluation.curated.settings.source_limit
    if record.evaluation.evalchemy is not None:
        return record.evaluation.evalchemy.max_eval_instances
    if record.evaluation.harbor is not None:
        return record.evaluation.harbor.task_limit
    return None


def measurement_from_record(record: EvalRunRecord) -> Measurement | None:
    """One record's benchmark measurement, or None when it produced no primary metric.

    The benchmark is the registry eval name (the leaderboard column); a record's task entries roll up
    to it exactly as the dashboard has always rolled them up, with the group-aggregate rule preserved.
    Readers decide whether the record's observed StepSpec state makes its score publishable.
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
    repeated = tuple(record.repeat_aggregates.values())
    if len(repeated) > 1:
        raise ValueError(f"{record.run_id}: cannot combine several repeated task aggregates")
    repeat_aggregation = repeated[0] if repeated else None
    selected = PUBLICATION_METRICS.get(record.evaluation.name) if record.evaluation.curated is None else None
    if selected is not None:
        # These policy metrics share their benchmark's scored-item denominator.
        # A saved numerator belongs to the evaluator's original headline metric.
        coverage = replace(coverage, n_correct=None)
    declared = all(score.declared for score in scores)
    declared_kinds = {score.kind for score in scores if score.kind is not None}
    declared_metrics = {score.protocol_metric for score in scores if score.protocol_metric is not None}
    identities = {score.protocol_identity for score in scores}
    if repeat_aggregation is not None:
        kind = MetricKind.CONTINUOUS
    elif declared and len(declared_kinds) == 1:
        kind = next(iter(declared_kinds))
    else:
        kind = MetricKind.BINARY if base_metric(metric) in BINARY_METRICS else MetricKind.CONTINUOUS
    stderr = (
        repeat_aggregation.stderr
        if repeat_aggregation is not None
        else _combined_stderr([score.stderr for score in scores])
    )
    if record.evaluation.name == "mrcr" and not declared:
        kind = MetricKind.CONTINUOUS
    n_correct = _successes(value, coverage) if kind is MetricKind.BINARY else None
    if n_correct is None and kind is MetricKind.BINARY:
        # A binary metric whose value is not k/n (an unweighted rollup across subtasks) has no
        # Bernoulli count behind it, so it takes the recorded-dispersion path instead.
        kind = MetricKind.CONTINUOUS

    if selected is not None:
        coverage = replace(coverage, n_correct=n_correct)
    item_cap = _item_cap(record)
    fewshot_values = {task.num_fewshot for task in record.evaluation.tasks}
    return Measurement(
        benchmark=record.evaluation.name,
        metric=canonical_metric_name(metric, record.evaluation.name),
        kind=kind,
        value=value,
        coverage=coverage,
        n_correct=n_correct,
        recorded_stderr=stderr,
        repeat_aggregation=repeat_aggregation,
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
        protocol_kind=(
            MetricKind.CONTINUOUS
            if repeat_aggregation is not None
            else next(iter(declared_kinds)) if declared and len(declared_kinds) == 1 else None
        ),
        protocol_identity=next(iter(identities)) if declared and len(identities) == 1 else None,
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
        selected = PUBLICATION_METRICS.get(record.evaluation.name) if record.evaluation.curated is None else None
        name = selected.name if selected is not None else benchmark.primary_metric
        if name not in metrics:
            return f"declared metric {name} not in canonical results"
    return None


@dataclass(frozen=True)
class MetricRow:
    """One metric a record reports for a task: protocol name, value, and whether it was declared."""

    task: str
    metric: str
    value: float
    declared: bool


def _metric_task_keys(record: EvalRunRecord) -> list[str]:
    """Every distinct non-scratch metrics key, from either the raw or the canonical map."""
    return [key for key in dict.fromkeys((*record.metrics, *record.canonical_metrics)) if not _is_scratch_task_key(key)]


def _declared_metric_value(record: EvalRunRecord, task_keys: Sequence[str], metric: BenchmarkMetricRef) -> float | None:
    """A declared metric's canonical value, or its raw source value under filter priority."""
    for task_key in task_keys:
        canonical = record.canonical_metrics.get(task_key, {})
        if metric.name in canonical:
            return canonical[metric.name]
    for task_key in task_keys:
        picked = declared_metric(record.metrics.get(task_key, {}), metric.source_name)
        if picked is not None:
            return picked[1]
    return None


def declared_metric_rows(record: EvalRunRecord) -> list[MetricRow]:
    """Every declared benchmark metric for each recorded task, the primary metric first.

    A declared metric reports its protocol name and takes the canonical value when the producer
    wrote one; otherwise the raw source metric supplies it under the same extraction-filter priority
    the statistics engine ranks with. Metrics with no recorded value are left out.
    """
    rows: list[MetricRow] = []
    task_keys = _metric_task_keys(record)
    for task in record.evaluation.tasks:
        benchmark = task.benchmark
        if benchmark is None:
            continue
        keys = [task_key for task_key in task_keys if _task_ref(record, task_key) is task]
        ordered = [metric for metric in benchmark.metrics if metric.name == benchmark.primary_metric]
        ordered += [metric for metric in benchmark.metrics if metric.name != benchmark.primary_metric]
        for metric in ordered:
            value = _declared_metric_value(record, keys, metric)
            if value is not None:
                rows.append(MetricRow(task=task.name, metric=metric.name, value=value, declared=True))
    return rows


def undeclared_metric_rows(record: EvalRunRecord) -> list[MetricRow]:
    """Raw metric rows for task keys that no declared benchmark covers.

    A record can mix tasks with benchmark metadata and tasks without it. The declared rows describe
    only the former, so the latter keep their raw rows; a record with no declarations at all returns
    its whole raw metric map, in stable task then metric order.
    """
    rows: list[MetricRow] = []
    for task_key, metrics in record.metrics.items():
        if _is_scratch_task_key(task_key):
            continue
        task = _task_ref(record, task_key)
        if task is not None and task.benchmark is not None:
            continue
        rows.extend(
            MetricRow(task=task_key, metric=name, value=value, declared=False) for name, value in metrics.items()
        )
    return sorted(rows, key=lambda row: (row.task, row.metric))


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
