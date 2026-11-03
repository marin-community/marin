"""Canonical benchmark metadata across Evalchemy task implementations."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol

from eval.contracts.task_outcome import TaskRoute
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.contracts.prompt_length import BenchmarkPromptLength

BENCHMARK_METADATA_SCHEMA_VERSION = 1


class MetricKind(StrEnum):
    """Statistical interpretation of one per-item benchmark measurement."""

    BINARY = "binary"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class MetricMetadata:
    """One evaluator metric normalized to Evalchemy's public vocabulary."""

    name: str
    source_name: str
    kind: MetricKind
    higher_is_better: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_name": self.source_name,
            "kind": self.kind,
            "higher_is_better": self.higher_is_better,
        }


@dataclass(frozen=True)
class BenchmarkMetadata:
    """Resolved benchmark protocol consumed without knowledge of task routes."""

    task: str
    primary_metric: str
    metric_kind: MetricKind
    metrics: tuple[MetricMetadata, ...]
    n_benchmark: int | None
    n_attempted: int | None
    schema_version: Literal[1] = BENCHMARK_METADATA_SCHEMA_VERSION
    source: SourceRef | None = None
    reducers: tuple[ReducerSpec, ...] = ()
    prompt_length: BenchmarkPromptLength | None = None
    requires_judge: bool = False

    def __post_init__(self) -> None:
        names = [metric.name for metric in self.metrics]
        if not self.task:
            raise ValueError("benchmark task name must not be empty")
        if not names:
            raise ValueError(f"{self.task}: benchmark metadata requires at least one metric")
        if len(names) != len(set(names)):
            raise ValueError(f"{self.task}: canonical metric names must be unique")
        if self.primary_metric not in names:
            raise ValueError(f"{self.task}: primary metric {self.primary_metric!r} is not declared")
        primary = self.metrics[names.index(self.primary_metric)]
        if self.metric_kind is not primary.kind:
            raise ValueError(f"{self.task}: primary metric kind does not match its metric declaration")
        for label, count in (("n_benchmark", self.n_benchmark), ("n_attempted", self.n_attempted)):
            if count is not None and count < 0:
                raise ValueError(f"{self.task}: {label} must be non-negative when known")
        if self.n_benchmark is not None and self.n_attempted is not None and self.n_attempted > self.n_benchmark:
            raise ValueError(f"{self.task}: n_attempted exceeds n_benchmark")

    def to_dict(self) -> dict[str, Any]:
        document = {
            "schema_version": self.schema_version,
            "task": self.task,
            "primary_metric": self.primary_metric,
            "metric_kind": self.metric_kind,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "n_benchmark": self.n_benchmark,
            "n_attempted": self.n_attempted,
        }
        if self.source is not None:
            document.update(source=asdict(self.source), reducers=[asdict(reducer) for reducer in self.reducers],
                            prompt_length=self.prompt_length.to_dict() if self.prompt_length else None,
                            requires_judge=self.requires_judge)
        return document


@dataclass(frozen=True)
class SourceMetric:
    """Metric spelling and direction reported by a benchmark implementation."""

    name: str
    higher_is_better: bool = True


_ALIASES = {
    "acc": "accuracy",
    "accuracy": "accuracy",
    "accuracy_avg": "accuracy",
    "em": "accuracy",
    "exact_match": "accuracy",
    "exact-match": "accuracy",
    "acc_norm": "normalized_accuracy",
    "acc_norm_nospace": "normalized_accuracy",
    "f1": "f1",
    "score": "score",
    "reward": "reward",
    "mean_reward": "reward",
}
_PRIMARY_PRIORITY = ("accuracy", "normalized_accuracy", "f1", "pass_at_1", "score", "reward")
_PASS_AT_K = re.compile(r"^pass(?:@|_at_)(\d+)$", re.IGNORECASE)
_FILTER_PRIORITY = ("flexible-extract", "none")


def canonical_metric_name(source_name: str) -> str:
    """Return Evalchemy's stable name for a source metric spelling."""
    normalized = source_name.strip().lower().replace(" ", "_")
    if match := _PASS_AT_K.fullmatch(normalized):
        return f"pass_at_{match.group(1)}"
    return _ALIASES.get(normalized, normalized)


def default_metric_kind(canonical_name: str) -> MetricKind:
    """Return the conservative default statistical kind for a canonical metric."""
    if canonical_name in {"accuracy", "normalized_accuracy", "pass_at_1"}:
        return MetricKind.BINARY
    return MetricKind.CONTINUOUS


def resolve_metric_metadata(
    source_metrics: Iterable[SourceMetric],
    *,
    primary_metric: str | None = None,
    name_overrides: Mapping[str, str] | None = None,
    kind_overrides: Mapping[str, MetricKind | str] | None = None,
) -> tuple[tuple[MetricMetadata, ...], str]:
    """Normalize source metrics and select one canonical headline metric."""
    name_overrides = name_overrides or {}
    kind_overrides = kind_overrides or {}
    resolved_metrics = [
        MetricMetadata(
            name=name_overrides.get(source.name, canonical_metric_name(source.name)),
            source_name=source.name,
            kind=MetricKind(
                kind_overrides.get(
                    source.name,
                    kind_overrides.get(
                        name_overrides.get(source.name, canonical_metric_name(source.name)),
                        default_metric_kind(name_overrides.get(source.name, canonical_metric_name(source.name))),
                    ),
                )
            ),
            higher_is_better=source.higher_is_better,
        )
        for source in source_metrics
    ]
    metrics_by_name: dict[str, MetricMetadata] = {}
    for metric in resolved_metrics:
        metrics_by_name.setdefault(metric.name, metric)
    metrics = tuple(metrics_by_name.values())
    if not metrics:
        raise ValueError("benchmark metadata requires at least one source metric")
    names = [metric.name for metric in metrics]

    if primary_metric is not None:
        by_source = {metric.source_name: metric.name for metric in metrics}
        selected = by_source.get(primary_metric, name_overrides.get(primary_metric, canonical_metric_name(primary_metric)))
        if selected not in names:
            raise ValueError(f"primary metric {primary_metric!r} is not among the benchmark metrics")
        return metrics, selected

    for candidate in _PRIMARY_PRIORITY:
        if candidate in names:
            return metrics, candidate
    return metrics, names[0]


class CustomBenchmark(Protocol):
    def describe(self, task_name: str | None = None) -> BenchmarkMetadata | None: ...


class CustomTaskManager(Protocol):
    def get_benchmark(self, task_name: str) -> CustomBenchmark | None: ...


class LMEvalTaskManager(Protocol):
    def load_task_or_group(self, task_list: list[str]) -> Mapping[str, object]: ...


class LMEvalTask(Protocol):
    eval_docs: Sequence[object]

    def get_config(self, name: str) -> Any: ...

    def higher_is_better(self) -> Mapping[str, bool]: ...

    def aggregation(self) -> Mapping[str, object]: ...


def _task_objects(loaded: Mapping[str, object]) -> list[tuple[str, Any]]:
    tasks: list[tuple[str, Any]] = []
    for name, value in loaded.items():
        if isinstance(value, Mapping):
            tasks.extend(_task_objects(value))
        elif callable(getattr(value, "get_config", None)):
            tasks.append((name, value))
    return tasks


def _lm_eval_metadata(task_name: str, task: LMEvalTask, limit: int | None) -> BenchmarkMetadata:
    configured_metrics = task.get_config("metric_list") or []
    source_metrics = tuple(
        SourceMetric(str(metric["metric"]), bool(metric.get("higher_is_better", True)))
        for metric in configured_metrics
        if isinstance(metric, Mapping) and metric.get("metric")
    )
    if not source_metrics:
        directions = task.higher_is_better()
        source_metrics = tuple(
            SourceMetric(str(name), bool(directions.get(name, True))) for name in task.aggregation()
        )
    task_metadata = task.get_config("metadata") or {}
    if not isinstance(task_metadata, Mapping):
        task_metadata = {}
    metrics, primary = resolve_metric_metadata(
        source_metrics,
        primary_metric=task_metadata.get("primary_metric"),
        name_overrides=task_metadata.get("metric_name_overrides"),
        kind_overrides=task_metadata.get("metric_kind_overrides"),
    )
    documents = task.eval_docs
    n_benchmark = len(documents)
    n_attempted = min(limit, n_benchmark) if limit is not None and limit > 0 else n_benchmark
    primary_kind = next(metric.kind for metric in metrics if metric.name == primary)
    return BenchmarkMetadata(task_name, primary, primary_kind, metrics, n_benchmark, n_attempted)


def describe_benchmarks(
    task_names: Sequence[str],
    task_routes: Mapping[str, TaskRoute],
    custom_task_manager: CustomTaskManager,
    lm_eval_task_manager: LMEvalTaskManager,
    *,
    limit: int | None,
) -> tuple[BenchmarkMetadata, ...]:
    """Return every discoverable benchmark descriptor without blocking evaluation."""
    descriptions: list[BenchmarkMetadata] = []
    for task_name in task_names:
        if task_routes[task_name] is TaskRoute.CUSTOM:
            benchmark = custom_task_manager.get_benchmark(task_name)
            if benchmark is None:
                raise LookupError(f"custom benchmark {task_name!r} was not constructed")
            try:
                description = benchmark.describe(task_name)
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                warnings.warn(f"Could not describe benchmark {task_name!r}: {exc}", RuntimeWarning, stacklevel=2)
                description = None
            if description is not None:
                descriptions.append(description)
            continue

        loaded = lm_eval_task_manager.load_task_or_group([task_name])
        task_objects = _task_objects(loaded)
        if not task_objects:
            raise ValueError(f"lm-eval task {task_name!r} resolved no leaf tasks")
        for name, task in task_objects:
            try:
                descriptions.append(_lm_eval_metadata(name, task, limit))
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                warnings.warn(f"Could not describe benchmark {name!r}: {exc}", RuntimeWarning, stacklevel=2)
    return tuple(descriptions)


_STRUCTURAL_RESULT_KEYS = frozenset(
    {
        "attempted",
        "completion_rate",
        "num_correct",
        "num_examples",
        "num_repeat",
        "num_samples",
        "num_solved",
        "num_total",
        "sample_len",
        "scored_count",
        "solved",
        "total",
    }
)


def infer_benchmark_metadata(
    task: str,
    results: Mapping[str, Any],
    *,
    n_attempted: int | None,
) -> BenchmarkMetadata:
    """Describe an undeclared custom benchmark from its completed scalar results."""
    source_metrics = tuple(
        SourceMetric(name)
        for name, value in results.items()
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and name not in _STRUCTURAL_RESULT_KEYS
        and not name.split(",", 1)[0].endswith(("_stderr", "_std_err"))
    )
    metrics, primary = resolve_metric_metadata(source_metrics)
    primary_kind = next(metric.kind for metric in metrics if metric.name == primary)
    return BenchmarkMetadata(task, primary, primary_kind, metrics, None, n_attempted)


def _metric_key(metrics: Mapping[str, Any], source_name: str) -> str | None:
    candidates = [
        name
        for name, value in metrics.items()
        if name.split(",", 1)[0] == source_name
        and not source_name.endswith("_stderr")
        and isinstance(value, int | float)
        and not isinstance(value, bool)
    ]
    for metric_filter in _FILTER_PRIORITY:
        if candidate := next((name for name in candidates if name.endswith(f",{metric_filter}")), None):
            return candidate
    return min(candidates) if candidates else None


def _stderr_key(metrics: Mapping[str, Any], source_key: str) -> str | None:
    source_name, separator, metric_filter = source_key.partition(",")
    source_base = source_name.removesuffix("_avg")
    candidates = [
        f"{source_name}_stderr{separator}{metric_filter}",
        f"{source_name}_stderr",
        f"{source_base}_stderr",
        f"{source_base}_std_err",
    ]
    return next((candidate for candidate in candidates if candidate in metrics), None)


def canonicalize_results(
    results: Mapping[str, Mapping[str, Any]],
    descriptions: Iterable[BenchmarkMetadata],
) -> dict[str, dict[str, float]]:
    """Project evaluator-native result keys into the canonical metric vocabulary."""
    canonical: dict[str, dict[str, float]] = {}
    for description in descriptions:
        task_results = results.get(description.task, {})
        normalized: dict[str, float] = {}
        for metric in description.metrics:
            source_key = _metric_key(task_results, metric.source_name)
            if source_key is None:
                continue
            normalized[metric.name] = float(task_results[source_key])
            stderr_key = _stderr_key(task_results, source_key)
            if stderr_key is not None and isinstance(task_results[stderr_key], int | float):
                normalized[f"{metric.name}_stderr"] = float(task_results[stderr_key])
        canonical[description.task] = normalized
    return canonical
