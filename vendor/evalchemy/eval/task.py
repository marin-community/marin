"""Evaluation source metadata and explicit source loading."""

from __future__ import annotations

import importlib
import inspect
import logging
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import Mapping

from eval.contracts.benchmark_metadata import BenchmarkMetadata, MetricKind, SourceMetric, resolve_metric_metadata
from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.contracts.preflight import ResourceRequirement, TaskPreparation, prepare_task
from eval.contracts.prompt_length import load_prompt_lengths, resolve_task_max_tokens
from eval.contracts.task_outcome import TaskRoute
from eval.passk import parse_pass_at_k
from eval import sources


class BaseBenchmark(ABC):
    """Data source: metadata, static preparation, and an iterator of EvalSpec."""

    RESOURCE_REQUIREMENTS: tuple[ResourceRequirement, ...] = ()
    METRICS: tuple[str, ...] = ()
    PRIMARY_METRIC: str | None = None
    REDUCER = ReducerSpec()
    SOURCE = SourceRef("local")
    METRIC_NAME_OVERRIDES: Mapping[str, str] = MappingProxyType({})
    METRIC_KIND_OVERRIDES: Mapping[str, MetricKind | str] = MappingProxyType({})
    REQUIRES_JUDGE = False

    def __init__(self, logger=None, system_instruction=None, num_samples=1, pass_at_k=None):
        self.logger = logger or logging.getLogger(type(self).__name__)
        self.system_instruction = system_instruction
        self.num_samples = int(num_samples or 1)
        self.pass_at_k = parse_pass_at_k(pass_at_k)
        self._registered_task_name = None
        self._evaluation_max_tokens = None
        self.evaluation_limit = None
        self.dataset_name = self.SOURCE.dataset
        self.dataset_split = self.SOURCE.split
        self.dataset_revision = self.SOURCE.revision
        self.dataset_cache: str | None = None
        self.prompt_length = None

    @abstractmethod
    def iter_specs(self, options):
        """Yield normalized examples; the shared runtime owns their execution."""

    @property
    def benchmark_name(self) -> str:
        """Return the registered task name, or the class-derived name before registration."""
        return self._registered_task_name or self.__class__.__name__.replace("Benchmark", "")


    def prepare(self, task_name: str | None = None) -> TaskPreparation:
        """Validate static resources and loaded data without contacting a model."""
        name = task_name or self.benchmark_name
        return prepare_task(
            name,
            TaskRoute.CUSTOM,
            self.RESOURCE_REQUIREMENTS,
            self.validate_prepared_data,
        )


    def benchmark_size(self) -> int | None:
        """Return the prepared pre-limit item count, or ``None`` when unknown."""
        return None


    def benchmark_metrics(self) -> tuple[str, ...]:
        """Return source metrics for this benchmark's resolved run configuration."""
        if self.num_samples > 1 and self.REDUCER.trial_metric is not None:
            return tuple(f"pass@{k}" for k in self.pass_at_k if k <= self.num_samples)
        return self.METRICS


    def benchmark_primary_metric(self) -> str | None:
        """Return the source spelling of the preferred headline metric."""
        if self.num_samples > 1 and self.REDUCER.trial_metric is not None:
            metrics = self.benchmark_metrics()
            return "pass@1" if "pass@1" in metrics else (metrics[0] if metrics else None)
        return self.PRIMARY_METRIC


    def describe(self, task_name: str | None = None) -> BenchmarkMetadata | None:
        """Return canonical metrics and coverage, or ``None`` when metrics are undeclared."""
        source_metrics = self.benchmark_metrics()
        if not source_metrics:
            return None
        name = task_name or self.benchmark_name
        metrics, primary = resolve_metric_metadata(
            tuple(SourceMetric(metric) for metric in source_metrics),
            primary_metric=self.benchmark_primary_metric(),
            name_overrides=self.METRIC_NAME_OVERRIDES,
            kind_overrides=self.METRIC_KIND_OVERRIDES,
        )
        n_benchmark = self.benchmark_size()
        n_attempted = n_benchmark
        if self.evaluation_limit is not None and n_benchmark is not None:
            n_attempted = int(self.evaluation_limit * n_benchmark) if self.evaluation_limit < 1 else min(int(self.evaluation_limit), n_benchmark)
        primary_kind = next(metric.kind for metric in metrics if metric.name == primary)
        return BenchmarkMetadata(
            name, primary, primary_kind, metrics, n_benchmark, n_attempted,
            source=SourceRef(self.dataset_name, self.dataset_split, self.dataset_revision),
            reducers=(self.REDUCER,), prompt_length=self.prompt_length, requires_judge=self.REQUIRES_JUDGE,
        )


    def validate_prepared_data(self) -> None:
        """Hook for dataset shape and representative-request validation."""


class TaskManager:
    """Load only explicitly selected sources, without executing files or changing sys.path."""

    def __init__(self, task_list=None, **benchmark_kwargs):
        self.tasks = {}
        self.benchmark_instances = {}
        self.load_failures = {}
        lengths = load_prompt_lengths()
        selected = sources.SOURCES if task_list is None else [name for name in task_list if name in sources.SOURCES]
        for name in selected:
            module_name, class_name = sources.SOURCES[name]
            try:
                source_type = getattr(importlib.import_module(module_name), class_name)
                parameters = inspect.signature(source_type).parameters
                arguments = {key: value for key, value in benchmark_kwargs.items() if key in parameters and value is not None}
                source = source_type(**arguments)
                source.dataset_cache = benchmark_kwargs.get("dataset_cache")
                source.num_samples = int(benchmark_kwargs.get("num_samples") or 1)
                source.pass_at_k = parse_pass_at_k(benchmark_kwargs.get("pass_at_k"))
                source._registered_task_name = name
                source.evaluation_limit = benchmark_kwargs.get("limit")
                source.prompt_length = lengths.benchmarks.get(name)
                source._evaluation_max_tokens = resolve_task_max_tokens(
                    name, context_length=benchmark_kwargs.get("max_length"),
                    requested_max_tokens=benchmark_kwargs.get("max_tokens"), prompt_lengths=lengths,
                )
                self.tasks[name] = source_type
                self.benchmark_instances[name] = source
            except Exception as error:
                self.load_failures[name] = error
                logging.getLogger(__name__).error("Cannot load source %s: %s", name, error)

    def get_benchmark(self, name):
        return self.benchmark_instances.get(name)

    def requires_judge_credentials(self, name):
        return name in self.tasks and self.tasks[name].REQUIRES_JUDGE
