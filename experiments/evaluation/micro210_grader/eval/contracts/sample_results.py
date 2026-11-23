"""Per-sample metric values every scored benchmark reports and persists.

A benchmark's grader is the only code that knows whether one sample was right.
Graders call :func:`record_sample_metrics` while scoring. With an output sink
active, that call emits the scored sample; the serialization boundary copies
the recorded values into each sample record with
:func:`sample_metric_fields`, and :func:`validate_sample_metrics` rejects a
persisted sample set that reached the artifact without them.

Records carry the metrics in lm-eval's native shape: ``metrics`` lists the
metric names and each name is a top-level key holding that sample's value, so
custom-benchmark records and lm-eval-native records read identically.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

import numpy as np

BOOLEAN_METRIC_TYPES = (bool, np.bool_)
"""Boolean spellings a grader may pass for a pass/fail metric."""

SAMPLE_METRICS_FIELD = "metrics"
"""Record field listing the metric names carried by one sample."""

SAMPLE_METRICS_ANNOTATION = "sample_metrics"
"""Field a grader annotates on a generated example to carry its own scores."""

SAMPLE_COORDINATES_ANNOTATION = "_sample_coordinates"
"""Stable namespace, ordinal, and output position copied with each generated example."""

SAMPLE_FAILURES_ANNOTATION = "sample_failure_categories"
"""Explicit failure category for each serialized trial, including successful trials as null."""

PER_TASK_PASS_RATE_FIELD = "per_task_pass_rate"
"""Grader-result field mapping a task id to the fraction of its completions that passed.

The vendored sandbox harnesses (HumanEvalPlus, MBPP, MBPPPlus, CruxEval) report
aggregate pass@k plus this map, so their benchmarks can record per-sample metrics
without re-running the sandbox.
"""

RESERVED_RECORD_FIELDS = frozenset(
    {
        "answer_extraction_errors",
        "arguments",
        "completion_responses",
        "doc",
        "doc_hash",
        "doc_id",
        "drop_extractions",
        "filter",
        "filter_variants",
        "failure_category",
        "loose_instruction_pass",
        "strict_instruction_pass",
        "filtered_resps",
        "prompt_hash",
        "resps",
        "sample_id",
        "sample_namespace",
        "sample_ordinal",
        "sample_repeat",
        "sample_shard",
        "schema_version",
        "source_id",
        "target",
        "target_hash",
        "task_name",
        SAMPLE_METRICS_FIELD,
    }
)
"""Sample-record field names a per-sample metric may not shadow."""


class SampleMetricsError(RuntimeError):
    """Raised when per-sample metrics are absent, misnamed, or not numeric."""


def record_sample_metrics(example: MutableMapping[str, Any], **values: bool | float) -> None:
    """Record one graded example's per-sample metric values.

    With a sample sink attached, this completes the example and emits it immediately.
    Call once, after adding every annotation that its sample adapter serializes.

    Args:
        example: The generated example the grader is scoring, annotated in place.
        **values: Metric name to that sample's value. Booleans become 1.0/0.0.
    """
    if not values:
        raise SampleMetricsError("a graded sample must record at least one metric value")
    example[SAMPLE_METRICS_ANNOTATION] = {name: _metric_value(name, value) for name, value in values.items()}


def sample_metric_fields(example: Mapping[str, Any]) -> dict[str, Any]:
    """Return the record fields carrying one example's recorded metrics."""
    metrics = example.get(SAMPLE_METRICS_ANNOTATION)
    if metrics is None:
        return {}
    if not isinstance(metrics, Mapping) or not metrics:
        raise SampleMetricsError(f"{SAMPLE_METRICS_ANNOTATION} must be a non-empty mapping of metric values")
    values = {name: _metric_value(name, value) for name, value in metrics.items()}
    return {SAMPLE_METRICS_FIELD: list(values), **values}


def sample_failure_fields(example: Mapping[str, Any], repeat: int = 0) -> dict[str, Any]:
    """Return a trial's explicit failure classification from restored grading evidence."""
    categories = example.get(SAMPLE_FAILURES_ANNOTATION)
    if categories is None or categories[repeat] is None:
        return {}
    return {"failure_category": categories[repeat]}


def validate_sample_metrics(task_name: str, records: Sequence[Mapping[str, Any]]) -> None:
    """Require every persisted sample record to carry named per-sample metrics."""
    for index, record in enumerate(records):
        names = record.get(SAMPLE_METRICS_FIELD)
        if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or not names:
            raise SampleMetricsError(
                f"{task_name}: sample {index} persists no per-sample metrics; its grader must call "
                "record_sample_metrics for every scored sample"
            )
        if len(set(names)) != len(names):
            raise SampleMetricsError(f"{task_name}: sample {index} lists a duplicate per-sample metric name")
        for name in names:
            if name not in record:
                raise SampleMetricsError(
                    f"{task_name}: sample {index} lists metric {name!r} but carries no value for it"
                )
            _metric_value(name, record[name])


def _metric_value(name: Any, value: Any) -> float:
    """Return one metric value as a finite float, rejecting unusable names."""
    if not isinstance(name, str) or not name:
        raise SampleMetricsError(f"per-sample metric names must be non-empty strings, got {name!r}")
    if name in RESERVED_RECORD_FIELDS:
        raise SampleMetricsError(f"per-sample metric {name!r} shadows a reserved sample-record field")
    if isinstance(value, BOOLEAN_METRIC_TYPES):
        return float(value)
    if not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise SampleMetricsError(f"per-sample metric {name!r} must be a finite number, got {value!r}")
    return float(value)
