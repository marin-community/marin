# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed eval-output artifacts and the aggregated report.

An eval step writes its backend's native output; the typed artifact reads the metrics back
*through* the artifact, so a consumer calls ``result.task_metrics()`` instead of guessing the
directory layout. :class:`EvalchemyResult` reads the evalchemy fork's output — lm-eval's native
nested tree (``{task}_{n}shot/<model>/results_<ts>.json``, one file per task) — by globbing and
merging each file's ``results`` block.

:func:`compile_eval_report` reads every dependency uniformly
(``dep.artifact_type.raw_load(path).task_metrics()``) and materializes one :class:`EvalReport` — a
value artifact carrying the merged per-task metrics and averages.
"""

import functools
import json
import logging
from dataclasses import dataclass

from pydantic import Field
from rigging.filesystem import StoragePath, prefix_join

from marin.execution.artifact import Artifact, result_type_name

logger = logging.getLogger(__name__)

_REPORT_FILE = "report.json"


def _numeric(values: dict) -> dict[str, float]:
    """The numeric entries of a metric dict, dropping string aliases and config echoes."""
    return {key: float(value) for key, value in values.items() if isinstance(value, bool | int | float)}


class EvalResult(Artifact):
    """One eval's output: per-task metrics and (where the backend provides them) cross-task averages.

    A path-ref artifact — ``raw_load`` returns a handle into the output directory and the metrics are
    parsed on demand. Subclasses implement the two accessors for their backend's on-disk shape.
    """

    def task_metrics(self) -> dict[str, dict[str, float]]:
        """Numeric metrics for every evaluated task, as ``{task: {metric: value}}``."""
        raise NotImplementedError

    def averages(self) -> dict[str, float]:
        """Cross-task averages the backend recorded, or ``{}`` if it records none."""
        raise NotImplementedError


class EvalchemyResult(EvalResult):
    """An evalchemy run's output: lm-eval's native ``{task}_{n}shot/<model>/results_<ts>.json`` tree.

    The evalchemy fork runs each task through lm-eval's ``EvaluationTracker`` and writes one
    ``results_<ts>.json`` per task, uploaded whole to the artifact path. The accessor globs them and
    merges each file's ``results`` block. evalchemy records no cross-task average, so :meth:`averages`
    is empty — :func:`compile_eval_report` computes suite-level rollups instead.
    """

    @functools.cached_property
    def _task_metrics(self) -> dict[str, dict[str, float]]:
        # StoragePath.glob reattaches the protocol to each match; a bare fs.glob result drops the
        # gs:// prefix and would reopen as a local path.
        result_files = sorted(StoragePath(prefix_join(self.path, "**/results_*.json")).glob(), key=str)
        if not result_files:
            raise FileNotFoundError(f"no evalchemy results_*.json under {self.path}")
        metrics: dict[str, dict[str, float]] = {}
        for result_file in result_files:
            payload = json.loads(result_file.read_text())
            for task, task_metrics in payload.get("results", {}).items():
                metrics[task] = _numeric(task_metrics)
        return metrics

    def task_metrics(self) -> dict[str, dict[str, float]]:
        return dict(self._task_metrics)

    def averages(self) -> dict[str, float]:
        return {}


class EvalReport(Artifact):
    """The aggregated report over a suite of :class:`EvalResult` artifacts.

    A value artifact: ``task_metrics`` and ``averages`` round-trip through the record, so a downstream
    step reads ``resolve(report).averages`` directly. :func:`compile_eval_report` also writes a
    human-readable ``report.json`` alongside for inspection.
    """

    task_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    """Every evaluated task across the suite, as ``{task: {metric: value}}``."""

    averages: dict[str, float] = Field(default_factory=dict)
    """Backend-recorded cross-task averages, namespaced ``{result_label}/{average}`` to keep the
    contributions from different results distinct."""


# result_type name -> reader class, so :func:`compile_eval_report` reconstructs the right accessor
# from the identity string a step records (the class itself cannot ride through the JSON config).
_EVAL_RESULT_TYPES: dict[str, type[EvalResult]] = {result_type_name(cls): cls for cls in (EvalchemyResult,)}


@dataclass(frozen=True)
class ReportEntry:
    """One result feeding :func:`compile_eval_report`.

    ``path`` is the result's output directory, ``result_type`` selects the reader (see
    :data:`_EVAL_RESULT_TYPES`), and ``label`` namespaces that result's averages.
    """

    path: str
    result_type: str
    label: str


def compile_eval_report(entries: list[ReportEntry], output_path: str) -> EvalReport:
    """Read each result's metrics and merge them into one :class:`EvalReport`.

    Writes ``report.json`` under ``output_path`` and returns the typed report (its fields persist via
    the record).
    """
    task_metrics: dict[str, dict[str, float]] = {}
    averages: dict[str, float] = {}
    for entry in entries:
        reader = _EVAL_RESULT_TYPES.get(entry.result_type)
        if reader is None:
            raise ValueError(f"no EvalResult reader for {entry.result_type!r}; known: {sorted(_EVAL_RESULT_TYPES)}")
        result = reader.raw_load(entry.path)
        for task, metrics in result.task_metrics().items():
            if task in task_metrics:
                raise ValueError(
                    f"duplicate task {task!r} while compiling the report (from {entry.label!r}); two "
                    "results evaluate the same task, so one would silently overwrite the other — give the "
                    "tasks distinct aliases or split them into separate reports"
                )
            task_metrics[task] = metrics
        for average, value in result.averages().items():
            averages[f"{entry.label}/{average}"] = value

    report = EvalReport(task_metrics=task_metrics, averages=averages)
    StoragePath(prefix_join(output_path, _REPORT_FILE)).write_text(
        json.dumps({"task_metrics": task_metrics, "averages": averages}, indent=2)
    )
    return report
