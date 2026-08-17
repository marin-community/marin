# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed readers for evaluation output artifacts and aggregate reports.

Eval steps write backend-native output. :class:`EvalResult` subclasses parse each backend's layout.
:class:`EvalchemyResult` reads lm-eval's ``<task_dir>/<model>/results_<ts>.json`` trees and keys the
metrics by task-config directory. :func:`compile_eval_report` merges several typed results.
"""

import functools
import json
import logging
from dataclasses import dataclass

from pydantic import Field
from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.evaluation.lm_eval_samples import is_scratch_artifact
from marin.execution.artifact import Artifact, result_type_name

logger = logging.getLogger(__name__)

_REPORT_FILE = "report.json"


def _numeric(values: dict) -> dict[str, float]:
    """The numeric entries of a metric dict, dropping string aliases and config echoes."""
    return {key: float(value) for key, value in values.items() if isinstance(value, bool | int | float)}


def _result_task_dir(result_file: StoragePath) -> str:
    """Return ``<task_dir>`` from ``<task_dir>/<model>/results_<ts>.json``.

    The evalchemy client assigns each task configuration a unique alias or ``name_Nshot`` directory.
    """
    task_dir = result_file.parent.parent.name
    if not task_dir:
        raise ValueError(f"unexpected evalchemy results layout (want <task_dir>/<model>/file): {result_file}")
    return task_dir


class EvalResult(Artifact):
    """Path-backed per-task metrics and cross-task averages for one evaluation."""

    def task_metrics(self) -> dict[str, dict[str, float]]:
        """Numeric metrics for every evaluated task, as ``{task: {metric: value}}``."""
        raise NotImplementedError

    def averages(self) -> dict[str, float]:
        """Cross-task averages the backend recorded, or ``{}`` if it records none."""
        raise NotImplementedError


class EvalchemyResult(EvalResult):
    """An evalchemy run's output: lm-eval's native ``<task_dir>/<model>/results_<ts>.json`` tree.

    The evalchemy fork writes one result file per task configuration under the directory chosen by
    :func:`~marin.evaluation.evalchemy.runner._task_dir`. Single-task files use that directory as the
    metric key. Group-task entries use ``<task_dir>/<subtask>``. The task directory keeps shot
    variants distinct. :func:`compile_eval_report` computes suite-level rollups because evalchemy
    does not record cross-task averages.
    """

    @functools.cached_property
    def _task_metrics(self) -> dict[str, dict[str, float]]:
        # StoragePath.glob reattaches the protocol to each match; a bare fs.glob result drops the
        # gs:// prefix and would reopen as a local path.
        root = StoragePath(self.path)
        found = sorted((root / "**/results_*.json").glob(), key=str)
        if root.scheme == "file":
            # fsspec's local glob returns bare paths, while ``relative_to`` compares protocols.
            found = [StoragePath(scheme="file", segments=path.segments, rooted=path.rooted) for path in found]
        if not found:
            raise FileNotFoundError(f"no evalchemy results_*.json under {self.path}")
        # A retried evaluation leaves a second complete tree under the harness's scratch directory,
        # scoring the same items again. Reading both would key one benchmark's panel twice and double
        # its item count, so the canonical tree wins wherever there is one to prefer.
        canonical = [path for path in found if not is_scratch_artifact(path.relative_to(root))]
        result_files = canonical or found
        metrics: dict[str, dict[str, float]] = {}
        for result_file in result_files:
            task_dir = _result_task_dir(result_file)
            results = json.loads(result_file.read_text()).get("results", {})
            for task, task_metrics in results.items():
                # One entry -> the dir is the whole identity; several (a group task) -> namespace them.
                key = task_dir if len(results) == 1 else f"{task_dir}/{task}"
                metrics[key] = _numeric(task_metrics)
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


# Map serialized result-type names to the readers used by :func:`compile_eval_report`.
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
