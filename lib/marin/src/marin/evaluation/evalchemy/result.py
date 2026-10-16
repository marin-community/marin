# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read Evalchemy result trees and FineStore archives and compile aggregate reports."""

import functools
import json
import logging
from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import Field
from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.evaluation.evalchemy.client import EVALCHEMY_RESULTS_PREFIX, EVALCHEMY_RESULTS_SUFFIX
from marin.evaluation.lm_eval_samples import (
    is_scratch_artifact,
    read_native_evalchemy_artifacts,
    read_native_evalchemy_results,
)
from marin.execution.artifact import Artifact, result_type_name

logger = logging.getLogger(__name__)

_REPORT_FILE = "report.json"


def _numeric(values: dict) -> dict[str, float]:
    """The numeric entries of a metric dict, dropping string aliases and config echoes."""
    return {key: float(value) for key, value in values.items() if isinstance(value, bool | int | float)}


def _result_task_dir(result_file: StoragePath) -> str:
    task_dir = result_file.parent.parent.name
    if not task_dir:
        raise ValueError(f"unexpected evalchemy results layout (want <task_dir>/<model>/file): {result_file}")
    return task_dir


def _metrics_for_task_dir(results: dict, task_dir: str) -> dict[str, dict[str, float]]:
    return {
        task_dir if len(results) == 1 else f"{task_dir}/{task}": _numeric(task_metrics)
        for task, task_metrics in results.items()
    }


class EvalResult(Artifact):
    """Path-backed per-task metrics and cross-task averages for one evaluation."""

    def task_metrics(self) -> dict[str, dict[str, float]]:
        """Numeric metrics for every evaluated task, as ``{task: {metric: value}}``."""
        raise NotImplementedError

    def averages(self) -> dict[str, float]:
        """Cross-task averages the backend recorded, or ``{}`` if it records none."""
        raise NotImplementedError


class EvalchemyResult(EvalResult):
    """Per-task metrics from an Evalchemy result tree.

    Keys use task-config directories, with subtask names appended for groups.
    Shot variants remain distinct. Cross-task averages are empty.
    Historical reports persist the ``EvalchemyResult`` type name, so this reader
    remains part of their decoding contract.
    """

    @functools.cached_property
    def _task_metrics(self) -> dict[str, dict[str, float]]:
        # StoragePath.glob reattaches the protocol to each match; a bare fs.glob result drops the
        # gs:// prefix and would reopen as a local path.
        root = StoragePath(self.path)
        found = sorted((root / f"**/{EVALCHEMY_RESULTS_PREFIX}*{EVALCHEMY_RESULTS_SUFFIX}").glob(), key=str)
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
            metrics.update(_metrics_for_task_dir(results, task_dir))
        return metrics

    def task_metrics(self) -> dict[str, dict[str, float]]:
        return dict(self._task_metrics)

    def averages(self) -> dict[str, float]:
        return {}


class FineStoreEvalchemyResult(EvalResult):
    """Per-task metrics from Evalchemy results artifacts stored in FineStore."""

    jobs: dict[str, str] = Field(default_factory=dict)

    @functools.cached_property
    def _task_metrics(self) -> dict[str, dict[str, float]]:
        documents = read_native_evalchemy_results(self.path)
        if documents:
            return {
                key: value
                for prefix, document in documents.items()
                for key, value in _metrics_for_task_dir(document.get("results", {}), prefix).items()
            }
        result_payloads = read_native_evalchemy_artifacts(self.path).result_payloads
        if not result_payloads:
            raise FileNotFoundError(f"no Evalchemy results artifacts in FineStore archive {self.path}")

        metrics: dict[str, dict[str, float]] = {}
        for name, payload in sorted(result_payloads.items()):
            task_dir = PurePosixPath(name).parent.parent.name
            results = json.loads(payload).get("results", {})
            metrics.update(_metrics_for_task_dir(results, task_dir))
        return metrics

    def task_metrics(self) -> dict[str, dict[str, float]]:
        return dict(self._task_metrics)

    def averages(self) -> dict[str, float]:
        return {}


class EvalReport(Artifact):
    """An evaluation report with serialized per-task metrics and cross-task averages."""

    task_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    """Every evaluated task across the suite, as ``{task: {metric: value}}``."""

    averages: dict[str, float] = Field(default_factory=dict)
    """Backend-recorded cross-task averages, namespaced ``{result_label}/{average}`` to keep the
    contributions from different results distinct."""


_EVAL_RESULT_TYPES: dict[str, type[EvalResult]] = {
    result_type_name(cls): cls for cls in (EvalchemyResult, FineStoreEvalchemyResult)
}


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
    """Merge result metrics, write ``output_path/report.json``, and return the report.

    Reject duplicate task keys and namespace averages by result label.
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
