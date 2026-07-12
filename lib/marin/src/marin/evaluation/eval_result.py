# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed result handle for a Levanter lm-eval-harness run.

The Levanter evaluator writes a single top-level ``results.json`` spanning every task it ran
(see :mod:`marin.evaluation.evaluators.levanter_lm_eval_evaluator`). :class:`LevanterEvalResult`
is the typed artifact for that output: a path ref (no payload pulled into the launcher) whose
accessors read the per-task metrics and the macro/micro averages on demand, so a downstream
consumer reads ``eval_result.averages()`` instead of guessing the directory layout.
"""

import functools
import json
import logging

from rigging.filesystem import StoragePath, url_to_fs

from marin.execution.artifact import Artifact

logger = logging.getLogger(__name__)

_RESULTS_FILE = "results.json"


def _numeric(values: dict) -> dict[str, float]:
    """The numeric entries of a metric dict, dropping string aliases and config echoes."""
    return {key: float(value) for key, value in values.items() if isinstance(value, bool | int | float)}


class LevanterEvalResult(Artifact):
    """A Levanter lm-eval-harness run's output: per-task metrics and cross-task averages.

    The realized artifact for a :func:`~experiments.evals.evals.evaluate_levanter_lm_evaluation_harness`
    handle. ``raw_load`` is a path ref; the metrics are parsed from ``results.json`` under the
    artifact's path the first time they are read.
    """

    @functools.cached_property
    def _results(self) -> dict:
        """The raw ``results.json`` payload (read once)."""
        path = f"{self.path}/{_RESULTS_FILE}"
        if not url_to_fs(path, use_listings_cache=False)[0].exists(path):
            raise FileNotFoundError(f"no {_RESULTS_FILE} for eval result at {self.path}")
        return json.loads(StoragePath(path).read_text())

    def task_metrics(self) -> dict[str, dict[str, float]]:
        """The numeric metrics for every evaluated task, as ``{task: {metric: value}}``."""
        return {task: _numeric(metrics) for task, metrics in self._results.get("results", {}).items()}

    def averages(self) -> dict[str, float]:
        """The cross-task ``macro_avg_*`` / ``micro_avg_*`` scalars Levanter records."""
        return _numeric(self._results.get("averages", {}))
