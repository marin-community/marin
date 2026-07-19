# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standard-error lookup paired with the canonical primary-metric selection.

Primary-metric selection (``PRIMARY_METRIC_PRIORITY``, ``FILTER_PRIORITY``, ``primary_metric``) is
defined once, in ``marin.evaluation.samples`` (the per-sample export contract), and re-exported here
so the matrix/leaderboard views and the sample browser rank metrics identically. ``stderr_for`` is the
one piece specific to run-level metric dicts: finding the stderr paired with a metric key.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from marin.evaluation.records import EvalRunRecord
from marin.evaluation.samples import primary_metric as primary_metric


def stderr_for(metrics: dict[str, float], metric_key: str) -> float | None:
    """The standard error paired with ``metric_key``: its ``<base>_stderr,<filter>`` value, or None.

    lm-eval names the stderr for ``acc,none`` as ``acc_stderr,none``; a filterless ``acc`` pairs with
    ``acc_stderr``.
    """
    base, _, flt = metric_key.partition(",")
    key = f"{base}_stderr,{flt}" if flt else f"{base}_stderr"
    value = metrics.get(key)
    return float(value) if value is not None else None


def _task_of(metric_key: str) -> str:
    """The task a metrics key belongs to: the prefix before ``/`` for a group subtask, else the key."""
    return metric_key.split("/", 1)[0]


def _combined_stderr(stderrs: list[float | None]) -> float | None:
    """Standard error of an unweighted mean of independent means: ``sqrt(sum se^2)/n``.

    None when any component stderr is missing, since the aggregate is then unknown.
    """
    values: list[float] = []
    for stderr in stderrs:
        if stderr is None:
            return None
        values.append(stderr)
    if not values:
        return None
    return math.sqrt(sum(value * value for value in values)) / len(values)


@dataclass(frozen=True)
class TaskScore:
    """One task's headline score for a record: the value, its metric label, and paired stderr."""

    value: float
    metric: str
    stderr: float | None


@dataclass(frozen=True)
class _MetricScore:
    subtask: str
    value: float
    metric: str
    stderr: float | None


def primary_metrics_by_task(record: EvalRunRecord) -> dict[str, TaskScore]:
    """One record's headline metric per task as ``{task: TaskScore}``.

    A group task writes namespaced ``prefix/subtask`` keys; those roll up to ``prefix`` by the
    unweighted mean of each subtask's primary metric, with the aggregate stderr combined across
    subtasks. A plain task keeps its single primary value. ``metric`` is the shared metric name, or
    ``mean`` when a rollup spans differing metrics.
    """
    by_task: dict[str, list[_MetricScore]] = {}
    for task_key, metrics in (record.metrics or {}).items():
        picked = primary_metric(metrics)
        if picked is None:
            continue
        name, value = picked
        subtask = task_key.rsplit("/", 1)[-1]
        by_task.setdefault(_task_of(task_key), []).append(
            _MetricScore(subtask=subtask, value=value, metric=name, stderr=stderr_for(metrics, name))
        )
    result: dict[str, TaskScore] = {}
    for task, entries in by_task.items():
        # lm-eval writes a group's doc-weighted aggregate as a subtask whose name prefixes every
        # other subtask (``mmlu_5shot/mmlu`` beside ``mmlu_5shot/mmlu_anatomy``); score from that
        # row alone rather than re-averaging it with the per-subject rows it already summarizes.
        aggregates = [entry for entry in entries if all(other.subtask.startswith(entry.subtask) for other in entries)]
        if len(aggregates) == 1 and len(entries) > 1:
            entries = aggregates
        mean = sum(entry.value for entry in entries) / len(entries)
        labels = {entry.metric for entry in entries}
        stderr = _combined_stderr([entry.stderr for entry in entries])
        result[task] = TaskScore(mean, next(iter(labels)) if len(labels) == 1 else "mean", stderr)
    return result


def record_score(record: EvalRunRecord) -> TaskScore | None:
    """One record's headline score: its per-task primaries rolled up by unweighted mean.

    Records here carry a single top-level task, so this is normally that task's primary metric;
    a multi-task eval averages its tasks with the stderrs combined. None when nothing scored.
    """
    per_task = primary_metrics_by_task(record) if record.metrics else {}
    if not per_task:
        return None
    entries = list(per_task.values())
    mean = sum(score.value for score in entries) / len(entries)
    labels = {score.metric for score in entries}
    metric = labels.pop() if len(labels) == 1 else "mean"
    return TaskScore(value=mean, metric=metric, stderr=_combined_stderr([score.stderr for score in entries]))


def build_matrix(records: list[EvalRunRecord]) -> dict:
    """Pivot runs into a ``model x eval`` matrix plus a per-model leaderboard.

    Columns are registry eval names (``record.evaluation.name``), so a failed run occupies the same
    column its succeeded retry fills. Each cell shows the latest succeeded run's rolled-up primary
    metric (with stderr) for that ``(model, eval)``; when no run for a cell ever succeeded, the cell
    carries the latest run's failure status instead, still linking to that run so a failure is
    visible and clickable rather than silently dropped. The leaderboard scores each model by the
    unweighted mean of its succeeded cells and reports coverage over the full eval set, sorted
    best-first with unscored models last.
    """
    succeeded: dict[tuple[str, str], dict] = {}
    latest_any: dict[tuple[str, str], dict] = {}
    tasks: set[str] = set()
    for record in records:
        model = record.model.name
        created_at = record.created_at or ""
        status = record.status.value
        eval_name = record.evaluation.name
        if eval_name.endswith("-smoke"):
            # Smoke suites are capped-instance launcher validation runs; keep them out of the
            # headline grid (they remain visible in the runs list and history).
            continue
        tasks.add(eval_name)
        key = (model, eval_name)
        latest = latest_any.get(key)
        if latest is None or created_at > latest["created_at"]:
            latest_any[key] = {"run_id": record.run_id, "created_at": created_at, "status": status}
        score = record_score(record)
        if score is not None:
            current = succeeded.get(key)
            if current is None or created_at > current["created_at"]:
                succeeded[key] = {
                    "value": score.value,
                    "stderr": score.stderr,
                    "metric": score.metric,
                    "run_id": record.run_id,
                    "created_at": created_at,
                }

    rows_by_model: dict[str, dict] = {}
    for key, latest in latest_any.items():
        model, task = key
        win = succeeded.get(key)
        if win is not None:
            cell = {
                "status": "succeeded",
                "value": win["value"],
                "stderr": win["stderr"],
                "metric": win["metric"],
                "run_id": win["run_id"],
                "created_at": win["created_at"],
            }
        else:
            cell = {
                "status": latest["status"],
                "value": None,
                "stderr": None,
                "metric": None,
                "run_id": latest["run_id"],
                "created_at": latest["created_at"],
            }
        rows_by_model.setdefault(model, {})[task] = cell

    task_list = sorted(tasks)
    rows = [{"model": model, "cells": rows_by_model[model]} for model in sorted(rows_by_model)]
    leaderboard = []
    for model, cells in rows_by_model.items():
        scored = [cell for cell in cells.values() if cell["value"] is not None]
        score = sum(cell["value"] for cell in scored) / len(scored) if scored else None
        stderr = _combined_stderr([cell["stderr"] for cell in scored]) if scored else None
        leaderboard.append(
            {"model": model, "score": score, "stderr": stderr, "covered": len(scored), "total": len(task_list)}
        )
    leaderboard.sort(key=lambda entry: (entry["score"] is not None, entry["score"] or 0.0), reverse=True)
    return {"tasks": task_list, "rows": rows, "leaderboard": leaderboard}


def build_meta(records: list[EvalRunRecord]) -> dict:
    """Distinct filter values (models, evals, users, statuses) across all records."""
    return {
        "models": sorted({r.model.name for r in records}),
        "evals": sorted({r.evaluation.name for r in records}),
        "users": sorted({r.user for r in records if r.user}),
        "statuses": sorted({r.status.value for r in records}),
    }
