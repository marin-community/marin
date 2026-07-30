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

# Presentation grouping of eval columns into suites for the dashboard's column tree. This mirrors the
# launcher's suite membership (experiments/evaluation/evals.py), which evaldash cannot import: it ships
# as a standalone image vendoring only the marin record contracts, and experiments depends on marin,
# not the reverse. Membership drift is graceful -- an eval not listed here just falls into "Other".
EVAL_SUITES: dict[str, tuple[str, ...]] = {
    "NLP": (
        "mmlu",
        "arc-challenge",
        "arc-easy",
        "hellaswag",
        "winogrande",
        "truthfulqa",
        "boolq",
        "piqa",
        "openbookqa",
        "lambada",
        "triviaqa",
        "nq-open",
        "drop",
        "gsm8k-0shot",
    ),
    "Chat / Math": ("math500", "aime24", "olympiadbench"),
    "Code": ("humaneval", "humanevalplus", "mbppplus"),
}


def eval_suites(evals: set[str]) -> list[dict]:
    """Group the eval names present into ordered presentation suites for the column tree.

    Each suite lists only the evals actually seen; any eval outside :data:`EVAL_SUITES` lands in a
    trailing ``Other`` bucket so an unmapped column is still selectable.
    """
    assigned = {name for names in EVAL_SUITES.values() for name in names}
    result = [
        {"suite": suite, "evals": present}
        for suite, names in EVAL_SUITES.items()
        if (present := sorted(name for name in names if name in evals))
    ]
    other = sorted(name for name in evals if name not in assigned)
    if other:
        result.append({"suite": "Other", "evals": other})
    return result


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


def _current_version(records: list[EvalRunRecord]) -> str | None:
    """The version label of a model's most recent run: the cohort the headline matrix shows for it.

    Runs are grouped into version cohorts so the matrix never mixes evals across model states -- a
    model's newest run picks the current version, and only that version's runs fill its row. An
    unlabelled launch (``version is None``) is its own cohort, so pre-version runs behave as before.
    """
    newest = max(records, key=lambda record: record.created_at or "")
    return newest.version


def _cohort_cells(records: list[EvalRunRecord]) -> dict[str, dict]:
    """One version cohort's ``{eval_name: cell}``: latest succeeded run per eval, else latest failure.

    A cell shows the latest succeeded run's rolled-up primary metric (with stderr); when no run for
    that eval in the cohort succeeded, it carries the latest run's failure status with a null value,
    still linking that run so the failure stays visible and clickable rather than silently dropped.
    """
    succeeded: dict[str, dict] = {}
    latest_any: dict[str, dict] = {}
    for record in records:
        eval_name = record.evaluation.name
        created_at = record.created_at or ""
        latest = latest_any.get(eval_name)
        if latest is None or created_at > latest["created_at"]:
            latest_any[eval_name] = {"run_id": record.run_id, "created_at": created_at, "status": record.status.value}
        score = record_score(record)
        if score is not None:
            current = succeeded.get(eval_name)
            if current is None or created_at > current["created_at"]:
                succeeded[eval_name] = {
                    "value": score.value,
                    "stderr": score.stderr,
                    "metric": score.metric,
                    "run_id": record.run_id,
                    "created_at": created_at,
                }
    cells: dict[str, dict] = {}
    for eval_name, latest in latest_any.items():
        win = succeeded.get(eval_name)
        if win is not None:
            cells[eval_name] = {"status": "succeeded", **win}
        else:
            cells[eval_name] = {
                "status": latest["status"],
                "value": None,
                "stderr": None,
                "metric": None,
                "run_id": latest["run_id"],
                "created_at": latest["created_at"],
            }
    return cells


def build_matrix(records: list[EvalRunRecord], archived_models: frozenset[str] = frozenset()) -> dict:
    """Pivot runs into a ``model x eval`` matrix plus a per-model leaderboard.

    Each model row reflects its latest version cohort: only the runs labelled with the version of the
    model's most recent launch fill its cells, so the headline never unions evals produced against
    different model states. Columns are the registry eval names present across those cohorts. Each row
    and leaderboard entry carries its cohort ``version`` and an ``archived`` flag. The leaderboard
    scores each model by the unweighted mean of its succeeded cells over the full eval set, sorted
    best-first with unscored models last.
    """
    by_model: dict[str, list[EvalRunRecord]] = {}
    for record in records:
        # Smoke suites are capped-instance launcher validation runs; keep them out of the headline
        # grid (they remain visible in the runs list and history).
        if record.evaluation.name.endswith("-smoke"):
            continue
        by_model.setdefault(record.model.name, []).append(record)

    tasks: set[str] = set()
    rows = []
    scored_by_model: dict[str, list[dict]] = {}
    for model in sorted(by_model):
        version = _current_version(by_model[model])
        cohort = [record for record in by_model[model] if record.version == version]
        cells = _cohort_cells(cohort)
        tasks.update(cells)
        rows.append({"model": model, "version": version, "archived": model in archived_models, "cells": cells})
        scored_by_model[model] = [cell for cell in cells.values() if cell["value"] is not None]

    task_list = sorted(tasks)
    leaderboard = []
    for row in rows:
        scored = scored_by_model[row["model"]]
        score = sum(cell["value"] for cell in scored) / len(scored) if scored else None
        stderr = _combined_stderr([cell["stderr"] for cell in scored]) if scored else None
        leaderboard.append(
            {
                "model": row["model"],
                "version": row["version"],
                "archived": row["archived"],
                "score": score,
                "stderr": stderr,
                "covered": len(scored),
                "total": len(task_list),
            }
        )
    leaderboard.sort(key=lambda entry: (entry["score"] is not None, entry["score"] or 0.0), reverse=True)
    return {"tasks": task_list, "rows": rows, "leaderboard": leaderboard}


def build_meta(records: list[EvalRunRecord], archived_models: frozenset[str] = frozenset()) -> dict:
    """Distinct filter values (models, evals, users, statuses) across all records, plus archived set."""
    eval_names = {r.evaluation.name for r in records}
    return {
        "models": sorted({r.model.name for r in records}),
        "evals": sorted(eval_names),
        "suites": eval_suites(eval_names),
        "users": sorted({r.user for r in records if r.user}),
        "statuses": sorted({r.status.value for r in records}),
        "archived_models": sorted(archived_models),
    }
