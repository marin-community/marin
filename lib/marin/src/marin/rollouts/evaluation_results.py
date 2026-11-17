# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read logical evaluation trials and coverage from a single archive snapshot."""

import json
from collections import defaultdict
from collections.abc import Iterator

from finestore.eval import ARCHIVE_SAMPLES_TABLE
from finestore.reader import ReadView
from taskcompendium.evaluation.selection import SELECTIONS_TABLE
from taskcompendium.evaluation.spec import EvaluationProtocol

from marin.evaluation.records import TaskCoverage
from marin.rollouts.evaluation import ATTEMPTS_TABLE, RUN_OBJECT

SCORED_OUTCOMES = frozenset({"graded", "submission_failure"})


def terminal_attempts(reader: ReadView) -> dict[tuple[str, str, int], dict]:
    """Choose the latest finished attempt for each logical trial, including failed grades."""
    latest = {}
    for row in reader.iter_rows(ATTEMPTS_TABLE):
        if row["phase"] != "finished":
            continue
        key = (row["dataset"], row["source_row"], row["trial"])
        if key not in latest or row["retry"] > latest[key]["retry"]:
            latest[key] = row
    return latest


def logical_samples(reader: ReadView, *, task: str | None = None) -> Iterator[dict]:
    """Yield the sample from the latest finished attempt of each logical trial."""
    trials = {
        (f"{row['dataset']}/{row['trial']}/{row['retry']}", row["task_id"]) for row in terminal_attempts(reader).values()
    }
    where = [("task", "==", task)] if task is not None else None
    for row in reader.iter_rows(ARCHIVE_SAMPLES_TABLE, where=where):
        if (row["trial_id"], row["doc_id"]) in trials:
            yield row


def evaluation_coverage(reader: ReadView) -> dict[str, TaskCoverage]:
    """Count declared, selected and scored trials without enlarging denominators on retry."""
    payload = reader.read_blob(RUN_OBJECT)
    assert payload is not None
    run = json.loads(payload)
    attempts = terminal_attempts(reader)
    samples = {(row["trial_id"], row["doc_id"]): row for row in logical_samples(reader)}
    selected = defaultdict(set)
    for row in reader.iter_rows(SELECTIONS_TABLE):
        if row["status"] == "selected":
            selected[row["dataset"]].add(row["source_row"])
    groups = defaultdict(list)
    for entry in run["datasets"]:
        protocol = EvaluationProtocol.model_validate(entry["protocol"])
        groups[protocol.benchmark].append((entry, protocol))
    coverage = {}
    for name, entries in groups.items():
        datasets = {entry["dataset"] for entry, _ in entries}
        rows = [row for row in attempts.values() if row["dataset"] in datasets]
        primary = next(metric for metric in entries[0][1].metrics if metric.name == entries[0][1].primary_metric)
        errors = defaultdict(int)
        scored, correct = 0, 0
        for attempt in rows:
            if attempt["outcome"] not in SCORED_OUTCOMES:
                errors[attempt["outcome"] or "unfinished"] += 1
                continue
            sample = samples[(f"{attempt['dataset']}/{attempt['trial']}/{attempt['retry']}", attempt["task_id"])]
            observed = dict(sample["metrics"] or {})
            value = observed.get(primary.name, (sample["grading"] or {}).get("score"))
            if value is None:
                errors["ungraded"] += 1
                continue
            scored += 1
            correct += value == 1
        attempted = sum(len(selected[entry["dataset"]]) * protocol.trials.count for entry, protocol in entries)
        errors["unfinished"] += max(0, attempted - len(rows))
        coverage[name] = TaskCoverage(
            n_benchmark=sum(entry["manifest"]["rows"] * protocol.trials.count for entry, protocol in entries),
            n_attempted=attempted,
            n_scored=scored,
            n_correct=correct if primary.kind == "binary" else None,
            errors={name: count for name, count in errors.items() if count},
        )
    return coverage
