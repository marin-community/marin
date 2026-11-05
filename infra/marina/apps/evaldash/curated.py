# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import raw TaskSpec observations and reduce their declared benchmark metrics."""

import json
import math
from collections import defaultdict
from collections.abc import Iterator

from finestore.eval import ARCHIVE_SAMPLES_TABLE
from finestore.reader import ReadView
from marin.evaluation.records import (
    BenchmarkMetadataRef,
    BenchmarkMetricRef,
    EvalRunRecord,
    EvalTaskRef,
    MetricKind,
    TaskCoverage,
)
from marin.rollouts.evaluation import ATTEMPTS_TABLE, RUN_OBJECT
from taskcompendium.evaluation.selection import SELECTIONS_TABLE
from taskcompendium.evaluation.spec import EvaluationProtocol, SummaryKind
from taskcompendium.identity import canonical_sha256

SUMMARY_REVISION = "2026.11.05.1"
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
    """Expose one physical sample per logical trial to summaries and the sample browser."""
    trials = {
        (f"{row['dataset']}/{row['trial']}/{row['retry']}", row["task_id"]) for row in terminal_attempts(reader).values()
    }
    where = [("task", "==", task)] if task is not None else None
    for row in reader.iter_rows(ARCHIVE_SAMPLES_TABLE, where=where):
        if (row["trial_id"], row["doc_id"]) in trials:
            yield row


def _reduce(values: list[tuple[str, float, float]], kind: SummaryKind, metric: str) -> tuple[float, float | None]:
    total_weight = sum(weight for _, _, weight in values)
    mean = sum(value * weight for _, value, weight in values) / total_weight
    by_source = defaultdict(float)
    for source, value, weight in values:
        by_source[source] += weight * (value - mean)
    count = len(by_source)
    stderr = (
        math.sqrt(count / (count - 1) * sum(value**2 for value in by_source.values())) / total_weight
        if count > 1
        else None
    )
    if kind is SummaryKind.PERPLEXITY and "perplexity" in metric:
        value = math.exp(-mean)
        return value, None if stderr is None else value * stderr
    return mean, stderr


def import_curated_record(record: EvalRunRecord) -> EvalRunRecord:
    """Derive dashboard scores from one FineStore snapshot, leaving the producer record unchanged."""
    if record.evaluation.curated is None:
        return record
    reader = ReadView(record.results_path)
    payload = reader.read_blob(RUN_OBJECT)
    if payload is None:
        return record
    run = json.loads(payload)
    protocols = {entry["dataset"]: EvaluationProtocol.model_validate(entry["protocol"]) for entry in run["datasets"]}
    manifests = {entry["dataset"]: entry["manifest"] for entry in run["datasets"]}
    attempts = terminal_attempts(reader)
    samples = {(row["trial_id"], row["doc_id"]): row for row in logical_samples(reader)}
    selected = defaultdict(set)
    for row in reader.iter_rows(SELECTIONS_TABLE):
        if row["status"] == "selected":
            selected[row["dataset"]].add(row["source_row"])
    metrics, coverage, tasks = {}, {}, []
    groups = defaultdict(list)
    for dataset, protocol in protocols.items():
        groups[protocol.benchmark].append(dataset)
    for name, datasets in groups.items():
        protocol = protocols[datasets[0]]
        for dataset in datasets[1:]:
            other = protocols[dataset]
            if other.model_dump(exclude={"expected_population"}) != protocol.model_dump(exclude={"expected_population"}):
                raise ValueError(f"Inconsistent summary protocols for benchmark {name}")
        rows = [row for row in attempts.values() if row["dataset"] in datasets]
        values = defaultdict(list)
        errors = defaultdict(int)
        scored, correct = 0, 0
        group_values = defaultdict(lambda: defaultdict(list))
        for attempt in rows:
            if attempt["outcome"] not in SCORED_OUTCOMES:
                errors[attempt["outcome"] or "unfinished"] += 1
                continue
            sample = samples[(f"{attempt['dataset']}/{attempt['trial']}/{attempt['retry']}", attempt["task_id"])]
            observed = dict(sample["metrics"] or {})
            if protocol.primary_metric not in observed:
                score = (sample["grading"] or {}).get("score")
                if score is None:
                    errors["ungraded"] += 1
                    continue
                observed[protocol.primary_metric] = score
            grade = json.loads((sample["grading"] or {}).get("detail") or "{}")
            detail = grade.get("detail") or {}
            weights = detail.get("weights", {})
            sample_groups = json.loads(sample["doc"] or "{}").get("groups", {})
            scored += 1
            correct += observed[protocol.primary_metric] == 1
            for metric in protocol.metrics:
                if metric.name not in observed:
                    continue
                value = observed[metric.name]
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite metric {metric.name} in {record.results_path}")
                weight = weights.get(metric.name, 1) if protocol.summary is SummaryKind.WEIGHTED_MEAN else 1
                if weight <= 0:
                    raise ValueError(f"Non-positive metric weight for {metric.name}")
                datum = (attempt["dataset"] + ":" + attempt["source_row"], value, weight)
                values[metric.name].append(datum)
                for group, label in sample_groups.items():
                    group_values[f"{group}={label}"][metric.name].append(datum)
        summary = {}
        for metric, observations in values.items():
            summary[metric], stderr = _reduce(observations, protocol.summary, metric)
            if stderr is not None:
                summary[metric + "_stderr"] = stderr
        if protocol.summary is SummaryKind.CLASSIFICATION and values:
            attempted = summary.get("accuracy", 0) + summary.get("incorrect", 0)
            accuracy = summary.get("accuracy", 0)
            summary["accuracy_given_attempted"] = accuracy / attempted if attempted else 0
            summary["f1"] = 2 * accuracy / (1 + attempted)
        for group, group_metrics in group_values.items():
            for metric, observations in group_metrics.items():
                summary[f"{group}/{metric}"] = _reduce(observations, protocol.summary, metric)[0]
        metrics[name] = summary
        n_attempted = sum(len(selected[dataset]) * protocols[dataset].trials.count for dataset in datasets)
        n_benchmark = sum(manifests[dataset]["rows"] * protocols[dataset].trials.count for dataset in datasets)
        errors["unfinished"] += max(0, n_attempted - len(rows))
        primary = next(metric for metric in protocol.metrics if metric.name == protocol.primary_metric)
        coverage[name] = TaskCoverage(
            n_benchmark=n_benchmark,
            n_attempted=n_attempted,
            n_scored=scored,
            n_correct=correct if primary.kind == "binary" else None,
            errors={name: count for name, count in errors.items() if count},
        )
        tasks.append(
            EvalTaskRef(
                name=name,
                num_fewshot=protocol.num_fewshot,
                benchmark=BenchmarkMetadataRef(
                    schema_version=1,
                    task=name,
                    primary_metric=protocol.primary_metric,
                    metric_kind=MetricKind(primary.kind),
                    metrics=tuple(
                        BenchmarkMetricRef(
                            name=metric.name,
                            source_name=metric.name,
                            kind=MetricKind(metric.kind),
                            higher_is_better=metric.higher_is_better,
                        )
                        for metric in protocol.metrics
                    ),
                    n_benchmark=n_benchmark,
                    n_attempted=n_attempted,
                    requires_judge=protocol.requires_judge,
                    identity=canonical_sha256({"datasets": sorted(datasets), "summary_revision": SUMMARY_REVISION}),
                    source={
                        "dataset": ",".join(manifests[dataset]["source_dataset"] for dataset in datasets),
                        "revision": ",".join(manifests[dataset]["source_revision"] for dataset in datasets),
                        "split": "eval",
                    },
                ),
            )
        )
    return record.model_copy(
        update={
            "metrics": metrics,
            "canonical_metrics": metrics,
            "coverage": coverage,
            "evaluation": record.evaluation.model_copy(update={"tasks": tuple(tasks)}),
        }
    )
