# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read logical evaluation trials, coverage and benchmark scores from a single archive snapshot."""

import json
import math
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from finestore.eval import ARCHIVE_SAMPLES_TABLE
from finestore.reader import ReadView
from taskcompendium.evaluation.selection import SELECTIONS_TABLE
from taskcompendium.evaluation.spec import EvaluationProtocol, SummaryKind
from taskcompendium.identity import canonical_sha256

from evalrecords.records import (
    BenchmarkMetadataRef,
    BenchmarkMetricRef,
    EvalRunRecord,
    EvalTaskRef,
    MetricKind,
    PromptFormat,
    ResolvedEvaluationSelection,
    TaskCoverage,
)
from evalrecords.rollout_attempts import ATTEMPTS_TABLE, RUN_OBJECT

SCORED_OUTCOMES = frozenset({"graded", "submission_failure"})
SUMMARY_REVISION = "2026.11.08.1"
# rolloutengine's stop reason for a rollout that reached a token limit.
LENGTH_STOP_REASON = "length"


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


def _reduce(values: list[tuple[str, float, float]], kind: SummaryKind) -> tuple[float, float | None]:
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
    if kind is SummaryKind.PERPLEXITY:
        value = math.exp(-mean)
        return value, None if stderr is None else value * stderr
    return mean, stderr


def _length_stops(prefix: str, stop_reasons: list[str | None]) -> dict[str, float]:
    """Count the scored attempts that stopped at a token limit, when every one of them recorded a stop reason."""
    if not stop_reasons or None in stop_reasons:
        return {}
    count = stop_reasons.count(LENGTH_STOP_REASON)
    return {f"{prefix}length_stop_count": count, f"{prefix}length_stop_fraction": count / len(stop_reasons)}


def _benchmark_summary(
    protocol: EvaluationProtocol,
    datasets: list[str],
    attempts: dict[tuple[str, str, int], dict],
    samples: dict[tuple[str, str], dict],
    results_path: str,
) -> dict[str, float]:
    """Reduce one benchmark's declared metrics, with standard errors, per-group scores and length stops."""
    log_likelihoods = {metric.name for metric in protocol.likelihood_metrics if metric.scoring == "log_likelihood"}
    if protocol.summary is SummaryKind.PERPLEXITY and not log_likelihoods:
        raise ValueError(f"Perplexity summary requires declared log-likelihood metrics for {protocol.benchmark}")
    reductions = {
        metric.name: (
            SummaryKind.MEAN
            if protocol.summary is SummaryKind.PERPLEXITY and metric.name not in log_likelihoods
            else protocol.summary
        )
        for metric in protocol.metrics
    }
    values = defaultdict(list)
    group_values = defaultdict(lambda: defaultdict(list))
    stop_reasons: list[str | None] = []
    group_stop_reasons: defaultdict[str, list[str | None]] = defaultdict(list)
    for attempt in attempts.values():
        if attempt["dataset"] not in datasets or attempt["outcome"] not in SCORED_OUTCOMES:
            continue
        sample = samples[(f"{attempt['dataset']}/{attempt['trial']}/{attempt['retry']}", attempt["task_id"])]
        observed = dict(sample["metrics"] or {})
        if protocol.primary_metric not in observed:
            score = (sample["grading"] or {}).get("score")
            if score is None:
                continue
            observed[protocol.primary_metric] = score
        weights = {}
        if protocol.summary is SummaryKind.WEIGHTED_MEAN:
            grade = json.loads(sample["grading"]["detail"])
            weights = grade["detail"].get("metric_weights", {})
        sample_groups = json.loads(sample["doc"] or "{}").get("groups", {})
        # Result roots written before attempts recorded stop reasons have no such column.
        stop_reason = attempt.get("stop_reason")
        stop_reasons.append(stop_reason)
        for group, label in sample_groups.items():
            group_stop_reasons[f"{group}={label}"].append(stop_reason)
        for metric in protocol.metrics:
            if metric.name not in observed:
                continue
            value = observed[metric.name]
            if not math.isfinite(value):
                raise ValueError(f"Non-finite metric {metric.name} in {results_path}")
            weight = weights.get(metric.name, 1) if protocol.summary is SummaryKind.WEIGHTED_MEAN else 1
            if weight <= 0:
                raise ValueError(f"Non-positive metric weight for {metric.name}")
            datum = (attempt["dataset"] + ":" + attempt["source_row"], value, weight)
            values[metric.name].append(datum)
            for group, label in sample_groups.items():
                group_values[f"{group}={label}"][metric.name].append(datum)
    summary = {}
    for metric, observations in values.items():
        summary[metric], stderr = _reduce(observations, reductions[metric])
        if stderr is not None:
            summary[metric + "_stderr"] = stderr
    if protocol.summary is SummaryKind.CLASSIFICATION and values:
        attempted = summary.get("accuracy", 0) + summary.get("incorrect", 0)
        accuracy = summary.get("accuracy", 0)
        summary["accuracy_given_attempted"] = accuracy / attempted if attempted else 0
        summary["f1"] = 2 * accuracy / (1 + attempted)
    for group, group_metrics in group_values.items():
        for metric, observations in group_metrics.items():
            summary[f"{group}/{metric}"] = _reduce(observations, reductions[metric])[0]
    summary.update(_length_stops("", stop_reasons))
    for group, reasons in group_stop_reasons.items():
        summary.update(_length_stops(f"{group}/", reasons))
    return summary


def _task_ref(name: str, protocol: EvaluationProtocol, manifests: list[dict], coverage: TaskCoverage) -> EvalTaskRef:
    """Describe one benchmark's metrics, population and corpus identity for comparison."""
    primary = next(metric for metric in protocol.metrics if metric.name == protocol.primary_metric)
    # Physical manifests include regional paths and conversion timing. The curation
    # identity binds source, compiler, grader and protocol independently of placement.
    identities = [manifest.get("curation_identity") for manifest in manifests]
    known = sorted(identity for identity in identities if identity is not None)
    return EvalTaskRef(
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
            n_benchmark=coverage.n_benchmark,
            n_attempted=coverage.n_attempted,
            requires_judge=protocol.requires_judge,
            identity=(
                canonical_sha256({"corpora": known, "summary_revision": SUMMARY_REVISION})
                if len(known) == len(identities)
                else None
            ),
            source={
                "dataset": ",".join(manifest["source_dataset"] for manifest in manifests),
                "revision": ",".join(manifest["source_revision"] for manifest in manifests),
                "split": "eval",
            },
        ),
    )


def _published_selection(
    selection: ResolvedEvaluationSelection | None,
    protocols: dict[str, EvaluationProtocol],
    manifests: dict[str, dict],
    groups: dict[str, list[str]],
) -> ResolvedEvaluationSelection | None:
    """Check a recorded selection against its single published benchmark, or recover it from the corpus."""
    if selection is not None and len(groups) != 1:
        raise ValueError("A benchmark selection requires exactly one declared benchmark")
    if len(groups) != 1:
        return selection
    benchmark, datasets = next(iter(groups.items()))
    protocol = protocols[datasets[0]]
    presets = {manifests[dataset].get("protocol_name") for dataset in datasets}
    if len(presets) != 1 or None in presets:
        return selection
    form = {"chat_generation": PromptFormat.CHAT, "text_generation": PromptFormat.COMPLETION}.get(protocol.operation)
    published = ResolvedEvaluationSelection(
        benchmark=benchmark, protocol=next(iter(presets)), num_fewshot=protocol.num_fewshot, prompt_format=form
    )
    if selection is not None and (
        selection.benchmark,
        selection.protocol,
        selection.num_fewshot,
        selection.prompt_format,
    ) != (published.benchmark, published.protocol, published.num_fewshot, published.prompt_format):
        raise ValueError("Recorded evaluation selection differs from the published corpus protocol")
    return selection or published


@dataclass(frozen=True)
class BenchmarkScores:
    """One benchmark's summary protocol, dataset snapshots, reduced metrics and trial coverage in an archive."""

    protocol: EvaluationProtocol
    datasets: tuple[str, ...]
    metrics: dict[str, float]
    coverage: TaskCoverage


def benchmark_scores(reader: ReadView) -> dict[str, BenchmarkScores]:
    """Reduce each benchmark's declared metrics over the latest finished attempt per logical trial.

    Each metric is reduced with its protocol's summary kind. The archive must hold a run descriptor.
    """
    payload = reader.read_blob(RUN_OBJECT)
    assert payload is not None
    protocols = {
        entry["dataset"]: EvaluationProtocol.model_validate(entry["protocol"])
        for entry in json.loads(payload)["datasets"]
    }
    attempts = terminal_attempts(reader)
    samples = {(row["trial_id"], row["doc_id"]): row for row in logical_samples(reader)}
    coverage = evaluation_coverage(reader)
    groups = defaultdict(list)
    for dataset, protocol in protocols.items():
        groups[protocol.benchmark].append(dataset)
    scores = {}
    for name, datasets in groups.items():
        protocol = protocols[datasets[0]]
        for dataset in datasets[1:]:
            other = protocols[dataset]
            if other.model_dump(exclude={"expected_population"}) != protocol.model_dump(exclude={"expected_population"}):
                raise ValueError(f"Inconsistent summary protocols for benchmark {name}")
        scores[name] = BenchmarkScores(
            protocol,
            tuple(datasets),
            _benchmark_summary(protocol, datasets, attempts, samples, reader.root),
            coverage[name],
        )
    return scores


def score_curated_record(record: EvalRunRecord) -> EvalRunRecord:
    """Derive benchmark scores from one FineStore snapshot, leaving the producer record unchanged.

    Reduces each benchmark's declared metrics with its protocol's summary kind over the latest
    finished attempt per logical trial. Non-curated records are returned unchanged.
    """
    if record.evaluation.curated is None:
        return record
    reader = ReadView(record.results_path)
    payload = reader.read_blob(RUN_OBJECT)
    if payload is None:
        return record
    manifests = {entry["dataset"]: entry["manifest"] for entry in json.loads(payload)["datasets"]}
    scores = benchmark_scores(reader)
    metrics = {name: score.metrics for name, score in scores.items()}
    tasks = [
        _task_ref(name, score.protocol, [manifests[dataset] for dataset in score.datasets], score.coverage)
        for name, score in scores.items()
    ]
    selection = _published_selection(
        record.evaluation.curated.selection,
        {dataset: score.protocol for score in scores.values() for dataset in score.datasets},
        manifests,
        {name: list(score.datasets) for name, score in scores.items()},
    )
    return record.model_copy(
        update={
            "metrics": metrics,
            "canonical_metrics": metrics,
            "coverage": {name: score.coverage for name, score in scores.items()},
            "evaluation": record.evaluation.model_copy(
                update={
                    "tasks": tuple(tasks),
                    "curated": record.evaluation.curated.model_copy(update={"selection": selection}),
                }
            ),
        }
    )
