# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import raw TaskSpec observations and reduce their declared benchmark metrics."""

import json
import math
from collections import defaultdict

from finestore.reader import ReadView
from marin.evaluation.records import (
    BenchmarkMetadataRef,
    BenchmarkMetricRef,
    EvalRunRecord,
    EvalTaskRef,
    MetricKind,
    PromptFormat,
    ResolvedEvaluationSelection,
)
from marin.rollouts.evaluation import RUN_OBJECT
from marin.rollouts.evaluation_results import SCORED_OUTCOMES, evaluation_coverage, logical_samples, terminal_attempts
from taskcompendium.evaluation.spec import EvaluationProtocol, SummaryKind
from taskcompendium.identity import canonical_sha256

SUMMARY_REVISION = "2026.11.08.1"


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
    metrics, tasks = {}, []
    coverage = evaluation_coverage(reader)
    groups = defaultdict(list)
    for dataset, protocol in protocols.items():
        groups[protocol.benchmark].append(dataset)
    for name, datasets in groups.items():
        protocol = protocols[datasets[0]]
        for dataset in datasets[1:]:
            other = protocols[dataset]
            if other.model_dump(exclude={"expected_population"}) != protocol.model_dump(exclude={"expected_population"}):
                raise ValueError(f"Inconsistent summary protocols for benchmark {name}")
        log_likelihoods = {metric.name for metric in protocol.likelihood_metrics if metric.scoring == "log_likelihood"}
        if protocol.summary is SummaryKind.PERPLEXITY and not log_likelihoods:
            raise ValueError(f"Perplexity summary requires declared log-likelihood metrics for {name}")
        reductions = {
            metric.name: (
                SummaryKind.MEAN
                if protocol.summary is SummaryKind.PERPLEXITY and metric.name not in log_likelihoods
                else protocol.summary
            )
            for metric in protocol.metrics
        }
        rows = [row for row in attempts.values() if row["dataset"] in datasets]
        values = defaultdict(list)
        group_values = defaultdict(lambda: defaultdict(list))
        for attempt in rows:
            if attempt["outcome"] not in SCORED_OUTCOMES:
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
        metrics[name] = summary
        primary = next(metric for metric in protocol.metrics if metric.name == protocol.primary_metric)
        # Physical manifests include regional paths and conversion timing. The curation
        # identity binds source, compiler, grader and protocol independently of placement.
        identities = [manifests[dataset].get("curation_identity") for dataset in datasets]
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
                    n_benchmark=coverage[name].n_benchmark,
                    n_attempted=coverage[name].n_attempted,
                    requires_judge=protocol.requires_judge,
                    identity=(
                        canonical_sha256({"corpora": sorted(identities), "summary_revision": SUMMARY_REVISION})
                        if all(identities)
                        else None
                    ),
                    source={
                        "dataset": ",".join(manifests[dataset]["source_dataset"] for dataset in datasets),
                        "revision": ",".join(manifests[dataset]["source_revision"] for dataset in datasets),
                        "split": "eval",
                    },
                ),
            )
        )
    selection = record.evaluation.curated.selection
    if selection is not None and len(groups) != 1:
        raise ValueError("A benchmark selection requires exactly one declared benchmark")
    if len(groups) == 1:
        benchmark = next(iter(groups))
        datasets = groups[benchmark]
        protocol = protocols[datasets[0]]
        presets = {manifests[dataset].get("protocol_name") for dataset in datasets}
        if len(presets) == 1 and None not in presets:
            preset = next(iter(presets))
            form = {"chat_generation": PromptFormat.CHAT, "text_generation": PromptFormat.COMPLETION}.get(
                protocol.operation
            )
            published = ResolvedEvaluationSelection(
                benchmark=benchmark, protocol=preset, num_fewshot=protocol.num_fewshot, prompt_format=form
            )
            if selection is not None and (
                selection.benchmark,
                selection.protocol,
                selection.num_fewshot,
                selection.prompt_format,
            ) != (published.benchmark, published.protocol, published.num_fewshot, published.prompt_format):
                raise ValueError("Recorded evaluation selection differs from the published corpus protocol")
            selection = selection or published
    return record.model_copy(
        update={
            "metrics": metrics,
            "canonical_metrics": metrics,
            "coverage": coverage,
            "evaluation": record.evaluation.model_copy(
                update={
                    "tasks": tuple(tasks),
                    "curated": record.evaluation.curated.model_copy(update={"selection": selection}),
                }
            ),
        }
    )
