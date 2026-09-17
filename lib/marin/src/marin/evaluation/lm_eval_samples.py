# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Support legacy lm-eval exports and summarize native Evalchemy output.

lm-eval (and evalchemy, which drives it) writes one ``samples_<task>_<timestamp>.jsonl`` row per
evaluated question in its own native shape. This module owns Marin's compatibility export and rebuild
path for outputs produced without native FineStore writing. It preserves each source file it reads
so an archive can be rebuilt from itself. For native runs, Marin reads Evalchemy's normalized table
to calculate record coverage; Evalchemy owns conversion from the lm-eval row shape.

The same pass combines the evaluator's benchmark metadata with the per-sample rows to measure each
task's coverage and preserve its canonical metrics.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Protocol

import rigging.filesystem.factory as factory
from finestore.eval import (
    ARCHIVE_SAMPLES_TABLE,
    ARCHIVE_STEPS_TABLE,
    SAMPLES_PREFIX,
    SCHEMA_VERSION,
    SOURCES_PREFIX,
    Choice,
    EvalSample,
    EvaluationStore,
    Grading,
    Message,
    SampleKind,
    sample_from_archive_row,
)
from finestore.layout import ARCHIVE_FILE, DATA_DIR, HEAD_FILE, MANIFESTS_DIR, SCHEMAS_DIR, BlobTables
from finestore.migrations.m0001_manifest import LEGACY_SEAL_FILE
from finestore.reader import ReadView
from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.evaluation.eval_stats import SAMPLE_COUNT_METRIC
from marin.evaluation.evaluation_config import eval_task_directory
from marin.evaluation.metric_selection import base_metric, declared_metric, primary_filter
from marin.evaluation.records import EVALCHEMY_INFRASTRUCTURE_ERROR, BenchmarkMetadataRef, EvalTaskRef, TaskCoverage


class TaskDeclaration(Protocol):
    """Fields shared by launch-time and recorded task declarations."""

    name: str
    num_fewshot: int | None
    task_alias: str | None
    generation: bool
    unsafe_code: bool
    completion_only: bool


logger = logging.getLogger(__name__)

# The archive's own objects, which share the run's results root and must never be preserved into
# themselves. See :func:`run_artifacts`.
_ARCHIVE_DIRS = frozenset(
    {
        DATA_DIR,
        MANIFESTS_DIR,
        SCHEMAS_DIR,
        ARCHIVE_SAMPLES_TABLE,
        ARCHIVE_STEPS_TABLE,
        BlobTables.DESCRIPTORS,
    }
)
_ARCHIVE_MARKERS = frozenset({ARCHIVE_FILE, HEAD_FILE, LEGACY_SEAL_FILE})

_CONTENT_TYPES = {
    ".jsonl": "application/x-ndjson",
    ".json": "application/json",
    ".parquet": "application/vnd.apache.parquet",
    ".txt": "text/plain",
}

# A ``tempfile.mkdtemp`` directory name that ended up inside a results tree. See
# :func:`is_scratch_artifact`.
_SCRATCH_SEGMENT = re.compile(r"(?:^|/)tmp[a-z0-9_]{6,}/")
_INFRASTRUCTURE_ERROR_PREFIX = f"[{EVALCHEMY_INFRASTRUCTURE_ERROR}]"
EVALCHEMY_SOURCE_ROOT = PurePosixPath(prefix_join(SOURCES_PREFIX, "evalchemy"))
EVALCHEMY_NATIVE_SOURCE_DIR = "native"


def is_scratch_artifact(relative_path: str) -> bool:
    """Whether an artifact under a run came from the harness's scratch directory.

    evalchemy runs the harness in a ``tempfile`` working directory and copies the tree into the
    results path, so a retried evaluation leaves a second complete tree under a ``tmp<random>/``
    segment. The canonical copy produced the metrics on the run's record. The scratch copy may have
    different loglikelihoods, so it is preserved as a source blob and excluded from the sample table.
    """
    return _SCRATCH_SEGMENT.search(relative_path) is not None


# Live runs are normalized by Evalchemy. These conversion helpers remain for historical exports and
# for rebuilding an archive's table from preserved sources after damage or an interrupted migration.
# FineStore itself owns only the normalized schema and storage API.
_LM_EVAL_STRUCTURAL_KEYS = frozenset(
    {
        "doc",
        "doc_id",
        "target",
        "arguments",
        "resps",
        "filtered_resps",
        "filter",
        "filter_variants",
        "metrics",
        "schema_version",
        "task_name",
        "doc_hash",
        "prompt_hash",
        "target_hash",
    }
)


def _loglikelihood_pair(entry) -> tuple[float, bool] | None:
    """Unwrap a response entry to ``(loglikelihood, is_greedy)``."""
    if isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    if (
        isinstance(entry, list)
        and len(entry) == 2
        and isinstance(entry[0], int | float)
        and not isinstance(entry[0], bool)
        and isinstance(entry[1], bool)
    ):
        return float(entry[0]), entry[1]
    return None


def _is_multiple_choice(arguments, responses) -> bool:
    if not isinstance(arguments, list) or len(arguments) <= 1:
        return False
    if not isinstance(responses, list) or len(responses) != len(arguments):
        return False
    return all(_loglikelihood_pair(entry) is not None for entry in responses)


def _choice_labels(doc, count: int) -> list[str]:
    choices = doc.get("choices") if isinstance(doc, dict) else None
    labels = choices.get("label") if isinstance(choices, dict) else None
    if isinstance(labels, list) and len(labels) == count and all(isinstance(label, str) for label in labels):
        return labels
    return [chr(ord("A") + index) for index in range(count)]


def _resolve_target_choice(target, choices: list[Choice]) -> int | None:
    if isinstance(target, bool):
        return None
    if isinstance(target, int):
        return target if 0 <= target < len(choices) else None
    if isinstance(target, str):
        trimmed = target.strip()
        for index, choice in enumerate(choices):
            if choice.label == trimmed or choice.text.strip() == trimmed:
                return index
        if trimmed.isdigit():
            index = int(trimmed)
            return index if 0 <= index < len(choices) else None
    return None


def _parse_chat_messages(text: str) -> list[Message] | None:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    messages = []
    for item in parsed:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("role"), str)
            or not isinstance(item.get("content"), str)
        ):
            return None
        messages.append(Message(role=item["role"], content=item["content"]))
    return messages


def _sample_metrics(raw: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in raw.items()
        if key not in _LM_EVAL_STRUCTURAL_KEYS and not isinstance(value, bool) and isinstance(value, int | float)
    }


def _lm_eval_grading(
    metrics: dict[str, float], extraction_filter: str | None, primary_metric_name: str | None
) -> Grading | None:
    """The explicit grading for an lm-eval sample: its headline metric, filter, score, and pass flag.

    Per-sample rows name the extraction filter in ``filter``. Aggregate metric keys use a
    ``,<filter>`` suffix. This accepts both encodings.
    """
    picked = declared_metric(metrics, primary_metric_name)
    if picked is None:
        return None
    name, value = picked
    metric_filter = extraction_filter or (name.split(",", 1)[1] if "," in name else None)
    return Grading(
        method=f"lm-eval:{base_metric(name)}",
        metric=name,
        filter=metric_filter,
        score=value,
        passed=value >= 1.0,
    )


def sample_from_lm_eval(task: str, raw: dict, primary_metric_name: str | None = None) -> EvalSample:
    """Normalize one historical lm-eval ``--log_samples`` filter row."""
    arguments = raw.get("arguments")
    responses = raw.get("resps")
    doc = raw.get("doc")
    target = raw.get("target")
    metrics = _sample_metrics(raw)
    extraction_filter = raw.get("filter")
    grading = _lm_eval_grading(
        metrics,
        extraction_filter if isinstance(extraction_filter, str) else None,
        primary_metric_name,
    )
    common = {
        "task": task,
        "doc_id": str(raw.get("doc_id")),
        "metrics": metrics,
        "correct": grading.passed if grading is not None else None,
        "grading": grading,
        "target_text": target if isinstance(target, str) else json.dumps(target, ensure_ascii=False),
        "doc": doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False),
    }

    if isinstance(arguments, list) and isinstance(responses, list) and _is_multiple_choice(arguments, responses):
        labels = _choice_labels(doc, len(arguments))
        choices = []
        for index, entry in enumerate(arguments):
            text = entry[1] if isinstance(entry, list) and len(entry) > 1 and isinstance(entry[1], str) else ""
            pair = _loglikelihood_pair(responses[index])
            loglikelihood, is_greedy = pair if pair is not None else (None, None)
            choices.append(Choice(label=labels[index], text=text, loglikelihood=loglikelihood, is_greedy=is_greedy))
        scored = [
            (choice.loglikelihood, index) for index, choice in enumerate(choices) if choice.loglikelihood is not None
        ]
        context = arguments[0][0] if isinstance(arguments[0], list) and isinstance(arguments[0][0], str) else ""
        return EvalSample(
            kind=SampleKind.MULTIPLE_CHOICE,
            prompt_text=context,
            choices=choices,
            model_choice=max(scored)[1] if scored else None,
            target_choice=_resolve_target_choice(target, choices),
            **common,
        )

    prompt = ""
    if isinstance(arguments, list) and arguments:
        first = arguments[0]
        candidate = first[0] if isinstance(first, list) and first else first
        if isinstance(candidate, str):
            prompt = candidate
    output = ""
    if isinstance(responses, list) and responses:
        first = responses[0]
        if isinstance(first, list) and first and isinstance(first[0], str):
            output = first[0]
        elif isinstance(first, str):
            output = first
    filtered = raw.get("filtered_resps")
    if isinstance(filtered, list) and filtered:
        filtered = filtered[0]
    messages = _parse_chat_messages(prompt)
    return EvalSample(
        kind=SampleKind.GENERATION,
        prompt_text=None if messages else prompt,
        prompt_messages=messages,
        output=output,
        extracted=filtered if isinstance(filtered, str) else json.dumps(filtered, ensure_ascii=False),
        **common,
    )


def samples_from_lm_eval(task: str, raw: dict, primary_metric_name: str | None = None) -> list[EvalSample]:
    """Normalize a historical Evalchemy record into its extraction-filter rows."""
    variants = raw.get("filter_variants")
    if not isinstance(variants, list) or not variants:
        return [sample_from_lm_eval(task, raw, primary_metric_name)]

    samples = []
    for variant in variants:
        if not isinstance(variant, dict):
            raise ValueError("lm-eval filter_variants entries must be mappings")
        metrics = variant.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError("lm-eval filter_variants metrics must be a mapping")
        samples.append(
            sample_from_lm_eval(
                task,
                {
                    **raw,
                    **metrics,
                    "filter": variant.get("filter"),
                    "filtered_resps": variant.get("filtered_resps", []),
                },
                primary_metric_name,
            )
        )
    return samples


# --------------------------------------------------------------------------------------------------
# Coverage: what a task's own sample rows say about how much of it ran.
# --------------------------------------------------------------------------------------------------


def _document_extent(doc_ids: Iterable[str]) -> int | None:
    """How many documents a task enumerated, from the indices its rows carry, or None if unknowable.

    lm-eval indexes a task's documents ``0..N-1`` and writes a row for each one it reached, so the
    highest index present establishes ``N``. Non-numeric document identifiers leave coverage
    unknown.
    """
    highest = -1
    for doc_id in doc_ids:
        if not doc_id.isdigit():
            return None
        highest = max(highest, int(doc_id))
    return highest + 1 if highest >= 0 else None


def _is_infrastructure_error(sample: EvalSample) -> bool:
    return any(
        value is not None and _INFRASTRUCTURE_ERROR_PREFIX in value for value in (sample.output, sample.extracted)
    )


def _validated_coverage(coverage: TaskCoverage) -> TaskCoverage:
    if coverage.n_attempted is not None and coverage.n_scored > coverage.n_attempted:
        raise ValueError(f"scored count {coverage.n_scored} exceeds intended count {coverage.n_attempted}")
    if coverage.n_benchmark is None:
        return coverage
    if coverage.n_benchmark <= 0:
        raise ValueError(f"benchmark size must be positive, got {coverage.n_benchmark}")
    if coverage.n_attempted is None:
        raise ValueError("benchmark size requires an intended attempted count")
    if coverage.n_attempted > coverage.n_benchmark:
        raise ValueError(f"intended count {coverage.n_attempted} exceeds benchmark size {coverage.n_benchmark}")
    return coverage


def task_coverage_and_metrics(
    samples: Sequence[EvalSample], *, n_benchmark: int | None = None, n_attempted: int | None = None
) -> tuple[TaskCoverage, dict[str, float]]:
    """Compute one task's coverage and metrics recovered after request failures.

    Ungraded documents and failed requests are unscored. Empty model completions remain scored and
    count as unanswered. For tasks with several extraction filters, coverage uses the filter chosen
    by :func:`~marin.evaluation.metric_selection.primary_filter`. Recovered metrics retain every filter.
    """
    graded: dict[str, list[EvalSample]] = {}
    recovered_values: dict[str, list[float]] = {}
    recovered_doc_ids: set[str] = set()
    seen: set[str] = set()
    for sample in samples:
        seen.add(sample.doc_id)
        if sample.grading is not None:
            graded.setdefault(sample.doc_id, []).append(sample)
            if not _is_infrastructure_error(sample):
                recovered_doc_ids.add(sample.doc_id)
                for name, value in sample.metrics.items():
                    metric = name if "," in name or sample.grading.filter is None else f"{name},{sample.grading.filter}"
                    recovered_values.setdefault(metric, []).append(value)

    headline = primary_filter(
        {sample.grading.filter for rows in graded.values() for sample in rows if sample.grading.filter}
    )
    graded_samples = [
        next((sample for sample in rows if sample.grading.filter == headline), rows[0]) for rows in graded.values()
    ]
    infrastructure_errors = [sample for sample in graded_samples if _is_infrastructure_error(sample)]
    scored = [sample for sample in graded_samples if not _is_infrastructure_error(sample)]
    ungraded = len(seen) - len(graded_samples)
    # A pass/fail grade is the only one with a Bernoulli count behind it; a partial-credit score
    # (a rubric, an edit distance) has no numerator to record.
    binary = all(sample.grading.score in (0.0, 1.0) for sample in scored)
    errors = {"ungraded": ungraded} if ungraded else {}
    if infrastructure_errors:
        errors[EVALCHEMY_INFRASTRUCTURE_ERROR] = len(infrastructure_errors)
    extent = _document_extent(seen)
    if n_attempted is not None and extent is not None and extent > n_attempted:
        raise ValueError(f"sample document extent {extent} exceeds intended count {n_attempted}")
    coverage = TaskCoverage(
        n_benchmark=n_benchmark,
        n_attempted=n_attempted if n_attempted is not None else extent,
        n_scored=len(scored),
        n_correct=sum(1 for sample in scored if sample.correct) if binary and scored else None,
        n_unanswered=sum(1 for sample in scored if sample.kind is SampleKind.GENERATION and not sample.extracted),
        errors=errors,
    )
    recovered_metrics: dict[str, float] = {}
    if infrastructure_errors:
        recovered_metrics = {name: sum(entries) / len(entries) for name, entries in recovered_values.items()}
        if recovered_metrics:
            recovered_metrics[SAMPLE_COUNT_METRIC] = float(len(recovered_doc_ids))
    return coverage, recovered_metrics


def _task_keys(
    sources: Sequence[str], benchmark_metadata: Mapping[str, Mapping[str, BenchmarkMetadataRef]] | None = None
) -> dict[str, str]:
    """The ``metrics`` key each sample file's coverage belongs to.

    A run's records key metrics by the task-config directory (``<task_dir>/<model>/<file>``), and
    namespace them ``<task_dir>/<task>`` when one config evaluated several tasks -- see
    :meth:`~marin.evaluation.evalchemy.result.EvalchemyResult.task_metrics`. Coverage uses the same
    keys. Multiple sample files or multiple benchmark metadata entries trigger subtask names.
    """
    by_directory: dict[PurePosixPath, list[str]] = {}
    for relative in sources:
        by_directory.setdefault(PurePosixPath(relative).parent.parent, []).append(relative)
    keys: dict[str, str] = {}
    for directory, files in by_directory.items():
        for relative in files:
            task = _task_from_filename(PurePosixPath(relative).name, ".jsonl")
            grouped = len(files) > 1 or len((benchmark_metadata or {}).get(directory.name, {})) > 1
            keys[relative] = f"{directory.name}/{task}" if grouped else directory.name
    return keys


def _result_contracts(
    result_payloads: Mapping[str, bytes],
) -> tuple[dict[str, dict[str, BenchmarkMetadataRef]], dict[str, dict[str, dict[str, float]]]]:
    """Read evaluator-owned benchmark metadata and canonical results by task directory."""
    benchmarks: dict[str, dict[str, BenchmarkMetadataRef]] = {}
    canonical: dict[str, dict[str, dict[str, float]]] = {}
    for relative, payload in result_payloads.items():
        directory = PurePosixPath(relative).parent.parent.name
        result = json.loads(payload)
        raw_benchmarks = result.get("benchmark_metadata") or {}
        if isinstance(raw_benchmarks, Mapping):
            for leaf, raw in raw_benchmarks.items():
                if isinstance(leaf, str):
                    benchmarks.setdefault(directory, {})[leaf] = BenchmarkMetadataRef.model_validate(raw)
        raw_canonical = result.get("canonical_results") or {}
        if isinstance(raw_canonical, Mapping):
            for leaf, values in raw_canonical.items():
                if isinstance(leaf, str) and isinstance(values, Mapping):
                    canonical.setdefault(directory, {})[leaf] = {
                        str(name): float(value)
                        for name, value in values.items()
                        if isinstance(value, int | float) and not isinstance(value, bool)
                    }
    return benchmarks, canonical


def _task_key(directory: str, leaf: str, leaf_count: int) -> str:
    return f"{directory}/{leaf}" if leaf_count > 1 else directory


def _recorded_tasks(
    tasks: Sequence[TaskDeclaration], benchmarks: Mapping[str, Mapping[str, BenchmarkMetadataRef]]
) -> tuple[EvalTaskRef, ...]:
    recorded: list[EvalTaskRef] = []
    for task in tasks:
        directory = eval_task_directory(task.name, task.num_fewshot, task.task_alias)
        task_benchmarks = benchmarks.get(directory, {})
        if not task_benchmarks:
            recorded.append(
                EvalTaskRef(
                    name=task.name,
                    num_fewshot=task.num_fewshot,
                    task_alias=task.task_alias,
                    generation=task.generation,
                    unsafe_code=task.unsafe_code,
                    completion_only=task.completion_only,
                )
            )
            continue
        for benchmark in task_benchmarks.values():
            recorded.append(
                EvalTaskRef(
                    name=benchmark.task,
                    num_fewshot=task.num_fewshot,
                    task_alias=directory if len(task_benchmarks) > 1 else task.task_alias,
                    generation=task.generation,
                    unsafe_code=task.unsafe_code,
                    completion_only=task.completion_only,
                    benchmark=benchmark,
                )
            )
    return tuple(recorded)


@dataclass(frozen=True)
class _EvalchemyContractSummary:
    benchmarks: dict[str, dict[str, BenchmarkMetadataRef]]
    canonical_metrics: dict[str, dict[str, float]]
    tasks: tuple[EvalTaskRef, ...]
    coverage: dict[str, TaskCoverage]


def _evalchemy_contract_summary(
    result_payloads: Mapping[str, bytes], tasks: Sequence[TaskDeclaration]
) -> _EvalchemyContractSummary:
    benchmarks, canonical_results = _result_contracts(result_payloads)
    return _EvalchemyContractSummary(
        benchmarks=benchmarks,
        canonical_metrics={
            _task_key(directory, leaf, len(benchmarks.get(directory, values))): metrics
            for directory, values in canonical_results.items()
            for leaf, metrics in values.items()
        },
        tasks=_recorded_tasks(tasks, benchmarks),
        coverage={
            _task_key(directory, leaf, len(values)): TaskCoverage(
                n_benchmark=benchmark.n_benchmark,
                n_attempted=benchmark.n_attempted,
                n_scored=0,
            )
            for directory, values in benchmarks.items()
            for leaf, benchmark in values.items()
        },
    )


# --------------------------------------------------------------------------------------------------
# Export: normalize a run's sample files into its finestore archive, preserving the sources read.
# --------------------------------------------------------------------------------------------------


def _is_sample_source(relative: str) -> bool:
    """Whether a preserved artifact is an lm-eval per-sample jsonl this module can normalize."""
    return relative.rsplit("/", 1)[-1].startswith(SAMPLES_PREFIX) and relative.endswith(".jsonl")


def _is_result_source(relative: str) -> bool:
    name = PurePosixPath(relative).name
    return name.startswith("results_") and name.endswith(".json") and not is_scratch_artifact(relative)


def run_artifacts(out_path: str) -> list[str]:
    """List harness artifacts under a run's results root, relative to the root.

    The results tree and archive share a root, so archive objects are excluded. A filesystem listing
    includes dot-directories such as evalchemy's ``.resume/`` state.
    """
    fs, key = factory.url_to_fs(out_path)
    artifacts = []
    for path in fs.find(key):
        relative = path[len(key) :].strip("/")
        if relative in _ARCHIVE_MARKERS or relative.split("/", 1)[0] in _ARCHIVE_DIRS:
            continue
        artifacts.append(relative)
    return sorted(artifacts)


def _content_type(relative: str) -> str:
    """The media type recorded on a preserved artifact, from its extension."""
    for suffix, media_type in _CONTENT_TYPES.items():
        if relative.endswith(suffix):
            return media_type
    return "application/octet-stream"


@dataclass(frozen=True)
class EvaluationArchiveSummary:
    """Rows, coverage, and evaluator metadata read or produced for an archive."""

    samples: int
    coverage: dict[str, TaskCoverage] = field(default_factory=dict)
    """Per-task coverage keyed like the run's ``metrics`` (see :func:`_task_keys`)."""

    recovered_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    """Metrics rebuilt from successful samples for tasks with request failures."""

    canonical_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    tasks: tuple[EvalTaskRef, ...] = ()


def _write_sample_archive(
    out_path: str,
    writer_id: str,
    artifacts: Sequence[str],
    result_payloads: Mapping[str, bytes],
    benchmarks: Mapping[str, Mapping[str, BenchmarkMetadataRef]],
    initial_coverage: Mapping[str, TaskCoverage],
) -> tuple[int, dict[str, TaskCoverage], dict[str, dict[str, float]]]:
    """Preserve evaluator artifacts and normalize their per-sample rows."""
    root = StoragePath(out_path)
    keys = _task_keys(
        [relative for relative in artifacts if _is_sample_source(relative) and not is_scratch_artifact(relative)],
        benchmarks,
    )
    store = EvaluationStore.open(out_path, writer_id=writer_id)
    count = 0
    coverage = dict(initial_coverage)
    recovered_metrics: dict[str, dict[str, float]] = {}
    try:
        for relative in artifacts:
            payload = result_payloads.get(relative)
            if payload is None:
                payload = StoragePath(prefix_join(str(root), relative)).read_bytes()
            store.add_source_artifact(relative, payload, content_type=_content_type(relative))
            # One shard per artifact keeps a multi-hundred-megabyte results tree from buffering whole.
            store.flush()
            task_key = keys.get(relative)
            if task_key is None:
                continue
            directory = PurePosixPath(relative).parent.parent.name
            leaf = _task_from_filename(PurePosixPath(relative).name, ".jsonl")
            benchmark = benchmarks.get(directory, {}).get(leaf)
            primary_source = None
            if benchmark is not None:
                primary_source = next(
                    metric.source_name for metric in benchmark.metrics if metric.name == benchmark.primary_metric
                )
            samples = _add_lm_eval_rows(
                store,
                relative.rsplit("/", 1)[-1],
                payload,
                primary_metric_name=primary_source,
            )
            count += len(samples)
            if not samples:
                continue
            task_coverage_result, task_metrics = task_coverage_and_metrics(
                samples,
                n_benchmark=benchmark.n_benchmark if benchmark is not None else None,
                n_attempted=benchmark.n_attempted if benchmark is not None else None,
            )
            coverage[task_key] = task_coverage_result
            if task_coverage_result.errors.get(EVALCHEMY_INFRASTRUCTURE_ERROR):
                recovered_metrics[task_key] = task_metrics
        store.seal()
    finally:
        store.close()
    return count, coverage, recovered_metrics


def export_lm_eval_samples(
    out_path: str,
    *,
    tasks: Sequence[TaskDeclaration] = (),
    writer_id: str = "evalchemy",
) -> EvaluationArchiveSummary:
    """Normalize every lm-eval ``samples_*.jsonl`` under ``out_path`` into the run's finestore archive.

    Returns the rows written, coverage, and metrics recovered from successful requests. The archive
    preserves every harness artifact and remains usable after the results tree is pruned. Re-running
    an unchanged source produces rows that collapse on the primary key. Runs without sample sources
    remain untouched.

    Raises if the archive holds samples written under an older contract. Finestore cannot collapse
    those rows against the current schema; an explicit migration must replace them.
    """
    artifacts = run_artifacts(out_path)
    sources = [relative for relative in artifacts if _is_sample_source(relative) and not is_scratch_artifact(relative)]
    root = StoragePath(out_path)
    result_payloads = {
        relative: StoragePath(prefix_join(str(root), relative)).read_bytes()
        for relative in artifacts
        if _is_result_source(relative)
    }
    contracts = _evalchemy_contract_summary(result_payloads, tasks)
    if not sources:
        return EvaluationArchiveSummary(
            samples=0,
            coverage=contracts.coverage,
            canonical_metrics=contracts.canonical_metrics,
            tasks=contracts.tasks,
        )
    require_current_samples(out_path)
    count, coverage, recovered_metrics = _write_sample_archive(
        out_path,
        writer_id,
        artifacts,
        result_payloads,
        contracts.benchmarks,
        contracts.coverage,
    )
    return EvaluationArchiveSummary(
        samples=count,
        coverage=coverage,
        recovered_metrics=recovered_metrics,
        canonical_metrics=contracts.canonical_metrics,
        tasks=contracts.tasks,
    )


def _is_native_evalchemy_source(name: str) -> bool:
    path = PurePosixPath(name)
    if not path.is_relative_to(EVALCHEMY_SOURCE_ROOT):
        return False
    relative = path.relative_to(EVALCHEMY_SOURCE_ROOT)
    return len(relative.parts) == 3 and relative.parts[1] == EVALCHEMY_NATIVE_SOURCE_DIR


@dataclass(frozen=True)
class NativeEvalchemyArtifacts:
    """Native Evalchemy source artifacts needed by Marin's archive readers."""

    sample_sources: tuple[str, ...]
    result_payloads: dict[str, bytes]


def read_native_evalchemy_artifacts(out_path: str) -> NativeEvalchemyArtifacts:
    """Discover native Evalchemy sources and read their results payloads."""
    reader = ReadView(out_path)
    source_names = tuple(
        sorted(
            key[0]
            for key in reader.keys(BlobTables.DESCRIPTORS)
            if isinstance(key[0], str) and _is_native_evalchemy_source(key[0])
        )
    )
    payloads: dict[str, bytes] = {}
    for name in source_names:
        if not _is_result_source(name):
            continue
        payload = reader.read_blob(name)
        if payload is None:
            raise FileNotFoundError(f"archive at {out_path!r} lists source blob {name!r} but cannot read it")
        payloads[name] = payload
    return NativeEvalchemyArtifacts(
        sample_sources=tuple(name for name in source_names if _is_sample_source(name)),
        result_payloads=payloads,
    )


def summarize_native_eval_samples(
    out_path: str,
    *,
    tasks: Sequence[TaskDeclaration] = (),
) -> EvaluationArchiveSummary:
    """Read coverage and evaluator metadata from native Evalchemy FineStore output."""
    reader = ReadView(out_path)
    artifacts = read_native_evalchemy_artifacts(out_path)
    if not artifacts.result_payloads:
        raise FileNotFoundError(f"archive at {out_path!r} preserves no native Evalchemy results artifacts")

    contracts = _evalchemy_contract_summary(artifacts.result_payloads, tasks)
    coverage = dict(contracts.coverage)

    table = reader.scan(ARCHIVE_SAMPLES_TABLE)
    if table is None and artifacts.sample_sources:
        raise FileNotFoundError(f"archive at {out_path!r} contains no normalized evaluation samples")
    by_task: dict[str, list[EvalSample]] = {}
    if table is not None:
        for row in table.to_pylist(maps_as_pydicts="strict"):
            sample = sample_from_archive_row(row)
            by_task.setdefault(sample.task, []).append(sample)

    task_keys = _task_keys(artifacts.sample_sources, contracts.benchmarks)
    sample_count = 0
    recovered_metrics: dict[str, dict[str, float]] = {}
    for name in artifacts.sample_sources:
        task_key = task_keys[name]
        samples = by_task.get(task_key, [])
        sample_count += len(samples)
        directory = PurePosixPath(name).parent.parent.name
        leaf = _task_from_filename(PurePosixPath(name).name, ".jsonl")
        benchmark = contracts.benchmarks.get(directory, {}).get(leaf)
        task_coverage_result, task_metrics = task_coverage_and_metrics(
            samples,
            n_benchmark=benchmark.n_benchmark if benchmark is not None else None,
            n_attempted=benchmark.n_attempted if benchmark is not None else None,
        )
        coverage[task_key] = task_coverage_result
        if task_coverage_result.errors.get(EVALCHEMY_INFRASTRUCTURE_ERROR):
            recovered_metrics[task_key] = task_metrics

    return EvaluationArchiveSummary(
        samples=sample_count,
        coverage=coverage,
        recovered_metrics=recovered_metrics,
        canonical_metrics=contracts.canonical_metrics,
        tasks=contracts.tasks,
    )


def require_current_samples(out_path: str) -> None:
    """Raise if the archive's samples predate the current contract.

    A widened primary key reads as null on older shards, leaving both old and current rows in the
    table. A migration must remove the old rows before writing the current contract.
    """
    stored_version = ReadView(out_path).schema_version(ARCHIVE_SAMPLES_TABLE)
    if stored_version is None or stored_version == SCHEMA_VERSION:
        return
    raise ValueError(
        f"{out_path} holds samples written under schema v{stored_version}; this writer is at "
        f"v{SCHEMA_VERSION}. Migrate the archive with experiments.evaluation.migrations.samples_v4."
    )


def _task_from_filename(name: str, suffix: str) -> str:
    # samples_<task>_<timestamp>.<suffix>; the timestamp contains no underscore.
    return name[len(SAMPLES_PREFIX) : -len(suffix)].rsplit("_", 1)[0]


def _add_lm_eval_rows(
    store: EvaluationStore, filename: str, payload: bytes, *, primary_metric_name: str | None = None
) -> list[EvalSample]:
    """Normalize one ``samples_*.jsonl`` payload into ``store``; return the samples added.

    Physical LF bytes delimit records. Literal U+2028/U+2029 characters remain inside JSON strings.
    """
    rows = [json.loads(line) for line in payload.decode().split("\n") if line.strip()]
    if not rows:
        logger.warning("samples file %s is empty; skipping archive export", filename)
        return []
    task = _task_from_filename(filename, ".jsonl")
    samples = [sample for raw in rows for sample in samples_from_lm_eval(task, raw, primary_metric_name)]
    for sample in samples:
        store.add_sample(sample)
    return samples


def preserved_sample_sources(out_path: str) -> tuple[str, ...]:
    """The ``sources/`` blob names holding this archive's ``samples_*.jsonl`` inputs.

    Empty for an archive written before source preservation, whose rebuild must come from the
    surrounding results tree instead.
    """
    return tuple(
        sorted(
            key[0]
            for key in ReadView(out_path).keys(BlobTables.DESCRIPTORS)
            if isinstance(key[0], str)
            and key[0].startswith(f"{SOURCES_PREFIX}/")
            and key[0].endswith(".jsonl")
            and SAMPLES_PREFIX in key[0]
        )
    )


def rebuild_lm_eval_samples(out_path: str, *, tasks: Sequence[EvalTaskRef] = (), writer_id: str = "rebuild") -> int:
    """Rebuild a run's ``samples`` table from the source artifacts preserved inside its archive.

    The inputs are the ``sources/`` blobs written by :func:`export_lm_eval_samples`, so this repairs
    an archive whose surrounding results tree has been pruned, or one an export left half written.
    Returns the number of samples written. Rows the archive already holds are reproduced exactly and
    collapse against themselves, so nothing is deleted; an archive at an older contract raises, and
    one preserving no sources raises too.
    """
    reader = ReadView(out_path)
    names = preserved_sample_sources(out_path)
    if not names:
        raise FileNotFoundError(f"archive at {out_path!r} preserves no sample sources to rebuild from")
    require_current_samples(out_path)
    store = EvaluationStore.open(out_path, writer_id=writer_id)
    task_configs = {eval_task_directory(task.name, task.num_fewshot, task.task_alias): task for task in tasks}
    count = 0
    try:
        for name in names:
            if is_scratch_artifact(name):
                continue
            payload = reader.read_blob(name)
            if payload is None:
                raise FileNotFoundError(f"archive at {out_path!r} lists source blob {name!r} but cannot read it")
            directory = PurePosixPath(name).parent.parent.name
            sample_task = _task_from_filename(PurePosixPath(name).name, ".jsonl")
            task = task_configs.get(directory)
            benchmark = task.benchmark if task is not None else None
            if benchmark is None:
                benchmark = next(
                    (
                        candidate.benchmark
                        for candidate in tasks
                        if candidate.benchmark is not None and candidate.benchmark.task == sample_task
                    ),
                    None,
                )
            primary_source = None
            if benchmark is not None:
                primary_source = next(
                    metric.source_name for metric in benchmark.metrics if metric.name == benchmark.primary_metric
                )
            count += len(
                _add_lm_eval_rows(
                    store,
                    name.rsplit("/", 1)[-1],
                    payload,
                    primary_metric_name=primary_source,
                )
            )
        store.seal()
    finally:
        store.close()
    return count
