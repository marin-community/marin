# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Preserve and summarize lm-eval's native output in a :mod:`finestore.eval` archive.

lm-eval (and evalchemy, which drives it) writes one ``samples_<task>_<timestamp>.jsonl`` row per
evaluated question in its own native shape. This module exports those rows into the run's finestore
archive, preserving each source file it read so the archive can be rebuilt from itself. The
published :mod:`finestore.eval` module owns the contract, archive tables, and row normalization so
Evalchemy can perform the same write natively.

The same pass measures each task's coverage from the document indices in its per-sample rows.
lm-eval's aggregate results omit attempted-item counts. The sample indices establish the intended
item count needed by the statistics engine.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import rigging.filesystem.factory as factory
from finestore.eval import (
    ARCHIVE_SAMPLES_TABLE,
    ARCHIVE_STEPS_TABLE,
    SAMPLES_PREFIX,
    SCHEMA_VERSION,
    SOURCES_PREFIX,
    EvalSample,
    EvaluationStore,
    SampleKind,
    primary_filter,
    samples_from_lm_eval,
)
from finestore.layout import ARCHIVE_FILE, DATA_DIR, HEAD_FILE, MANIFESTS_DIR, SCHEMAS_DIR, BlobTables
from finestore.migrations.m0001_manifest import LEGACY_SEAL_FILE
from finestore.reader import ReadView
from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.evaluation.eval_stats import SAMPLE_COUNT_METRIC
from marin.evaluation.records import EVALCHEMY_INFRASTRUCTURE_ERROR, TaskCoverage

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
_NATIVE_EVALCHEMY_SOURCE_PREFIX = f"{SOURCES_PREFIX}/evalchemy/"


def is_scratch_artifact(relative_path: str) -> bool:
    """Whether an artifact under a run came from the harness's scratch directory.

    evalchemy runs the harness in a ``tempfile`` working directory and copies the tree into the
    results path, so a retried evaluation leaves a second complete tree under a ``tmp<random>/``
    segment. The canonical copy produced the metrics on the run's record. The scratch copy may have
    different loglikelihoods, so it is preserved as a source blob and excluded from the sample table.
    """
    return _SCRATCH_SEGMENT.search(relative_path) is not None


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


def task_coverage_and_metrics(samples: Sequence[EvalSample]) -> tuple[TaskCoverage, dict[str, float]]:
    """Compute one task's coverage and metrics recovered after request failures.

    Ungraded documents and failed requests are unscored. Empty model completions remain scored and
    count as unanswered. For tasks with several extraction filters, coverage uses the filter chosen
    by :func:`~finestore.eval.primary_filter`. Recovered metrics retain every filter.
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
    coverage = TaskCoverage(
        n_attempted=_document_extent(seen),
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


def _task_keys(sources: Sequence[str]) -> dict[str, str]:
    """The ``metrics`` key each sample file's coverage belongs to.

    A run's records key metrics by the task-config directory (``<task_dir>/<model>/<file>``), and
    namespace them ``<task_dir>/<task>`` when one config evaluated several tasks -- see
    :meth:`~marin.evaluation.evalchemy.result.EvalchemyResult.task_metrics`. Coverage uses the same
    keys. The full source set determines whether a task-config directory needs subtask names.
    """
    by_directory: dict[PurePosixPath, list[str]] = {}
    for relative in sources:
        by_directory.setdefault(PurePosixPath(relative).parent.parent, []).append(relative)
    keys: dict[str, str] = {}
    for directory, files in by_directory.items():
        for relative in files:
            task = _task_from_filename(PurePosixPath(relative).name, ".jsonl")
            keys[relative] = f"{directory.name}/{task}" if len(files) > 1 else directory.name
    return keys


# --------------------------------------------------------------------------------------------------
# Export: normalize a run's sample files into its finestore archive, preserving the sources read.
# --------------------------------------------------------------------------------------------------


def _is_sample_source(relative: str) -> bool:
    """Whether a preserved artifact is an lm-eval per-sample jsonl this module can normalize."""
    return relative.rsplit("/", 1)[-1].startswith(SAMPLES_PREFIX) and relative.endswith(".jsonl")


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
class SampleExport:
    """Rows, coverage, and recovered metrics produced by a sample export."""

    samples: int
    coverage: dict[str, TaskCoverage] = field(default_factory=dict)
    """Per-task coverage keyed like the run's ``metrics`` (see :func:`_task_keys`)."""

    recovered_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    """Metrics rebuilt from successful samples for tasks with request failures."""


def export_lm_eval_samples(out_path: str, *, writer_id: str = "evalchemy") -> SampleExport:
    """Normalize every lm-eval ``samples_*.jsonl`` under ``out_path`` into the run's finestore archive.

    Returns the rows written, coverage, and metrics recovered from successful requests. The archive
    preserves every harness artifact and remains usable after the results tree is pruned. Re-running
    an unchanged source produces rows that collapse on the primary key. Runs without sample sources
    remain untouched.

    Raises if the archive holds samples written under an older contract. Finestore cannot collapse
    those rows against the current schema; an explicit migration must replace them.
    """
    root = StoragePath(out_path)
    artifacts = run_artifacts(out_path)
    sources = [relative for relative in artifacts if _is_sample_source(relative) and not is_scratch_artifact(relative)]
    if not any(_is_sample_source(relative) for relative in artifacts):
        # With no source there is nothing to write, and nothing that could stand in for what is
        # already stored, so a run evaluated by another mechanism keeps the archive it has.
        return SampleExport(samples=0)
    require_current_samples(out_path)
    keys = _task_keys(sources)
    store = EvaluationStore.open(out_path, writer_id=writer_id)
    count = 0
    coverage: dict[str, TaskCoverage] = {}
    recovered_metrics: dict[str, dict[str, float]] = {}
    try:
        for relative in artifacts:
            payload = StoragePath(prefix_join(str(root), relative)).read_bytes()
            store.add_source_artifact(relative, payload, content_type=_content_type(relative))
            # One shard per artifact keeps a multi-hundred-megabyte results tree from buffering whole.
            store.flush()
            if relative not in keys:
                continue
            samples = _add_lm_eval_rows(store, relative.rsplit("/", 1)[-1], payload)
            count += len(samples)
            if samples:
                task_key = keys[relative]
                task_coverage_result, task_metrics = task_coverage_and_metrics(samples)
                coverage[task_key] = task_coverage_result
                if task_coverage_result.errors.get(EVALCHEMY_INFRASTRUCTURE_ERROR):
                    recovered_metrics[task_key] = task_metrics
        store.seal()
    finally:
        store.close()
    return SampleExport(samples=count, coverage=coverage, recovered_metrics=recovered_metrics)


def summarize_native_eval_samples(out_path: str) -> SampleExport:
    """Summarize Evalchemy's native FineStore sources without rewriting its sample table."""
    reader = ReadView(out_path)
    sources = tuple(
        name
        for name in preserved_sample_sources(out_path)
        if name.startswith(_NATIVE_EVALCHEMY_SOURCE_PREFIX) and "/native/" in name
    )
    if not sources:
        raise FileNotFoundError(f"archive at {out_path!r} preserves no native Evalchemy sample sources")

    keys = _task_keys(sources)
    count = 0
    coverage: dict[str, TaskCoverage] = {}
    recovered_metrics: dict[str, dict[str, float]] = {}
    for name in sources:
        payload = reader.read_blob(name)
        if payload is None:
            raise FileNotFoundError(f"archive at {out_path!r} lists source blob {name!r} but cannot read it")
        samples = _lm_eval_samples(name.rsplit("/", 1)[-1], payload)
        count += len(samples)
        if not samples:
            continue
        task_key = keys[name]
        task_coverage_result, task_metrics = task_coverage_and_metrics(samples)
        coverage[task_key] = task_coverage_result
        if task_coverage_result.errors.get(EVALCHEMY_INFRASTRUCTURE_ERROR):
            recovered_metrics[task_key] = task_metrics
    return SampleExport(samples=count, coverage=coverage, recovered_metrics=recovered_metrics)


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


def _add_lm_eval_rows(store: EvaluationStore, filename: str, payload: bytes) -> list[EvalSample]:
    """Normalize one ``samples_*.jsonl`` payload into ``store``; return the samples added.

    Physical LF bytes delimit records. Literal U+2028/U+2029 characters remain inside JSON strings.
    """
    samples = _lm_eval_samples(filename, payload)
    for sample in samples:
        store.add_sample(sample)
    return samples


def _lm_eval_samples(filename: str, payload: bytes) -> list[EvalSample]:
    """Normalize one evaluator-native JSONL payload without writing it."""
    rows = [json.loads(line) for line in payload.decode().split("\n") if line.strip()]
    if not rows:
        logger.warning("samples file %s is empty; skipping archive export", filename)
        return []
    task = _task_from_filename(filename, ".jsonl")
    return [sample for raw in rows for sample in samples_from_lm_eval(task, raw)]


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


def rebuild_lm_eval_samples(out_path: str, *, writer_id: str = "rebuild") -> int:
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
    count = 0
    try:
        for name in names:
            if is_scratch_artifact(name):
                continue
            payload = reader.read_blob(name)
            if payload is None:
                raise FileNotFoundError(f"archive at {out_path!r} lists source blob {name!r} but cannot read it")
            count += len(_add_lm_eval_rows(store, name.rsplit("/", 1)[-1], payload))
        store.seal()
    finally:
        store.close()
    return count
