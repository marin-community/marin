# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert lm-eval's ``--log_samples`` output into the :mod:`finestore.eval` contract.

lm-eval (and evalchemy, which drives it) writes one ``samples_<task>_<timestamp>.jsonl`` row per
evaluated question in its own native shape. This module normalizes those rows into
:class:`~finestore.eval.EvalSample` and exports them into the run's finestore archive, preserving
each source file it read so the archive can be rebuilt from itself. ``finestore.eval`` owns the
contract and the archive tables; knowledge of lm-eval's row shape lives here.

The same pass measures each task's coverage. lm-eval publishes no attempted-item count in its
aggregate results, but its per-sample rows carry the document indices it enumerated, so the rows are
the only place a run can establish what it set out to grade -- which is what lets the statistics
engine identify an interval instead of reporting sampling error alone.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

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
    base_metric,
    primary_filter,
    primary_metric,
)
from finestore.layout import ARCHIVE_FILE, BLOBS_TABLE, SEALED_MARKER
from finestore.reader import CompositeReader
from rigging.filesystem import StoragePath, factory, prefix_join

from marin.evaluation.records import TaskCoverage

logger = logging.getLogger(__name__)

# Filename prefix of the aggregate results json lm-eval writes beside its per-sample jsonl.
RESULTS_PREFIX = "results_"

# The archive's own objects, which share the run's results root and must never be preserved into
# themselves. See :func:`run_artifacts`.
_ARCHIVE_TABLES = frozenset({ARCHIVE_SAMPLES_TABLE, ARCHIVE_STEPS_TABLE, BLOBS_TABLE})
_ARCHIVE_MARKERS = frozenset({ARCHIVE_FILE, SEALED_MARKER})

_CONTENT_TYPES = {
    ".jsonl": "application/x-ndjson",
    ".json": "application/json",
    ".parquet": "application/vnd.apache.parquet",
    ".txt": "text/plain",
}

# A ``tempfile.mkdtemp`` directory name that ended up inside a results tree. See
# :func:`is_scratch_artifact`.
_SCRATCH_SEGMENT = re.compile(r"(?:^|/)tmp[a-z0-9_]{6,}/")


def is_scratch_artifact(relative_path: str) -> bool:
    """Whether an artifact under a run came from the harness's scratch directory.

    evalchemy runs the harness in a ``tempfile`` working directory and copies the tree into the
    results path, so a retried evaluation leaves a second complete tree under a ``tmp<random>/``
    segment. Both trees are real, and their loglikelihoods differ because inference is not
    deterministic, but only the canonical copy produced the metrics on the run's record. Indexing
    both would put two different rows on one primary key; the scratch copy is still preserved as a
    source blob, so nothing is discarded by leaving it out of the table.
    """
    return _SCRATCH_SEGMENT.search(relative_path) is not None


# --------------------------------------------------------------------------------------------------
# lm-eval adapter: normalize one --log_samples row into the contract.
# --------------------------------------------------------------------------------------------------

# lm-eval per-sample fields that are structural rather than metric values. ``schema_version`` is
# lm-eval's own row-format stamp and ``metrics`` its list of metric *names*, so both would otherwise
# be mistaken for scores by the numeric sweep below.
_LM_EVAL_STRUCTURAL_KEYS = frozenset(
    {
        "doc",
        "doc_id",
        "target",
        "arguments",
        "resps",
        "filtered_resps",
        "filter",
        "metrics",
        "schema_version",
        "task_name",
        "doc_hash",
        "prompt_hash",
        "target_hash",
    }
)


def _loglikelihood_pair(entry) -> tuple[float, bool] | None:
    """Unwrap a response entry to ``(loglikelihood, is_greedy)``, tolerating a singleton wrapper."""
    if isinstance(entry, list) and len(entry) == 1:
        entry = entry[0]
    if (
        isinstance(entry, list)
        and len(entry) == 2
        and isinstance(entry[0], (int, float))
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
    # arc-style docs carry {"choices": {"label": [...], "text": [...]}}; other tasks (mmlu,
    # hellaswag) store choices as a plain list, which gets the A/B/C default.
    choices = doc.get("choices") if isinstance(doc, dict) else None
    labels = choices.get("label") if isinstance(choices, dict) else None
    if isinstance(labels, list) and len(labels) == count and all(isinstance(label, str) for label in labels):
        return labels
    return [chr(ord("A") + i) for i in range(count)]


def _resolve_target_choice(target, choices: list[Choice]) -> int | None:
    if isinstance(target, bool):
        return None
    if isinstance(target, int):
        return target if 0 <= target < len(choices) else None
    if isinstance(target, str):
        trimmed = target.strip()
        for i, choice in enumerate(choices):
            if choice.label == trimmed or choice.text.strip() == trimmed:
                return i
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
    metrics: dict[str, float] = {}
    for key, value in raw.items():
        if key in _LM_EVAL_STRUCTURAL_KEYS or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def _correct(metrics: dict[str, float]) -> bool | None:
    picked = primary_metric(metrics)
    if picked is None:
        return None
    return picked[1] >= 1.0


def _lm_eval_grading(metrics: dict[str, float], extraction_filter: str | None) -> Grading | None:
    """The explicit grading for an lm-eval sample: its headline metric, filter, score, and pass flag.

    A per-sample row scores one extraction filter and names it in its own ``filter`` field, unlike an
    aggregate result key which carries the filter as a ``,<filter>`` suffix. Both spellings are
    accepted so the grading records the filter whichever shape the row uses.
    """
    picked = primary_metric(metrics)
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


def sample_from_lm_eval(task: str, raw: dict) -> EvalSample:
    """Normalize one lm-eval ``--log_samples`` row into an :class:`EvalSample`.

    Multiple-choice rows carry one ``arguments`` entry per choice (``[context, continuation]``,
    context identical across entries) and per-choice ``[loglikelihood, is_greedy]`` responses; the
    model's pick is the loglikelihood argmax and the gold index is resolved from ``target`` (an
    index, a label, or the choice text). Generation rows carry a single prompt -- raw text, or a
    JSON-encoded chat message list when the chat API served the request.

    A task that applies several extraction filters emits one row per (document, filter), each with
    its own responses and score. The row's filter is recorded on the grading and forms part of the
    sample's archive identity, so the variants coexist instead of overwriting one another.
    """
    arguments = raw.get("arguments")
    responses = raw.get("resps")
    doc = raw.get("doc")
    target = raw.get("target")
    metrics = _sample_metrics(raw)
    extraction_filter = raw.get("filter")
    common = {
        "task": task,
        "doc_id": str(raw.get("doc_id")),
        "metrics": metrics,
        "correct": _correct(metrics),
        "grading": _lm_eval_grading(metrics, extraction_filter if isinstance(extraction_filter, str) else None),
        "target_text": target if isinstance(target, str) else json.dumps(target, ensure_ascii=False),
        "doc": doc if isinstance(doc, str) else json.dumps(doc, ensure_ascii=False),
    }

    if isinstance(arguments, list) and isinstance(responses, list) and _is_multiple_choice(arguments, responses):
        labels = _choice_labels(doc, len(arguments))
        choices = []
        for i, entry in enumerate(arguments):
            text = entry[1] if isinstance(entry, list) and len(entry) > 1 and isinstance(entry[1], str) else ""
            pair = _loglikelihood_pair(responses[i])
            loglikelihood, is_greedy = pair if pair is not None else (None, None)
            choices.append(Choice(label=labels[i], text=text, loglikelihood=loglikelihood, is_greedy=is_greedy))
        scored = [(choice.loglikelihood, i) for i, choice in enumerate(choices) if choice.loglikelihood is not None]
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


# --------------------------------------------------------------------------------------------------
# Coverage: what a task's own sample rows say about how much of it ran.
# --------------------------------------------------------------------------------------------------


def _document_extent(doc_ids: Iterable[str]) -> int | None:
    """How many documents a task enumerated, from the indices its rows carry, or None if unknowable.

    lm-eval indexes a task's documents ``0..N-1`` and writes a row for each one it reached, so the
    highest index present establishes ``N``. A task whose ids are not indices (another harness's
    identifiers) establishes nothing, and coverage stays unknown rather than being guessed at.
    """
    highest = -1
    for doc_id in doc_ids:
        if not doc_id.isdigit():
            return None
        highest = max(highest, int(doc_id))
    return highest + 1 if highest >= 0 else None


def task_coverage(samples: Sequence[EvalSample]) -> TaskCoverage:
    """One task's coverage, read from the samples it produced.

    A document that yielded no grading is counted as attempted but unscored, and one whose grader
    extracted no answer is counted as unanswered -- it scored, but on nothing the model actually
    supplied. A task scoring each document under several extraction filters holds one sample per
    (document, filter); the tallies come from the filter :func:`~finestore.eval.primary_filter`
    picks, which is the one the run's headline metric is also reported under.
    """
    graded: dict[str, list[EvalSample]] = {}
    seen: set[str] = set()
    for sample in samples:
        seen.add(sample.doc_id)
        if sample.grading is not None:
            graded.setdefault(sample.doc_id, []).append(sample)

    headline = primary_filter(
        {sample.grading.filter for rows in graded.values() for sample in rows if sample.grading.filter}
    )
    scored = [
        next((sample for sample in rows if sample.grading.filter == headline), rows[0]) for rows in graded.values()
    ]
    ungraded = len(seen) - len(scored)
    # A pass/fail grade is the only one with a Bernoulli count behind it; a partial-credit score
    # (a rubric, an edit distance) has no numerator to record.
    binary = all(sample.grading.score in (0.0, 1.0) for sample in scored)
    return TaskCoverage(
        n_attempted=_document_extent(seen),
        n_scored=len(scored),
        n_correct=sum(1 for sample in scored if sample.correct) if binary and scored else None,
        n_unanswered=sum(1 for sample in scored if sample.kind is SampleKind.GENERATION and not sample.extracted),
        errors={"ungraded": ungraded} if ungraded else {},
    )


def _task_key(relative: str, group: bool) -> str:
    """The ``metrics`` key a sample file's coverage belongs to.

    A run's records key metrics by the task-config directory (``<task_dir>/<model>/<file>``), and
    namespace them ``<task_dir>/<task>`` when one config evaluated several tasks -- see
    :meth:`~marin.evaluation.evalchemy.result.EvalchemyResult.task_metrics`. Coverage keys the same
    way so a reader can line the two up.
    """
    path = PurePosixPath(relative)
    task_dir = path.parent.parent.name
    return f"{task_dir}/{_task_from_filename(path.name, '.jsonl')}" if group else task_dir


# --------------------------------------------------------------------------------------------------
# Export: normalize a run's sample files into its finestore archive, preserving the sources read.
# --------------------------------------------------------------------------------------------------


def _is_sample_source(relative: str) -> bool:
    """Whether a preserved artifact is an lm-eval per-sample jsonl this module can normalize."""
    return relative.rsplit("/", 1)[-1].startswith(SAMPLES_PREFIX) and relative.endswith(".jsonl")


def run_artifacts(out_path: str) -> list[str]:
    """Every object under a run's results root that is not part of its own archive, relative to it.

    A run's results tree and its archive share a root, so "preserve everything" has to exclude the
    archive or a re-export would fold it into itself. Listing rather than globbing is what reaches
    the harness's dot-directories: evalchemy's resume state lives under ``.resume/``, which a ``*``
    glob does not match.
    """
    fs, key = factory.url_to_fs(out_path)
    artifacts = []
    for path in fs.find(key):
        relative = path[len(key) :].strip("/")
        if relative in _ARCHIVE_MARKERS or relative.split("/", 1)[0] in _ARCHIVE_TABLES:
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
    """What a run's sample export wrote, and what those samples say about the run's completeness."""

    samples: int
    coverage: dict[str, TaskCoverage] = field(default_factory=dict)
    """Per-task coverage keyed like the run's ``metrics`` (see :func:`_task_key`)."""


def export_lm_eval_samples(out_path: str, *, writer_id: str = "evalchemy") -> SampleExport:
    """Normalize every lm-eval ``samples_*.jsonl`` under ``out_path`` into the run's finestore archive.

    Returns the rows written and the coverage they establish. Every artifact the harness left in the
    results tree is preserved inside the archive, not only the files this reads, so the archive stands
    alone if the tree is pruned. Re-running is idempotent: an unchanged source produces identical rows,
    which collapse on the primary key without conflict. A run with no sample source is left untouched.

    Raises if the archive holds samples written under an older contract. Those rows cannot collapse
    against rows written under the current one, and finestore does not delete, so bringing them
    forward is a migration rather than an export.
    """
    root = StoragePath(out_path)
    artifacts = run_artifacts(out_path)
    sources = [relative for relative in artifacts if _is_sample_source(relative) and not is_scratch_artifact(relative)]
    if not any(_is_sample_source(relative) for relative in artifacts):
        # With no source there is nothing to write, and nothing that could stand in for what is
        # already stored, so a run evaluated by another mechanism keeps the archive it has.
        return SampleExport(samples=0)
    require_current_samples(out_path)
    # A task config that evaluated several tasks wrote several sample files under one directory, and
    # its metrics are namespaced to match.
    grouped = {relative for relative in sources if _dir_source_count(sources, relative) > 1}
    store = EvaluationStore.open(out_path, writer_id=writer_id)
    count = 0
    coverage: dict[str, TaskCoverage] = {}
    try:
        for relative in artifacts:
            payload = StoragePath(prefix_join(str(root), relative)).read_bytes()
            store.add_source_artifact(relative, payload, content_type=_content_type(relative))
            # One shard per artifact keeps a multi-hundred-megabyte results tree from buffering whole.
            store.flush()
            if relative not in sources:
                continue
            samples = _add_lm_eval_rows(store, relative.rsplit("/", 1)[-1], payload)
            count += len(samples)
            if samples:
                coverage[_task_key(relative, relative in grouped)] = task_coverage(samples)
        store.seal()
    finally:
        store.close()
    return SampleExport(samples=count, coverage=coverage)


def _dir_source_count(sources: Sequence[str], relative: str) -> int:
    """How many sample files share ``relative``'s task-config directory."""
    parent = PurePosixPath(relative).parent.parent
    return sum(1 for other in sources if PurePosixPath(other).parent.parent == parent)


def require_current_samples(out_path: str) -> None:
    """Raise if the archive's samples predate the current contract.

    A widened primary key reads as null on the older shards, so those rows would survive beside the
    ones a writer adds now rather than collapsing against them. Removing them is a migration's job,
    not a writer's.
    """
    stored_version = CompositeReader(out_path).schema_version(ARCHIVE_SAMPLES_TABLE)
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

    Splits only on the physical LF delimiter, so a literal U+2028/U+2029 inside a JSON string stays
    part of its record rather than being taken for a record boundary.
    """
    rows = [json.loads(line) for line in payload.decode().split("\n") if line.strip()]
    if not rows:
        logger.warning("samples file %s is empty; skipping archive export", filename)
        return []
    task = _task_from_filename(filename, ".jsonl")
    samples = [sample_from_lm_eval(task, raw) for raw in rows]
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
            for key in CompositeReader(out_path).keys(BLOBS_TABLE)
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
    reader = CompositeReader(out_path)
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
