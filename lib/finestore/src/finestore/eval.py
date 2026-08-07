# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The per-sample contract: one typed record per evaluated question, with its writer and reader.

Every eval mechanism normalizes its native per-question output into :class:`EvalSample` at export
time, and every consumer (the dashboard's sample browser, ad-hoc parquet analysis) reads that
contract back through this module -- the producer and consumer share one schema definition, so a
format change is a change to this file and its round-trip test, never a guessing game in a viewer.

A sample is either ``multiple_choice`` (the model scored a fixed choice list by loglikelihood;
``choices`` carries the per-choice scores with the model's pick and the gold index resolved at
export time) or ``generation`` (the model produced free text; ``output`` is the raw completion and
``extracted`` the post-filter answer). Prompts are either raw text (``prompt_text``, the completions
API) or a chat message list (``prompt_messages``); exactly one is set. ``correct`` is resolved once
at export time from the primary metric.

A run's durable storage is one finestore archive rooted at its results directory (see the archive
adapter below). The archive pins each table's column types from the pydantic model (via
``finestore.arrow_schema``), so the writer is ``model_dump`` and the reader is ``model_validate``,
with nested fields stored as parquet structs/lists and the dynamic-keyed ``metrics`` as a
``map<string,double>`` that cannot drift its type across shards. ``schema_version`` is a model field,
so it rides in every row for future evolution. The legacy per-(sub)task
``samples_<task>_<timestamp>.parquet`` layout is still read by the dashboard's migration fallback.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable
from enum import StrEnum

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel
from rigging.filesystem import StoragePath, prefix_join

from finestore.layout import BLOBS_TABLE
from finestore.reader import CompositeReader
from finestore.schema import arrow_schema
from finestore.store import DataStore, replace_table

logger = logging.getLogger(__name__)

# 3: metrics moved from an inferred struct to a pinned ``map<string,double>`` and the unused
# ``exchange_uri`` was dropped (finestore now pins both eval tables' schemas from these models).
# 4: the lm-eval extraction filter joined the samples merge key. A task that scores one document
# under several filters (gsm8k under strict-match and flexible-extract) emits one sample per filter,
# and each is a distinct row rather than one overwriting the other.
SCHEMA_VERSION = 4

SAMPLES_PREFIX = "samples_"
SAMPLES_SUFFIX = ".parquet"

# Headline metric for a task, matched on the base metric name (lm-eval's ",<filter>" suffix
# stripped). The first base name present wins; the alphabetically-first metric is the fallback.
# acc_norm outranks acc: where both are emitted (arc, hellaswag, piqa, openbookqa) the
# length-normalized score is the conventional headline. "accuracy" is the evalchemy chat-native
# benchmarks' key (MATH500).
PRIMARY_METRIC_PRIORITY = ("exact_match", "accuracy", "acc_norm", "acc", "pass@1")

# Tie-break among same-base metrics that differ only in lm-eval filter. flexible-extract outranks
# strict-match: chat models solve gsm8k-style problems but rarely emit the strict "#### N" format.
FILTER_PRIORITY = ("flexible-extract",)

# A ``tempfile.mkdtemp`` directory name that ended up inside a results tree. See
# :func:`is_scratch_artifact`.
_SCRATCH_SEGMENT = re.compile(r"(?:^|/)tmp[a-z0-9_]{6,}/")


class SampleKind(StrEnum):
    """How the model was queried for this sample."""

    MULTIPLE_CHOICE = "multiple_choice"
    GENERATION = "generation"
    AGENTIC = "agentic"


class Message(BaseModel):
    """One chat turn of a messages-based prompt."""

    role: str
    content: str


class Choice(BaseModel):
    """One scored option of a multiple-choice sample."""

    label: str
    text: str
    loglikelihood: float | None = None
    is_greedy: bool | None = None


class Grading(BaseModel):
    """How one prediction was scored, made explicit so the UI can show *why* a sample is (in)correct.

    ``method`` names the grader: ``lm-eval:<metric>`` for a harness metric, ``harbor:<verifier>`` for a
    Harbor trial's verifier, ``judge:<model>`` for an LLM judge. ``metric`` is the full headline key
    (with lm-eval's ``,<filter>`` suffix), ``filter`` the extraction filter that produced it, ``score``
    its value, and ``passed`` whether it cleared the pass threshold. ``detail`` carries the grader's raw
    output verbatim (the verifier/judge JSON) as the escape hatch for anything the fields do not.
    """

    method: str
    metric: str | None = None
    filter: str | None = None
    score: float | None = None
    passed: bool | None = None
    detail: str = "{}"


class EvalSample(BaseModel):
    """One evaluated question: the prompt, the model's answer, the gold answer, and its scores.

    ``prompt_text`` and ``prompt_messages`` are mutually exclusive; ``choices``/``model_choice``/
    ``target_choice`` are set for ``multiple_choice`` samples, ``output``/``extracted`` for
    ``generation`` samples, and ``trajectory_uri`` for ``agentic`` (Harbor) samples. ``grading`` makes
    the scoring decision explicit for the UI. ``doc`` is the source dataset row as a JSON string, kept
    verbatim as the escape hatch for anything the normalized fields do not carry.

    Rows stay bounded: the one unbounded payload, an agentic trajectory, is stored as a sibling blob
    and referenced here by ``trajectory_uri``, so the columnar reader never has to materialize it to
    page the light columns.
    """

    schema_version: int = SCHEMA_VERSION
    task: str
    doc_id: str
    kind: SampleKind
    prompt_text: str | None = None
    prompt_messages: list[Message] | None = None
    choices: list[Choice] | None = None
    model_choice: int | None = None
    target_choice: int | None = None
    output: str | None = None
    extracted: str | None = None
    target_text: str | None = None
    trajectory_uri: str | None = None
    grading: Grading | None = None
    metrics: dict[str, float] = {}
    correct: bool | None = None
    doc: str = "{}"


def primary_filter(filters: Iterable[str]) -> str | None:
    """Pick the extraction filter a reader should show by default, or ``None`` if there are none.

    A task that scores each document under several filters stores one sample per filter, so a viewer
    showing every row would list each question more than once. ``FILTER_PRIORITY`` decides which one
    leads, and the alphabetically-first name is the fallback — the same ranking
    :func:`primary_metric` applies to the aggregate keys, so the samples a reader sees by default and
    the headline score it compares them against come from the same filter.
    """
    names = {name for name in filters if name}
    if not names:
        return None
    for preferred in FILTER_PRIORITY:
        if preferred in names:
            return preferred
    return min(names)


def is_scratch_artifact(relative_path: str) -> bool:
    """Whether an artifact under a run came from the harness's scratch directory.

    evalchemy runs the harness in a ``tempfile`` working directory and copies the tree into the
    results path, so a retried evaluation leaves a second complete tree under a ``tmp<random>/``
    segment. Both trees are real, and their loglikelihoods differ because inference is not
    deterministic, but only the canonical copy produced the metrics on the run's record. Indexing
    both would put two different rows on one merge key; the scratch copy is still preserved as a
    source blob, so nothing is discarded by leaving it out of the table.
    """
    return _SCRATCH_SEGMENT.search(relative_path) is not None


def base_metric(name: str) -> str:
    """A metric key without lm-eval's ``,<filter>`` suffix (``exact_match,none`` -> ``exact_match``)."""
    return name.split(",", 1)[0]


def primary_metric(metrics: dict[str, float]) -> tuple[str, float] | None:
    """Pick the headline ``(key, value)`` from a metric dict, or None if it is empty.

    ``*_stderr`` never headlines; among the rest ``PRIMARY_METRIC_PRIORITY`` wins by base name,
    ``FILTER_PRIORITY`` breaks ties between same-base filters, and the alphabetically-first key is
    the final fallback at each step.
    """
    candidates = {name: value for name, value in metrics.items() if not base_metric(name).endswith("_stderr")}
    if not candidates:
        return None
    for preferred in PRIMARY_METRIC_PRIORITY:
        matches = {name: value for name, value in candidates.items() if base_metric(name) == preferred}
        if not matches:
            continue
        for metric_filter in FILTER_PRIORITY:
            for name, value in matches.items():
                if name.endswith(f",{metric_filter}"):
                    return name, value
        name = min(matches)
        return name, matches[name]
    name = min(candidates)
    return name, candidates[name]


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
# Writer: the parquet schema is the pydantic model, so ``EvalSample.model_validate(row)`` on any
# ``to_pylist`` row is the reader.
# --------------------------------------------------------------------------------------------------


def write_sample_parquet(fs, dest: str, samples: Iterable[EvalSample]) -> None:
    """Write samples as one parquet at ``dest`` on ``fs``."""
    rows = [sample.model_dump() for sample in samples]
    # Parquet cannot store a zero-field struct; the model default restores {} on read.
    if all(not row["metrics"] for row in rows):
        for row in rows:
            del row["metrics"]
    table = pa.Table.from_pylist(rows)
    with fs.open(dest, "wb") as handle:
        pq.write_table(table, handle)


def _task_from_filename(name: str, suffix: str) -> str:
    # samples_<task>_<timestamp>.<suffix>; the timestamp contains no underscore.
    return name[len(SAMPLES_PREFIX) : -len(suffix)].rsplit("_", 1)[0]


# --------------------------------------------------------------------------------------------------
# finestore archive adapter: an eval run's durable output is one finestore archive rooted at its
# results directory, with a ``samples`` table (one row per evaluated question, this contract), a
# ``steps`` table (agentic trajectories flattened for column projection), and finestore's reserved
# ``blobs`` table (raw trajectories and other opaque attachments). The samples row is the sample's
# JSON dump plus the archive's ``trial_id`` key; the reader ignores that extra key when validating.
# --------------------------------------------------------------------------------------------------

ARCHIVE_SAMPLES_TABLE = "samples"
ARCHIVE_STEPS_TABLE = "steps"

# Blob-name prefix for preserved evaluator-native inputs. A blob under this prefix is the verbatim
# bytes of a file the export read, keyed by its path relative to the run's results root, so a rebuild
# can recover the same inputs from the archive alone.
SOURCES_PREFIX = "sources"
RESULTS_PREFIX = "results_"

# The archive's own keys on a samples row, added at write time (not EvalSample fields): a sample is
# unique within a run by its task, its document, the trial that produced it (for multi-attempt Harbor
# runs), and the extraction filter that scored it (for lm-eval tasks that apply more than one).
# evalchemy leaves ``trial_id`` empty (one attempt per document); a task with a single filter, or a
# Harbor trial, leaves ``filter`` empty.
TRIAL_ID_COLUMN = "trial_id"
FILTER_COLUMN = "filter"
SAMPLES_MERGE_KEY = ("task", "doc_id", TRIAL_ID_COLUMN, FILTER_COLUMN)
STEPS_MERGE_KEY = ("task", "doc_id", TRIAL_ID_COLUMN, "step_id")


def samples_schema() -> pa.Schema:
    """The pinned ``samples`` table schema: the :class:`EvalSample` columns plus the archive's
    ``trial_id`` and ``filter`` keys. Pinning fixes each column's type up front, so ``metrics`` stays
    a ``map<string,double>`` and no per-flush inference can drift a column's type across shards."""
    return pa.schema(
        [
            *arrow_schema(EvalSample),
            pa.field(TRIAL_ID_COLUMN, pa.string()),
            pa.field(FILTER_COLUMN, pa.string()),
        ]
    )


def steps_schema() -> pa.Schema:
    """The pinned ``steps`` table schema, derived from :class:`StepRecord` (its own columns carry the
    merge key), so token-id and logprob columns keep their list types across shards."""
    return arrow_schema(StepRecord)


def sample_to_archive_row(sample: EvalSample, *, trial_id: str = "") -> dict:
    """One archive ``samples`` row: the sample's JSON-mode dump plus its ``trial_id`` and ``filter``
    archive keys. The filter comes from the sample's own grading, so a caller sets it by grading the
    sample, not by passing it here."""
    row = sample.model_dump(mode="json")
    row[TRIAL_ID_COLUMN] = trial_id
    row[FILTER_COLUMN] = (sample.grading.filter or "") if sample.grading else ""
    return row


def sample_from_archive_row(row: dict) -> EvalSample:
    """Reconstruct an :class:`EvalSample` from an archive ``samples`` row.

    ``metrics`` is a pinned ``map<string,double>``, so a batch that wrote no metrics reads back a null
    map and a partly-populated map can carry null values; normalize null, absent, and null-valued
    metrics to a plain dict (a null value means the metric is absent, not zero) before validation. The
    caller must materialize map columns as dicts (``to_pylist(maps_as_pydicts="strict")``). Archive-only
    keys (``trial_id`` and finestore's ``_seq``/``_writer``/``_gen`` columns) are ignored by the model.
    """
    metrics = row.get("metrics") or {}
    return EvalSample.model_validate(
        {**row, "metrics": {name: value for name, value in metrics.items() if value is not None}}
    )


def _content_to_text(value: object) -> str | None:
    """A trajectory step's message, coerced to text (multimodal content parts become JSON)."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


@dataclasses.dataclass(frozen=True)
class StepRecord:
    """One flattened agentic-trajectory step: the row schema of the archive ``steps`` table.

    Token-id and logprob arrays stay as native list columns (the RL signal a reader projects); nested
    tool calls and observations are kept as JSON strings. Sub-agent trajectories are preserved in the
    raw-trajectory blob rather than flattened here.
    """

    task: str
    doc_id: str
    trial_id: str
    step_id: int | None
    source: str | None
    model_name: str | None
    message: str | None
    reasoning_content: str | None
    tool_calls_json: str | None
    observation_json: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    prompt_token_ids: list[int] | None
    completion_token_ids: list[int] | None
    logprobs: list[float] | None


def trajectory_step_rows(trajectory: dict, *, task: str, doc_id: str, trial_id: str) -> list[StepRecord]:
    """Flatten a trajectory's top-level steps into typed :class:`StepRecord`s."""
    records = []
    for step in trajectory.get("steps") or []:
        metrics = step.get("metrics") or {}
        tool_calls = step.get("tool_calls")
        observation = step.get("observation")
        records.append(
            StepRecord(
                task=task,
                doc_id=doc_id,
                trial_id=trial_id,
                step_id=step.get("step_id"),
                source=step.get("source"),
                model_name=step.get("model_name"),
                message=_content_to_text(step.get("message")),
                reasoning_content=step.get("reasoning_content"),
                tool_calls_json=json.dumps(tool_calls, ensure_ascii=False) if tool_calls is not None else None,
                observation_json=json.dumps(observation, ensure_ascii=False) if observation is not None else None,
                prompt_tokens=metrics.get("prompt_tokens"),
                completion_tokens=metrics.get("completion_tokens"),
                cost_usd=metrics.get("cost_usd"),
                prompt_token_ids=metrics.get("prompt_token_ids"),
                completion_token_ids=metrics.get("completion_token_ids"),
                logprobs=metrics.get("logprobs"),
            )
        )
    return records


@dataclasses.dataclass(frozen=True)
class StoredTrajectory:
    """The result of archiving one raw trajectory: its blob reference and its flattened steps."""

    uri: str
    steps: list[StepRecord]


class EvaluationStore:
    """One eval run's finestore archive: a ``samples`` table (the :class:`EvalSample` contract), a
    ``steps`` table (flattened agentic trajectories), and raw trajectories held as blobs and
    referenced by a ``finestore://`` URI. A caller adds samples and trajectories through this wrapper
    without handling the individual tables; reads go through ``CompositeReader`` over the same root.
    """

    def __init__(self, store: DataStore) -> None:
        self._store = store
        self._samples = store.table(
            ARCHIVE_SAMPLES_TABLE, schema=samples_schema(), merge_key=SAMPLES_MERGE_KEY, schema_version=SCHEMA_VERSION
        )
        self._steps = store.table(
            ARCHIVE_STEPS_TABLE, schema=steps_schema(), merge_key=STEPS_MERGE_KEY, schema_version=SCHEMA_VERSION
        )

    @classmethod
    def open(cls, root: str, *, writer_id: str) -> EvaluationStore:
        """Open the archive for the run rooted at ``root``, written by ``writer_id``."""
        return cls(DataStore.open(root, writer_id=writer_id))

    def add_sample(self, sample: EvalSample, *, trial_id: str = "") -> None:
        """Append one evaluated question to the ``samples`` table."""
        self._samples.append(sample_to_archive_row(sample, trial_id=trial_id))

    def add_source_artifact(self, name: str, raw: bytes, *, content_type: str) -> str:
        """Preserve one evaluator-native source file inside the archive; return its blob URI.

        ``name`` is the file's path relative to the run's results root, which
        :func:`rebuild_lm_eval_samples` reads back to re-derive the tables without the surrounding
        results tree still being intact.
        """
        return self._store.write(
            prefix_join(SOURCES_PREFIX, name),
            {"content_type": content_type, "sha256": hashlib.sha256(raw).hexdigest()},
            raw,
        )

    def add_trajectory(self, raw: bytes, *, task: str, doc_id: str, trial_id: str) -> StoredTrajectory:
        """Store a raw trajectory as a blob and flatten its steps into the ``steps`` table.

        The payload is written once under ``<trial_id>/trajectory.json``; an unparseable payload is
        still stored but yields no steps. Returns the blob's ``finestore://`` URI and the steps
        written, so a caller can reference the trajectory from its sample and count what was flattened.
        """
        uri = self._store.write(
            f"{trial_id}/trajectory.json", {"task": task, "doc_id": doc_id, "trial_id": trial_id}, raw
        )
        try:
            trajectory = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("trajectory for trial %s is unparseable; storing the blob without steps", trial_id)
            return StoredTrajectory(uri=uri, steps=[])
        steps = trajectory_step_rows(trajectory, task=task, doc_id=doc_id, trial_id=trial_id)
        if steps:
            self._steps.extend(dataclasses.asdict(step) for step in steps)
        return StoredTrajectory(uri=uri, steps=steps)

    def flush(self) -> None:
        """Write buffered rows to shards now, bounding how much a large payload holds in memory."""
        self._store.flush()

    def seal(self) -> None:
        """Flush every table and mark the run complete."""
        self._store.seal()

    def close(self) -> None:
        """Stop the writer and flush any remaining rows."""
        self._store.close()

    def __enter__(self) -> EvaluationStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def export_lm_eval_samples(
    out_path: str,
    *,
    writer_id: str = "evalchemy",
    superseded_prefix: Callable[[int], str] | None = None,
) -> int:
    """Normalize every lm-eval ``samples_*.jsonl`` under ``out_path`` into the run's finestore archive.

    Returns the number of samples written. The source jsonl is kept as the mechanism's native
    artifact and is additionally preserved inside the archive, so the run can be rebuilt from the
    archive alone after a contract change. Re-running is idempotent: an unchanged source produces
    identical rows, which collapse on the merge key without conflict. A run with no sample source is
    left untouched.

    Replacing a table written under an older contract needs somewhere to put the rows first.
    ``superseded_prefix`` maps that stored version to a destination outside the run; without it an
    export that would have to replace rows raises instead.
    """
    root = StoragePath(out_path)
    sources = list(StoragePath(prefix_join(out_path, f"**/{SAMPLES_PREFIX}*.jsonl")).glob())
    if not sources:
        # With no source there is nothing to write, and nothing that could stand in for what is
        # already stored, so a run evaluated by another mechanism keeps the archive it has.
        return 0
    _replace_superseded_samples(out_path, superseded_prefix)
    store = EvaluationStore.open(out_path, writer_id=writer_id)
    count = 0
    try:
        for path in StoragePath(prefix_join(out_path, f"**/{RESULTS_PREFIX}*.json")).glob():
            store.add_source_artifact(path.relative_to(root), path.read_bytes(), content_type="application/json")
            # One shard per artifact keeps a multi-hundred-megabyte results tree from buffering whole.
            store.flush()
        for path in sources:
            relative = path.relative_to(root)
            payload = path.read_bytes()
            store.add_source_artifact(relative, payload, content_type="application/x-ndjson")
            store.flush()
            if is_scratch_artifact(relative):
                continue
            count += _add_lm_eval_rows(store, path.name, payload)
        store.seal()
    finally:
        store.close()
    return count


def _replace_superseded_samples(out_path: str, superseded_prefix: Callable[[int], str] | None) -> None:
    """Replace a samples table written under an older contract, or leave a current one alone.

    Rows an older contract left cannot collapse against the new ones, because the widened merge key
    reads as null on their shards, so a version change has to replace them wholesale. An export at
    the current version merges idempotently instead, because identical rows share a digest.
    """
    stored_version = CompositeReader(out_path).schema_version(ARCHIVE_SAMPLES_TABLE)
    if stored_version is None or stored_version == SCHEMA_VERSION:
        return
    _preserve_and_drop_samples(out_path, superseded_prefix, stored_version)


def _preserve_and_drop_samples(
    out_path: str, superseded_prefix: Callable[[int], str] | None, stored_version: int
) -> None:
    """Copy the samples table out of the run, then drop it.

    The copy runs to completion and is size-verified before the drop, so a failure at any point
    leaves either the original table or a full snapshot of it outside the run; an existing snapshot
    is never overwritten. Nothing here is allowed to be the only copy of anything, so a caller with
    nowhere to put the rows is refused rather than served.

    Agentic samples come from Harbor, which writes no lm-eval source, so an lm-eval path that
    replaced a table holding them would return a table missing that half. It stops instead.
    """
    reader = CompositeReader(out_path)
    stored = reader.scan(ARCHIVE_SAMPLES_TABLE, columns=["kind"])
    if stored is None:
        return
    agentic = sum(1 for kind in stored["kind"].to_pylist() if kind == SampleKind.AGENTIC)
    if agentic:
        raise ValueError(
            f"{out_path} holds {agentic} agentic sample(s) that an lm-eval export cannot regenerate; "
            "migrate the run instead of replacing its samples table"
        )
    if superseded_prefix is None:
        raise ValueError(
            f"{out_path} holds schema v{stored_version} samples that an export at v{SCHEMA_VERSION} has "
            "to replace; pass superseded_prefix to say where they are preserved first"
        )
    destination = superseded_prefix(stored_version)
    logger.info(
        "replacing %s samples written under schema v%s at v%s, preserved at %s",
        out_path,
        stored_version,
        SCHEMA_VERSION,
        destination,
    )
    replace_table(out_path, ARCHIVE_SAMPLES_TABLE, destination)


def _add_lm_eval_rows(store: EvaluationStore, filename: str, payload: bytes) -> int:
    """Normalize one ``samples_*.jsonl`` payload into ``store``; return the rows added.

    Splits only on the physical LF delimiter, so a literal U+2028/U+2029 inside a JSON string stays
    part of its record rather than being taken for a record boundary.
    """
    rows = [json.loads(line) for line in payload.decode().split("\n") if line.strip()]
    if not rows:
        logger.warning("samples file %s is empty; skipping archive export", filename)
        return 0
    task = _task_from_filename(filename, ".jsonl")
    for raw in rows:
        store.add_sample(sample_from_lm_eval(task, raw))
    return len(rows)


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


def rebuild_lm_eval_samples(
    out_path: str,
    *,
    writer_id: str = "rebuild",
    superseded_prefix: Callable[[int], str] | None = None,
) -> int:
    """Rebuild a run's ``samples`` table from the source artifacts preserved inside its archive.

    The inputs are the ``sources/`` blobs written by :func:`export_lm_eval_samples`, so this works on
    an archive whose surrounding results tree has been pruned. Returns the number of samples written.
    Check :func:`preserved_sample_sources` first: an archive holding none cannot be rebuilt this way
    and raises. ``superseded_prefix`` says where the rows being replaced are preserved, as in
    :func:`export_lm_eval_samples`.
    """
    reader = CompositeReader(out_path)
    names = preserved_sample_sources(out_path)
    if not names:
        raise FileNotFoundError(f"archive at {out_path!r} preserves no sample sources to rebuild from")
    # The preserved sources are the complete input, so the existing rows are replaced rather than
    # merged into: rows written under an older merge key would otherwise survive beside the new ones.
    stored_version = reader.schema_version(ARCHIVE_SAMPLES_TABLE)
    if stored_version is not None:
        _preserve_and_drop_samples(out_path, superseded_prefix, stored_version)
    store = EvaluationStore.open(out_path, writer_id=writer_id)
    count = 0
    try:
        for name in names:
            if is_scratch_artifact(name):
                continue
            payload = reader.read_blob(name)
            if payload is None:
                raise FileNotFoundError(f"archive at {out_path!r} lists source blob {name!r} but cannot read it")
            count += _add_lm_eval_rows(store, name.rsplit("/", 1)[-1], payload)
        store.seal()
    finally:
        store.close()
    return count
