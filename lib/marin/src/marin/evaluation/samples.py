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

Storage is one ``samples_<task>_<timestamp>.parquet`` per (sub)task next to the mechanism's native
output, and the parquet schema is the pydantic model itself: the writer is ``model_dump`` and the
reader is ``model_validate``, with nested fields stored as parquet structs/lists. ``schema_version``
is a model field, so it rides in every row for future evolution.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from enum import StrEnum

import pyarrow as pa
import pyarrow.parquet as pq
from finestore import DataStore, DataTable
from fsspec.core import url_to_fs
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

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

    Rows stay bounded: the two unbounded payloads (an agentic trajectory, a prediction's raw
    request/response exchange) are stored as sibling artifact files and referenced here by URI, so the
    columnar reader never has to materialize them to page the light columns.
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
    exchange_uri: str | None = None
    grading: Grading | None = None
    metrics: dict[str, float] = {}
    correct: bool | None = None
    doc: str = "{}"


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

# lm-eval per-sample fields that are structural rather than metric values.
_LM_EVAL_STRUCTURAL_KEYS = frozenset(
    {"doc", "doc_id", "target", "arguments", "resps", "filtered_resps", "doc_hash", "prompt_hash", "target_hash"}
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


def _lm_eval_grading(metrics: dict[str, float]) -> Grading | None:
    """The explicit grading for an lm-eval sample: its headline metric, filter, score, and pass flag."""
    picked = primary_metric(metrics)
    if picked is None:
        return None
    name, value = picked
    metric_filter = name.split(",", 1)[1] if "," in name else None
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
    """
    arguments = raw.get("arguments")
    responses = raw.get("resps")
    doc = raw.get("doc")
    target = raw.get("target")
    metrics = _sample_metrics(raw)
    common = {
        "task": task,
        "doc_id": str(raw.get("doc_id")),
        "metrics": metrics,
        "correct": _correct(metrics),
        "grading": _lm_eval_grading(metrics),
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
# ``steps`` table (agentic trajectories flattened for column projection), and a ``blobs`` table (raw
# trajectories and other opaque attachments). The samples row is the sample's JSON dump plus the
# archive's ``run_id``/``trial_id`` keys; the reader ignores those extra keys when validating.
# --------------------------------------------------------------------------------------------------

ARCHIVE_SAMPLES_TABLE = "samples"
ARCHIVE_STEPS_TABLE = "steps"
ARCHIVE_BLOBS_TABLE = "blobs"
ARCHIVE_BLOB_NAME_COLUMN = "name"

# A sample is unique within a run by its task, its document, and (for multi-attempt Harbor runs) the
# trial that produced it. evalchemy leaves ``trial_id`` empty: one attempt per document.
SAMPLES_PRIMARY_KEY = ("task", "doc_id", "trial_id")
STEPS_PRIMARY_KEY = ("task", "doc_id", "trial_id", "step_id")


def sample_to_archive_row(sample: EvalSample, *, run_id: str = "", trial_id: str = "") -> dict:
    """One archive ``samples`` row: the sample's JSON-mode dump plus its run/trial archive keys."""
    row = sample.model_dump(mode="json")
    row["run_id"] = run_id
    row["trial_id"] = trial_id
    return row


def sample_from_archive_row(row: dict) -> EvalSample:
    """Reconstruct an :class:`EvalSample` from an archive ``samples`` row.

    Parquet unifies the metrics struct across a shard, so a row missing a metric another row carries
    reads back with that key null; a null means the metric is absent (an ungraded sample), not zero,
    so it is dropped before validation. Archive-only keys (``run_id``, ``trial_id``, and finestore's
    ``_seq``/``_writer``/``_gen`` bookkeeping columns) are ignored by the model.
    """
    metrics = row.get("metrics")
    if metrics:
        row = {**row, "metrics": {name: value for name, value in metrics.items() if value is not None}}
    return EvalSample.model_validate(row)


def _content_to_text(value: object) -> str | None:
    """A trajectory step's message, coerced to text (multimodal content parts become JSON)."""
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def trajectory_step_rows(trajectory: dict, *, task: str, doc_id: str, trial_id: str, run_id: str = "") -> list[dict]:
    """Flatten a trajectory's top-level steps into archive ``steps`` rows.

    Token-id and logprob arrays stay as native list columns (the RL signal a reader projects); nested
    tool calls and observations are kept as JSON strings. Sub-agent trajectories are preserved in the
    raw-trajectory blob rather than flattened here.
    """
    rows = []
    for step in trajectory.get("steps") or []:
        metrics = step.get("metrics") or {}
        tool_calls = step.get("tool_calls")
        observation = step.get("observation")
        rows.append(
            {
                "run_id": run_id,
                "task": task,
                "doc_id": doc_id,
                "trial_id": trial_id,
                "step_id": step.get("step_id"),
                "source": step.get("source"),
                "model_name": step.get("model_name"),
                "message": _content_to_text(step.get("message")),
                "reasoning_content": step.get("reasoning_content"),
                "tool_calls_json": json.dumps(tool_calls, ensure_ascii=False) if tool_calls is not None else None,
                "observation_json": json.dumps(observation, ensure_ascii=False) if observation is not None else None,
                "prompt_tokens": metrics.get("prompt_tokens"),
                "completion_tokens": metrics.get("completion_tokens"),
                "cost_usd": metrics.get("cost_usd"),
                "prompt_token_ids": metrics.get("prompt_token_ids"),
                "completion_token_ids": metrics.get("completion_token_ids"),
                "logprobs": metrics.get("logprobs"),
            }
        )
    return rows


def open_eval_archive(root: str, *, writer_id: str) -> DataStore:
    """Open the finestore archive for an eval run rooted at ``root``, written by ``writer_id``."""
    return DataStore.open(root, writer_id=writer_id)


def archive_samples_table(store: DataStore) -> DataTable:
    """Register and return the run's ``samples`` table."""
    return store.table(ARCHIVE_SAMPLES_TABLE, primary_key=SAMPLES_PRIMARY_KEY, schema_version=SCHEMA_VERSION)


def archive_steps_table(store: DataStore) -> DataTable:
    """Register and return the run's ``steps`` table (flattened agentic trajectories)."""
    return store.table(ARCHIVE_STEPS_TABLE, primary_key=STEPS_PRIMARY_KEY, schema_version=SCHEMA_VERSION)


def archive_blobs_table(store: DataStore) -> DataTable:
    """Register and return the run's ``blobs`` table (raw trajectories and opaque attachments)."""
    return store.table(ARCHIVE_BLOBS_TABLE, primary_key=(ARCHIVE_BLOB_NAME_COLUMN,))


def export_lm_eval_samples_to_archive(out_path: str, *, writer_id: str = "evalchemy") -> int:
    """Normalize every lm-eval ``samples_*.jsonl`` under ``out_path`` into the run's finestore archive.

    Returns the number of samples written. The source jsonl is kept as the mechanism's native
    artifact. Re-running is idempotent: samples dedupe on ``(task, doc_id, trial_id)``.
    """
    fs, root = url_to_fs(out_path)
    store = open_eval_archive(out_path, writer_id=writer_id)
    samples = archive_samples_table(store)
    count = 0
    try:
        for path in fs.find(root):
            name = path.rsplit("/", 1)[-1]
            if not (name.startswith(SAMPLES_PREFIX) and name.endswith(".jsonl")):
                continue
            with fs.open(path, "r") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            if not rows:
                logger.warning("samples file %s is empty; skipping archive export", path)
                continue
            task = _task_from_filename(name, ".jsonl")
            for raw in rows:
                sample = sample_from_lm_eval(task, raw)
                samples.append(sample_to_archive_row(sample))
                count += 1
        store.seal()
    finally:
        store.close()
    return count
