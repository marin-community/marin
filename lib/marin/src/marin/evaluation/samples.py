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


def export_lm_eval_samples(out_path: str) -> list[str]:
    """Write a contract parquet sibling for every lm-eval ``samples_*.jsonl`` under ``out_path``.

    Files stay keyed per (sub)task. The source jsonl is kept: it is the mechanism's native artifact,
    the parquet is the contract view.
    """
    fs, root = url_to_fs(out_path)
    written: list[str] = []
    for path in fs.find(root):
        name = path.rsplit("/", 1)[-1]
        if not (name.startswith(SAMPLES_PREFIX) and name.endswith(".jsonl")):
            continue
        with fs.open(path, "r") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows:
            logger.warning("samples file %s is empty; skipping parquet export", path)
            continue
        task = _task_from_filename(name, ".jsonl")
        samples = [sample_from_lm_eval(task, raw) for raw in rows]
        dest = path[: -len(".jsonl")] + SAMPLES_SUFFIX
        write_sample_parquet(fs, dest, samples)
        written.append(dest)
    return written
