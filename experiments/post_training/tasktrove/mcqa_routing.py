# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classify mechanically valid TaskTrove MCQA rows as RL, SFT, or garbage.

Malformed or incomplete model responses fail closed to SFT. RL requires a
verified answer-key match, high confidence, prompt-contained evidence, and
multi-step or multi-constraint reasoning.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import os
import re
import time
import tomllib
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import JobName
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.tasktrove.taskbinary import INSTRUCTION, read_task_binary

logger = logging.getLogger(__name__)

MCQA_SOURCE = "laion__nemotron-gym-knowledge-mcqa-v2"
POLICY_VERSION = "tasktrove-mcqa-glm53-v1"
GLM_BULK_TOKEN_ENV = "GLM_BULK_TOKEN"
DECISIONS_FILENAME = "decisions.jsonl"
ROUTE_MAPPINGS_FILENAME = "route-mappings.jsonl"
ROUTE_MAPPING_FIELDS = ("task_id", "route", "route_source", "policy_version", "reason_codes")
WORKER_SUMMARY_FILENAME = "summary.json"
VERIFIER_TOML = "tests/verifier.toml"
_OPTION_RE = re.compile(r"(?m)^\s*\(?([A-J])[.):]\s+(.+?)\s*$")
_ALL_OR_NONE_RE = re.compile(r"(?i)\b(?:all|none) of the above\b")
_EXTERNAL_CONTEXT_RE = re.compile(r"(?i)\b(?:the (?:above|following) (?:passage|text|article)|as discussed above)\b")


class Route(StrEnum):
    RL = "rl"
    SFT = "sft"
    GARBAGE = "garbage"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AnswerStatus(StrEnum):
    MATCH = "match"
    TIE = "tie"
    MISMATCH = "mismatch"
    UNCERTAIN = "uncertain"


class OperationType(StrEnum):
    CHAIN = "chain"
    CONSTRAINTS = "constraints"
    SINGLE = "single"
    RULE = "rule"
    SUBSTITUTION = "substitution"
    RECOGNITION = "recognition"
    RECALL = "recall"
    EXTRACT = "extract"
    UNCLEAR = "unclear"


class EvidenceSource(StrEnum):
    PROMPT = "prompt"
    RULE = "rule"
    RECALL = "recall"
    MISSING = "missing"


class Defect(StrEnum):
    NONE = "none"
    MISSING = "missing"
    MULTIPLE = "multiple"
    WRONG_KEY = "wrong_key"
    UNSTABLE = "unstable"
    OPTIONS = "options"
    INCOHERENT = "incoherent"
    OTHER = "other"


class Subject(StrEnum):
    MATH = "math"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    MEDICINE = "medicine"
    COMPUTER_SCIENCE = "computer_science"
    ENGINEERING = "engineering"
    ECONOMICS_BUSINESS = "economics_business"
    LAW_POLICY = "law_policy"
    SOCIAL_SCIENCE = "social_science"
    HUMANITIES = "humanities"
    OTHER = "other"


@dataclass(frozen=True)
class RoutingTask:
    task_id: str
    question: str
    options: dict[str, str]
    expected: str
    mechanical_flags: tuple[str, ...]


@dataclass(frozen=True)
class RoutingConfig:
    input_path: str
    output_path: str
    source: str
    git_revision: str
    sample_size: int
    sample_seed: str
    request_batch_size: int
    relay_job: str
    poll_seconds: float
    expected_workers: int | None = None


@dataclass(frozen=True)
class GlmDecision:
    row_id: int
    route: Route
    confidence: Confidence
    derived_choice: str
    answer_status: AnswerStatus
    operation_type: OperationType
    evidence_source: EvidenceSource
    defect: Defect
    subject: Subject
    check: str


RUBRIC = """Classify each MCQA task as rl, sft, or garbage. False-negative RL
decisions are acceptable. Uncertain tasks must not be RL.

rl: The prompt is self-contained, has one defensible keyed answer, and requires
either a chained derivation whose intermediate result feeds a second operation
or two independently derived constraints that must be combined. RL requires
high confidence, answer=match, defect=none, and evidence=prompt.
sft: Coherent material dominated by recall, recognition, direct extraction,
one rule or substitution, or uncertain possible reasoning.
garbage: A wrong key, reversed quantity, tie, missing input, unsupported premise,
unstable local standard, malformed option, or other material defect prevents
reliable use.

Solve each task independently before comparing with expected. Try to falsify
the key: check every option, units, directionality, numerical consistency, and
whether the premise supports the keyed choice. A long scenario around a fact is
SFT. A conclusion disclosed in the stem is SFT. Use choice=? if no choice is
defensible. Keep check concrete and under 160 characters. Return exactly one
result for every row_id."""


def _enum_values(enum_type: type[StrEnum]) -> list[str]:
    return [member.value for member in enum_type]


def tool_schema(expected_rows: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["results"],
        "properties": {
            "results": {
                "type": "array",
                "minItems": expected_rows,
                "maxItems": expected_rows,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "r", "c", "choice", "answer", "op", "evidence", "defect", "subject", "check"],
                    "properties": {
                        "id": {"type": "integer", "enum": list(range(expected_rows))},
                        "r": {"type": "string", "enum": _enum_values(Route)},
                        "c": {"type": "string", "enum": _enum_values(Confidence)},
                        "choice": {"type": "string", "enum": [*"ABCDEFGHIJ", "?"]},
                        "answer": {"type": "string", "enum": _enum_values(AnswerStatus)},
                        "op": {"type": "string", "enum": _enum_values(OperationType)},
                        "evidence": {"type": "string", "enum": _enum_values(EvidenceSource)},
                        "defect": {"type": "string", "enum": _enum_values(Defect)},
                        "subject": {"type": "string", "enum": _enum_values(Subject)},
                        "check": {"type": "string", "maxLength": 160},
                    },
                },
            }
        },
    }


def _question_body(instruction: str) -> str:
    return instruction.split("---", 1)[-1].strip()


def parse_mcqa_task(task_id: str, task_binary: bytes) -> tuple[RoutingTask | None, str | None]:
    """Parse one packed MCQA task, returning a structural rejection reason on failure."""
    try:
        task = read_task_binary(task_binary)
        instruction = task.text(INSTRUCTION)
        verifier = tomllib.loads(task.text(VERIFIER_TOML))
    except (KeyError, OSError, EOFError, ValueError, tomllib.TOMLDecodeError) as error:
        return None, f"malformed:{type(error).__name__}"

    body = _question_body(instruction)
    matches = list(_OPTION_RE.finditer(body))
    if not matches:
        return None, "malformed:no_option_lines"

    labels = [match.group(1) for match in matches]
    if labels.count("A") > 1:
        return None, "malformed:multiple_option_blocks"
    expected_labels = [chr(ord("A") + index) for index in range(len(labels))]
    if labels != expected_labels:
        return None, "malformed:noncontiguous_option_labels"
    if len(labels) < 2:
        return None, "fewer_than_two_options"

    contract_options = verifier.get("options")
    if not isinstance(contract_options, int) or contract_options != len(labels):
        return None, "contract_option_count_mismatch"
    expected = verifier.get("expected")
    if not isinstance(expected, str) or expected not in labels:
        return None, "gold_outside_options"

    options = {match.group(1): match.group(2).strip() for match in matches}
    normalized_options = [" ".join(value.casefold().split()).rstrip(" .") for value in options.values()]
    if len(normalized_options) != len(set(normalized_options)):
        return None, "duplicate_options"

    flags = []
    if any(_ALL_OR_NONE_RE.search(option) for option in options.values()):
        flags.append("all_or_none_option")
    question = body[: matches[0].start()].strip()
    if _EXTERNAL_CONTEXT_RE.search(question):
        flags.append("external_context_reference")
    return RoutingTask(task_id, question, options, expected, tuple(flags)), None


def _sample_score(seed: str, task_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{task_id}".encode()).digest(), "big")


def select_tasks(
    input_path: str,
    *,
    source: str,
    worker_index: int,
    worker_count: int,
    sample_size: int | None,
    sample_seed: str,
) -> tuple[list[RoutingTask], dict[str, Any]]:
    """Select a deterministic sample from this worker's disjoint Parquet row groups."""
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError(f"worker_index {worker_index} is outside worker_count {worker_count}")

    selected: list[tuple[int, str, RoutingTask]] = []
    reject_counts: Counter[str] = Counter()
    reject_rows = 0
    scanned = 0
    source_rows = 0
    survivor_rows = 0
    assigned_groups = []
    with StoragePath(input_path).open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        is_mechanical_ledger = {
            "path",
            "status",
            "reasons",
            "flags",
            "question",
            "expected",
            "options_json",
        }.issubset(parquet.schema_arrow.names)
        for row_group in range(parquet.num_row_groups):
            if row_group % worker_count != worker_index:
                continue
            assigned_groups.append(row_group)
            if is_mechanical_ledger:
                table = parquet.read_row_group(
                    row_group,
                    columns=["path", "status", "reasons", "flags", "question", "expected", "options_json"],
                )
            else:
                table = parquet.read_row_group(row_group, columns=["path", "source", "task_binary"])
            for row in table.to_pylist():
                scanned += 1
                source_rows += 1
                if is_mechanical_ledger:
                    if row["status"] != "keep":
                        reject_rows += 1
                        for reason in row["reasons"]:
                            reject_counts[reason] += 1
                        continue
                    task = RoutingTask(
                        row["path"],
                        row["question"],
                        json.loads(row["options_json"]),
                        row["expected"],
                        tuple(row["flags"]),
                    )
                    rejection = None
                else:
                    if row["source"] != source:
                        source_rows -= 1
                        continue
                    task, rejection = parse_mcqa_task(row["path"], row["task_binary"])
                if rejection is not None:
                    reject_rows += 1
                    reject_counts[rejection] += 1
                    continue
                assert task is not None
                survivor_rows += 1
                if sample_size is None:
                    selected.append((0, task.task_id, task))
                    continue
                if sample_size == 0:
                    continue
                score = _sample_score(sample_seed, task.task_id)
                item = (-score, task.task_id, task)
                if len(selected) < sample_size:
                    heapq.heappush(selected, item)
                elif score < -selected[0][0]:
                    heapq.heapreplace(selected, item)

    if sample_size is not None and len(selected) != sample_size:
        raise ValueError(f"worker {worker_index} selected {len(selected)} rows, expected {sample_size}")
    tasks = [item[2] for item in sorted(selected, key=lambda item: (-item[0], item[1]))]
    summary = {
        "assigned_row_groups": assigned_groups,
        "input_kind": "mechanical_ledger" if is_mechanical_ledger else "clean_release",
        "scanned_rows": scanned,
        "source_rows": source_rows,
        "mechanical_survivors": survivor_rows,
        "mechanical_rejects": reject_rows,
        "mechanical_reject_counts": dict(reject_counts),
        "selected_rows": len(tasks),
    }
    return tasks, summary


def resolve_base_url(relay_job: str) -> str:
    client = iris_ctx().client
    if client is None:
        raise RuntimeError("Iris client is unavailable inside the task")
    endpoints = client.resolver_for_job(JobName.from_string(relay_job)).resolve("glm-5.3").endpoints
    if not endpoints:
        raise RuntimeError("The GLM relay has no registered endpoint")
    base_url = endpoints[0].url.rstrip("/")
    return base_url if base_url.endswith("/v1") else f"{base_url}/v1"


def _request(url: str, token: str, *, data: bytes | None = None, content_type: str | None = None) -> bytes:
    headers = {"Authorization": f"Bearer {token}", "x-priority": "bulk"}
    if content_type is not None:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def _request_json(url: str, token: str, *, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, separators=(",", ":")).encode()
    raw = _request(url, token, data=data, content_type="application/json" if data is not None else None)
    return json.loads(raw)


def completion_body(tasks: list[RoutingTask], batch_id: int) -> dict[str, Any]:
    visible = [
        {"row_id": row_id, "question": task.question, "options": task.options, "expected": task.expected}
        for row_id, task in enumerate(tasks)
    ]
    prompt = (
        RUBRIC
        + "\n\nTasks (JSONL):\n"
        + "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in visible)
    )
    return {
        "model": "glm-5.3",
        "messages": [
            {"role": "system", "content": "Apply the routing rubric conservatively and call submit_routes once."},
            {"role": "user", "content": prompt},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "submit_routes",
                    "description": "Submit one conservative routing decision for every supplied row.",
                    "strict": True,
                    "parameters": tool_schema(len(tasks)),
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "submit_routes"}},
        "parallel_tool_calls": False,
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "prompt_cache_key": f"tasktrove-mcqa-routing-{batch_id % 16}",
        "max_tokens": 12000,
    }


def batch_lines(
    tasks: list[RoutingTask], batch_size: int, worker_index: int, label: str = "batch"
) -> tuple[list[dict[str, Any]], dict[str, list[RoutingTask]]]:
    lines = []
    by_custom_id = {}
    for start in range(0, len(tasks), batch_size):
        batch = tasks[start : start + batch_size]
        custom_id = f"worker-{worker_index:03d}-{label}-{start // batch_size:05d}"
        lines.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": completion_body(batch, start // batch_size),
            }
        )
        by_custom_id[custom_id] = batch
    return lines, by_custom_id


def _jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows)


def submit_batch(base_url: str, token: str, lines: list[dict[str, Any]], filename: str) -> tuple[str, str]:
    query = urllib.parse.urlencode({"purpose": "batch", "filename": filename})
    file_response = json.loads(
        _request(f"{base_url}/files?{query}", token, data=_jsonl(lines).encode(), content_type="application/jsonl")
    )
    file_id = file_response["id"]
    batch = _request_json(
        f"{base_url}/batches",
        token,
        body={"input_file_id": file_id, "endpoint": "/v1/chat/completions", "priority": "bulk"},
    )
    return file_id, batch["id"]


def wait_for_batch(base_url: str, token: str, batch_id: str, poll_seconds: float) -> dict[str, Any]:
    terminal = {"completed", "failed", "expired", "cancelled"}
    last_counts = None
    while True:
        batch = _request_json(f"{base_url}/batches/{batch_id}", token)
        counts = batch.get("request_counts")
        if counts != last_counts:
            logger.info("GLM batch %s status=%s counts=%s", batch_id, batch.get("status"), counts)
            last_counts = counts
        if batch.get("status") in terminal:
            return batch
        time.sleep(poll_seconds)


def _tool_arguments(response_body: dict[str, Any]) -> dict[str, Any]:
    message = response_body["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "submit_routes":
        raise ValueError("expected exactly one submit_routes tool call")
    return json.loads(calls[0]["function"]["arguments"])


def parse_decision(row: dict[str, Any]) -> GlmDecision:
    derived_choice = row["choice"]
    if not isinstance(derived_choice, str) or derived_choice not in {*"ABCDEFGHIJ", "?"}:
        raise ValueError(f"invalid derived choice: {derived_choice!r}")
    check = row["check"]
    if not isinstance(check, str) or len(check) > 160:
        raise ValueError("check must be a string of at most 160 characters")
    return GlmDecision(
        row_id=row["id"],
        route=Route(row["r"]),
        confidence=Confidence(row["c"]),
        derived_choice=derived_choice,
        answer_status=AnswerStatus(row["answer"]),
        operation_type=OperationType(row["op"]),
        evidence_source=EvidenceSource(row["evidence"]),
        defect=Defect(row["defect"]),
        subject=Subject(row["subject"]),
        check=check,
    )


def final_route(task: RoutingTask, decision: GlmDecision) -> Route:
    """Apply fail-closed policy gates to a model routing decision."""
    key_conflict = decision.answer_status is AnswerStatus.MATCH and decision.derived_choice != task.expected
    if (
        decision.route is Route.GARBAGE
        or decision.defect is not Defect.NONE
        or decision.answer_status in {AnswerStatus.TIE, AnswerStatus.MISMATCH}
        or key_conflict
    ):
        return Route.GARBAGE
    if (
        decision.route is Route.RL
        and decision.confidence is Confidence.HIGH
        and decision.answer_status is AnswerStatus.MATCH
        and decision.derived_choice == task.expected
        and decision.operation_type in {OperationType.CHAIN, OperationType.CONSTRAINTS}
        and decision.evidence_source is EvidenceSource.PROMPT
    ):
        return Route.RL
    return Route.SFT


def decision_row(task: RoutingTask, decision: GlmDecision) -> dict[str, Any]:
    route = final_route(task, decision)
    return {
        "task_id": task.task_id,
        "route": route.value,
        "model_route": decision.route.value,
        "route_source": "glm-5.3",
        "policy_version": POLICY_VERSION,
        "reason_codes": [
            f"model_route:{decision.route.value}",
            f"confidence:{decision.confidence.value}",
            f"answer:{decision.answer_status.value}",
            f"operation:{decision.operation_type.value}",
            f"evidence:{decision.evidence_source.value}",
            f"defect:{decision.defect.value}",
        ],
        "expected": task.expected,
        "derived_choice": decision.derived_choice,
        "confidence": decision.confidence.value,
        "answer_status": decision.answer_status.value,
        "operation_type": decision.operation_type.value,
        "evidence_source": decision.evidence_source.value,
        "defect": decision.defect.value,
        "subject": decision.subject.value,
        "check": decision.check,
        "mechanical_flags": list(task.mechanical_flags),
        "classification_status": "classified",
    }


def fallback_row(task: RoutingTask, reason: str) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "route": Route.SFT.value,
        "model_route": None,
        "route_source": "best-effort-fallback",
        "policy_version": POLICY_VERSION,
        "reason_codes": [f"fallback:{reason}"],
        "expected": task.expected,
        "derived_choice": "?",
        "confidence": Confidence.LOW.value,
        "answer_status": AnswerStatus.UNCERTAIN.value,
        "operation_type": OperationType.UNCLEAR.value,
        "evidence_source": EvidenceSource.MISSING.value,
        "defect": Defect.OTHER.value,
        "subject": Subject.OTHER.value,
        "check": "GLM response was unavailable or invalid; routed to SFT by the best-effort fallback.",
        "mechanical_flags": list(task.mechanical_flags),
        "classification_status": "fallback",
    }


def read_batch_output(base_url: str, token: str, batch: dict[str, Any]) -> tuple[str, str | None]:
    output_id = batch.get("output_file_id")
    output = "" if not output_id else _request(f"{base_url}/files/{output_id}/content", token).decode()
    error_id = batch.get("error_file_id")
    errors = None if not error_id else _request(f"{base_url}/files/{error_id}/content", token).decode()
    return output, errors


def parse_batch_output(
    output: str, by_custom_id: dict[str, list[RoutingTask]]
) -> tuple[list[dict[str, Any]], list[str]]:
    routed = []
    degraded = []
    seen = set()
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            custom_id = row["custom_id"]
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
        if custom_id not in by_custom_id or custom_id in seen:
            continue
        seen.add(custom_id)
        tasks = by_custom_id[custom_id]
        response = row.get("response") or {}
        if row.get("error") or response.get("status_code") != 200:
            degraded.append(custom_id)
            routed.extend(fallback_row(task, "request_error") for task in tasks)
            continue
        try:
            raw_results = _tool_arguments(response["body"])["results"]
            if not isinstance(raw_results, list):
                raise TypeError("results must be an array")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            degraded.append(custom_id)
            routed.extend(fallback_row(task, "invalid_response") for task in tasks)
            continue

        decisions: dict[int, GlmDecision] = {}
        invalid_row_ids = set()
        for raw_result in raw_results:
            row_id = raw_result.get("id") if isinstance(raw_result, dict) else None
            if not isinstance(row_id, int) or isinstance(row_id, bool) or row_id < 0 or row_id >= len(tasks):
                continue
            if row_id in decisions or row_id in invalid_row_ids:
                decisions.pop(row_id, None)
                invalid_row_ids.add(row_id)
                continue
            try:
                decisions[row_id] = parse_decision(raw_result)
            except (KeyError, TypeError, ValueError):
                invalid_row_ids.add(row_id)

        missing_rows = set(range(len(tasks))) - decisions.keys()
        if missing_rows:
            degraded.append(custom_id)
        for row_id, task in enumerate(tasks):
            decision = decisions.get(row_id)
            routed.append(fallback_row(task, "invalid_row") if decision is None else decision_row(task, decision))

    for custom_id in sorted(set(by_custom_id) - seen):
        degraded.append(custom_id)
        routed.extend(fallback_row(task, "missing_response") for task in by_custom_id[custom_id])
    return routed, sorted(set(degraded))


def _worker_coordinates() -> tuple[int, int]:
    job_info = get_job_info()
    if job_info is None:
        return 0, 1
    return job_info.task_index, job_info.num_tasks


def _worker_sample_size(total: int, worker_index: int, worker_count: int) -> int:
    assert total > 0
    quotient, remainder = divmod(total, worker_count)
    return quotient + (worker_index < remainder)


def run_worker(config: RoutingConfig) -> None:
    worker_index, worker_count = _worker_coordinates()
    if config.expected_workers is not None and worker_count != config.expected_workers:
        raise ValueError(f"Iris launched {worker_count} workers, expected {config.expected_workers}")
    selected_rows = (
        None if config.sample_size == 0 else _worker_sample_size(config.sample_size, worker_index, worker_count)
    )
    output_root = StoragePath(config.output_path) / f"worker-{worker_index:03d}"
    output_root.mkdirs()
    summary_path = output_root / WORKER_SUMMARY_FILENAME
    if summary_path.exists():
        logger.info("worker %d already completed: %s", worker_index, summary_path)
        return

    started = time.time()
    tasks, scan_summary = select_tasks(
        config.input_path,
        source=config.source,
        worker_index=worker_index,
        worker_count=worker_count,
        sample_size=selected_rows,
        sample_seed=config.sample_seed,
    )
    logger.info("worker %d selected %d rows from %d source rows", worker_index, len(tasks), scan_summary["source_rows"])

    base_url = resolve_base_url(config.relay_job)
    token = os.environ[GLM_BULK_TOKEN_ENV]
    lines, by_custom_id = batch_lines(tasks, config.request_batch_size, worker_index)
    request_path = output_root / "requests.jsonl"
    if not request_path.exists():
        request_path.write_text(_jsonl(lines))

    state_path = output_root / "batch-state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        file_id, batch_id = state["file_id"], state["batch_id"]
        logger.info("worker %d resuming GLM batch %s", worker_index, batch_id)
    else:
        file_id, batch_id = submit_batch(
            base_url,
            token,
            lines,
            f"tasktrove-mcqa-worker-{worker_index:03d}.jsonl",
        )
        state_path.write_text(json.dumps({"file_id": file_id, "batch_id": batch_id}, indent=2) + "\n")
        logger.info(
            "worker %d submitted GLM batch %s with %d requests of at most %d tasks",
            worker_index,
            batch_id,
            len(lines),
            config.request_batch_size,
        )

    batch = wait_for_batch(base_url, token, batch_id, config.poll_seconds)
    raw_output, raw_errors = read_batch_output(base_url, token, batch)
    (output_root / "raw-output.jsonl").write_text(raw_output)
    if raw_errors is not None:
        (output_root / "raw-errors.jsonl").write_text(raw_errors)
    routed, degraded_requests = parse_batch_output(raw_output, by_custom_id)
    (output_root / "degraded-requests.json").write_text(json.dumps(degraded_requests, indent=2) + "\n")
    if len(routed) != len(tasks):
        raise RuntimeError(f"worker {worker_index} routed {len(routed)} of {len(tasks)} tasks")

    routed.sort(key=lambda row: row["task_id"])
    (output_root / DECISIONS_FILENAME).write_text(_jsonl(routed))
    mappings = [{key: row[key] for key in ROUTE_MAPPING_FIELDS} for row in routed]
    (output_root / ROUTE_MAPPINGS_FILENAME).write_text(_jsonl(mappings))

    route_subject = defaultdict(Counter)
    for row in routed:
        route_subject[row["route"]][row["subject"]] += 1
    summary = {
        "worker_index": worker_index,
        "worker_count": worker_count,
        "input": config.input_path,
        "output": str(output_root),
        "source": config.source,
        "sample_seed": config.sample_seed,
        "sample_size": config.sample_size,
        "request_batch_size": config.request_batch_size,
        "requests": len(lines),
        "git_revision": config.git_revision,
        "policy_version": POLICY_VERSION,
        "glm_batch": {
            "file_id": file_id,
            "batch_id": batch_id,
            "request_counts": batch.get("request_counts"),
            "created_at": batch.get("created_at"),
            "completed_at": batch.get("completed_at"),
        },
        "degraded_requests": len(degraded_requests),
        "fallback_rows": sum(row["classification_status"] == "fallback" for row in routed),
        "scan": scan_summary,
        "routed_rows": len(routed),
        "route_counts": dict(Counter(row["route"] for row in routed)),
        "confidence_counts": dict(Counter(row["confidence"] for row in routed)),
        "answer_status_counts": dict(Counter(row["answer_status"] for row in routed)),
        "operation_type_counts": dict(Counter(row["operation_type"] for row in routed)),
        "defect_counts": dict(Counter(row["defect"] for row in routed)),
        "subject_counts": dict(Counter(row["subject"] for row in routed)),
        "route_subject_counts": {route: dict(counts) for route, counts in route_subject.items()},
        "derived_choice_matches": sum(row["derived_choice"] == row["expected"] for row in routed),
        "wall_seconds": time.time() - started,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    local_output = Path(os.environ["IRIS_OUTPUT_DIR"])
    local_output.mkdir(parents=True, exist_ok=True)
    (local_output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("MCQA_ROUTING_SUMMARY=" + json.dumps(summary, separators=(",", ":")), flush=True)
