# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic task and rollout scoring for curriculum SFT ablations."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from experiments.post_training.curriculum_sft.ablation.tasks import (
    EVIDENCE_IDS,
    FACT_FIELDS,
    RESULT_FIELDS,
    TASK_PAYLOAD_FIELDS,
    SyntheticFinanceTask,
)


@dataclass(frozen=True)
class TaskVerification:
    """Independent checks applied to a generated task or response."""

    format_valid: bool
    arithmetic_valid: bool
    evidence_valid: bool

    @property
    def accepted(self) -> bool:
        return self.format_valid and self.arithmetic_valid and self.evidence_valid


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate deterministic metrics; no model judge is involved."""

    count: int
    format_rate: float
    arithmetic_rate: float
    evidence_rate: float
    exact_rate: float


def _strict_object(value: object, required: set[str]) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping) or set(value) != required:
        return None
    return value


def _question_contains_integer(question: str, value: object) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    normalized = question.replace(",", "")
    return re.search(rf"(?<!\d){re.escape(str(value))}(?!\d)", normalized) is not None


def task_payload_has_schema(payload: object) -> bool:
    """Return whether GLM honored the shared task-object wire schema."""

    task = _strict_object(payload, TASK_PAYLOAD_FIELDS)
    if task is None:
        return False
    facts = _strict_object(task["facts"], FACT_FIELDS)
    answer = _strict_object(task["answer"], RESULT_FIELDS)
    evidence = task["evidence"]
    return (
        isinstance(task["task_id"], str)
        and bool(task["task_id"])
        and isinstance(task["issuer"], str)
        and bool(task["issuer"])
        and isinstance(task["question"], str)
        and bool(task["question"])
        and facts is not None
        and all(isinstance(facts[field], int) and not isinstance(facts[field], bool) for field in facts)
        and answer is not None
        and all(isinstance(answer[field], int) and not isinstance(answer[field], bool) for field in answer)
        and isinstance(evidence, list)
        and all(isinstance(item, str) for item in evidence)
    )


def verify_task_payload(payload: object) -> TaskVerification:
    """Validate a generated task against its supplied facts and answer."""

    task = _strict_object(payload, TASK_PAYLOAD_FIELDS)
    if task is None:
        return TaskVerification(False, False, False)
    facts = _strict_object(task["facts"], FACT_FIELDS)
    answer = _strict_object(task["answer"], RESULT_FIELDS)
    evidence = task["evidence"]
    format_valid = task_payload_has_schema(task)
    if not format_valid or facts is None or answer is None or not isinstance(evidence, list):
        return TaskVerification(False, False, False)
    format_valid = (
        format_valid
        and _question_contains_integer(task["question"], facts["revenue"])
        and _question_contains_integer(task["question"], facts["operating_cost"])
    )
    try:
        revenue = facts["revenue"]
        operating_cost = facts["operating_cost"]
        gross_profit = revenue - operating_cost
        margin_bps, remainder = divmod(gross_profit * 10_000, revenue)
        arithmetic_valid = (
            isinstance(revenue, int)
            and not isinstance(revenue, bool)
            and isinstance(operating_cost, int)
            and not isinstance(operating_cost, bool)
            and isinstance(answer["gross_profit"], int)
            and not isinstance(answer["gross_profit"], bool)
            and isinstance(answer["margin_bps"], int)
            and not isinstance(answer["margin_bps"], bool)
            and revenue > 0
            and 0 <= operating_cost < revenue
            and remainder == 0
            and answer["gross_profit"] == gross_profit
            and answer["margin_bps"] == margin_bps
        )
    except (ArithmeticError, TypeError):
        arithmetic_valid = False
    evidence_valid = evidence == list(EVIDENCE_IDS)
    return TaskVerification(format_valid, arithmetic_valid, evidence_valid)


def _parse_response(response: object) -> tuple[Mapping[str, Any] | None, bool]:
    exact_envelope = True
    if isinstance(response, str):
        reasoning = re.fullmatch(r"\s*<think>.*?</think>\s*(.*)", response, re.DOTALL)
        if reasoning is not None:
            response = reasoning.group(1)
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            match = re.fullmatch(r"\s*```(?:json)?\s*(\{.*\})\s*```\s*", response, re.DOTALL)
            if match is None:
                return None, False
            try:
                response = json.loads(match.group(1))
            except json.JSONDecodeError:
                return None, False
            exact_envelope = False
    if not isinstance(response, Mapping) or not {"result", "evidence"}.issubset(response):
        return None, False
    exact_envelope = exact_envelope and set(response) == {"result", "evidence"}
    return response, exact_envelope


def score_response(task: SyntheticFinanceTask, response: object) -> TaskVerification:
    """Score a model response using exact arithmetic, evidence, and format checks."""

    parsed, exact_envelope = _parse_response(response)
    if parsed is None:
        return TaskVerification(False, False, False)
    result = _strict_object(parsed["result"], RESULT_FIELDS)
    if result is None or not isinstance(parsed["evidence"], list):
        return TaskVerification(False, False, False)
    arithmetic_valid = result == task.expected_result
    evidence_valid = parsed["evidence"] == list(task.evidence_ids)
    return TaskVerification(exact_envelope, arithmetic_valid, evidence_valid)


def evaluate_responses(tasks: Sequence[SyntheticFinanceTask], responses: Sequence[object]) -> EvaluationSummary:
    """Compute component rates over a held-out set."""

    if len(tasks) != len(responses):
        raise ValueError("tasks and responses must have equal length")
    if not tasks:
        raise ValueError("at least one response is required")
    scores = [score_response(task, response) for task, response in zip(tasks, responses, strict=True)]
    count = len(scores)
    return EvaluationSummary(
        count=count,
        format_rate=sum(score.format_valid for score in scores) / count,
        arithmetic_rate=sum(score.arithmetic_valid for score in scores) / count,
        evidence_rate=sum(score.evidence_valid for score in scores) / count,
        exact_rate=sum(score.accepted for score in scores) / count,
    )
