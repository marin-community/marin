# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate static judge contracts without loading provider transports."""

import math
import string
from dataclasses import dataclass

from verifyit.grade import InvalidTask, empty_output_policy
from verifyit.modes.grade_ifeval import resolve_checks
from verifyit.modes.ifeval import Check
from verifyit.spec import (
    RUBRIC_CHECKLIST,
    RUBRIC_LABELS,
    RUBRIC_REFERENCE,
    RUBRICS,
    Constraint,
    JudgeCompletionPolicy,
    JudgeSpec,
)


@dataclass(frozen=True)
class ValidatedJudgeSpec:
    """Resolved references, criteria and deterministic gates for one judge contract."""

    references: tuple[str, ...]
    criteria: tuple[str, ...]
    checks: list[tuple[Constraint, Check]]


def validated_judge_spec(spec: JudgeSpec) -> ValidatedJudgeSpec:
    """Validate the static judge contract and resolve its deterministic gates."""
    if type(spec.context_char_limit) is not int or spec.context_char_limit < 1:
        raise InvalidTask("judge context character limit must be positive")
    if type(spec.max_turns) is not int or spec.max_turns != 1:
        raise InvalidTask("the judge harness supports exactly one conversation turn")
    if type(spec.parse_attempts) is not int or spec.parse_attempts < 1:
        raise InvalidTask("judge parse attempts must be positive")
    if spec.temperature is not None and (
        isinstance(spec.temperature, bool)
        or not isinstance(spec.temperature, (int, float))
        or not 0 <= spec.temperature <= 2
    ):
        raise InvalidTask("judge temperature must be finite and between zero and two")
    if spec.chat_template_kwargs and spec.api != "chat_completions":
        raise InvalidTask("chat template options require Chat Completions")
    if not isinstance(spec.completion_policy, JudgeCompletionPolicy):
        raise InvalidTask("Unknown judge completion policy")
    if type(spec.transport_attempts) is not int or spec.transport_attempts < 1:
        raise InvalidTask("Judge transport attempts must be positive")
    if (
        isinstance(spec.transport_budget, bool)
        or not isinstance(spec.transport_budget, (int, float))
        or not math.isfinite(spec.transport_budget)
        or spec.transport_budget <= 0
    ):
        raise InvalidTask("Judge transport budget must be finite and positive")
    if not isinstance(spec.api, str) or spec.api not in {"chat_completions", "responses"}:
        raise InvalidTask("judge api must be chat_completions or responses")
    if spec.api == "responses" and spec.rubric != RUBRIC_LABELS:
        raise InvalidTask("Responses transport requires the labels rubric")
    if not isinstance(spec.label_case, str) or spec.label_case not in {"sensitive", "upper"}:
        raise InvalidTask("label_case must be sensitive or upper")
    if spec.rubric not in RUBRICS:
        raise InvalidTask(f"unknown judge rubric {spec.rubric!r}; known rubrics: {sorted(RUBRICS)}")
    if not isinstance(spec.strip_candidate_reasoning_blocks, bool):
        raise InvalidTask("strip_candidate_reasoning_blocks must be boolean")
    if spec.rubric == RUBRIC_LABELS and (
        not isinstance(spec.system_prompt, str)
        or not isinstance(spec.prompt_template, str)
        or not isinstance(spec.question, str)
        or not all(isinstance(reference, str) for reference in spec.references)
    ):
        raise InvalidTask("label rubric templates, question and references must be strings")
    references = tuple(reference for reference in spec.references if reference.strip())
    criteria = tuple(criterion for criterion in spec.criteria if criterion.strip())
    if spec.rubric == RUBRIC_REFERENCE and not references:
        raise InvalidTask("judge rubric 'reference' needs non-empty reference answers")
    if spec.rubric == RUBRIC_CHECKLIST and not criteria:
        raise InvalidTask("judge rubric 'checklist' needs non-empty criteria")
    if spec.rubric == RUBRIC_LABELS:
        _validate_label_spec(spec, references)
    for budget in (spec.max_completion_tokens, spec.incomplete_retry_tokens):
        if type(budget) is not int or budget < 0:
            raise InvalidTask("judge token budgets must be nonnegative integers")
    if spec.max_completion_tokens == 0:
        raise InvalidTask("judge token budget must be positive")
    if spec.incomplete_retry_tokens and spec.incomplete_retry_tokens <= spec.max_completion_tokens:
        raise InvalidTask("retry token budget must exceed the initial budget")
    if (
        isinstance(spec.request_timeout, bool)
        or not isinstance(spec.request_timeout, (int, float))
        or spec.request_timeout <= 0
        or not math.isfinite(spec.request_timeout)
    ):
        raise InvalidTask("judge request timeout must be finite and positive")
    if (
        not isinstance(spec.exact_gate_answers, tuple)
        or any(not isinstance(answer, str) or not answer for answer in spec.exact_gate_answers)
        or not isinstance(spec.exact_gate_label, str)
    ):
        raise InvalidTask("label exact gate requires nonempty answer strings and a label")
    if spec.exact_gate_answers or spec.exact_gate_label:
        if spec.rubric != RUBRIC_LABELS or not spec.exact_gate_answers or spec.exact_gate_label not in spec.label_scores:
            raise InvalidTask("label exact gate requires answers and a declared label")
    return ValidatedJudgeSpec(
        references=references, criteria=criteria, checks=resolve_checks(spec.constraints) if spec.constraints else []
    )


def validate_judge_spec(spec: JudgeSpec) -> None:
    """Validate a complete judge contract before any provider side effects."""
    validated_judge_spec(spec)
    empty_output_policy(spec)


def _validate_label_spec(spec: JudgeSpec, references: tuple[str, ...]) -> None:
    if len(references) != 1 or len(spec.references) != 1 or not spec.prompt_template.strip() or not spec.label_scores:
        raise InvalidTask("label rubric requires one reference, a prompt template and labels")
    if not isinstance(spec.label_scan, str) or spec.label_scan not in {"literal", "lines", "whole"}:
        raise InvalidTask("label_scan must be literal, lines or whole")
    if type(spec.strip_reasoning_blocks) is not bool:
        raise InvalidTask("strip_reasoning_blocks must be boolean")
    if not isinstance(spec.label_scores, dict):
        raise InvalidTask("verdict labels must be a label/reward table")
    for label, score in spec.label_scores.items():
        if not isinstance(label, str) or not label.strip() or label != label.strip() or "\n" in label:
            raise InvalidTask("verdict labels must be nonempty single lines")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= score <= 1
            or not math.isfinite(score)
        ):
            raise InvalidTask("verdict label rewards must be finite unit scalars")
    normalized_labels = [label.upper() if spec.label_case == "upper" else label for label in spec.label_scores]
    if len(set(normalized_labels)) != len(normalized_labels):
        raise InvalidTask("verdict labels collide after case normalization")
    if spec.label_metrics and set(spec.label_metrics) != set(spec.label_scores):
        raise InvalidTask("label metrics must cover every declared verdict")
    for metrics in spec.label_metrics.values():
        if (
            not isinstance(metrics, dict)
            or not metrics
            or any(
                not isinstance(name, str) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for name, value in metrics.items()
            )
        ):
            raise InvalidTask("label metrics require named finite observations")
    used = set()
    for template in (spec.system_prompt, spec.prompt_template):
        try:
            for _, name, format_spec, conversion in string.Formatter().parse(template):
                if name is None:
                    continue
                if name not in {"question", "reference", "candidate", "context"} or format_spec or conversion:
                    raise InvalidTask("judge template has an unsupported field")
                used.add(name)
        except ValueError as error:
            raise InvalidTask("malformed judge template") from error
    if not {"reference", "candidate"} <= used:
        raise InvalidTask("judge template must include reference and candidate")
