# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate static judge contracts without loading provider transports."""

import math
import re
import string
from collections.abc import Callable
from dataclasses import dataclass

from verifyit.grade import InvalidTask, empty_output_policy
from verifyit.modes.grade_ifeval import resolve_checks
from verifyit.modes.ifeval import Check
from verifyit.reasoning_markup import REASONING_DELIMITERS
from verifyit.spec import (
    RUBRIC_ANSWERS,
    RUBRIC_CHECKLIST,
    RUBRIC_LABELS,
    RUBRIC_PAIRWISE,
    RUBRIC_REFERENCE,
    RUBRICS,
    CandidateStep,
    Constraint,
    ExtractGroup,
    JudgeCompletionPolicy,
    JudgeSpec,
    StripOuterWhitespace,
    StripReasoningBlocks,
)

ANSWERS_FIELDS = frozenset({"question", "references", "candidate"})
LABEL_FIELDS = frozenset({"question", "reference", "candidate", "context"})
PAIRWISE_FIELDS = frozenset({"question", "answer_a", "answer_b"})


@dataclass(frozen=True)
class ValidatedJudgeSpec:
    """Resolved references, criteria and deterministic gates for one judge contract."""

    references: tuple[str, ...]
    criteria: tuple[str, ...]
    checks: list[tuple[Constraint, Check]]
    candidate_steps: tuple[Callable[[str], str], ...]


def validated_judge_spec(spec: JudgeSpec) -> ValidatedJudgeSpec:
    """Validate the static judge contract and resolve its deterministic gates."""
    if type(spec.context_char_limit) is not int or spec.context_char_limit < 1:
        raise InvalidTask("judge context character limit must be positive")
    if type(spec.max_turns) is not int or spec.max_turns != 1:
        raise InvalidTask("the judge harness supports exactly one conversation turn")
    if type(spec.parse_attempts) is not int or spec.parse_attempts < 1:
        raise InvalidTask("judge parse attempts must be positive")
    temperature = spec.sampling.temperature
    if temperature is not None and (type(temperature) not in (int, float) or not 0 <= temperature <= 2):
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
    if spec.system_role not in ("system", "developer"):
        raise InvalidTask("judge system_role must be system or developer")
    if not isinstance(spec.response_format, dict):
        raise InvalidTask("judge response_format must be a table")
    if spec.response_format and (spec.rubric != RUBRIC_LABELS or spec.api != "chat_completions"):
        raise InvalidTask("judge response_format requires the labels rubric over Chat Completions")
    if spec.rubric in (RUBRIC_LABELS, RUBRIC_PAIRWISE, RUBRIC_ANSWERS) and (
        not isinstance(spec.system_prompt, str)
        or not isinstance(spec.prompt_template, str)
        or not isinstance(spec.question, str)
        or not all(isinstance(reference, str) for reference in spec.references)
    ):
        raise InvalidTask("judge templates, question and references must be strings")
    references = tuple(reference for reference in spec.references if reference.strip())
    criteria = tuple(criterion for criterion in spec.criteria if criterion.strip())
    if spec.rubric == RUBRIC_REFERENCE and not references:
        raise InvalidTask("judge rubric 'reference' needs non-empty reference answers")
    if spec.rubric == RUBRIC_CHECKLIST and not criteria:
        raise InvalidTask("judge rubric 'checklist' needs non-empty criteria")
    if spec.rubric == RUBRIC_LABELS:
        _validate_label_spec(spec, references)
    if spec.rubric == RUBRIC_ANSWERS:
        _validate_answers_spec(spec, references)
    if spec.rubric == RUBRIC_PAIRWISE:
        _validate_pairwise_spec(spec)
    elif spec.label_weights or spec.verdict_patterns:
        raise InvalidTask("label weights and verdict patterns apply only to pairwise judges")
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
        references=references,
        criteria=criteria,
        checks=resolve_checks(spec.constraints) if spec.constraints else [],
        candidate_steps=tuple(_candidate_step(step) for step in spec.candidate_steps),
    )


def validate_judge_spec(spec: JudgeSpec) -> None:
    """Validate a complete judge contract before any provider side effects."""
    validated_judge_spec(spec)
    empty_output_policy(spec)


def answer_text(text: str, strip_reasoning: bool, *, unfinished_error: str | None = None) -> str:
    """Strip outer whitespace and optionally remove the reasoning preceding an answer."""
    if not strip_reasoning:
        return text.strip()
    for opening, closing in REASONING_DELIMITERS:
        text = re.sub(re.escape(opening) + ".*?" + re.escape(closing), "", text, flags=re.DOTALL)
    closing = max((text.rfind(end) + len(end) for _, end in REASONING_DELIMITERS if end in text), default=0)
    text = text[closing:]
    if any(opening in text for opening, _ in REASONING_DELIMITERS):
        if unfinished_error is not None:
            raise RuntimeError(unfinished_error)
        return ""
    return text.strip().removesuffix("<|eot_id|>").strip()


def _candidate_step(step: CandidateStep) -> Callable[[str], str]:
    match step:
        case StripReasoningBlocks():
            return lambda text: answer_text(text, strip_reasoning=True)
        case StripOuterWhitespace():
            return str.strip
        case ExtractGroup(pattern=pattern):
            try:
                compiled = re.compile(pattern)
            except re.error as error:
                raise InvalidTask(f"malformed extract_group pattern: {error}") from error
            if not compiled.groups:
                raise InvalidTask("extract_group pattern must capture the answer in a group")
            return lambda text: found[1] if (found := compiled.search(text)) else text
    raise InvalidTask(f"unknown candidate step {step!r}")


def _validate_label_spec(spec: JudgeSpec, references: tuple[str, ...]) -> None:
    if len(references) != 1 or len(spec.references) != 1 or not spec.prompt_template.strip() or not spec.label_scores:
        raise InvalidTask("label rubric requires one reference, a prompt template and labels")
    if not isinstance(spec.label_scan, str) or spec.label_scan not in {"literal", "lines", "whole", "json"}:
        raise InvalidTask("label_scan must be literal, lines, whole or json")
    if not isinstance(spec.label_field, str) or (spec.label_scan == "json") != bool(spec.label_field.strip()):
        raise InvalidTask("label_field names the verdict member exactly when label_scan is json")
    if type(spec.strip_reasoning_blocks) is not bool:
        raise InvalidTask("strip_reasoning_blocks must be boolean")
    _validate_verdict_labels(spec)
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
    if not {"reference", "candidate"} <= _template_fields(spec, LABEL_FIELDS):
        raise InvalidTask("judge template must include reference and candidate")


def _validate_pairwise_spec(spec: JudgeSpec) -> None:
    # The baseline may legitimately be blank: Arena-Hard judges an empty baseline answer too.
    if len(spec.references) != 1 or not spec.prompt_template.strip():
        raise InvalidTask("pairwise rubric requires one baseline answer and a prompt template")
    if spec.label_metrics or spec.exact_gate_answers or spec.exact_gate_label:
        raise InvalidTask("pairwise rubric reports its own metric and has no exact gate")
    if not spec.verdict_patterns:
        raise InvalidTask("pairwise rubric requires verdict patterns")
    for pattern in spec.verdict_patterns:
        try:
            groups = re.compile(pattern).groups
        except re.error as error:
            raise InvalidTask(f"malformed verdict pattern {pattern!r}") from error
        if groups != 1:
            raise InvalidTask("each verdict pattern must capture exactly one group")
    _validate_verdict_labels(spec)
    if not spec.label_scores:
        raise InvalidTask("pairwise rubric requires verdict labels")
    if not isinstance(spec.label_weights, dict) or not set(spec.label_weights) <= set(spec.label_scores):
        raise InvalidTask("label weights must name declared verdicts")
    for weight in spec.label_weights.values():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 < weight < math.inf:
            raise InvalidTask("label weights must be finite and positive")
    if not {"answer_a", "answer_b"} <= _template_fields(spec, PAIRWISE_FIELDS):
        raise InvalidTask("pairwise template must include answer_a and answer_b")


def _validate_verdict_labels(spec: JudgeSpec) -> None:
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
    if len(set(normalized_labels)) != len(spec.label_scores):
        raise InvalidTask("verdict labels collide after case normalization")


def _template_fields(spec: JudgeSpec, allowed: frozenset[str]) -> set[str]:
    """The fields the judge templates use, each of which must be in ``allowed``."""
    used = set()
    for template in (spec.system_prompt, spec.prompt_template):
        try:
            for _, name, format_spec, conversion in string.Formatter().parse(template):
                if name is None:
                    continue
                if name not in allowed or format_spec or conversion:
                    raise InvalidTask("judge template has an unsupported field")
                used.add(name)
        except ValueError as error:
            raise InvalidTask("malformed judge template") from error
    return used


def _validate_answers_spec(spec: JudgeSpec, references: tuple[str, ...]) -> None:
    if not references or len(references) != len(spec.references) or not spec.prompt_template.strip():
        raise InvalidTask("answers rubric requires nonempty references and a prompt template")
    if not {"references", "candidate"} <= _template_fields(spec, ANSWERS_FIELDS):
        raise InvalidTask("judge template must include references and candidate")
