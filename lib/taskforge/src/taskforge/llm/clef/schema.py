# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copied from the WS1 clefclient package, phase11/ws1-clef-client/clefclient/schema.py in the WS1 scratch
# workspace (2026-10-15). That package has no repository; the source file's sha256 is
# b424aed8c3c1fb4038ced1669c5a25a57c4657ac2c266873fdb262ff505ed564.
# Changed: nothing.

"""Request and response model for Clef's ``POST /v1/systemone``.

Shapes follow vLLM's structured-decisions docs and Cloudflare's reference
``joint_schema_model.systemone``: a request is ``{model, state, questions}``; a question is
``{type, instructions, criteria}``; Clef has a fixed prompt, so a top-level ``instructions``
field is a 400 and is rejected here before it is sent.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

MODEL = "clef"
MAX_QUESTIONS = 64
MAX_CHOICE_OPTIONS = 26
MAX_SCORE_LEVELS = 10
REQUEST_FIELDS = frozenset({"model", "state", "questions", "cache_salt"})
QUESTION_FIELDS = frozenset({"type", "instructions", "criteria"})
# vLLM's Clef proxy: a 200 whose input_tokens is within this many of the window is a truncated read.
TRUNCATION_SLACK = 8

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class ClefRequestError(ValueError):
    """The request breaks the /v1/systemone contract (the server would answer 400)."""


class QuestionType(StrEnum):
    CHOICE = "choice"
    NOUL = "noul"
    SCORE = "score"


@dataclass(frozen=True)
class ChoiceQuestion:
    """Pick one named option. ``options`` maps option name to a description (or None)."""

    instructions: str
    options: tuple[tuple[str, str | None], ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.options) <= MAX_CHOICE_OPTIONS:
            raise ClefRequestError(f"choice needs 1..{MAX_CHOICE_OPTIONS} options, got {len(self.options)}")
        names = [name for name, _ in self.options]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ClefRequestError(f"choice option names must be non-empty and unique: {names}")

    @classmethod
    def of(cls, instructions: str, criteria: Mapping[str, str | None]) -> ChoiceQuestion:
        return cls(instructions, tuple(criteria.items()))

    def to_json(self) -> dict[str, Any]:
        return {"type": QuestionType.CHOICE.value, "instructions": self.instructions, "criteria": dict(self.options)}


@dataclass(frozen=True)
class NoulQuestion:
    """Yes/no. ``true``/``false`` optionally describe what each answer means."""

    instructions: str
    true: str | None = None
    false: str | None = None

    def to_json(self) -> dict[str, Any]:
        criteria = {k: v for k, v in (("true", self.true), ("false", self.false)) if v is not None}
        return {"type": QuestionType.NOUL.value, "instructions": self.instructions, "criteria": criteria or None}


@dataclass(frozen=True)
class ScoreQuestion:
    """An ordered scale; the answer is the expected 0-indexed level."""

    instructions: str
    levels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 2 <= len(self.levels) <= MAX_SCORE_LEVELS:
            raise ClefRequestError(f"score needs 2..{MAX_SCORE_LEVELS} levels, got {len(self.levels)}")

    def to_json(self) -> dict[str, Any]:
        return {"type": QuestionType.SCORE.value, "instructions": self.instructions, "criteria": list(self.levels)}


Question = ChoiceQuestion | NoulQuestion | ScoreQuestion


def question_from_json(qid: str, payload: Mapping[str, Any]) -> Question:
    extra = set(payload) - QUESTION_FIELDS
    if extra:
        raise ClefRequestError(f"question {qid!r}: unknown fields {sorted(extra)}")
    kind = QuestionType(payload["type"])
    instructions = payload.get("instructions") or ""
    criteria = payload.get("criteria")
    if kind is QuestionType.CHOICE:
        if not isinstance(criteria, Mapping):
            raise ClefRequestError(f"choice {qid!r}: criteria must map option name to description")
        return ChoiceQuestion.of(instructions, criteria)
    if kind is QuestionType.SCORE:
        if not isinstance(criteria, list):
            raise ClefRequestError(f"score {qid!r}: criteria must be a list of level names")
        return ScoreQuestion(instructions, tuple(criteria))
    criteria = criteria or {}
    if set(criteria) - {"true", "false"}:
        raise ClefRequestError(f"noul {qid!r}: criteria keys must be true/false")
    return NoulQuestion(instructions, criteria.get("true"), criteria.get("false"))


@dataclass(frozen=True)
class SystemOneRequest:
    """One Clef read: a state and up to 64 typed questions, answered in one forward pass."""

    state: JsonValue
    questions: Mapping[str, Question]
    model: str = MODEL

    def __post_init__(self) -> None:
        if not 1 <= len(self.questions) <= MAX_QUESTIONS:
            raise ClefRequestError(f"need 1..{MAX_QUESTIONS} questions, got {len(self.questions)}")
        if any(not isinstance(qid, str) or not qid for qid in self.questions):
            raise ClefRequestError("question ids must be non-empty strings")
        json.dumps(self.state)  # TypeError here, not on the wire

    @property
    def question_ids(self) -> tuple[str, ...]:
        return tuple(self.questions)

    def to_json(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "state": self.state,
            "questions": {qid: q.to_json() for qid, q in self.questions.items()},
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> SystemOneRequest:
        if "instructions" in payload:
            raise ClefRequestError("Clef's prompt is fixed: a top-level 'instructions' field is rejected")
        extra = set(payload) - REQUEST_FIELDS
        if extra:
            raise ClefRequestError(f"unknown request fields {sorted(extra)}")
        if "state" not in payload or not isinstance(payload.get("questions"), Mapping):
            raise ClefRequestError("state and questions are required")
        questions = {qid: question_from_json(qid, q) for qid, q in payload["questions"].items()}
        return cls(state=payload["state"], questions=questions, model=payload.get("model", MODEL))


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float


@dataclass(frozen=True)
class NoulAnswer:
    noul: float  # P(yes)
    probabilities: dict[str, float] | None = None  # vLLM: {"yes", "no"}; the HF reference omits it
    confidence: float | None = None


@dataclass(frozen=True)
class ScoreAnswer:
    score: float  # expected 0-indexed level
    probabilities: dict[str, float]  # by level index as a string
    confidence: float
    legend: dict[str, str] = field(default_factory=dict)


Answer = ChoiceAnswer | NoulAnswer | ScoreAnswer


@dataclass(frozen=True)
class Diagnostics:
    label_mass: float
    argmax_is_label: bool


def answer_from_json(payload: Mapping[str, Any]) -> Answer:
    kind = QuestionType(payload["type"])
    if kind is QuestionType.CHOICE:
        return ChoiceAnswer(payload["choice"], dict(payload["probabilities"]), float(payload["confidence"]))
    if kind is QuestionType.SCORE:
        return ScoreAnswer(
            float(payload["score"]),
            dict(payload["probabilities"]),
            float(payload["confidence"]),
            dict(payload.get("legend") or {}),
        )
    confidence = payload.get("confidence")
    return NoulAnswer(
        float(payload["noul"]),
        payload.get("probabilities"),
        None if confidence is None else float(confidence),
    )


@dataclass(frozen=True)
class SystemOneResponse:
    """A /v1/systemone answer plus the routing headers of Clef's proxy."""

    model: str
    answers: dict[str, Answer]
    input_tokens: int
    output_tokens: int
    diagnostics: dict[str, Diagnostics]
    context_window: int | None  # x-clef-context: the window of the engine that answered
    proxy_retries: int | None  # x-clef-retries: overflow retries inside the proxy
    id: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)  # the body as received

    @property
    def truncated(self) -> bool:
        """Clef silently truncates an overlong state; this is the proxy's truncation signature."""
        return self.context_window is not None and self.input_tokens >= self.context_window - TRUNCATION_SLACK

    @classmethod
    def from_json(cls, payload: Mapping[str, Any], headers: Mapping[str, str]) -> SystemOneResponse:
        context, retries = headers.get("x-clef-context"), headers.get("x-clef-retries")
        usage = payload["usage"]
        return cls(
            model=payload["model"],
            answers={qid: answer_from_json(a) for qid, a in payload["answers"].items()},
            input_tokens=int(usage["input_tokens"]),
            output_tokens=int(usage.get("output_tokens", 0)),
            diagnostics={
                qid: Diagnostics(float(d["label_mass"]), bool(d["argmax_is_label"]))
                for qid, d in (payload.get("diagnostics") or {}).items()
            },
            context_window=None if context is None else int(context),
            proxy_retries=None if retries is None else int(retries),
            id=payload.get("id"),
            raw=payload,
        )


Decision = str | bool | float


def decisions(response: SystemOneResponse, noul_cutoff: float = 0.5) -> dict[str, Decision]:
    """Plain decisions per question id: choice -> the chosen option name; noul -> P(yes) >= ``noul_cutoff``;
    score -> the expected 0-indexed level as Clef gives it. Probabilities stay on ``response.answers``."""
    out: dict[str, Decision] = {}
    for qid, answer in response.answers.items():
        if isinstance(answer, ChoiceAnswer):
            out[qid] = answer.choice
        elif isinstance(answer, NoulAnswer):
            out[qid] = answer.noul >= noul_cutoff
        else:
            out[qid] = answer.score
    return out
