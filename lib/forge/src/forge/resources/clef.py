# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Answer VerifyIt's reference-judge prompt with one Clef decision read.

Clef (``POST /v1/systemone`` behind the GLM gateway, model ``clef``) answers typed questions about a
state and does not generate text. This backend takes the prompt VerifyIt's ``reference`` rubric sends,
recovers its question, references and candidate from the fixed template, asks Clef one whole-answer score
question, and replies with the ``SCORE:`` line VerifyIt parses. It is a ``CompletionBackend``, so the
judge's ``InferenceRequest`` effects record and replay it like any other inference.

Every failure to reach a serving Clef (transport errors, 404 no route, 408/429, any 5xx such as the
gateway's 503 ``clef_unavailable``) waits out the response's ``retry-after`` and raises ``ModelUnavailable``,
which ``WaitingBackend`` holds on; other refusals are ``ModelRequestRejected``.
"""

import asyncio
import math
import string
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, override

import httpx
from taskforge.llm.client import (
    Attempt,
    AttemptOutcome,
    Completion,
    FinishReason,
    ModelRequestRejected,
    ModelUnavailable,
    Usage,
)
from taskforge.llm.glm_client import EndpointSource
from taskforge.llm.policy import LLMPolicy
from verifyit.modes.grade_judge import REFERENCE_PROMPT

from forge.resources.inference import CompletionBackend

CLEF_MODEL = "clef"
QUESTION_ID = "whole"
# Source: mark/taskforge-clef-read 080513645e, taskforge/spec/decision.py (the form measured at AUC 0.999,
# glm-infer/notes/clef.md 2026-10-16).
QUESTION = (
    "Compare the whole candidate answer with the reference answer (which earned full credit), against the whole "
    "rubric. Judge meaning, not wording: accept the listed equivalences; a negated, quoted, hypothetical or "
    "retracted mention does not count as asserting something; keyword lists without the corresponding claim earn "
    "nothing; when the answer states several values, the designated final block governs."
)
LEVELS = (
    "missing: the candidate does not address this (empty, off-task, or only keywords)",
    "contradicts the reference: the candidate addresses it but its content disagrees with the reference",
    "partially matches: part agrees with the reference, part is wrong or missing",
    "equivalent to the reference: different notation, wording, units or order, same correct content",
    "matches exactly: the same content as the reference",
)
FULL_LEVEL = 2.5
"""An expected level at or above this (between "partially matches" and "equivalent") scores 1, anything below 0:
fooling cases earn no credit, and the live smoke read a hedge over several values at 1.78."""
_SEGMENTS = tuple(literal for literal, *_ in string.Formatter().parse(REFERENCE_PROMPT))
_QUESTION_BLOCK = ("\nQuestion:\n", "\n")
UNAVAILABLE_STATUSES = frozenset({404, 408, 429})


def reference_fields(prompt: str) -> tuple[str, str, str]:
    """The question, reference lines and candidate of a VerifyIt ``reference`` prompt.

    The question and references are author text and the candidate is untrusted, so the candidate is taken
    between the first candidate header and the template's fixed tail.
    """
    head, references, candidate, tail = _SEGMENTS
    if not prompt.startswith(head) or not prompt.endswith(tail):
        raise ValueError("Clef grades only VerifyIt's reference-rubric prompt")
    body = prompt[len(head) : len(prompt) - len(tail)]
    question, found, rest = body.partition(references)
    refs, found_candidate, answer = rest.partition(candidate)
    if not found or not found_candidate:
        raise ValueError("Clef grades only VerifyIt's reference-rubric prompt")
    if question.startswith(_QUESTION_BLOCK[0]) and question.endswith(_QUESTION_BLOCK[1]):
        question = question[len(_QUESTION_BLOCK[0]) : -len(_QUESTION_BLOCK[1])]
    return question, refs, answer


def score(probabilities: Mapping[str, Any]) -> tuple[float, float]:
    """The expected level of a five-level read and the VerifyIt score it earns."""
    weights = [float(probabilities[str(level)]) for level in range(len(LEVELS))]
    if any(not math.isfinite(w) or w < 0 for w in weights) or not sum(weights) > 0:
        raise ValueError(f"Clef probabilities are not a distribution: {weights}")
    level = sum(index * weight for index, weight in enumerate(weights)) / sum(weights)
    return level, 1.0 if level >= FULL_LEVEL else 0.0


def _retry_after(response: httpx.Response) -> float:
    try:
        return max(0.0, float(response.headers.get("retry-after", 0)))
    except ValueError:
        return 0.0


@dataclass(frozen=True, slots=True)
class ClefBackend(CompletionBackend):
    """Clef behind the gateway that ``source`` resolves; the bearer token is the endpoint's."""

    source: EndpointSource
    http: httpx.AsyncClient
    timeout: float

    @property
    @override
    def model(self) -> str:
        return CLEF_MODEL

    @override
    async def complete(
        self,
        messages: Sequence[dict[str, object]],
        policy: LLMPolicy,
        request_fields: Mapping[str, object] | None = None,
    ) -> Completion:
        if len(messages) != 1 or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str):
            raise ValueError("Clef grades one user prompt")
        question, references, candidate = reference_fields(messages[0]["content"])
        body = {
            "model": CLEF_MODEL,
            "state": {
                "task_and_rubric": question,
                "y_reference_answer": references,
                "z_candidate_answer": candidate,
            },
            "questions": {QUESTION_ID: {"type": "score", "instructions": QUESTION, "criteria": list(LEVELS)}},
        }
        started = time.monotonic()

        def attempt(outcome: AttemptOutcome, status: int | None, detail: str) -> Attempt:
            return Attempt(0, outcome, 0, started, time.monotonic() - started, status, detail, ())

        try:
            endpoint = await asyncio.to_thread(self.source.current)
            response = await self.http.post(
                f"{endpoint.base_url}/systemone",
                json=body,
                headers={"Authorization": f"Bearer {endpoint.token}"},
                timeout=self.timeout,
            )
        except (httpx.TransportError, RuntimeError) as error:
            failed = attempt(AttemptOutcome.TRANSPORT_ERROR, None, type(error).__name__)
            raise ModelUnavailable(f"Clef unreachable: {type(error).__name__}", (failed,)) from error
        status = response.status_code
        if status in UNAVAILABLE_STATUSES or status >= 500:
            failed = attempt(AttemptOutcome.RETRYABLE_STATUS, status, response.text[:500])
            # The gateway already queued this read; honour its pacing before WaitingBackend's own backoff.
            await asyncio.sleep(min(_retry_after(response), self.timeout))
            raise ModelUnavailable(f"Clef answered {status}", (failed,))
        if status != 200:
            raise ModelRequestRejected(status, response.text[:500], (attempt(AttemptOutcome.REJECTED, status, ""),))
        try:
            payload = response.json()
            level, earned = score(payload["answers"][QUESTION_ID]["probabilities"])
            tokens = int(payload["usage"]["input_tokens"])
        except (KeyError, TypeError, ValueError) as error:
            raise ModelRequestRejected(200, f"Clef answer is not a score read: {error}", ()) from error
        return Completion(
            f"Clef expected level {level:.3f} of {len(LEVELS) - 1}.\nSCORE: {earned:g}",
            "",
            (),
            FinishReason.STOP,
            Usage(tokens, 0, 0, 0),
            time.monotonic() - started,
            0.0,
            0.0,
            0,
            (attempt(AttemptOutcome.COMPLETED, status, ""),),
        )
