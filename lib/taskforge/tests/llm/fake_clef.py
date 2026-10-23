# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Vendored from the WS1 clefclient package (clefclient/fake.py, 2026-10-15); only imports changed.

"""FakeClef: an in-process /v1/systemone that validates requests and returns canned answers.

Plug it into an ``httpx.AsyncClient`` as ``transport=httpx.MockTransport(fake.handle)`` so a client runs end to end
with no model.
Answers carry the vLLM shape (``probabilities``, ``confidence``, ``diagnostics``) and the
proxy's ``x-clef-context`` header; ``input_tokens`` is exact when a tokenizer is given.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from taskforge.llm.clef.budget import MAX_LENGTH, ClefTokenizer, input_tokens, render
from taskforge.llm.clef.schema import ChoiceQuestion, ClefRequestError, NoulQuestion, Question, SystemOneRequest

# Canned distribution for a question: option key -> probability (choice: names; noul: yes/no;
# score: level index strings). Defaults to 0.7 on the first option.
Prober = Callable[[str, Question], dict[str, float]]


def option_keys(question: Question) -> list[str]:
    if isinstance(question, ChoiceQuestion):
        return [name for name, _ in question.options]
    if isinstance(question, NoulQuestion):
        return ["yes", "no"]
    return [str(i) for i in range(len(question.levels))]


def first_option_prober(_: str, question: Question) -> dict[str, float]:
    first, *rest = option_keys(question)
    if not rest:
        return {first: 1.0}
    return {first: 0.7} | {k: 0.3 / len(rest) for k in rest}


def answer_json(question: Question, p: dict[str, float]) -> dict[str, Any]:
    best = max(p, key=p.__getitem__)
    if isinstance(question, ChoiceQuestion):
        return {"type": "choice", "choice": best, "probabilities": p, "confidence": p[best]}
    if isinstance(question, NoulQuestion):
        return {"type": "noul", "noul": p["yes"], "probabilities": p, "confidence": p[best]}
    return {
        "type": "score",
        "score": round(sum(int(k) * v for k, v in p.items()), 4),
        "legend": {str(i): level for i, level in enumerate(question.levels)},
        "probabilities": p,
        "confidence": p[best],
    }


@dataclass
class FakeClef:
    tokenizer: ClefTokenizer | None = None
    prober: Prober = first_option_prober
    max_length: int = MAX_LENGTH
    token: str | None = None  # when set, require this bearer
    unavailable: list[int] = field(default_factory=list)  # status per leading call: 503 / 502 before answering
    retry_after: float | None = None  # sent with 503s when set (Clef's proxy); None = front-door style body only
    seen: list[SystemOneRequest] = field(default_factory=list)
    calls: int = 0

    def tokens(self, request: SystemOneRequest, max_length: int) -> int:
        if self.tokenizer is not None:
            return input_tokens(self.tokenizer, request, max_length)
        return min(max_length, len(render(request.state)) // 4 + 64 * len(request.questions))

    def answer(self, body: dict[str, Any], max_length: int | None = None) -> dict[str, Any]:
        """The response body for a request body; raises ClefRequestError for a 400. ``max_length`` is a
        pinned engine window (x-clef-window); the default is the fake's own."""
        max_length = max_length or self.max_length
        request = SystemOneRequest.from_json(body)
        self.seen.append(request)
        answers = {qid: answer_json(q, self.prober(qid, q)) for qid, q in request.questions.items()}
        return {
            "id": f"decision-fake-{len(self.seen)}",
            "object": "structured_decision",
            "model": request.model,
            "answers": answers,
            "usage": {"input_tokens": self.tokens(request, max_length), "output_tokens": 0},
            "diagnostics": {qid: {"label_mass": 1.0, "argmax_is_label": True} for qid in answers},
        }

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.token is not None and request.headers.get("authorization") != f"Bearer {self.token}":
            return httpx.Response(401, json={"error": "unauthorized"})
        if request.url.path == "/admin/clef":
            return httpx.Response(
                200,
                json={
                    "backend": "fake",
                    "healthy": not self.unavailable,
                    "queued": 0,
                    "inflight": 0,
                    "last_request_ts": None,
                    "requests_total": len(self.seen),
                },
            )
        if request.method != "POST" or request.url.path != "/v1/systemone":
            return httpx.Response(404, json={"error": "not found"})
        if self.unavailable:
            status = self.unavailable.pop(0)
            headers = {"Retry-After": str(self.retry_after)} if self.retry_after is not None else {}
            return httpx.Response(status, json={"error": {"code": "clef_unavailable"}}, headers=headers)
        pinned = request.headers.get("x-clef-window")
        window = int(pinned) if pinned is not None else self.max_length
        try:
            body = self.answer(json.loads(request.content), window)
        except (ClefRequestError, ValueError, KeyError, TypeError) as e:  # the server's 400
            return httpx.Response(400, json={"error": str(e)})
        return httpx.Response(200, json=body, headers={"x-clef-context": str(window), "x-clef-retries": "0"})
