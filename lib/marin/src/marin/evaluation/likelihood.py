# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced likelihood scoring with exact token and target boundaries."""

import json
import math
import time
from dataclasses import dataclass, field
from typing import TextIO

import requests


def continuation_start(prefix_ids: list[int], full_ids: list[int], single_token: bool) -> int:
    """Reject boundary merges instead of silently scoring conditioning text."""
    assert full_ids[: len(prefix_ids)] == prefix_ids, "Tokenizer merges across the target boundary"
    assert len(full_ids) > len(prefix_ids)
    if single_token:
        assert len(full_ids) == len(prefix_ids) + 1, "Candidate must be exactly one token"
    return len(prefix_ids)


def completion_score(response: dict, token_ids: list[int], start: int, target: str) -> float:
    """Score observed continuation tokens, checking the echoed token identities."""
    assert 0 < start < len(token_ids), "Scored tokens require preceding context, including BOS"
    assert len(response["choices"]) == 1
    probabilities = response["choices"][0]["logprobs"]
    assert probabilities["tokens"] == [f"token_id:{token}" for token in token_ids]
    assert len(probabilities["token_logprobs"]) == len(token_ids)
    assert response["usage"]["prompt_tokens"] == len(token_ids)
    # Some vLLM builds account for one internal token when max_tokens=0.
    # Exact echoed-token equality above ensures no generated token is scored.
    assert response["usage"]["completion_tokens"] in {0, 1}
    native = response["choices"][0]["prompt_logprobs"]
    assert len(native) == len(token_ids)
    decoded = "".join(native[i][str(token_ids[i])]["decoded_token"] for i in range(start, len(token_ids)))
    assert decoded == target, "Scored tokens must decode to exactly the target"
    scores = probabilities["token_logprobs"][start:]
    assert all(score is not None and math.isfinite(score) for score in scores)
    return math.fsum(scores)


@dataclass
class ScoringClient:
    """Caller supplies an authenticated session and a private output stream."""

    session: requests.Session
    base_url: str
    model: str
    evidence: TextIO
    max_requests: int
    max_seconds: float
    started: float = field(default_factory=time.monotonic)
    requests_sent: int = 0
    fingerprint: str | None = None
    tokenizations: dict[str, list[int]] = field(default_factory=dict)

    def request(self, route: str, payload: dict | None) -> dict:
        remaining = self.max_seconds - (time.monotonic() - self.started)
        if self.requests_sent >= self.max_requests or remaining <= 0:
            raise RuntimeError("Request or elapsed-time budget exhausted")
        self.requests_sent += 1
        started = time.monotonic()
        # Never write the base URL or authentication headers into evidence.
        try:
            response = self.session.request(
                "GET" if payload is None else "POST",
                self.base_url.rstrip("/") + "/" + route,
                json=payload,
                timeout=min(45, remaining),
                allow_redirects=False,
            )
        except requests.RequestException:
            self.evidence.write(json.dumps({"route": route, "request": payload, "error": "transport failure"}) + "\n")
            self.evidence.flush()
            raise RuntimeError("Scoring transport failed; endpoint details suppressed") from None
        record = {
            "route": route,
            "request": payload,
            "status": response.status_code,
            "elapsed": time.monotonic() - started,
            "response_text": response.text,
        }
        self.evidence.write(json.dumps(record) + "\n")
        self.evidence.flush()
        if response.status_code != 200:
            raise RuntimeError(f"Scoring request failed with HTTP {response.status_code}; inspect private evidence")
        return response.json()

    def verify_model(self) -> None:
        models = self.request("v1/models", None)
        assert self.model in {model["id"] for model in models["data"]}

    def tokenize(self, text: str) -> list[int]:
        if text not in self.tokenizations:
            result = self.request("tokenize", {"model": self.model, "prompt": text, "add_special_tokens": True})
            tokens = result["tokens"]
            assert all(isinstance(token, int) for token in tokens)
            assert result["count"] == len(tokens)
            if len(self.tokenizations) >= 128:
                self.tokenizations.pop(next(iter(self.tokenizations)))
            self.tokenizations[text] = tokens
        return self.tokenizations[text]

    def score(self, prefix: str, target: str, single_token: bool) -> dict:
        prefix_ids = self.tokenize(prefix)
        tokens = self.tokenize(prefix + target)
        start = continuation_start(prefix_ids, tokens, single_token)
        assert start > 0, "Tokenizer must provide model-required BOS for unconditioned likelihood"
        payload = {
            "model": self.model,
            "prompt": tokens,
            "add_special_tokens": False,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
            "temperature": 1.0,
            "return_tokens_as_token_ids": True,
        }
        response = self.request("v1/completions", payload)
        assert response["model"] == self.model
        fingerprint = response["system_fingerprint"]
        assert isinstance(fingerprint, str) and fingerprint
        if self.fingerprint is None:
            self.fingerprint = fingerprint
        assert fingerprint == self.fingerprint, "Serving build changed during scoring"
        return {
            "logprob": completion_score(response, tokens, start, target),
            "target_ids": tokens[start:],
            "prefix_tokens": start,
            "input_tokens": len(tokens),
            "usage": response["usage"],
        }
