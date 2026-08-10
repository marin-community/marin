# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A bounded Harbor agent for one-shot AIME smoke evaluations."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent  # pyrefly: ignore[missing-import]
from harbor.environments.base import BaseEnvironment  # pyrefly: ignore[missing-import]
from harbor.models.agent.context import AgentContext  # pyrefly: ignore[missing-import]
from upath import UPath  # pyrefly: ignore[missing-import]

logger = logging.getLogger(__name__)

_BOXED_ANSWER = re.compile(r"\\boxed\s*\{\s*([0-9]{1,3})\s*\}")
_STANDALONE_INTEGER = re.compile(r"(?<![0-9])([0-9]{1,3})(?![0-9])")
_INVALID_AIME_ANSWER = "-1"
_RESPONSE_LOG = "response.txt"
_RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_MAX_RETRY_DELAY = 5.0


def _model_request_should_retry(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _RETRYABLE_HTTP_STATUS
    return isinstance(exc, (TimeoutError, urllib.error.URLError))


def _request_with_retry(call: Callable[[], object], *, max_attempts: int, retry_initial: float) -> object:
    """Call ``call``, retrying transient endpoint failures with capped exponential backoff.

    The backoff doubles each attempt with no jitter and is clamped to ``_MAX_RETRY_DELAY`` so a
    disappearing endpoint cannot stall the smoke past its bounded budget. Inlined rather than pulled
    from ``rigging`` so the agent imports cleanly in the isolated Harbor driver environment, which
    does not install ``marin-rigging``.
    """
    for attempt in range(max_attempts):
        try:
            return call()
        except Exception as exc:
            if attempt + 1 >= max_attempts or not _model_request_should_retry(exc):
                raise
            delay = min(retry_initial * 2.0**attempt, _MAX_RETRY_DELAY)
            logger.warning(
                "Harbor model request failed (attempt %d/%d), retrying in %.2fs: %s",
                attempt + 1,
                max_attempts,
                delay,
                exc,
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def _completion_content(
    *,
    api_base: str,
    model: str,
    instruction: str,
    max_tokens: int,
    timeout: float,
    max_attempts: int,
    retry_initial: float,
) -> str:
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{instruction}\n\nSolve the problem and end with the final integer answer "
                            "between 0 and 999."
                        ),
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def request_payload() -> object:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    payload = _request_with_retry(request_payload, max_attempts=max_attempts, retry_initial=retry_initial)
    if not isinstance(payload, Mapping):
        raise ValueError("model response must be a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("model response must contain at least one choice")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ValueError("model response choice must contain text content")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("model response choice must contain text content")
    return content


def _aime_answer(content: str) -> str:
    boxed = _BOXED_ANSWER.findall(content)
    if boxed:
        return boxed[-1]
    integers = _STANDALONE_INTEGER.findall(content)
    return integers[-1] if integers else _INVALID_AIME_ANSWER


class SingleTurnAimeAgent(BaseAgent):
    """Write the final boxed value, or last AIME-sized integer, from a bounded request.

    Unlike an interactive terminal agent, this smoke agent deliberately accepts an OpenAI
    completion whose finish reason is ``length``. That keeps weak or newly initialized models
    gradeable without letting missing EOS behavior turn a Harbor integration smoke into an
    unbounded agent retry loop.
    """

    @staticmethod
    def name() -> str:
        return "single-turn-aime"

    def __init__(
        self,
        logs_dir: Path | UPath,
        model_name: str,
        api_base: str,
        answer_path: str,
        max_tokens: int,
        request_timeout: float = 300,
        request_max_attempts: int = 4,
        request_retry_initial: float = 1.0,
        **kwargs: Any,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if request_max_attempts <= 0:
            raise ValueError("request_max_attempts must be positive")
        if request_retry_initial <= 0:
            raise ValueError("request_retry_initial must be positive")
        self._api_base = api_base
        self._answer_path = answer_path
        self._max_tokens = max_tokens
        self._request_timeout = request_timeout
        self._request_max_attempts = request_max_attempts
        self._request_retry_initial = request_retry_initial

    def version(self) -> str:
        return "1.1.0"

    async def setup(self, _environment: BaseEnvironment) -> None:
        return

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        _context: AgentContext,
    ) -> None:
        assert self.model_name is not None
        served_model = self.model_name.split("/", maxsplit=1)[-1]
        content = await asyncio.to_thread(
            _completion_content,
            api_base=self._api_base,
            model=served_model,
            instruction=instruction,
            max_tokens=self._max_tokens,
            timeout=self._request_timeout,
            max_attempts=self._request_max_attempts,
            retry_initial=self._request_retry_initial,
        )
        (self.logs_dir / _RESPONSE_LOG).write_text(content)
        answer = _aime_answer(content)
        command = f"printf '%s\\n' {shlex.quote(answer)} > {shlex.quote(self._answer_path)}"
        result = await environment.exec(command=command)
        if result.return_code != 0:
            detail = result.stderr or result.stdout or "no output"
            raise RuntimeError(f"failed to write AIME answer: {detail}")
