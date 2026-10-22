# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Clef's ``POST /v1/systemone`` behind the Orion gateway: the production ``spec.decision.DecisionClient``, and a
``ClefReader`` for any mix of question types.

The host selects it from its own configuration (``ClefDecisionConfig``: the gateway URL, the name of the
environment variable holding the bearer token, the engine window); no proposal, author or task names it.

``read`` asks any mix of choice, noul and score questions and returns Clef's whole response; ``decide`` is one
``read`` of a grader's score questions, answered as level probabilities. Both share the client's HTTP pool and
in-flight limit. One read is bounded as a whole by the caller's ``timeout`` (for a grade, the verifier
deadline): the wait for an in-flight slot, fitting the state and the HTTP exchange. Nothing runs on the event
loop's default executor; fitting the state to the window tokenizes on the client's own thread pool, and the
HTTP call is async. The client retries nothing: a transient failure (a connection error or timeout, a 5xx, the
front door's ``clef_unavailable``, the deadline) raises ``DecisionUnavailable``, and for a grade the host
retries once (``spec.decision.host_graded``); any other failure (a 4xx, an answer that does not match the
questions) raises ``DecisionRequestError`` and is permanent.

The state is fitted with ``budget.fit_request`` to the pinned window, or to Clef's 131,072-token maximum
when none is pinned; only the ``elastic`` string the caller names is cut, with a visible marker (``decide``
cuts the solver's output from the middle). A pinned window is sent as ``x-clef-window``. The token is sent as
a bearer header and appears in no log, error or repr.
"""

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Protocol, Self

import httpx
from pydantic import JsonValue

from taskforge.llm.clef.budget import MAX_LENGTH, ClefTokenizer, TruncateFrom, fit_request, load_tokenizer
from taskforge.llm.clef.schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)
from taskforge.spec import decision

logger = logging.getLogger(__name__)

ENDPOINT = "/v1/systemone"
WINDOW_HEADER = "x-clef-window"
WINDOWS = (16384, 32768, 65536, 131072)
"""The engine windows of a Clef node; a pinned window is one of them."""
UNAVAILABLE_CODE = "clef_unavailable"
"""The front door's error code while Clef swaps in."""
FIT_WORKERS = 4
"""Threads of the client's own pool that tokenize states to fit them."""


ANSWER_TYPES: dict[type[Question], type[Answer]] = {
    ChoiceQuestion: ChoiceAnswer,
    NoulQuestion: NoulAnswer,
    ScoreQuestion: ScoreAnswer,
}
"""The answer type Clef gives each question type."""


class DecisionRequestError(ValueError):
    """Clef refused the read or answered something other than the questions asked: a permanent failure."""


class ClefReader(Protocol):
    """One Clef read of any mix of question types; the client that grades decisions is one."""

    async def read(
        self,
        state: Mapping[str, JsonValue],
        questions: Mapping[str, Question],
        *,
        elastic: Sequence[str],
        truncate_from: TruncateFrom,
        timeout: float,
    ) -> SystemOneResponse: ...


@dataclass(frozen=True)
class ClefDecisionConfig:
    """Where a host's decision model is.

    Attributes:
        base_url: The Orion gateway Clef sits behind (the GLM router's base URL, without ``/v1``).
        token_env: The environment variable holding the gateway's bearer token.
        window: The engine window every read is pinned to (one of ``WINDOWS``), or ``None`` for an unpinned read
            fitted to ``MAX_LENGTH``.
        max_inflight: Reads in flight at once from this host.
    """

    base_url: str
    token_env: str
    window: int | None
    max_inflight: int

    def __post_init__(self) -> None:
        if self.window is not None and self.window not in WINDOWS:
            raise ValueError(f"window must be one of {WINDOWS}, got {self.window}")
        if self.max_inflight < 1:
            raise ValueError("max_inflight must be at least 1")


class ClefDecisionClient:
    """A ``DecisionClient`` and a ``ClefReader`` over Clef; open one per host with ``open_clef_decision_client``."""

    def __init__(
        self,
        config: ClefDecisionConfig,
        token: str,
        tokenizer: ClefTokenizer,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._window = config.window
        self._max_length = config.window or MAX_LENGTH
        self._tokenizer = tokenizer
        self._slots = asyncio.Semaphore(config.max_inflight)
        self._executor = ThreadPoolExecutor(max_workers=FIT_WORKERS, thread_name_prefix="clef-fit")
        self._http = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            transport=transport,
            limits=httpx.Limits(max_connections=config.max_inflight),
        )

    def __repr__(self) -> str:
        return f"ClefDecisionClient(window={self._window}, max_length={self._max_length})"

    async def aclose(self) -> None:
        await self._http.aclose()
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def decide(
        self,
        state: Mapping[str, JsonValue],
        questions: Mapping[str, decision.ScoreQuestion],
        *,
        elastic: str,
        timeout: float,
    ) -> Mapping[str, tuple[float, ...]]:
        """Each score question's level probabilities from one ``read`` of ``questions`` over ``state``.

        ``elastic`` is the state key ``read`` may cut, from the middle, to fit the window.

        Raises:
            DecisionUnavailable: a connection error or timeout, a 5xx or ``clef_unavailable``, or the deadline.
            DecisionRequestError: a 4xx, or an answer that is not a score for each question asked.
            taskforge.llm.clef.schema.ClefRequestError: the questions and the non-elastic state do not fit the window.
        """
        response = await self.read(
            state,
            {qid: ScoreQuestion(q.prompt, q.levels) for qid, q in questions.items()},
            elastic=(elastic,),
            truncate_from=TruncateFrom.MIDDLE,
            timeout=timeout,
        )
        return _probabilities(response, questions)

    async def read(
        self,
        state: Mapping[str, JsonValue],
        questions: Mapping[str, Question],
        *,
        elastic: Sequence[str],
        truncate_from: TruncateFrom,
        timeout: float,
    ) -> SystemOneResponse:
        """One Clef read of ``questions`` over ``state``, within ``timeout`` seconds, sent once.

        ``state`` is fitted to the window by cutting the string at the key path ``elastic`` from
        ``truncate_from``. The wait for an in-flight slot, the fit and the HTTP exchange share ``timeout``. Nothing
        is retried: a caller that wants another read after ``DecisionUnavailable`` makes it.

        Raises:
            DecisionUnavailable: a connection error or timeout, a 5xx or ``clef_unavailable``, or the deadline.
            DecisionRequestError: a 4xx, or answers other than one of its own type for each question asked.
            taskforge.llm.clef.schema.ClefRequestError: the questions and the non-elastic state do not fit the window.
        """
        request = SystemOneRequest(state=dict(state), questions=dict(questions))
        try:
            async with asyncio.timeout(timeout):
                async with self._slots:
                    fitted, budget = await asyncio.get_running_loop().run_in_executor(
                        self._executor,
                        fit_request,
                        self._tokenizer,
                        request,
                        elastic,
                        truncate_from,
                        self._max_length,
                    )
                    response = await self._post(fitted, timeout)
        except TimeoutError as error:
            raise decision.DecisionUnavailable(f"clef gave no answer within {timeout:.1f}s") from error
        if budget.truncated:
            logger.info("clef read: cut %d tokens of %r to fit %d", budget.tokens_cut, list(elastic), self._max_length)
        return _matched(response, questions)

    async def _post(self, request: SystemOneRequest, timeout: float) -> SystemOneResponse:
        headers = {} if self._window is None else {WINDOW_HEADER: str(self._window)}
        try:
            response = await self._http.post(ENDPOINT, json=request.to_json(), headers=headers, timeout=timeout)
        except httpx.TransportError as error:
            raise decision.DecisionUnavailable(f"clef transport error: {type(error).__name__}") from error
        if response.status_code == 200:
            try:
                return SystemOneResponse.from_json(response.json(), response.headers)
            except (KeyError, TypeError, ValueError) as error:
                raise DecisionRequestError(f"clef answered 200 with a body that is not a decision: {error}") from error
        code = _error_code(response)
        if response.status_code >= 500 or code == UNAVAILABLE_CODE:
            raise decision.DecisionUnavailable(f"clef answered {response.status_code} {code or ''}".rstrip())
        raise DecisionRequestError(f"clef answered {response.status_code}: {response.text[:500]}")


def _error_code(response: httpx.Response) -> str | None:
    """``error.code`` of a JSON error body, as the front door's ``clef_unavailable``."""
    try:
        error = response.json().get("error")
    except ValueError:
        return None
    return error.get("code") if isinstance(error, dict) else None


def _matched(response: SystemOneResponse, questions: Mapping[str, Question]) -> SystemOneResponse:
    """``response``, once it answers each question exactly once with that question's answer type.

    Raises:
        DecisionRequestError: an answer is missing, extra, or of another question type.
    """
    if set(response.answers) != set(questions):
        raise DecisionRequestError(f"clef answered {sorted(response.answers)}, not the questions {sorted(questions)}")
    for qid, question in questions.items():
        answer = response.answers[qid]
        if not isinstance(answer, ANSWER_TYPES[type(question)]):
            raise DecisionRequestError(f"clef answered {type(question).__name__} {qid!r} with a {type(answer).__name__}")
    return response


def _probabilities(
    response: SystemOneResponse, questions: Mapping[str, decision.ScoreQuestion]
) -> dict[str, tuple[float, ...]]:
    """Each question's level probabilities, in level order.

    Raises:
        DecisionRequestError: an answer is not a score, or lacks a level.
    """
    reads: dict[str, tuple[float, ...]] = {}
    for qid, question in questions.items():
        answer = response.answers[qid]
        if not isinstance(answer, ScoreAnswer):
            raise DecisionRequestError(f"clef answered question {qid!r} with a {type(answer).__name__}, not a score")
        try:
            reads[qid] = tuple(float(answer.probabilities[str(level)]) for level in range(len(question.levels)))
        except KeyError as error:
            raise DecisionRequestError(f"clef's answer to {qid!r} lacks level {error}") from error
    return reads


@asynccontextmanager
async def open_clef_decision_client(
    config: ClefDecisionConfig, tokenizer: ClefTokenizer | None = None
) -> AsyncIterator[ClefDecisionClient]:
    """The host's Clef decision client, open for the ``async with`` block.

    The token is read from ``config.token_env``; the tokenizer is Clef's own (``budget.load_tokenizer``) unless
    one is given.

    Raises:
        ValueError: ``config.token_env`` is not set.
    """
    token = os.environ.get(config.token_env, "")
    if not token:
        raise ValueError(f"the decision client reads its token from {config.token_env}, which is not set")
    async with ClefDecisionClient(config, token, tokenizer or load_tokenizer()) as client:
        yield client
