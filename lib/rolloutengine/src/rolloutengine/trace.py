# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-mode token bookkeeping for the rollout turn loop."""

from typing import Protocol

from rolloutengine.contracts import ModelTurn, RolloutContractError
from rolloutengine.spec import TraceMode


class _TokenTrace(Protocol):
    """Token fields of the current training record and the prefix for the next request."""

    @property
    def prefix(self) -> tuple[int, ...]: ...

    @property
    def prompt(self) -> tuple[int, ...]: ...

    @property
    def response(self) -> tuple[int, ...]: ...

    @property
    def loss_mask(self) -> tuple[int, ...]: ...

    @property
    def logprobs(self) -> tuple[float, ...] | None: ...

    @property
    def response_end(self) -> int: ...

    def append(self, turn: ModelTurn) -> None: ...


class _ExactTokenTrace:
    """Enforce the exact-token contract and accumulate served tokens, loss masks, and log probabilities.

    Masks and log probabilities grow by each turn's new tokens; the tuples are built only when read.
    """

    def __init__(self) -> None:
        self.started = False
        self.prompt: tuple[int, ...] = ()
        self.prefix: tuple[int, ...] = ()
        self._loss_mask: list[int] = []
        self._logprobs: list[float] | None = []

    @property
    def response(self) -> tuple[int, ...]:
        return self.prefix[len(self.prompt) :]

    @property
    def loss_mask(self) -> tuple[int, ...]:
        return tuple(self._loss_mask)

    @property
    def logprobs(self) -> tuple[float, ...] | None:
        return None if self._logprobs is None else tuple(self._logprobs)

    @property
    def response_end(self) -> int:
        return len(self.prefix) - len(self.prompt) - 1

    def append(self, turn: ModelTurn) -> None:
        if turn.logprobs is not None and len(turn.logprobs) != len(turn.response_token_ids):
            raise RolloutContractError("Model logprobs must align with response tokens")
        if not turn.response_token_ids:
            raise RolloutContractError("Model response contains no token evidence")
        if not self.started:
            self.started = True
            self.prompt = self.prefix = turn.prompt_token_ids
        if turn.prompt_token_ids[: len(self.prefix)] != self.prefix:
            raise RolloutContractError("Model transport changed the served token prefix")
        observation_count = len(turn.prompt_token_ids) - len(self.prefix)
        self._loss_mask += [0] * observation_count + [1] * len(turn.response_token_ids)
        if turn.logprobs is None:
            self._logprobs = None
        elif self._logprobs is not None:
            self._logprobs += [0.0] * observation_count + list(turn.logprobs)
        self.prefix = turn.prompt_token_ids + turn.response_token_ids


class _MessageTrace:
    """Keep the token IDs a model reports, if any, without checking them or making them trainable.

    A chat endpoint re-tokenizes the conversation for every request, so a later prompt need not extend
    the earlier sequence. The recorded response is the last served sequence after the first prompt; its
    loss mask is all zeros, one per response token, and there are no log probabilities. A model that
    reports no token IDs leaves every token field empty.
    """

    prefix: tuple[int, ...] = ()
    logprobs: tuple[float, ...] | None = None

    def __init__(self) -> None:
        self.started = False
        self.prompt: tuple[int, ...] = ()
        self.tokens: tuple[int, ...] = ()

    @property
    def response(self) -> tuple[int, ...]:
        return self.tokens[len(self.prompt) :]

    @property
    def loss_mask(self) -> tuple[int, ...]:
        return (0,) * len(self.response)

    @property
    def response_end(self) -> int:
        return len(self.tokens) - len(self.prompt) - 1

    def append(self, turn: ModelTurn) -> None:
        if not self.started:
            self.started = True
            self.prompt = turn.prompt_token_ids
        self.tokens = turn.prompt_token_ids + turn.response_token_ids


def _token_trace(mode: TraceMode) -> _TokenTrace:
    match mode:
        case TraceMode.EXACT_TOKENS:
            return _ExactTokenTrace()
        case TraceMode.MESSAGES:
            return _MessageTrace()
