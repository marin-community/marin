"""Carry native chat candidate heads through the public engine interface."""

from collections.abc import AsyncGenerator, Iterator
from contextlib import aclosing, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Protocol, Self

import numpy as np

from skyrl_train.inference_engines.response_topk import ResponseTopK, select_flat_chat_response_topk


class ChatSamplingParams(Protocol):
    logprobs: int | None
    flat_logprobs: bool

    def clone(self) -> Self: ...


@dataclass
class ChatTopKCapture:
    top_k: int
    response: tuple[list[int], ResponseTopK] | None = None


class CompactChatEngineClient:
    """Compose vLLM chat serving with numeric logprobs from its engine API.

    Chat rendering, sampling options, parsing and abort handling stay with
    vLLM. Only an internal, nonstreaming request enters a capture scope.
    Its serving request omits display logprobs; its engine request collects
    flat heads. Other requests pass through unchanged.
    """

    def __init__(self, engine: Any, id_only_requests: set[str]):
        self._engine = engine
        self._id_only_requests = id_only_requests
        self._capture: ContextVar[ChatTopKCapture | None] = ContextVar("chat_topk_capture", default=None)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    @contextmanager
    def capture(self, top_k: int) -> Iterator[ChatTopKCapture]:
        """Keep one request's numeric evidence separate from concurrent chats."""
        capture = ChatTopKCapture(top_k)
        token = self._capture.set(capture)
        try:
            yield capture
        finally:
            self._capture.reset(token)

    async def generate(
        self, prompt: Any, sampling_params: ChatSamplingParams, request_id: str, **kwargs: Any
    ) -> AsyncGenerator:
        capture = self._capture.get()
        if capture is not None:
            sampling_params = sampling_params.clone()
            sampling_params.logprobs = capture.top_k + 1
            sampling_params.flat_logprobs = True
            self._id_only_requests.add(request_id)
        try:
            async with aclosing(self._engine.generate(prompt, sampling_params, request_id, **kwargs)) as outputs:
                async for output in outputs:
                    if capture is not None and output.finished:
                        if len(output.outputs) != 1:
                            raise ValueError("Compact chat capture requires one response")
                        response = output.outputs[0]
                        response_ids = list(response.token_ids)
                        if response_ids:
                            selected = select_flat_chat_response_topk(response.logprobs, response_ids, capture.top_k)
                        else:
                            # Weight publication can abort a request before its first token.
                            selected = ResponseTopK(
                                [],
                                np.empty((0, capture.top_k), dtype=np.int32),
                                np.empty((0, capture.top_k), dtype=np.float32),
                            )
                        capture.response = response_ids, selected
                    yield output
        finally:
            if capture is not None:
                self._id_only_requests.discard(request_id)
