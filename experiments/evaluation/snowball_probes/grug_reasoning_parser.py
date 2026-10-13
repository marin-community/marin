# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import Any

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.utils import partial_tag_overlap

TOOL_CALL_START = "<tool_call>"


class GrugReasoningParser(BaseThinkingReasoningParser):
    """Separate Grug's special-token reasoning block from its final answer."""

    def __init__(self, tokenizer: TokenizerLike, *args: Any, **kwargs: Any):
        chat_kwargs = kwargs.get("chat_template_kwargs") or {}
        self._thinking_enabled = chat_kwargs.get("enable_thinking") is not False
        super().__init__(tokenizer, *args, **kwargs)
        self._header_token_id = self.vocab.get("<|start_header_id|>")
        self._streamed_reasoning_length = 0
        self._tool_call_started = False

    @property
    def start_token(self) -> str:
        return "<|start_think|>"

    @property
    def end_token(self) -> str:
        return "<|end_think|>"

    def _nothink_content(self, input_ids: Sequence[int]) -> bool:
        return not self._thinking_enabled and self.start_token_id not in input_ids

    def _current_turn_ids(self, input_ids: Sequence[int]) -> Sequence[int]:
        if self._header_token_id is None:
            return input_ids
        for index in range(len(input_ids) - 1, -1, -1):
            if input_ids[index] == self._header_token_id:
                return input_ids[index:]
        return input_ids

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        current_turn_ids = self._current_turn_ids(input_ids)
        if self._nothink_content(current_turn_ids):
            return True
        return super().is_reasoning_end(current_turn_ids)

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Iterable[int]) -> bool:
        return self._tool_call_started or super().is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        current_turn_ids = list(self._current_turn_ids(input_ids))
        if self._nothink_content(current_turn_ids):
            return current_turn_ids
        return super().extract_content_ids(current_turn_ids)

    def adjust_request(
        self, request: ChatCompletionRequest | ResponsesRequest
    ) -> ChatCompletionRequest | ResponsesRequest:
        request.skip_special_tokens = False
        return request

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output
        return super().extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if not self._thinking_enabled and self.start_token not in current_text:
            return DeltaMessage(content=delta_text)

        start = current_text.find(self.start_token)
        body_start = start + len(self.start_token) if start >= 0 else 0
        end = current_text.find(self.end_token, body_start)
        tool = current_text.find(TOOL_CALL_START, body_start)
        if tool >= 0 and (end < 0 or tool < end):
            marker_start = tool
            marker_end = tool
            self._tool_call_started = True
        elif end >= 0:
            marker_start = end
            marker_end = end + len(self.end_token)
        else:
            if start < 0:
                # An unmarked prefix may still be reasoning if a close follows.
                return None
            body = current_text[body_start:]
            withheld = max(
                partial_tag_overlap(body, self.end_token),
                partial_tag_overlap(body, TOOL_CALL_START),
            )
            sendable_length = len(body) - withheld
            reasoning = body[self._streamed_reasoning_length : sendable_length]
            self._streamed_reasoning_length = sendable_length
            return DeltaMessage(reasoning=reasoning) if reasoning else None

        body = current_text[body_start:marker_start]
        reasoning = body[self._streamed_reasoning_length :]
        self._streamed_reasoning_length = len(body)
        content = current_text[marker_end:]
        return DeltaMessage(
            reasoning=reasoning or None,
            content=content or None,
        )

    def get_streaming_fallback_content(
        self,
        previous_text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        if self.start_token in previous_text or self.end_token in previous_text:
            return None
        return previous_text or None


# Source: marin-community/vllm@ac3b5ea5e8510c002f3d349ea5c0f8b3f20e74ac.
# Register the fork parser while its next GPU wheel is being published.
ReasoningParserManager.register_module(name="grug", module=GrugReasoningParser)
