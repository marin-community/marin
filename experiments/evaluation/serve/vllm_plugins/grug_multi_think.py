# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM reasoning-parser plugin that routes every Grug think span to reasoning.

Grug models such as Marin Snowball can emit several ``<|start_think|>...<|end_think|>`` spans in one
turn. vLLM's ``grug`` reasoning parser sends only the first span to reasoning, so later spans, their
markers and stray ``<|end_think|>`` markers reach the answer, and agent harnesses send them back.

Loading this file with ``--reasoning-parser-plugin`` registers the reasoning parser
``grug_multi_think``. It routes every span to reasoning in order and drops stray end markers, in
streaming and non-streaming responses. Its unified parser keeps a free-text stream in the reasoning
phase past the first span and hands off to the tool parser at ``<tool_call>``. The engine's
structured-output gate, which builds its own instance of the reasoning parser, still ends reasoning at
the first end marker. The file imports vLLM, so it runs only inside vLLM server and engine processes.
"""

from collections.abc import Iterable, Sequence
from enum import Enum, auto
from typing import Any

from openai.types.responses import ToolChoiceFunction
from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import DelegatingParser, Parser
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.grug_reasoning_parser import TOOL_CALL_START, GrugReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.utils import partial_tag_overlap

REASONING_PARSER_NAME = "grug_multi_think"
START_THINK = "<|start_think|>"
END_THINK = "<|end_think|>"


class _Mode(Enum):
    # No marker seen yet: reasoning if an end marker closes it, else content.
    UNMARKED = auto()
    REASONING = auto()
    CONTENT = auto()
    # Everything left is content, markers included.
    RAW = auto()


class _ThinkSpanScanner:
    """Route Grug output between reasoning and content as text arrives.

    Every ``<|start_think|>...<|end_think|>`` span is reasoning and a stray end marker is dropped.
    Each ``feed`` takes the whole text so far and returns only the newly routed reasoning and content,
    so total work is linear.
    """

    def __init__(self, *, unmarked_is_reasoning: bool, tool_handoff: bool):
        self._markers = (START_THINK, END_THINK) + ((TOOL_CALL_START,) if tool_handoff else ())
        self._mode = _Mode.UNMARKED if unmarked_is_reasoning else _Mode.CONTENT
        # Text before ``_pos`` is routed; no marker starts before ``_scan``.
        self._pos = 0
        self._scan = 0
        # Text from the first tool call onward, once one appears.
        self.tool_call_text: str | None = None

    def feed(self, text: str, *, final: bool, raw_after_end: bool) -> tuple[str, str]:
        """Route the unrouted end of ``text``; hold a partial marker unless ``final``.

        With ``raw_after_end``, text after the first end marker is content.
        """
        reasoning: list[str] = []
        content: list[str] = []
        while self._mode is not _Mode.RAW and self.tool_call_text is None:
            found = self._next_marker(text)
            if found is None:
                break
            index, marker = found
            to_reasoning = self._mode is _Mode.REASONING or (self._mode is _Mode.UNMARKED and marker == END_THINK)
            (reasoning if to_reasoning else content).append(text[self._pos : index])
            self._pos = self._scan = index + len(marker)
            if marker == TOOL_CALL_START:
                self.tool_call_text = text[index:]
                self._pos = len(text)
            elif marker == START_THINK:
                self._mode = _Mode.REASONING
            else:
                self._mode = _Mode.RAW if raw_after_end else _Mode.CONTENT
        if self.tool_call_text is not None:
            return "".join(reasoning), "".join(content)
        tail_end = len(text)
        if not final and self._mode is not _Mode.RAW:
            longest = max(len(marker) for marker in self._markers)
            window = text[max(self._pos, len(text) - longest) :]
            tail_end -= max(partial_tag_overlap(window, m) for m in self._markers)
            self._scan = max(self._scan, tail_end)
        if self._mode is _Mode.REASONING:
            reasoning.append(text[self._pos : tail_end])
            self._pos = tail_end
        elif self._mode is not _Mode.UNMARKED or final:
            content.append(text[self._pos : tail_end])
            self._pos = tail_end
        return "".join(reasoning), "".join(content)

    def _next_marker(self, text: str) -> tuple[int, str] | None:
        hits = [(i, m) for m in self._markers if (i := text.find(m, self._scan)) >= 0]
        return min(hits) if hits else None


class MultiThinkGrugReasoningParser(GrugReasoningParser):
    """``GrugReasoningParser`` that routes every think span to reasoning, joined in order."""

    def __init__(self, tokenizer: TokenizerLike, *args: Any, **kwargs: Any):
        super().__init__(tokenizer, *args, **kwargs)
        # When False, streaming leaves the reasoning phase at the first end marker, as the engine's
        # gate does; ``GrugMultiThinkParser`` sets it for requests whose answer is free text.
        self.stream_all_spans = False
        self._stream = self._scanner(tool_handoff=True)

    def _scanner(self, *, tool_handoff: bool) -> _ThinkSpanScanner:
        return _ThinkSpanScanner(unmarked_is_reasoning=self._thinking_enabled, tool_handoff=tool_handoff)

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Iterable[int]) -> bool:
        if self.stream_all_spans and not self._tool_call_started:
            return False
        return super().is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output
        reasoning, content = self._scanner(tool_handoff=False).feed(model_output, final=True, raw_after_end=False)
        return reasoning, content or None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        reasoning, content = self._stream.feed(current_text, final=False, raw_after_end=not self.stream_all_spans)
        if self._stream.tool_call_text is not None:
            self._tool_call_started = True
            content += self._stream.tool_call_text
        if not (reasoning or content):
            return None
        return DeltaMessage(reasoning=reasoning or None, content=content or None)

    def get_streaming_fallback_content(
        self,
        previous_text: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> str | None:
        _, content = self._stream.feed(previous_text, final=True, raw_after_end=not self.stream_all_spans)
        return content or None


class GrugMultiThinkParser(DelegatingParser):
    """Stream every Grug think span as reasoning.

    When the tool parser serves required and named tool choices as bare JSON, that JSON follows the
    first think span, so those streams still hand off to the tool parser at its end marker.
    """

    def parse_delta(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: ChatCompletionRequest | ResponsesRequest,
        prompt_token_ids: list[int] | None = None,
        *,
        finished: bool,
    ) -> DeltaMessage | None:
        reasoning_parser = self._reasoning_parser
        assert isinstance(reasoning_parser, MultiThinkGrugReasoningParser)
        reasoning_parser.stream_all_spans = not self._tool_choice_constrained(request)
        return super().parse_delta(delta_text, delta_token_ids, request, prompt_token_ids, finished=finished)

    def _tool_choice_constrained(self, request: ChatCompletionRequest | ResponsesRequest) -> bool:
        if self._tool_parser is None or not self._tool_parser.supports_required_and_named:
            return False
        return request.tool_choice == "required" or isinstance(
            request.tool_choice, (ToolChoiceFunction, ChatCompletionNamedToolChoiceParam)
        )


_previous_get_parser = ParserManager.get_parser


def _get_parser(
    cls,
    tool_parser_name: str | None = None,
    reasoning_parser_name: str | None = None,
    *args,
    **kwargs,
) -> type[Parser] | None:
    """Select ``GrugMultiThinkParser`` for ``REASONING_PARSER_NAME`` and pass other names through.

    vLLM composes a generic delegating parser for an unrecognized reasoning-parser name, which would
    leave the stream after the first span. vLLM can import this file more than once; each import wraps
    the previous selector, which passes every other name through unchanged.
    """
    parser = _previous_get_parser(tool_parser_name, reasoning_parser_name, *args, **kwargs)
    if reasoning_parser_name != REASONING_PARSER_NAME:
        return parser
    assert parser is not None and issubclass(parser, DelegatingParser), parser
    return type(
        GrugMultiThinkParser.__name__,
        (GrugMultiThinkParser,),
        {"reasoning_parser_cls": parser.reasoning_parser_cls, "tool_parser_cls": parser.tool_parser_cls},
    )


ParserManager.get_parser = classmethod(_get_parser)
ReasoningParserManager.register_module(REASONING_PARSER_NAME, module=MultiThinkGrugReasoningParser)
