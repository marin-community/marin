# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""vLLM tool-parser plugin that serves Cohere Command 4 tool calls with unique IDs.

vLLM's Cohere Command parser returns the model's own ``tool_call_id`` as the API tool-call ID. The
chat template renumbers calls by position in every turn, so the model emits "0", "1", ... again and
the IDs collide with earlier calls in the conversation, binding tool results to the wrong call.

Loading this file with ``--tool-parser-plugin`` registers the tool-call parser
``cohere_command4_unique_ids``: the same unified Cohere parser, except that tool-call IDs are minted
by vLLM. It imports vLLM, so it runs only inside the vLLM server process.
"""

from collections.abc import Sequence

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.generate.base.protocol import DeltaToolCall, FunctionCall
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import Parser
from vllm.parser.cohere_command import CohereCommandParser
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.cohere_command_tool_parser import CohereCommand4ToolParser

TOOL_PARSER_NAME = "cohere_command4_unique_ids"
_COHERE_TOOL_PARSER_NAME = "cohere_command4"


class UniqueIdCohereCommandParser(CohereCommandParser):
    """``CohereCommandParser`` that ignores the model's per-turn ``tool_call_id``."""

    def parse(
        self,
        model_output: str,
        request,
        enable_auto_tools: bool = False,
        model_output_token_ids: Sequence[int] = (),
    ) -> tuple[str | None, str | None, list[FunctionCall] | None]:
        reasoning, content, tool_calls = super().parse(model_output, request, enable_auto_tools, model_output_token_ids)
        if tool_calls is None:
            return reasoning, content, None
        # Serving mints an ID with make_tool_call_id() for a call whose ID is unset.
        return reasoning, content, [FunctionCall(name=tc.name, arguments=tc.arguments) for tc in tool_calls]

    def _tool_call_deltas(self, melody_tool_calls) -> list[DeltaToolCall]:
        deltas = super()._tool_call_deltas(melody_tool_calls)
        # A call's ID is sent once, with the delta that first carries its name.
        for delta in deltas:
            if delta.function is not None and delta.function.name is not None:
                delta.id = make_tool_call_id()
        return deltas


_previous_get_parser = ParserManager.get_parser


def _get_parser(
    cls,
    tool_parser_name: str | None = None,
    reasoning_parser_name: str | None = None,
    *args,
    **kwargs,
) -> type[Parser] | None:
    """Select ``UniqueIdCohereCommandParser`` for ``TOOL_PARSER_NAME`` and pass other names through.

    vLLM chooses its unified Cohere parser by parser name, so the new tool-parser name alone would
    not reach it. vLLM can import this file more than once; each import wraps the previous selector,
    which passes every other name through unchanged.
    """
    if tool_parser_name != TOOL_PARSER_NAME:
        return _previous_get_parser(tool_parser_name, reasoning_parser_name, *args, **kwargs)
    cohere_parser = _previous_get_parser(_COHERE_TOOL_PARSER_NAME, reasoning_parser_name, *args, **kwargs)
    if cohere_parser is None:
        return None
    assert issubclass(cohere_parser, CohereCommandParser), cohere_parser
    return type(
        UniqueIdCohereCommandParser.__name__,
        (UniqueIdCohereCommandParser,),
        {
            "reasoning_parser_cls": cohere_parser.reasoning_parser_cls,
            "tool_parser_cls": cohere_parser.tool_parser_cls,
        },
    )


ParserManager.get_parser = classmethod(_get_parser)
# vLLM validates --tool-call-parser against the tool-parser registry before serving starts.
ToolParserManager.register_module(TOOL_PARSER_NAME, module=CohereCommand4ToolParser)
