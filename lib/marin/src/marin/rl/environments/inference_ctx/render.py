# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import re
from typing import Literal, NotRequired, TypedDict

import pydantic
from levanter.tokenizers import MarinTokenizer

logger = logging.getLogger(__name__)

TOOL_CALL_ARGUMENTS_KEY = "arguments"
LEGACY_TOOL_CALL_ARGUMENTS_KEY = "args"
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
QWEN_TOOL_PROMPT = (
    "# Tools\n\n"
    "You may call one or more functions to help with the user request.\n"
    "The available tools are provided inside <tools></tools> XML tags.\n"
    "When you call a tool, respond with a JSON object inside <tool_call></tool_call> "
    'and use the keys "name" and "arguments".'
)


class StrictBase(pydantic.BaseModel):
    """
    Pydantic base class that's immutable and doesn't silently ignore extra fields.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    def __str__(self) -> str:
        return repr(self)


class ToolCall(StrictBase):
    """
    Structured tool invocation following OpenAI/kosong format.

    This represents a request to invoke a tool/function. The structure follows
    the OpenAI function calling format for compatibility with various LLM APIs.

    Example:
        tool_call = ToolCall(
            function=ToolCall.FunctionBody(
                name="search",
                arguments='{"query_list": ["python async", "pydantic validation"]}'
            ),
            id="call_abc123"
        )
    """

    class FunctionBody(pydantic.BaseModel):
        """
        Tool call function body containing the tool name and arguments.

        The arguments field must be a valid JSON string that will be parsed
        by the tool implementation.
        """

        name: str
        """The name of the tool to be called."""
        arguments: str
        """Arguments of the tool call in JSON string format."""

    type: Literal["function"] = "function"
    """Tool call type, must be 'function' for compatibility."""

    id: str | None = None
    """Optional unique identifier for tracking this specific tool call."""

    function: FunctionBody
    """The function body containing tool name and arguments."""


class ToolSpec(StrictBase):
    """Model-facing tool schema following the OpenAI function-tool format."""

    class FunctionBody(pydantic.BaseModel):
        """Tool definition with JSON-schema parameters."""

        name: str
        description: str
        parameters: dict[str, object]

    type: Literal["function"] = "function"
    function: FunctionBody


class ToolOk(StrictBase):
    """
    Successful tool execution result.

    Used to indicate that a tool call completed successfully, with
    the main output and optional metadata fields.
    """

    output: str
    """The main output/result from the tool execution."""

    message: str = ""
    """Optional human-readable message about the execution."""

    brief: str = ""
    """Optional brief summary of the result for logging."""


class ToolError(StrictBase):
    """
    Tool execution error result.

    Used to indicate that a tool call failed or encountered an error,
    with details about what went wrong.
    """

    output: str = ""
    """Any partial output that was generated before the error."""

    message: str = ""
    """Error message describing what went wrong."""

    brief: str = ""
    """Brief error summary for logging."""


ToolReturnType = ToolOk | ToolError
"""Union type for tool execution results - either success or error."""


class ToolResult(StrictBase):
    """
    Complete tool execution result with tracking ID.

    Wraps the actual result (ToolOk or ToolError) with the corresponding
    tool call ID for correlation in multi-tool scenarios.

    Note: This class is defined for future use in handling multiple
    concurrent tool calls with result correlation.
    """

    tool_call_id: str | None
    """ID of the tool call this result corresponds to."""

    result: ToolReturnType
    """The actual execution result (success or error)."""


class GeneratedAssistantTurn(StrictBase):
    """Parsed assistant output suitable for environment tool execution."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()


class AssistantTurnParseResult(StrictBase):
    """Structured parse result for a generated assistant turn."""

    assistant_turn: GeneratedAssistantTurn
    parse_success: bool


# NOTE: we use a broad type definition for the role to be flexible
# Common roles are "user", "assistant", "system", "tool"
Role = str


class Message(TypedDict):
    role: Role
    content: str
    tool_calls: NotRequired[list[ToolCall]]
    thinking: NotRequired[str]
    trainable: NotRequired[bool]
    tool_call_id: NotRequired[str]
    name: NotRequired[str]


class Renderer:
    def __init__(self, tokenizer: MarinTokenizer):
        self.tokenizer = tokenizer

    def get_stop_sequences(self) -> list[str] | list[int]:
        raise NotImplementedError

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> list[int]:
        raise NotImplementedError

    def parse_response(self, response: list[int]) -> AssistantTurnParseResult:
        raise NotImplementedError


def _tool_call_payload(tool_call: ToolCall) -> dict[str, object]:
    """Minimal JSON payload for embedding in <tool_call> blocks."""
    payload = {
        "name": tool_call.function.name,
        TOOL_CALL_ARGUMENTS_KEY: json.loads(tool_call.function.arguments),
    }
    if tool_call.id is not None:
        payload["id"] = tool_call.id
    return payload


def _tool_spec_payload(tool_spec: ToolSpec) -> dict[str, object]:
    """Serialize a tool specification for prompt rendering."""
    return tool_spec.model_dump(mode="python")


def _compact_json_dumps(value: object) -> str:
    """Serialize tool payloads compactly while preserving deliberate field order."""
    return json.dumps(value, separators=(",", ":"))


def _tool_call_arguments(payload: dict[str, object]) -> dict[str, object] | None:
    arguments = payload.get(TOOL_CALL_ARGUMENTS_KEY)
    if arguments is None:
        arguments = payload.get(LEGACY_TOOL_CALL_ARGUMENTS_KEY)
    if not isinstance(arguments, dict):
        return None
    return arguments


def _qwen_tool_prompt(tools: list[ToolSpec]) -> str:
    tool_lines = "\n".join(_compact_json_dumps(_tool_spec_payload(tool_spec)) for tool_spec in tools)
    return f"{QWEN_TOOL_PROMPT}\n<tools>\n{tool_lines}\n</tools>"


def _assistant_message_from_turn(turn: GeneratedAssistantTurn) -> Message:
    message = Message(role="assistant", content=turn.content)
    if turn.tool_calls:
        message["tool_calls"] = list(turn.tool_calls)
    return message


def _parse_response_text(str_response: str, parse_success: bool) -> AssistantTurnParseResult:
    return AssistantTurnParseResult(
        assistant_turn=GeneratedAssistantTurn(content=str_response),
        parse_success=parse_success,
    )


def _messages_with_tool_prompt(messages: list[Message], tools: list[ToolSpec]) -> list[Message]:
    tool_prompt = _qwen_tool_prompt(tools)
    if messages and messages[0]["role"] == "system":
        updated_first_message = Message(
            role=messages[0]["role"],
            content=f"{messages[0]['content']}\n\n{tool_prompt}",
        )
        return [updated_first_message, *messages[1:]]
    return [Message(role="system", content=tool_prompt), *messages]


def _assistant_content_without_tool_calls(content: str) -> str:
    """Remove raw tool-call XML from assistant text after structured parsing."""
    without_tool_calls = TOOL_CALL_BLOCK_RE.sub("", content).strip()
    return MULTI_NEWLINE_RE.sub("\n\n", without_tool_calls)


def parse_response_for_stop_token(
    response: list[int], tokenizer: MarinTokenizer, stop_token: int
) -> AssistantTurnParseResult:
    """Parse response for a single stop token.

    We expect a properly rendered response to have exactly one stop token; but it may have zero if e.g. the model
    ran out of tokens when sampling, which will incur a format error. If there are > 1, there is likely a bug in the
    sampler and we should error.
    """
    emt_count = response.count(stop_token)
    if emt_count == 0:
        str_response = tokenizer.decode(response)
        logger.debug(f"Response is not a valid assistant response: {str_response}")
        return _parse_response_text(str_response, parse_success=False)
    elif emt_count == 1:
        str_response = tokenizer.decode(response[: response.index(stop_token)])
        return _parse_response_text(str_response, parse_success=True)
    else:
        raise ValueError(
            f"When parsing response, expected to split into 1 or 2 pieces using stop tokens, but got {emt_count}. "
            "You probably are using the wrong stop tokens when sampling"
        )


class Llama3Renderer(Renderer):
    """
    Format like this:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant for travel tips and
        recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

        What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """

    def _render_message(self, message: Message) -> tuple[list[int], list[int], list[int]]:
        assert message.get("thinking") is None, "CoT tokens not supported in Llama3"
        ob_str = f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n"
        # Observation (prompt) part
        ac_str = f"{message['content']}<|eot_id|>"
        # Action part
        ac_tail_str = ""  # No action tail needed for Llama3 format
        # Action part that's only included in the last message in SFT
        return (
            self.tokenizer.encode(ob_str, add_special_tokens=False),
            self.tokenizer.encode(ac_str, add_special_tokens=False),
            self.tokenizer.encode(ac_tail_str, add_special_tokens=False),
        )

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> list[int]:
        if tools:
            raise NotImplementedError("Tool calling is not implemented for Llama3Renderer.")
        tokens: list[int] = []
        tokens.extend(self._bos_tokens)
        for message in messages:
            ob_part, action_part, _action_tail = self._render_message(message)
            tokens.extend(ob_part)
            tokens.extend(action_part)
        new_partial_message = Message(role=role, content="")
        ob_part, _action_part, _action_tail = self._render_message(new_partial_message)
        tokens.extend(ob_part)
        tokens.extend(self.tokenizer.encode(prefill or "", add_special_tokens=False))
        return tokens

    @property
    def _bos_tokens(self) -> list[int]:
        return self.tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)

    @property
    def _end_message_token(self) -> int:
        (token,) = self.tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        return token

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def parse_response(self, response: list[int]) -> AssistantTurnParseResult:
        return parse_response_for_stop_token(response, self.tokenizer, self._end_message_token)


class Qwen3Renderer(Renderer):
    """
    Format like this:
        <|im_start|>system
        You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
        <|im_start|>user
        What can you help me with?<|im_end|>
        <|im_start|>assistant
        <think>

        </think>
        I can help you with...<|im_end|>
    """

    def __init__(self, tokenizer: MarinTokenizer, strip_thinking_from_history: bool = True):
        """
        Args:
            tokenizer: The tokenizer to use for encoding.
            strip_thinking_from_history: When True (default), strips <think>...</think> blocks
                from assistant messages in multi-turn history. This matches how Qwen3 models
                were trained - they only see their own thinking during the current turn, not
                from previous turns. Set to False to preserve thinking in history (useful for
                certain RL scenarios where you want the extension property for efficiency).

        See https://tinker-docs.thinkingmachines.ai/rl/sequence-extension for details on
        how this option affects multi-turn RL compute efficiency.
        """
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history

    def _render_message(self, idx: int, message: Message) -> tuple[list[int], list[int], list[int]]:
        assert message.get("thinking") is None, "TODO: support CoT in Qwen3 renderer"
        maybe_newline = "\n" if idx > 0 else ""
        ob_str = f"{maybe_newline}<|im_start|>{message['role']}\n"
        ac_content = message["content"]
        if self.strip_thinking_from_history and message["role"] == "assistant" and "</think>" in ac_content:
            # Multi-turn conversation, we remove the thinking section from the assistant message.
            # This matches how Qwen3 models were trained - they only see their own thinking
            # during the current turn, not from previous turns.
            ac_content = ac_content.split("</think>")[1].lstrip()
        elif message["role"] == "assistant" and "<think>" not in ac_content:
            # Matching the paper, we force the assistant to start with <think>. Some SFT datasets include
            # <think> in the assistant messages, we so don't need to re-add it in those cases.
            ob_str += "<think>\n"
        # Observation (prompt) part
        if "tool_calls" in message:
            ac_content += "\n".join(
                [
                    f"<tool_call>\n{_compact_json_dumps(_tool_call_payload(tool_call))}\n</tool_call>"
                    for tool_call in message["tool_calls"]
                ]
            )
        ac_content += "<|im_end|>"
        # Action part
        ac_tail_str = ""  # No action tail needed for Qwen format
        # Action part that's only included in the last message in SFT
        return (
            self.tokenizer.encode(ob_str, add_special_tokens=False),
            self.tokenizer.encode(ac_content, add_special_tokens=False),
            self.tokenizer.encode(ac_tail_str, add_special_tokens=False),
        )

    def build_generation_prompt(
        self,
        messages: list[Message],
        role: Role = "assistant",
        prefill: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> list[int]:
        if tools:
            messages = _messages_with_tool_prompt(messages, tools)
        tokens: list[int] = []  # No BOS token for Qwen
        for idx, message in enumerate(messages):
            ob_part, action_part, _ = self._render_message(idx, message)
            tokens.extend(ob_part)
            tokens.extend(action_part)
        # Add generation prompt
        new_partial_message = Message(role=role, content="")
        ob_part, _, _ = self._render_message(len(messages), new_partial_message)
        tokens.extend(ob_part)
        tokens.extend(self.tokenizer.encode(prefill or "", add_special_tokens=False))
        return tokens

    @property
    def _end_message_token(self) -> int:
        tokens = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for <|im_end|>, got {len(tokens)}"
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        return [self._end_message_token]

    def _parse_tool_call(self, tool_call_str: str) -> list[ToolCall] | None:
        try:
            tool_call = json.loads(tool_call_str)
        except json.JSONDecodeError:
            return None

        if not isinstance(tool_call, dict):
            return None
        name = tool_call.get("name")
        arguments = _tool_call_arguments(tool_call)
        tool_id = tool_call.get("id")
        if not isinstance(name, str) or arguments is None:
            return None
        if tool_id is not None and not isinstance(tool_id, str):
            tool_id = None
        # Convert to nested structure with arguments as JSON string
        return [
            ToolCall(
                function=ToolCall.FunctionBody(name=name, arguments=_compact_json_dumps(arguments)),
                id=tool_id,
            )
        ]

    def parse_response(self, response: list[int]) -> AssistantTurnParseResult:
        parse_result = parse_response_for_stop_token(response, self.tokenizer, self._end_message_token)
        if not parse_result.parse_success:
            return parse_result

        # Follow Qwen docs and Qwen-Agent's tool calling prompt to use
        # <tool_call>...</tool_call> tags to wrap the tool call.
        # - https://qwen.readthedocs.io/en/latest/getting_started/concepts.html#tool-calling
        # - https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py#L279-L282
        assistant_message = _assistant_message_from_turn(parse_result.assistant_turn)
        matches = TOOL_CALL_BLOCK_RE.findall(assistant_message["content"])
        if not matches:
            return parse_result

        tool_calls: list[ToolCall] = []
        for match in matches:
            parsed_tool_calls = self._parse_tool_call(match)
            if parsed_tool_calls is None:
                return AssistantTurnParseResult(
                    assistant_turn=parse_result.assistant_turn,
                    parse_success=False,
                )
            tool_calls.extend(parsed_tool_calls)

        return AssistantTurnParseResult(
            assistant_turn=GeneratedAssistantTurn(
                content=_assistant_content_without_tool_calls(assistant_message["content"]),
                tool_calls=tuple(tool_calls),
            ),
            parse_success=True,
        )
