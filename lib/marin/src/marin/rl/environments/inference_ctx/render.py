# ruff: noqa
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# https://github.com/thinking-machines-lab/tinker-cookbook/blob/989f84926245b227634797b8eac46abe232f9c24/tinker_cookbook/renderers.py#L459

from typing import Literal, NotRequired, TypedDict
from transformers import PreTrainedTokenizerBase as Tokenizer
import pydantic
import json
import re
import logging

logger = logging.getLogger(__name__)


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
    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    def get_stop_sequences(self) -> list[str] | list[int]:
        raise NotImplementedError

    def build_generation_prompt(
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> list[int]:
        raise NotImplementedError

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        raise NotImplementedError


def _tool_call_payload(tool_call: ToolCall) -> dict[str, object]:
    """Minimal JSON payload for embedding in <tool_call> blocks."""
    # Convert from nested structure to flat format for compatibility
    return {
        "name": tool_call.function.name,
        "args": json.loads(tool_call.function.arguments),
    }


def parse_response_for_stop_token(response: list[int], tokenizer: Tokenizer, stop_token: int) -> tuple[Message, bool]:
    """Parse response for a single stop token.

    We expect a properly rendered response to have exactly one stop token; but it may have zero if e.g. the model
    ran out of tokens when sampling, which will incur a format error. If there are > 1, there is likely a bug in the
    sampler and we should error.
    """
    emt_count = response.count(stop_token)
    if emt_count == 0:
        str_response = tokenizer.decode(response)
        logger.debug(f"Response is not a valid assistant response: {str_response}")
        return Message(role="assistant", content=str_response), False
    elif emt_count == 1:
        str_response = tokenizer.decode(response[: response.index(stop_token)])
        return Message(role="assistant", content=str_response), True
    else:
        raise ValueError(
            f"When parsing response, expected to split into 1 or 2 pieces using stop tokens, but got {emt_count}. "
            "You probably are using the wrong stop tokens when sampling"
        )


class Llama3Renderer(Renderer):
    """
    Format like this:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>

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
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> list[int]:
        tokens: list[int] = []
        tokens.extend(self._bos_tokens)
        for message in messages:
            ob_part, action_part, action_tail = self._render_message(message)
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

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
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

    def __init__(self, tokenizer: Tokenizer, strip_thinking_from_history: bool = True):
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
                    f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
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
        self, messages: list[Message], role: Role = "assistant", prefill: str | None = None
    ) -> list[int]:
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
        args = tool_call.get("args")
        tool_id = tool_call.get("id")
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        if tool_id is not None and not isinstance(tool_id, str):
            tool_id = None
        # Convert to nested structure with arguments as JSON string
        return [
            ToolCall(
                function=ToolCall.FunctionBody(name=name, arguments=json.dumps(args)),
                id=tool_id,
            )
        ]

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            return assistant_message, False

        # Follow Qwen docs and Qwen-Agent's tool calling prompt to use <tool_call>...</tool_call> tags to wrap the tool call.
        # - https://qwen.readthedocs.io/en/latest/getting_started/concepts.html#tool-calling
        # - https://github.com/QwenLM/Qwen-Agent/blob/main/qwen_agent/llm/fncall_prompts/nous_fncall_prompt.py#L279-L282
        match = re.search(r"<tool_call>(.*?)</tool_call>", assistant_message["content"], re.DOTALL)
        if match:
            tool_calls = self._parse_tool_call(match.group(1))
            if tool_calls is None:
                return assistant_message, False
            else:
                assistant_message["tool_calls"] = tool_calls
                return assistant_message, True
        return assistant_message, True
