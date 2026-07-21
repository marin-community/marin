# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from marin.core.conversation import OpenAIChatMessage


class InputDatasetFormat(str, Enum):
    """Format of the SFT Dataset

    SINGLE_COLUMN_MULTI_TURN example:
    In the huggingface dataset, there exists a single column with a list of messages.
    |                  Messages                 |
    | ----------------------------------------- |
    | [{"role": "user", "content": "..."},      |
    |  {"role": "assistant", "content": "..."}] |
    | ----------------------------------------- |


    INSTRUCTION_RESPONSE example:
    In the huggingface dataset, there exists two columns with a single message each.
    |             Instruction              | Response |
    | ------------------------------------ | -------- |
    | "What is the capital of France?"     | "Paris"  |
    | "What is 2 + 2?"                     |   "4"    |


    INSTRUCT_COLUMN_RESPONSE example:
    In the huggingface dataset, there exists a question column and a responses column with a list
    containing a single dictionary with model name and response.
    |             Question              |                 Responses                |
    | --------------------------------- | ---------------------------------------- |
    | "What is 2 + 2?"                  | [{"response_model": "Model-X",           |
    |                                   |   "response": "The answer is 4"}]        |


    INSTRUCT_MSG_RESPONSE example:
    In the huggingface dataset, there exists an Instruction column with a single message and a
    response column with a string.
    |             Question              |                 Responses                |
    | --------------------------------- | ---------------------------------------- |
    |[ { "role": "user", "content": "a  | "The car's speed is calculated by        |
    |  car runs 375 km in 3 hours.      |  dividing the distance traveled by the   |
    |  what's the car's speed ?" }]     |  time taken. Answer is 375/3 = 125 kmph" |
    """

    SINGLE_COLUMN_MULTI_TURN = "messages"
    INSTRUCTION_RESPONSE = "instruction_response"
    INSTRUCT_COLUMN_RESPONSE = "instruct_column_response"
    INSTRUCT_MSG_RESPONSE = "instruct_msg_response"


@dataclass
class TransformAdapter:
    dataset_format: InputDatasetFormat = InputDatasetFormat.INSTRUCTION_RESPONSE

    # Instruction Response
    instruction_column: str = ""
    response_column: str = ""

    """
    Example of role_key, user_value, assistant_value, and system_value:
    In OpenHermes-2.5, a conversation can look like this:
    [ { "from": "human", "value": "..."},
      { "from": "gpt", "value": "..."} ]

    In this example, the role_key is "from", the user_value is "human", the assistant_value is "gpt",
    and the system_value is "system". This helps us map the roles to the correct values in the OpenAI
    format from "from" -> "role" and "human"/"gpt" -> "user"/"assistant".
    """
    conversation_column: str = "messages"
    role_key: str = "role"
    user_value: str = "user"
    assistant_value: str = "assistant"
    system_value: str = "system"
    content_key: str = "content"
    tool_value: str = "tool"

    # If specified, the key will be used to select the message with
    # best metric in multiple turn conversations
    filter_on_key: str = ""
    metadata_remap: dict[str, str] = field(default_factory=dict)
    replacements: dict[str, str] | None = None
    extra_metadata_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    def transform_conversation_to_openai_format(
        self,
        row: dict[str, Any],
    ) -> list[OpenAIChatMessage] | None:
        """Convert a raw dataset *row* into OpenAI-format messages.

        Returns ``None`` for rows that should be dropped (missing data or a
        shape the adapter intentionally does not process); callers skip those.
        """
        if self.dataset_format == InputDatasetFormat.INSTRUCTION_RESPONSE:
            messages = []
            instruction = row[self.instruction_column]
            response = row[self.response_column]
            # Check data
            if instruction is None or response is None:
                return None  # Do not process rows with missing data
            if self.filter_on_key:
                best_completion = None
                best_metric = -float("inf")  # TODO: Make this a config

                for completion in response:
                    if completion[self.filter_on_key] > best_metric:
                        best_metric = completion[self.filter_on_key]
                        best_completion = completion
                assert best_completion is not None, "filter_on_key requires a non-empty response list"
                response = best_completion[self.content_key]
            messages.append(OpenAIChatMessage(role="user", content=instruction))
            messages.append(OpenAIChatMessage(role="assistant", content=response))
            return messages
        elif self.dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN:
            messages = []
            role_to_openai_role = {
                self.user_value: "user",
                self.assistant_value: "assistant",
                self.system_value: "system",
                self.tool_value: "tool",
            }
            conversation = row[self.conversation_column]
            for conv in conversation:
                role = role_to_openai_role[conv[self.role_key]]
                message = OpenAIChatMessage(role=role, content=conv[self.content_key])
                # Preserve structured tool-calling fields when present (agentic / tool-calling SFT):
                # the plain role/content mapping otherwise drops assistant ``tool_calls`` and the
                # ``tool_call_id``/``name`` that carry the <tool_call>/<tool_response> structure a
                # tools-aware chat template renders. No-op for conversations without these keys, so it
                # is backward-compatible for plain chat datasets.
                tool_calls = conv.get("tool_calls")
                if tool_calls:
                    message.tool_calls = tool_calls
                tool_call_id = conv.get("tool_call_id")
                if tool_call_id is not None:
                    message.tool_call_id = tool_call_id
                name = conv.get("name")
                if name is not None:
                    message.name = name
                messages.append(message)
            return messages
        elif self.dataset_format == InputDatasetFormat.INSTRUCT_COLUMN_RESPONSE:
            messages = []
            instruction = row[self.instruction_column]
            responses = row[self.response_column]

            # Get the first (and only) response from the list
            response_dict = responses[0]
            response_content = response_dict[self.content_key]

            messages.append(OpenAIChatMessage(role="user", content=instruction))
            messages.append(OpenAIChatMessage(role="assistant", content=response_content))
            return messages
        elif self.dataset_format == InputDatasetFormat.INSTRUCT_MSG_RESPONSE:
            messages = []  # Initialize
            # Get data
            instruction = row[self.instruction_column]  # List of dict
            responses = row[self.response_column]  # Single string
            if (responses is None) or (len(instruction) > 1) or (self.role_key not in instruction[0]):
                # We do not process rows that have more than one messages.
                # This occurs in Dolphin-R1 reasoning, where instructions are
                # sometimes part of the 'system' prompt instead of 'user' prompt.
                # Return None (not []) so the caller drops the row instead of
                # emitting an empty conversation.
                return None
            else:
                instruction_content = instruction[0][self.content_key]
                messages.append(OpenAIChatMessage(role="user", content=instruction_content))
                messages.append(OpenAIChatMessage(role="assistant", content=responses))
                return messages
        else:
            raise ValueError(f"Invalid dataset format: {self.dataset_format}")

    def copy(self) -> "TransformAdapter":
        return dataclasses.replace(self)


def tools_column_to_chat_template_kwargs(row: dict[str, Any], *, tools_column: str = "tools") -> dict[str, Any]:
    """Relocate a per-row ``tools`` column into a ``chat_template_kwargs`` column.

    Intended as a ``TransformAdapter.extra_metadata_fn`` for tool-calling / agentic SFT datasets
    (e.g. the opencode serve-parity data) whose rows carry the callable function schemas in a
    ``tools`` column. Levanter's ``ChatLmDatasetFormat`` has no ``tools`` column wiring: it renders
    the template with ``apply_chat_template(..., **chat_template_kwargs)`` and, by default, reads
    those kwargs from a ``chat_template_kwargs`` column. Emitting ``{"chat_template_kwargs":
    {"tools": [...]}}`` here makes the template's ``{% if tools %}`` / ``<tools>`` system block
    render per row, byte-for-byte matching a HuggingFace ``apply_chat_template(tools=...)`` render.

    ``tools`` may be a JSON string (Arrow-safe datasets store the schema list as a string) or an
    already-parsed list; both are handled. Returns ``{}`` (no extra column) when ``tools`` is
    missing, empty, or an unparseable string, so rows without tools are unaffected. Mirrors the
    ``chat_template_kwargs``-emitting pattern of ``ReasoningToChatKwargs`` in
    ``experiments/datasets/instruction.py``.

    Assistant ``tool_calls.function.arguments`` JSON-string parsing is handled separately by
    ``transform_conversation._normalize_tool_structures`` (already applied by ``transform_row``),
    and empty ``tool_calls`` lists are dropped by the ``SINGLE_COLUMN_MULTI_TURN`` adapter's
    truthiness guard, so this function only concerns the ``tools`` schema relocation.
    """
    tools = row.get(tools_column)
    if tools is None:
        return {}
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            return {}
    if not tools:
        return {}
    return {"chat_template_kwargs": {"tools": tools}}
