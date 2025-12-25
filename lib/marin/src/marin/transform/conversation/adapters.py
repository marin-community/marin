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

import dataclasses
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

    SINGLE_COLUMN_MULTI_TURN: str = "messages"
    INSTRUCTION_RESPONSE: str = "instruction_response"
    INSTRUCT_COLUMN_RESPONSE: str = "instruct_column_response"
    INSTRUCT_MSG_RESPONSE: str = "instruct_msg_response"


def _replace_special_keys(string: str | None, special_keys_mapping: dict[str, str]) -> str | None:
    if (not special_keys_mapping) or (string is None):
        return string
    pattern = "|".join(map(re.escape, special_keys_mapping.keys()))
    return re.sub(pattern, lambda m: special_keys_mapping[m.group(0)], string)


def transform_instruction_response(
    row: dict[str, dict[str, str]],
    instruction_column: str,
    response_column: str,
    filter_on_key: str | None,
    content_key: str,
    special_keys_mapping: dict[str, str] = dataclasses.field(default_factory=dict),
) -> list[OpenAIChatMessage] | None:
    messages: list[OpenAIChatMessage] = []
    instruction = row[instruction_column]
    response = row[response_column]
    # Check data
    if instruction is None or response is None:
        return None  # Do not process rows with missing data
    if filter_on_key:
        best_completion = None
        best_metric = -float("inf")  # TODO: Make this a config

        for completion in response:
            if completion[filter_on_key] > best_metric:
                best_metric = completion[filter_on_key]
                best_completion = completion
        response = best_completion[content_key]
    # Replace special keys
    instruction = _replace_special_keys(instruction, special_keys_mapping)
    response = _replace_special_keys(response, special_keys_mapping)
    # Save
    messages.append(OpenAIChatMessage(role="user", content=instruction))
    messages.append(OpenAIChatMessage(role="assistant", content=response))
    return messages


def transform_single_column_multi_turn(
    row: dict[str, list[str]],
    conversation_column: str,
    role_key: str,
    user_value: str,
    assistant_value: str,
    system_value: str,
    content_key: str,
    special_keys_mapping: dict[str, str] = dataclasses.field(default_factory=dict),
) -> list[OpenAIChatMessage]:
    messages: list[OpenAIChatMessage] = []
    role_to_openai_role: dict[str, str] = {
        user_value: "user",
        assistant_value: "assistant",
        system_value: "system",
    }
    conversation = row[conversation_column]
    for conv in conversation:
        this_role, this_content = conv[role_key], conv[content_key]
        openai_role = role_to_openai_role[this_role]
        # Replace special keys
        this_content = _replace_special_keys(this_content, special_keys_mapping)
        # if this_content is None:
        #     raise ValueError(f"Content is None, original conv: {conv}")
        messages.append(OpenAIChatMessage(role=openai_role, content=this_content))
    return messages


def transform_instruct_column_response(
    row: dict[str, dict[str, str]],
    instruction_column: str,
    response_column: str,
    content_key: str,
    special_keys_mapping: dict[str, str] = dataclasses.field(default_factory=dict),
) -> list[OpenAIChatMessage]:
    messages: list[OpenAIChatMessage] = []
    instruction = row[instruction_column]
    responses = row[response_column]

    # Get the first (and only) response from the list
    response_dict = responses[0]
    response_content = response_dict[content_key]
    # Replace special keys
    instruction = _replace_special_keys(instruction, special_keys_mapping)
    response_content = _replace_special_keys(response_content, special_keys_mapping)
    # Save
    messages.append(OpenAIChatMessage(role="user", content=instruction))
    messages.append(OpenAIChatMessage(role="assistant", content=response_content))
    return messages


def transform_instruct_msg_response(
    row: dict[str, dict[str, str]],
    instruction_column: str,
    response_column: str,
    role_key: str,
    content_key: str,
    special_keys_mapping: dict[str, str] = dataclasses.field(default_factory=dict),
) -> list[OpenAIChatMessage] | None:
    messages: list[OpenAIChatMessage] = []  # Initialize
    # Get data
    instruction = row[instruction_column]  # List of dict
    responses = row[response_column]  # Single string
    if (responses is None) or (len(instruction) > 1) or (role_key not in instruction[0]):
        # We do not process rows that have more than one messages.
        # This occurs in Dolphin-R1 reasoning, where instructions are
        # sometimes part of the 'system' prompt instead of 'user' prompt.
        # We handle misaligned data gracefully rather than crash.
        return None
    else:
        instruction_content = instruction[0][content_key]
        # Replace special keys
        instruction_content = _replace_special_keys(instruction_content, special_keys_mapping)
        responses = _replace_special_keys(responses, special_keys_mapping)
        # Save
        messages.append(OpenAIChatMessage(role="user", content=instruction_content))
        messages.append(OpenAIChatMessage(role="assistant", content=responses))
        return messages


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
    ) -> list[OpenAIChatMessage]:
        if self.dataset_format == InputDatasetFormat.INSTRUCTION_RESPONSE:
            return transform_instruction_response(
                row,
                self.instruction_column,
                self.response_column,
                self.filter_on_key,
                self.content_key,
                special_keys_mapping=self.special_keys_mapping,
            )
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
                messages.append(OpenAIChatMessage(role=role, content=conv[self.content_key]))
            return messages
        elif self.dataset_format == InputDatasetFormat.INSTRUCT_COLUMN_RESPONSE:
            return transform_instruct_column_response(
                row,
                self.instruction_column,
                self.response_column,
                self.content_key,
                special_keys_mapping=self.special_keys_mapping,
            )
        elif self.dataset_format == InputDatasetFormat.INSTRUCT_MSG_RESPONSE:
            messages = []  # Initialize
            # Get data
            instruction = row[self.instruction_column]  # List of dict
            responses = row[self.response_column]  # Single string
            if (responses is None) or (len(instruction) > 1) or (self.role_key not in instruction[0]):
                # We do not process rows that have more than one messages.
                # This occurs in Dolphin-R1 reasoning, where instructions are
                # sometimes part of the 'system' prompt instead of 'user' prompt.
                # We handle misaligned data gracefully rather than crash.
                return []
            else:
                instruction_content = instruction[0][self.content_key]
                messages.append(OpenAIChatMessage(role="user", content=instruction_content))
                messages.append(OpenAIChatMessage(role="assistant", content=responses))
                return messages
        else:
            raise ValueError(f"Invalid dataset format: {self.dataset_format}")

    def copy(self) -> "TransformAdapter":
        return dataclasses.replace(self)
