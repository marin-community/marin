from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel


class DatasetFormat(str, Enum):
    """Format of the SFT Dataset"""

    SINGLE_COLUMN_MULTI_TURN: str = "messages"
    INSTRUCTION_RESPONSE: str = "instruction_response"


class OpenAIChatMessage(BaseModel):
    role: str
    content: str


@dataclass
class TransformAdapter:
    source: str
    dataset_format: DatasetFormat = DatasetFormat.INSTRUCTION_RESPONSE

    # Instruction Response
    instruction_column: str = ""
    response_column: str = ""

    # Single Column Multi Turn
    """
    Example of role_key, user_value, assistant_value, and system_value:
    In OpenHermes-2.5, a conversation can look like this:
    [ { "from": "human", "value": "..."},
      { "from": "gpt", "value": "..."} ]

    In this example, the role_key is "from", the user_value is "human", the assistant_value is "gpt",
    and the system_value is "system". This helps us map the roles to the correct values in the OpenAI
    format from "from" -> "role" and "human"/"gpt" -> "user"/"assistant".
    """
    conversation_column: str = ""
    role_key: str = ""
    user_value: str = ""
    assistant_value: str = ""
    system_value: str = ""
    content_key: str = ""

    def transform_conversation_to_openai_format(
        self,
        row: dict[str, Any],
    ) -> list[OpenAIChatMessage]:
        if self.dataset_format == DatasetFormat.INSTRUCTION_RESPONSE:
            messages = []
            instruction = row[self.instruction_column]
            response = row[self.response_column]
            messages.append(OpenAIChatMessage(role="user", content=instruction))
            messages.append(OpenAIChatMessage(role="assistant", content=response))
            return messages
        elif self.dataset_format == DatasetFormat.SINGLE_COLUMN_MULTI_TURN:
            messages = []
            role_to_openai_role = {
                self.user_value: "user",
                self.assistant_value: "assistant",
                self.system_value: "system",
            }
            conversation = row[self.conversation_column]
            for conv in conversation:
                role = role_to_openai_role[conv[self.role_key]]
                messages.append(OpenAIChatMessage(role=role, content=conv[self.content_key]))
            return messages
        else:
            raise ValueError(f"Invalid dataset format: {self.dataset_format}")

    def copy(self) -> "TransformAdapter":
        return TransformAdapter(
            source=self.source,
            dataset_format=self.dataset_format,
            instruction_column=self.instruction_column,
            response_column=self.response_column,
            conversation_column=self.conversation_column,
            role_key=self.role_key,
            user_value=self.user_value,
            assistant_value=self.assistant_value,
            system_value=self.system_value,
            content_key=self.content_key,
        )


transform_templates: dict[str, TransformAdapter] = {}


def register_adapter(adapter: TransformAdapter):
    transform_templates[adapter.source] = adapter


def get_adapter(source: str) -> TransformAdapter:
    if source not in transform_templates:
        raise ValueError(f"No adapter found for source: {source}")
    return transform_templates[source].copy()


register_adapter(
    TransformAdapter(
        source="teknium/OpenHermes-2.5",
        dataset_format=DatasetFormat.SINGLE_COLUMN_MULTI_TURN,
        conversation_column="conversations",
        role_key="from",
        user_value="human",
        assistant_value="gpt",
        system_value="system",
        content_key="value",
    )
)

register_adapter(
    TransformAdapter(
        source="meta-math/MetaMathQA",
        dataset_format=DatasetFormat.INSTRUCTION_RESPONSE,
        instruction_column="query",
        response_column="response",
    )
)

register_adapter(
    TransformAdapter(
        source="allenai/tulu-v2-sft-mixture",
        dataset_format=DatasetFormat.SINGLE_COLUMN_MULTI_TURN,
        conversation_column="messages",
        role_key="role",
        user_value="user",
        assistant_value="assistant",
        system_value="system",
        content_key="content",
    )
)

register_adapter(
    TransformAdapter(
        source="openbmb/UltraInteract_sft",
        dataset_format=DatasetFormat.INSTRUCTION_RESPONSE,
        instruction_column="instruction",
        response_column="response",
    )
)
