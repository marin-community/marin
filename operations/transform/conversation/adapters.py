import dataclasses
from dataclasses import dataclass
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


@dataclass
class TransformAdapter:
    source: str
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
    conversation_column: str = ""
    role_key: str = ""
    user_value: str = ""
    assistant_value: str = ""
    system_value: str = ""
    content_key: str = ""

    # If specified, the key will be used to select the message with
    # best metric in multiple turn conversations
    filter_on_key: str = ""

    def transform_conversation_to_openai_format(
        self,
        row: dict[str, Any],
    ) -> list[OpenAIChatMessage]:
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
            }
            conversation = row[self.conversation_column]
            for conv in conversation:
                role = role_to_openai_role[conv[self.role_key]]
                messages.append(OpenAIChatMessage(role=role, content=conv[self.content_key]))
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
                # We handle misaligned data gracefully rather than crash.
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
        dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
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
        dataset_format=InputDatasetFormat.INSTRUCTION_RESPONSE,
        instruction_column="query",
        response_column="response",
    )
)

register_adapter(
    TransformAdapter(
        source="allenai/tulu-v2-sft-mixture",
        dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
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
        source="allenai/tulu-v2-sft-mixture-olmo-4096",
        dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
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
        source="allenai/tulu-3-sft-mixture",
        dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
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
        source="open-r1/OpenThoughts-114k-math",
        dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
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
        source="sherryy/tulu-3-sft-personas-instruction-following-expanded",
        dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
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
        dataset_format=InputDatasetFormat.INSTRUCTION_RESPONSE,
        instruction_column="instruction",
        response_column="response",
    )
)

register_adapter(
    TransformAdapter(
        source="TIGER-Lab/AceCode-89K",
        dataset_format=InputDatasetFormat.INSTRUCTION_RESPONSE,
        instruction_column="question",
        response_column="inferences",
        filter_on_key="pass_rate",
        content_key="completion",
    )
)

# Define adapter (parser) for dataset
register_adapter(
    TransformAdapter(
        source="cognitivecomputations/dolphin-r1-nonreasoning",
        dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
        conversation_column="messages",
        role_key="role",
        user_value="user",
        assistant_value="assistant",
        system_value="system",
        content_key="content",
    )
)

# Define adapter (parser) for dataset
register_adapter(
    TransformAdapter(
        source="cognitivecomputations/dolphin-r1-reasoning",
        dataset_format=InputDatasetFormat.INSTRUCT_MSG_RESPONSE,
        instruction_column="messages",
        response_column="answer",
        role_key="role",
        user_value="user",
        assistant_value="assistant",
        system_value="system",
        content_key="content",
    )
)

# Define adapter (parser) for dataset
register_adapter(
    TransformAdapter(
        source="bespokelabs/Bespoke-Stratos-17k",
        dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
        instruction_column="system",
        conversation_column="conversations",
        role_key="from",
        user_value="user",
        assistant_value="assistant",
        content_key="value",
    )
)

register_adapter(
    TransformAdapter(
        source="HuggingFaceTB/smoltalk",
        dataset_format=InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN,
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
        source="PrimeIntellect/verifiable-math-problems",
        dataset_format=InputDatasetFormat.INSTRUCTION_RESPONSE,
        instruction_column="prompt",
        response_column="gold_standard_solution",
    )
)

register_adapter(
    TransformAdapter(
        source="facebook/natural_reasoning",
        dataset_format=InputDatasetFormat.INSTRUCT_COLUMN_RESPONSE,
        instruction_column="question",
        response_column="responses",
        content_key="response",
    )
)

register_adapter(
    TransformAdapter(
        source="GeneralReasoning/GeneralThought-195K-modelanswer",
        dataset_format=InputDatasetFormat.INSTRUCTION_RESPONSE,
        instruction_column="question",
        response_column="model_answer",
    )
)

register_adapter(
    TransformAdapter(
        source="GeneralReasoning/GeneralThought-195K-modelreasoning",
        dataset_format=InputDatasetFormat.INSTRUCTION_RESPONSE,
        instruction_column="question",
        response_column="model_reasoning",
    )
)
