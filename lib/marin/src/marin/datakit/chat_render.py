# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render normalized Harmony conversations for the Marin tokenizer."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import groupby

import pyarrow as pa
from fray.types import ResourceConfig
from openai_harmony import Message, Role
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from marin.datakit.chat_normalize import ChatChannel, message_text
from marin.datakit.normalize import DEFAULT_MAX_WORKERS
from marin.execution.step_spec import StepSpec

CHAT_RENDER_VERSION = "marin-v1"
MARIN_BOS_TOKEN = "<|begin_of_text|>"
START_THINK = "<|start_think|>"
END_THINK = "<|end_think|>"
RENDERED_CHAT_SCHEMA = pa.schema(
    [pa.field("id", pa.string(), nullable=False), pa.field("text", pa.string(), nullable=False)]
)


def _tool_instructions(tools: Sequence[dict]) -> str:
    definitions = "".join(str(tool) for tool in tools)
    return f"""
### Tools

You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools> </tools> tags:

<tools>
{definitions}</tools>

For each function call, pass a json object with function name and arguments within <tool_call> </tool_call> tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

"""


@dataclass(frozen=True)
class _Turn:
    role: Role
    content: str
    separator: str


def _assistant_content(message: Message) -> str:
    text = message_text(message)
    match message.channel:
        case ChatChannel.ANALYSIS:
            return f"{START_THINK}{text}{END_THINK}"
        case ChatChannel.COMMENTARY | ChatChannel.FINAL:
            return text
        case _:
            raise ValueError(f"Unsupported assistant channel: {message.channel!r}")


def _message_turn(message: Message) -> _Turn:
    text = message_text(message)
    match message.author.role:
        case Role.SYSTEM | Role.DEVELOPER | Role.USER:
            return _Turn(message.author.role, text.strip(), "\n")
        case Role.ASSISTANT:
            recipient = message.recipient
            if message.channel != ChatChannel.COMMENTARY or recipient is None or not recipient.startswith("functions."):
                raise ValueError("Tool calls require commentary addressed to functions.<name>")
            arguments = json.loads(text)
            if not isinstance(arguments, dict):
                raise ValueError("Tool-call arguments must be a JSON object")
            name = recipient.removeprefix("functions.")
            body = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
            return _Turn(Role.ASSISTANT, body, "\n")
        case Role.TOOL:
            name = message.author.name
            if name is None or not name.startswith("functions."):
                raise ValueError("Tool observations require a functions.<name> author")
            name = name.removeprefix("functions.")
            return _Turn(Role.TOOL, f'<tool_response name="{name}">{text}</tool_response>', "\n")


def _chat_turns(messages: Sequence[Message]) -> Iterator[_Turn]:
    # Analysis and final are separate Harmony messages but one Marin assistant
    # turn. Tool calls retain their own turn, as required by the inference template.
    for assistant_text, group in groupby(
        messages, key=lambda message: message.author.role == Role.ASSISTANT and message.recipient is None
    ):
        if assistant_text:
            content = "".join(_assistant_content(message) for message in group).strip()
            yield _Turn(Role.ASSISTANT, content, "")
        else:
            yield from (_message_turn(message) for message in group)


def render_marin_chat(
    messages: Sequence[Message],
    *,
    bos_token: str = MARIN_BOS_TOKEN,
    tools: Sequence[dict] = (),
    enable_thinking: bool | None = None,
    custom_instructions: str = "",
    add_generation_prompt: bool = False,
) -> str:
    """Render Harmony as Marin chat text, retaining reasoning from every turn.

    Uses the existing Marin inference template's headers, whitespace, JSON and
    tool syntax. Tool call IDs are absent from normalized Harmony; ordered calls
    and named observations supply their association.
    """
    auxiliary: list[str] = []
    if enable_thinking is not None:
        auxiliary.append("Reasoning: /think" if enable_thinking else "Reasoning: /nothink")
    if custom_instructions:
        auxiliary.append(custom_instructions.strip())
    if tools:
        auxiliary.append(_tool_instructions(tools))
    system = f"<|start_header_id|>system<|end_header_id|>{''.join(auxiliary)}<|eot_id|>" if auxiliary else ""
    conversation = "".join(
        f"<|start_header_id|>{turn.role.value}<|end_header_id|>\n{turn.content}<|eot_id|>{turn.separator}"
        for turn in _chat_turns(messages)
    )
    prefix = "<|start_header_id|>assistant<|end_header_id|>\n" if add_generation_prompt else ""
    return f"{bos_token}{system}{conversation}{prefix}"


def render_chat_record(record: dict) -> dict:
    """Project a normalized chat row into the standard normalizer's id/text input."""
    messages = [Message.from_dict(message) for message in record["messages"]]
    kwargs = json.loads(record["chat_template_kwargs"]) if record.get("chat_template_kwargs") else {}
    return {"id": record["id"], "text": render_marin_chat(messages, **kwargs)}


def render_chat_to_parquet(
    *,
    input_path: str,
    output_path: str,
    worker_resources: ResourceConfig | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> None:
    """Write id/text Parquet from a directory of normalized Harmony chat shards."""
    pipeline = (
        Dataset.from_files(prefix_join(input_path, "*.parquet"))
        .flat_map(load_parquet)
        .map(render_chat_record)
        .write_parquet(prefix_join(output_path, "part-{shard:05d}-of-{total:05d}.parquet"), schema=RENDERED_CHAT_SCHEMA)
    )
    ZephyrContext(
        name="render-chat", resources=worker_resources or ResourceConfig(cpu=2, ram="4g"), max_workers=max_workers
    ).execute(pipeline)


def render_chat_step(*, name: str, chat: StepSpec) -> StepSpec:
    """Create an id/text rendering step from a chat normalization step."""
    return StepSpec(
        name=name,
        fn=lambda output_path: render_chat_to_parquet(
            input_path=prefix_join(chat.output_path, "outputs/main"), output_path=output_path
        ),
        deps=[chat],
        hash_attrs={"version": CHAT_RENDER_VERSION},
    )
