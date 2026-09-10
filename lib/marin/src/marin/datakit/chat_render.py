# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render normalized Harmony conversations for the Marin tokenizer."""

import json
from collections.abc import Iterator, Sequence
from itertools import groupby

import pyarrow as pa
from fray.types import ResourceConfig
from openai_harmony import Message, Role
from rigging.filesystem.storage_path import prefix_join
from transformers.utils.chat_template_utils import render_jinja_template
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from marin.datakit.chat_normalize import ChatChannel, message_text
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.datakit.normalize import DEFAULT_MAX_WORKERS
from marin.execution.step_spec import StepSpec

CHAT_RENDER_VERSION = "marin-v2"
MARIN_BOS_TOKEN = "<|begin_of_text|>"
START_THINK = "<|start_think|>"
END_THINK = "<|end_think|>"
RENDERED_CHAT_SCHEMA = pa.schema(
    [pa.field("id", pa.string(), nullable=False), pa.field("text", pa.string(), nullable=False)]
)


def _assistant_content(message: Message) -> str:
    text = message_text(message)
    match message.channel:
        case ChatChannel.ANALYSIS:
            return f"{START_THINK}{text}{END_THINK}"
        case ChatChannel.COMMENTARY | ChatChannel.FINAL:
            return text
        case _:
            raise ValueError(f"Unsupported assistant channel: {message.channel!r}")


def _inference_message(message: Message) -> dict:
    text = message_text(message)
    match message.author.role:
        case Role.SYSTEM | Role.DEVELOPER | Role.USER:
            return {"role": message.author.role.value, "content": text}
        case Role.ASSISTANT:
            recipient = message.recipient
            if message.channel != ChatChannel.COMMENTARY or recipient is None or not recipient.startswith("functions."):
                raise ValueError("Tool calls require commentary addressed to functions.<name>")
            arguments = json.loads(text)
            if not isinstance(arguments, dict):
                raise ValueError("Tool-call arguments must be a JSON object")
            name = recipient.removeprefix("functions.")
            return {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
            }
        case Role.TOOL:
            name = message.author.name
            if name is None or not name.startswith("functions."):
                raise ValueError("Tool observations require a functions.<name> author")
            name = name.removeprefix("functions.")
            return {"role": "tool", "name": name, "content": text}


def _inference_messages(messages: Sequence[Message]) -> Iterator[dict]:
    # A tool handoff ends an assistant turn. Analysis, commentary, and parallel
    # calls before that handoff must share one end-of-turn token.
    for role, group in groupby(messages, key=lambda message: message.author.role):
        if role == Role.ASSISTANT:
            turn = list(group)
            content = "".join(_assistant_content(message) for message in turn if message.recipient is None)
            calls = [_inference_message(message)["tool_calls"][0] for message in turn if message.recipient is not None]
            yield {"role": "assistant", "content": content, "tool_calls": calls}
        else:
            yield from (_inference_message(message) for message in group)


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
    # Omit absent options: the inference template distinguishes an undefined
    # reasoning mode from an explicitly supplied value.
    kwargs = {"enable_thinking": enable_thinking} if enable_thinking is not None else {}
    rendered, _ = render_jinja_template(
        conversations=[list(_inference_messages(messages))],
        chat_template=MARIN_CHAT_TEMPLATE,
        bos_token=bos_token,
        tools=list(tools),
        custom_instructions=custom_instructions,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    return rendered[0]


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
        hash_attrs={"version": CHAT_RENDER_VERSION, "chat_template": MARIN_CHAT_TEMPLATE},
    )
