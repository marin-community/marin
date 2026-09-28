# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project generated Datakit chats into Levanter's masked chat input."""

from dataclasses import dataclass

import pyarrow as pa
from fray.types import ResourceConfig
from marin.datakit.chat_normalize import message_text
from marin.execution.artifact import Artifact
from openai_harmony import Message, Role
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

SFT_CHAT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string(), nullable=False),
        pa.field(
            "messages",
            pa.list_(pa.struct([pa.field("role", pa.string()), pa.field("content", pa.string())])),
            nullable=False,
        ),
    ]
)


@dataclass(frozen=True)
class PrepareConfig:
    input_path: str
    output_path: str


def prepare_chat_record(record: dict) -> dict:
    """Project generated text-only Harmony into Levanter's OpenAI chat schema."""
    messages = [Message.from_dict(message) for message in record["messages"]]
    if any(message.author.role not in (Role.USER, Role.ASSISTANT) or message.recipient for message in messages):
        raise ValueError("curriculum SFT requires text-only user/assistant conversations")
    return {
        "id": record["id"],
        "messages": [{"role": message.author.role.value, "content": message_text(message)} for message in messages],
    }


def prepare_generated_chat(config: PrepareConfig) -> Artifact:
    """Write OpenAI-style chat Parquet for assistant-only SFT."""
    pipeline = (
        Dataset.from_files(prefix_join(prefix_join(config.input_path, "chat"), "*.parquet"))
        .flat_map(load_parquet)
        .map(prepare_chat_record)
        .write_parquet(
            prefix_join(config.output_path, "part-{shard:05d}-of-{total:05d}.parquet"), schema=SFT_CHAT_SCHEMA
        )
    )
    ZephyrContext(name="prepare-curriculum-chat", resources=ResourceConfig(cpu=2, ram="4g"), max_workers=4).execute(
        pipeline
    )
    return Artifact(path=config.output_path)
