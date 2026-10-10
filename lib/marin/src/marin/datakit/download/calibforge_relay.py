# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Terminus-2 relay SFT traces from laion/calibforge-relay-traces.

In a relay episode a student model (Grug Datakit 09-21, or an SFT checkpoint trained from it) works a CalibForge
terminal task until a hand-over trigger fires, and Qwen3.8-27B plays every later turn; only the teacher's turns are
trained. The ``terminus2_relay_sft`` config is the full training set of the best Terminus-2 recipe: 5,916 relay
episodes plus 4,392 Kimi-2.5 SWE-smith Terminus-2 traces, every assistant turn of the Kimi traces trained.

Each source message carries ``trained``. It is kept per source message as ``source_train_turns``, as the Nemotron v3
adapter does; neither the renderer nor the trainer applies it yet, so a mixture built from the rendered text also
trains the student's turns.
"""

import json
import re

import pyarrow as pa
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import prefix_join
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import (
    TRAJECTORY_FAILED_TAG,
    TRAJECTORY_SOLVED_TAG,
    checked_openai_chat_document,
    load_parquet_batched,
)
from marin.datakit.download.terminus import ThinkTokens, terminus_protocol_messages
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "laion/calibforge-relay-traces"
HF_REVISION = "66feab4d24bfc37af4c38d95a6a26187b5648308"
SOURCE_NAME = "calibforge-relay/terminus2"
TRANSFORM_VERSION = "2026.10.10.2"
THINK_TOKENS = ThinkTokens("<|start_think|>", "<|end_think|>")
COUNTER_PREFIX = "calibforge_relay/chat"
SOURCE_CHAT_SCHEMA = pa.schema(
    [
        *CHAT_SCHEMA,
        pa.field("source_train_turns", pa.list_(pa.bool_())),
        pa.field("teacher", pa.string()),
        pa.field("task_source", pa.string()),
        pa.field("outcome", pa.string()),
    ]
)

_THINK_MARKER = re.compile(r"<\|(?:start|end)_think\|>")
# Terminus-2 accepts a reply whose JSON stops before its closing brackets; these suffixes close the common cases.
_CLOSING_SUFFIXES = ("", "}", "]}", "}]}")


def _command_payload(text: str) -> tuple[dict, str] | None:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if stripped.startswith(THINK_TOKENS.start) and stripped.endswith(THINK_TOKENS.end):
        wrapped = stripped[len(THINK_TOKENS.start) : -len(THINK_TOKENS.end)].strip()
        try:
            payload = decoder.decode(wrapped)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(payload, dict) and isinstance(payload.get("commands"), list):
                return payload, ""
    # Closed reasoning spans can contain example commands that were never executed.
    reasoning_spans = list(re.finditer(r"<\|start_think\|>.*?<\|end_think\|>", text, re.DOTALL))
    for index, char in enumerate(text):
        if char != "{" or any(span.start() <= index < span.end() for span in reasoning_spans):
            continue
        body = text[index:].rstrip()
        for suffix in _CLOSING_SUFFIXES:
            try:
                payload, _ = decoder.raw_decode(body + suffix)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("commands"), list):
                return payload, text[:index]
            break
    return None


def canonical_reply(content: str) -> str | None:
    """Rewrite a recorded reply as one leading reasoning span plus its command JSON, or None without a command.

    The student's replies were accepted by the harness with several or unclosed reasoning spans, prose between the
    reasoning and the JSON, or JSON missing its closing brackets. When the reply opens with reasoning, everything
    recorded before the JSON becomes one reasoning span with the markers removed; a reply without reasoning drops its
    prose, as ``terminus_protocol_messages`` does.
    """
    found = _command_payload(content)
    if found is None:
        return None
    payload, before = found
    reasoning = _THINK_MARKER.sub("", before).strip() if THINK_TOKENS.start in before else ""
    command = json.dumps(payload, ensure_ascii=False)
    return f"{THINK_TOKENS.start}{reasoning}{THINK_TOKENS.end}{command}" if reasoning else command


def row_to_chat_doc(row: dict) -> list[dict]:
    source_messages = row.get("messages")
    if not source_messages:
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/empty_filtered", 1)
        return []
    conversation: list[dict] = []
    train_turns: list[bool] = []
    for message in source_messages:
        # The only system message is the student template's "Reasoning: /think" line, not part of the task.
        if message["role"] == "system":
            continue
        content = message["content"]
        if message["role"] == "assistant":
            content = canonical_reply(content)
            if content is None:
                counters.pipeline.update_counter(f"{COUNTER_PREFIX}/reply_without_command_filtered", 1)
                return []
        conversation.append({"role": message["role"], "content": content})
        train_turns.append(bool(message["trained"]))
    messages = terminus_protocol_messages(conversation, THINK_TOKENS)
    if messages is None:
        counters.pipeline.update_counter(f"{COUNTER_PREFIX}/terminus_protocol_filtered", 1)
        return []
    return checked_openai_chat_document(
        messages,
        HF_DATASET_ID,
        counter_prefix=COUNTER_PREFIX,
        source_id=row["sid"],
        source_train_turns=train_turns,
        teacher=row["teacher"],
        task_source=row["source"],
        outcome=TRAJECTORY_SOLVED_TAG if row["passed"] else TRAJECTORY_FAILED_TAG,
    )


def transform_chat(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(prefix_join(input_path, "**/*.parquet"))
        .flat_map(load_parquet_batched)
        .flat_map(row_to_chat_doc)
        .write_parquet(
            prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
            schema=SOURCE_CHAT_SCHEMA,
            skip_existing=True,
        )
    )
    ZephyrContext(name="calibforge-relay-chat-transform", resources=ResourceConfig(cpu=1, ram="16g")).execute(pipeline)


def calibforge_relay_chat_normalize_steps() -> tuple[StepSpec, ...]:
    """Download the Terminus-2 relay SFT set and normalize its structured chat."""
    raw = download_hf_step(
        f"raw/{SOURCE_NAME}",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=["data/terminus2_relay_sft/*.parquet"],
    )
    processed = StepSpec(
        name=f"processed-chat/{SOURCE_NAME}",
        deps=[raw],
        fn=lambda output_path: transform_chat(raw.output_path, output_path),
        hash_attrs={"version": TRANSFORM_VERSION, "think_tokens": THINK_TOKENS},
    )
    return processed, normalize_chat_step(
        name=f"normalized-chat/{SOURCE_NAME}", download=processed, output_schema=SOURCE_CHAT_SCHEMA
    )
