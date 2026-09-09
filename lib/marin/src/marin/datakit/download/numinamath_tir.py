# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AI-MO/NuminaMath-TIR dataset download and transform.

NuminaMath-TIR is a math SFT corpus where assistant turns contain
tool-integrated reasoning traces, including Python snippets and their outputs.
The Hugging Face rows already expose OpenAI-style ``messages``; this module
renders those messages into the tagged transcript format used by Marin's
datakit reasoning sources.
"""

import re
from typing import Any

from fray.types import ResourceConfig
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from marin.datakit.chat import CHAT_SCHEMA
from marin.datakit.chat_normalize import normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import openai_chat_document, text_document
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "AI-MO/NuminaMath-TIR"
HF_REVISION = "77a91d7"
TRAIN_PARQUET_GLOB = "data/train-*.parquet"
VALID_ROLES = frozenset({"assistant", "system", "tool", "user"})
_PYTHON_EXECUTION = re.compile(
    r"```python\s*\n(?P<code>.*?)\n```\s*```output\s*\n(?P<output>.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)
PYTHON_TOOL = {
    "type": "function",
    "name": "python",
    "description": "Execute Python code and return its output.",
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
    },
}


def _message_text(message: Any) -> str | None:
    if not isinstance(message, dict):
        return None

    role = message.get("role")
    content = message.get("content")
    if role not in VALID_ROLES or not isinstance(content, str):
        return None

    content = content.strip()
    if not content:
        return None

    return f"<{role}>\n{content}\n</{role}>"


def render_messages(messages: Any) -> str | None:
    """Render OpenAI-style messages as a tagged datakit transcript."""
    if not isinstance(messages, list):
        return None

    parts: list[str] = []
    for message in messages:
        text = _message_text(message)
        if text is None:
            return None
        parts.append(text)

    if not parts:
        return None

    return "\n\n".join(parts)


def row_to_doc(row: dict) -> list[dict]:
    text = render_messages(row.get("messages"))
    if text is None:
        counters.pipeline.update_counter("numinamath_tir/dropped", 1)
        return []

    counters.pipeline.update_counter("numinamath_tir/kept", 1)
    return [text_document(text, HF_DATASET_ID)]


def row_to_chat_doc(row: dict) -> list[dict]:
    messages = row.get("messages")
    if not isinstance(messages, list) or any(_message_text(message) is None for message in messages):
        return []

    canonical: list[dict] = []
    call_index = 0
    for message in messages:
        if message["role"] != "assistant":
            canonical.append(dict(message))
            continue

        content = message["content"]
        matches = list(_PYTHON_EXECUTION.finditer(content))
        if not matches:
            counters.pipeline.update_counter("numinamath_tir/chat_without_execution_filtered", 1)
            return []
        cursor = 0
        for match in matches:
            reasoning = content[cursor : match.start()].strip()
            call_id = f"call_python_{call_index}"
            call_index += 1
            canonical.append(
                {
                    "role": "assistant",
                    "content": f"<think>{reasoning}</think>" if reasoning else None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": "python", "arguments": {"code": match.group("code")}},
                        }
                    ],
                }
            )
            canonical.append(
                {"role": "tool", "content": match.group("output"), "name": "python", "tool_call_id": call_id}
            )
            cursor = match.end()
        final_answer = content[cursor:].strip()
        if not final_answer:
            counters.pipeline.update_counter("numinamath_tir/chat_without_final_answer_filtered", 1)
            return []
        canonical.append({"role": "assistant", "content": final_answer})

    return [openai_chat_document(canonical, HF_DATASET_ID, chat_template_kwargs={"tools": [PYTHON_TOOL]})]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="numinamath-tir-transform", resources=ResourceConfig(cpu=1, ram="4g"))
    ctx.execute(pipeline)


def transform_chat(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet)
        .flat_map(row_to_chat_doc)
        .write_parquet(
            f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", schema=CHAT_SCHEMA, skip_existing=True
        )
    )
    ZephyrContext(name="numinamath-tir-chat-transform", resources=ResourceConfig(cpu=1, ram="4g")).execute(pipeline)


def download_numinamath_tir_step() -> StepSpec:
    """Download and transform NuminaMath-TIR train rows into tagged transcript documents."""
    dl = download_hf_step(
        "raw/numinamath-tir",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[TRAIN_PARQUET_GLOB],
    )

    return StepSpec(
        name="processed/numinamath-tir",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )


def numinamath_tir_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for NuminaMath-TIR."""
    processed = download_numinamath_tir_step()
    return (
        processed,
        normalize_step(name="normalized/numinamath-tir", download=processed),
    )


def numinamath_tir_chat_normalize_steps() -> tuple[StepSpec, ...]:
    download = download_hf_step(
        "raw/numinamath-tir",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[TRAIN_PARQUET_GLOB],
    )
    processed = StepSpec(
        name="processed-chat/numinamath-tir",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path),
        hash_attrs={"version": "2026.09.05.harmony-arrow"},
    )
    return processed, normalize_chat_step(
        output_schema=CHAT_SCHEMA, name="normalized-chat/numinamath-tir", download=processed
    )
