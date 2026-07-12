# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AI-MO/NuminaMath-TIR dataset download and transform.

NuminaMath-TIR is a math SFT corpus where assistant turns contain
tool-integrated reasoning traces, including Python snippets and their outputs.
The Hugging Face rows already expose OpenAI-style ``messages``; this module
renders those messages into the tagged transcript format used by Marin's
datakit reasoning sources.
"""

from typing import Any

from fray.types import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_parquet

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import text_document
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "AI-MO/NuminaMath-TIR"
HF_REVISION = "77a91d7"
TRAIN_PARQUET_GLOB = "data/train-*.parquet"
VALID_ROLES = frozenset({"assistant", "system", "tool", "user"})


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


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="numinamath-tir-transform", resources=ResourceConfig(cpu=1, ram="4g"))
    ctx.execute(pipeline)


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
