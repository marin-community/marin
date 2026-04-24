# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""nvidia/Nemotron-Terminal-Corpus dataset download and transform.

Terminal agent rollouts (command-line task solving). Each row contains a
multi-turn conversation between a user and an AI assistant solving Linux
CLI tasks, with thinking traces.
"""

import hashlib

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.datakit.download.rollout_transforms import load_parquet_batched
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "nvidia/Nemotron-Terminal-Corpus"
HF_REVISION = "a1667c4"


def render_message(msg: dict) -> str:
    role = msg.get("role", "unknown")
    content = msg.get("content", "")
    return f"<{role}>\n{content}\n</{role}>"


def row_to_doc(row: dict) -> list[dict]:
    conversations = row.get("conversations")
    if not conversations:
        counters.increment("nemotron_terminal/dropped")
        return []

    text = "\n\n".join(render_message(m) for m in conversations)

    counters.increment("nemotron_terminal/kept")
    return [
        {
            "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
            "source": "nvidia/Nemotron-Terminal-Corpus",
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet_batched)
        .flat_map(row_to_doc)
        .reshard(64)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="nemotron-terminal-transform", resources=ResourceConfig(cpu=1, ram="32g"))
    list(ctx.execute(pipeline))


def download_nemotron_terminal_step() -> StepSpec:
    """Download and transform Nemotron-Terminal-Corpus into JSONL documents."""
    dl = download_hf_step(
        "raw/nemotron-terminal-corpus",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
    )

    return StepSpec(
        name="processed/nemotron-terminal-corpus",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v2"},
    )


def nemotron_terminal_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for nemotron-terminal."""
    processed = download_nemotron_terminal_step()
    return (
        processed,
        normalize_step(name="data/normalized/nemotron-terminal", download=processed),
    )
