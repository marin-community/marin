# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""lambda/hermes-agent-reasoning-traces dataset download and transform.

Multi-turn tool-calling trajectories produced by the Hermes agent harness,
with two source-model configs (kimi, glm-5.1). Each row carries a ShareGPT
conversation (``from``/``value`` turns) interleaving ``<think>`` reasoning,
``<tool_call>`` invocations, and ``<tool_response>`` execution results, plus
a JSON ``tools`` definition listing the tools available to the agent.
"""

import hashlib

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.datakit.download.rollout_transforms import load_parquet_batched
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "lambda/hermes-agent-reasoning-traces"
HF_REVISION = "b92885e"

CONFIGS = ("kimi", "glm-5.1")


def render_message(msg: dict) -> str:
    role = msg.get("from", "unknown")
    value = msg.get("value") or ""
    return f"<{role}>\n{value}\n</{role}>"


def row_to_doc(row: dict) -> list[dict]:
    conversations = row.get("conversations")
    if not conversations:
        counters.increment("hermes_agent_reasoning_traces/dropped")
        return []

    tools = row.get("tools") or ""
    rendered = "\n\n".join(render_message(m) for m in conversations)
    text = f"<tools>\n{tools}\n</tools>\n\n{rendered}" if tools else rendered

    counters.increment("hermes_agent_reasoning_traces/kept")
    return [
        {
            "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
            "source": HF_DATASET_ID,
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/data/**/train.parquet")
        .flat_map(load_parquet_batched)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="hermes-agent-reasoning-traces-transform", resources=ResourceConfig(cpu=1, ram="32g"))
    ctx.execute(pipeline)


def download_hermes_agent_reasoning_traces_step() -> StepSpec:
    """Download and transform hermes-agent-reasoning-traces into JSONL documents."""
    dl = download_hf_step(
        "raw/hermes-agent-reasoning-traces",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[f"data/{config}/train.parquet" for config in CONFIGS],
    )

    return StepSpec(
        name="processed/hermes-agent-reasoning-traces",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )


def hermes_agent_reasoning_traces_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for hermes-agent-reasoning-traces."""
    processed = download_hermes_agent_reasoning_traces_step()
    return (
        processed,
        normalize_step(name="normalized/hermes-agent-reasoning-traces", download=processed),
    )
