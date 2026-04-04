# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""togethercomputer/CoderForge-Preview dataset download and transform.

Downloads raw parquet files from HuggingFace, then transforms each trajectory
into a single document by rendering the chat messages as readable text with a
reward tag prefix so the model learns to distinguish successful and failed
rollouts.
"""

import hashlib
import json

from fray.v2 import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters, load_parquet

from marin.datakit.download.huggingface import download_hf_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "togethercomputer/CoderForge-Preview"
HF_REVISION = "060fca9"

SPLITS = ["SWE_Rebench", "SWE_Smith", "R2E_Gym"]


def reward_to_tag(reward: float | None) -> str:
    if reward is None:
        return "This trajectory has an unknown outcome."
    if reward >= 1.0:
        return "This trajectory solved the task successfully."
    return "This trajectory failed to solve the task."


def render_tool_call(tc: dict) -> str:
    func = tc.get("function", {})
    name = func.get("name", "unknown")
    args = func.get("arguments", {})
    if isinstance(args, str):
        args = json.loads(args)
    parts = [f"<tool_call:{name}>"]
    for k, v in args.items():
        parts.append(f"  {k}: {v}")
    parts.append(f"</tool_call:{name}>")
    return "\n".join(parts)


def render_message(msg: dict) -> str:
    role = msg.get("role", "unknown")
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls")

    parts = [f"<{role}>"]
    if content:
        parts.append(content)
    if tool_calls:
        for tc in tool_calls:
            parts.append(render_tool_call(tc))
    parts.append(f"</{role}>")
    return "\n".join(parts)


def row_to_doc(row: dict) -> list[dict]:
    messages_raw = row.get("messages", "")
    if not messages_raw:
        counters.increment("coderforge/dropped")
        return []

    messages = json.loads(messages_raw) if isinstance(messages_raw, str) else messages_raw
    if not messages:
        counters.increment("coderforge/dropped")
        return []

    tag = reward_to_tag(row.get("reward"))
    rendered = "\n\n".join(render_message(m) for m in messages)
    text = f"{tag}\n\n{rendered}"

    counters.increment("coderforge/kept")
    return [
        {
            "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
            "source": "togethercomputer/CoderForge-Preview",
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    # The download already filters to only the splits we want via hf_urls_glob
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="coderforge-transform", resources=ResourceConfig(cpu=1, ram="8g"))
    list(ctx.execute(pipeline))


def download_coderforge_step() -> StepSpec:
    """Download and transform CoderForge-Preview into JSONL documents."""
    dl = download_hf_step(
        "raw/coderforge-preview",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[f"trajectories/{split}-*.parquet" for split in SPLITS],
    )

    return StepSpec(
        name="processed/coderforge-preview",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )
