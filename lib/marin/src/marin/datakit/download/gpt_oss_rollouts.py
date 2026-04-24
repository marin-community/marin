# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""andyrdt/gpt-oss-20b-rollouts dataset download and transform.

GPT-OSS-20B rollouts with parsed reasoning chains. We include only the
non-benchmark subsets: NuminaMath-CoT, WildChat-1M, and ultrachat_200k.

Each row has a user prompt, the model's internal thinking, and the final
assistant response. We render these into a single document.
"""

import hashlib

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters
from zephyr.readers import load_jsonl

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "andyrdt/gpt-oss-20b-rollouts"
HF_REVISION = "f47b4a2"

SUBSETS = {
    "NuminaMath-CoT": "train",
    "WildChat-1M": "train",
    "ultrachat_200k": "train_sft",
}


def row_to_doc(row: dict) -> list[dict]:
    user = row.get("user_content") or ""
    thinking = row.get("assistant_thinking") or ""
    response = row.get("assistant_content") or ""
    if not user or not response:
        counters.increment("gpt_oss_rollouts/dropped")
        return []

    parts = [f"<user>\n{user}\n</user>"]
    if thinking:
        parts.append(f"<thinking>\n{thinking}\n</thinking>")
    parts.append(f"<assistant>\n{response}\n</assistant>")

    text = "\n\n".join(parts)

    counters.increment("gpt_oss_rollouts/kept")
    return [
        {
            "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
            "source": "andyrdt/gpt-oss-20b-rollouts",
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    input_files = [f"{input_path}/{subset}/{split}.jsonl" for subset, split in SUBSETS.items()]
    pipeline = (
        Dataset.from_list(input_files)
        .flat_map(load_jsonl)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="gpt-oss-rollouts-transform", resources=ResourceConfig(cpu=1, ram="8g"))
    list(ctx.execute(pipeline))


def download_gpt_oss_rollouts_step() -> StepSpec:
    """Download and transform non-benchmark GPT-OSS-20B rollouts into JSONL documents."""
    hf_urls_glob = []
    for subset, split in SUBSETS.items():
        hf_urls_glob.append(f"{subset}/{split}.jsonl")

    dl = download_hf_step(
        "raw/gpt-oss-20b-rollouts",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=hf_urls_glob,
    )

    return StepSpec(
        name="processed/gpt-oss-20b-rollouts",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v2"},
    )


def gpt_oss_rollouts_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for gpt-oss-rollouts."""
    processed = download_gpt_oss_rollouts_step()
    return (
        processed,
        normalize_step(name="data/normalized/gpt-oss-rollouts", download=processed),
    )
