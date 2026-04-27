# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PrimeIntellect/SYNTHETIC-1 dataset download and transform.

Downloads raw parquet files from HuggingFace, then transforms each row into a
single document by concatenating prompt + score tag + response.
"""

import hashlib

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters, load_parquet

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.datakit.download.rollout_transforms import strip_think_tags
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "PrimeIntellect/SYNTHETIC-1"
HF_REVISION = "f08fe8c"

SCORE_TAGS = [
    (0.8, "This is a mostly correct solution."),
    (0.5, "This is a partially correct solution."),
    (0.2, "This is a mostly incorrect solution."),
    (0.0, "This is an incorrect solution."),
]


def score_to_tag(score: float | None) -> str:
    if score is None:
        return "This is an unscored solution."
    if score >= 1.0:
        return "This is a correct solution."
    for threshold, tag in SCORE_TAGS:
        if score >= threshold:
            return tag
    return "This is an incorrect solution."


def row_to_doc(row: dict) -> list[dict]:
    prompt = row.get("prompt", "")
    response = row.get("llm_response", "")
    if not prompt or not response:
        counters.increment("synthetic1/dropped")
        return []

    response = strip_think_tags(response)
    if not response.strip():
        counters.increment("synthetic1/dropped")
        return []

    tag = score_to_tag(row.get("score"))
    text = f"{prompt}\n\n{tag}\n\n{response}"

    counters.increment("synthetic1/kept")
    return [
        {
            "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "text": text,
            "source": "PrimeIntellect/SYNTHETIC-1",
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="synthetic1-transform", resources=ResourceConfig(cpu=1, ram="4g"))
    list(ctx.execute(pipeline))


def download_synthetic1_step() -> StepSpec:
    """Download and transform PrimeIntellect/SYNTHETIC-1 into JSONL documents."""
    dl = download_hf_step(
        "raw/synthetic-1",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
    )

    return StepSpec(
        name="processed/synthetic-1",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )


def synthetic1_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for synthetic-1."""
    processed = download_synthetic1_step()
    return (
        processed,
        normalize_step(name="normalized/synthetic-1", download=processed),
    )
