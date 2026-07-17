# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""facebook/principia-collection dataset download and transform.

GPT-OSS-generated math problems with answers. Each row has a problem statement,
answer, topic, and answer type. We render these into a single document.
"""

from fray.types import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_parquet

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import text_document
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "facebook/principia-collection"
HF_REVISION = "f4413ee"


def row_to_doc(row: dict) -> list[dict]:
    problem = row.get("problem_statement", "")
    answer = row.get("answer", "")
    if not problem or not answer:
        counters.pipeline.update_counter("principia/dropped", 1)
        return []

    topic = row.get("topic", "")
    answer_type = row.get("answer_type", "")

    parts = []
    if topic:
        parts.append(f"Topic: {topic}")
    parts.append(problem)
    if answer_type:
        parts.append(f"Answer ({answer_type}): {answer}")
    else:
        parts.append(f"Answer: {answer}")

    text = "\n\n".join(parts)

    counters.pipeline.update_counter("principia/kept", 1)
    return [text_document(text, "facebook/principia-collection")]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(load_parquet)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="principia-transform", resources=ResourceConfig(cpu=1, ram="4g"))
    ctx.execute(pipeline)


def download_principia_step() -> StepSpec:
    """Download and transform facebook/principia-collection into JSONL documents."""
    dl = download_hf_step(
        "raw/principia-collection",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
    )

    return StepSpec(
        name="processed/principia-collection",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1"},
    )
