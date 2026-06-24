# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""facebook/principia-collection dataset download and transform.

GPT-OSS-generated math problems with answers. Each row has a problem statement,
answer, topic, and answer type. We render these into a single document.
"""

from zephyr import counters

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import run_document_transform, text_document
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "facebook/principia-collection"
HF_REVISION = "f4413ee"


def row_to_doc(row: dict) -> list[dict]:
    problem = row.get("problem_statement", "")
    answer = row.get("answer", "")
    if not problem or not answer:
        counters.increment("principia/dropped")
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

    counters.increment("principia/kept")
    return [text_document(text, "facebook/principia-collection")]


def transform(input_path: str, output_path: str) -> None:
    run_document_transform(
        input_path=input_path,
        output_path=output_path,
        row_to_doc=row_to_doc,
        name="principia-transform",
        ram="4g",
    )


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
