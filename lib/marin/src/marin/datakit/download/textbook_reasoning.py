# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned MegaScience TextbookReasoning chat source."""

import hashlib

from fray.types import ResourceConfig
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.readers import load_parquet

from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_step
from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import checked_openai_chat_document
from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
)
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "MegaScience/TextbookReasoning"
HF_REVISION = "ca7ecbec76d01bff2e99f3dc17735b02f87d4e96"
MARIN_NAME = "megascience/textbook-reasoning"
ROUGH_TOKEN_COUNT_B = 0.3

TEXTBOOK_REASONING_MANIFEST = IngestionSourceManifest(
    dataset_key=HF_DATASET_ID,
    slice_key=MARIN_NAME,
    source_label=MARIN_NAME,
    source_urls=(f"https://huggingface.co/datasets/{HF_DATASET_ID}/tree/{HF_REVISION}",),
    source_license="CC BY-NC-SA 4.0",
    source_format="huggingface_parquet_question_answer_rows",
    surface_form="one_user_assistant_conversation_per_row",
    policy=IngestionPolicy(
        usage_policy=UsagePolicy.TRAINING_ALLOWED,
        use_policy=(
            "Noncommercial ShareAlike training approved; retain dataset attribution in the training record and "
            "distributed model documentation."
        ),
        contamination_risk="medium: questions are derived from university textbooks and were decontaminated upstream",
        provenance_notes="Approved for noncommercial, ShareAlike model training on 2026-09-20.",
        identity_treatment=IdentityTreatment.PRESERVE,
        secret_redaction=SecretRedaction.NONE,
    ),
    staging=StagingMetadata(
        transform_name="transform_textbook_reasoning_chat",
        split="train",
        metadata={
            "question_field": "question",
            "answer_field": "answer",
            "source_id": "sha256(question)",
        },
    ),
    rough_tokens_b=ROUGH_TOKEN_COUNT_B,
    source_metadata={"hf_revision": HF_REVISION, "rows": 651_840},
)


def row_to_chat_doc(row: dict) -> list[dict]:
    """Convert a source row into a two-turn Harmony conversation."""
    question = row.get("question") or ""
    answer = row.get("answer") or ""
    if not question.strip() or not answer.strip():
        return []
    source_id = hashlib.sha256(question.encode()).hexdigest()
    return checked_openai_chat_document(
        [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        HF_DATASET_ID,
        counter_prefix="textbook_reasoning/chat",
        source_id=source_id,
    )


def transform_chat(input_path: str, output_path: str) -> None:
    """Transform pinned parquet rows into normalized chat records."""
    pipeline = (
        Dataset.from_files(prefix_join(input_path, "data/train-*.parquet"))
        .flat_map(load_parquet)
        .flat_map(row_to_chat_doc)
        .write_parquet(
            prefix_join(output_path, "data-{shard:05d}-of-{total:05d}.parquet"),
            schema=CHAT_SCHEMA,
            skip_existing=True,
        )
    )
    ZephyrContext(
        name="textbook-reasoning-chat-transform",
        resources=ResourceConfig(cpu=1, ram="8g"),
    ).execute(pipeline)


def textbook_reasoning_chat_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the download, transform, and chat-normalization chain."""
    download = download_hf_step(
        "raw/textbook-reasoning",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=["data/train-*.parquet"],
    )
    processed = StepSpec(
        name="processed-chat/textbook-reasoning",
        deps=[download],
        fn=lambda output_path: transform_chat(download.output_path, output_path),
        hash_attrs={"manifest_content_fingerprint": TEXTBOOK_REASONING_MANIFEST.fingerprint()},
    )
    return processed, normalize_chat_step(
        name="normalized-chat/textbook-reasoning",
        download=processed,
    )
