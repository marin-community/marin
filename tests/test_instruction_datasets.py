# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import datasets
import pytest
from datasets import load_dataset
from marin.execution.executor import unwrap_versioned_value
from marin.transform.conversation.adapters import InputDatasetFormat
from marin.transform.conversation.transform_conversation import _load_dataset_features_from_spec, transform_row

from experiments.posttrain.instruction_datasets import (
    FINEPROOFS_SFT_METADATA_COLUMNS,
    FINEPROOFS_SFT_REVISION,
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    NEMOTRON_SFT_SAFETY_V2_HF_ID,
    NEMOTRON_SFT_SAFETY_V2_LOAD_DATASET_FEATURES,
    NEMOTRON_SFT_SAFETY_V2_METADATA_COLUMNS,
    NEMOTRON_SFT_SAFETY_V2_REVISION,
    SYNTHETIC2_SFT_VERIFIED_HF_ID,
    SYNTHETIC2_SFT_VERIFIED_METADATA_COLUMNS,
    SYNTHETIC2_SFT_VERIFIED_REVISION,
    get_instruction_dataset,
)

SYNTHETIC2_SFT_VERIFIED_SAMPLE = {
    "problem_id": "prime_rl_code_21747",
    "task_type": "prime_rl_code",
    "reward": 1.0,
    "messages": [
        {"role": "user", "content": "Write Python code to count the intersection of two sets."},
        {"role": "assistant", "content": "<think>Use set intersection.</think>\n```python\nprint(len(a & b))\n```"},
    ],
}

NEMOTRON_SFT_SAFETY_V2_SAMPLE = {
    "uuid": "safety-row-1",
    "used_in": ["super_v3", "ultra_v3"],
    "prompt_source": "Aegis v2",
    "response_policy": "refusal_with_explanation_redirection",
    "translation_languages": None,
    "metadata": {
        "uuid": "safety-row-1",
        "source": "nemotron_safety_sft_blend_v0.3.0",
        "sdg_model": "nemotron-nano-v2-9b",
        "source_l1": "Aegis v2",
        "source_id": "source-1",
        "source_l2": None,
        "categories": "Needs Caution",
        "subcategories": None,
        "adversarial_persona": None,
        "prompt_label": "unsafe",
        "response_policy": "refusal_with_explanation_redirection",
        "date_created": "20251217",
    },
    "messages": [
        {"role": "user", "content": "Can you help me be mean to a classmate?"},
        {
            "role": "assistant",
            "content": "I can't help with harassing someone. I can help you resolve the conflict respectfully.",
            "reasoning_content": "The request asks for interpersonal harm, so the response should refuse and redirect.",
        },
    ],
}

NEMOTRON_SFT_SAFETY_V2_TRANSLATED_SAMPLE = {
    **NEMOTRON_SFT_SAFETY_V2_SAMPLE,
    "uuid": "safety-row-1-fr",
    "translation_languages": "en_fr",
    "metadata": {
        **NEMOTRON_SFT_SAFETY_V2_SAMPLE["metadata"],
        "title": "Translated safety sample",
        "authors": ["NVIDIA"],
        "publication_year": 2026,
        "prompt": "Can you help me be mean to a classmate?",
        "generated_reasoning": "The request asks for interpersonal harm, so the response should refuse and redirect.",
        "generated_response": "I can't help with harassing someone.",
        "type": "translated",
        "id": "translated-1",
        "quote": "translation smoke",
        "generator": "riva-translate",
        "source_l1_id": "source-l1-1",
    },
}


def test_fineproofs_sft_datasets_are_registered():
    raw_dataset = INSTRUCTION_DATASET_NAME_TO_CONFIG["lm-provers/FineProofs-SFT"]
    proof_only_dataset = INSTRUCTION_DATASET_NAME_TO_CONFIG["lm-provers/FineProofs-SFT/proof-only"]

    assert raw_dataset.hf_dataset_id == "lm-provers/FineProofs-SFT"
    assert raw_dataset.revision == FINEPROOFS_SFT_REVISION
    assert raw_dataset.subsets == ["default"]
    assert raw_dataset.splits == ["train"]
    assert raw_dataset.metadata_columns == FINEPROOFS_SFT_METADATA_COLUMNS
    assert raw_dataset.adapter.dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN

    assert proof_only_dataset.hf_dataset_id == "lm-provers/FineProofs-SFT"
    assert proof_only_dataset.revision == FINEPROOFS_SFT_REVISION
    assert proof_only_dataset.subsets == ["default"]
    assert proof_only_dataset.splits == ["train"]
    assert proof_only_dataset.metadata_columns == FINEPROOFS_SFT_METADATA_COLUMNS
    assert proof_only_dataset.adapter.dataset_format == InputDatasetFormat.INSTRUCTION_RESPONSE
    assert proof_only_dataset.adapter.instruction_column == "problem"
    assert proof_only_dataset.adapter.response_column == "proof"


def test_get_instruction_dataset_preserves_fineproofs_config():
    raw_step = get_instruction_dataset("lm-provers/FineProofs-SFT")
    raw_cfg = raw_step.config

    assert unwrap_versioned_value(raw_cfg.source) == "lm-provers/FineProofs-SFT"
    assert unwrap_versioned_value(raw_cfg.revision) == FINEPROOFS_SFT_REVISION
    assert unwrap_versioned_value(raw_cfg.subsets) == ["default"]
    assert unwrap_versioned_value(raw_cfg.splits) == ["train"]
    assert unwrap_versioned_value(raw_cfg.metadata_columns) == FINEPROOFS_SFT_METADATA_COLUMNS
    assert unwrap_versioned_value(raw_cfg.adapter).dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN
    assert raw_step.name == "documents/lm-provers/FineProofs-SFT"

    proof_only_step = get_instruction_dataset("lm-provers/FineProofs-SFT/proof-only")
    proof_only_cfg = proof_only_step.config

    assert unwrap_versioned_value(proof_only_cfg.source) == "lm-provers/FineProofs-SFT"
    assert unwrap_versioned_value(proof_only_cfg.revision) == FINEPROOFS_SFT_REVISION
    assert unwrap_versioned_value(proof_only_cfg.subsets) == ["default"]
    assert unwrap_versioned_value(proof_only_cfg.splits) == ["train"]
    assert unwrap_versioned_value(proof_only_cfg.metadata_columns) == FINEPROOFS_SFT_METADATA_COLUMNS
    assert unwrap_versioned_value(proof_only_cfg.adapter).dataset_format == InputDatasetFormat.INSTRUCTION_RESPONSE
    assert proof_only_step.name == "documents/lm-provers/FineProofs-SFT/proof-only"


def test_synthetic2_sft_verified_step_transforms_chat_rows():
    step = get_instruction_dataset(SYNTHETIC2_SFT_VERIFIED_HF_ID)
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)

    assert step.name == "documents/PrimeIntellect/SYNTHETIC-2-SFT-verified"
    assert step.override_output_path is not None
    assert step.override_output_path.startswith(
        f"documents/PrimeIntellect--SYNTHETIC-2-SFT-verified-{SYNTHETIC2_SFT_VERIFIED_REVISION}-"
    )
    assert unwrap_versioned_value(cfg.source) == SYNTHETIC2_SFT_VERIFIED_HF_ID
    assert unwrap_versioned_value(cfg.revision) == SYNTHETIC2_SFT_VERIFIED_REVISION
    assert unwrap_versioned_value(cfg.subsets) == ["default"]
    assert unwrap_versioned_value(cfg.splits) == ["train"]
    assert unwrap_versioned_value(cfg.metadata_columns) == SYNTHETIC2_SFT_VERIFIED_METADATA_COLUMNS
    assert adapter.dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN

    result = transform_row(SYNTHETIC2_SFT_VERIFIED_SAMPLE, cfg, adapter)

    assert result is not None
    assert result.source == SYNTHETIC2_SFT_VERIFIED_HF_ID
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert result.messages[0].content == "Write Python code to count the intersection of two sets."
    assert result.messages[1].content == "\n".join(
        [
            "<|start_think|>Use set intersection.<|end_think|>",
            "```python\nprint(len(a & b))\n```",
        ]
    )
    assert result.metadata == {
        "problem_id": "prime_rl_code_21747",
        "task_type": "prime_rl_code",
        "reward": 1.0,
    }


def test_nemotron_sft_safety_v2_load_features_fix_inferred_schema_failure(tmp_path):
    data_path = tmp_path / "nemotron_sft_safety_v2.jsonl"
    with data_path.open("w") as f:
        for row in [NEMOTRON_SFT_SAFETY_V2_SAMPLE, NEMOTRON_SFT_SAFETY_V2_TRANSLATED_SAMPLE]:
            f.write(json.dumps(row) + "\n")

    inferred_from_early_rows = datasets.Features(
        {
            "uuid": datasets.Value("string"),
            "messages": datasets.List(
                {
                    "role": datasets.Value("string"),
                    "content": datasets.Value("string"),
                    "reasoning_content": datasets.Value("string"),
                }
            ),
            "used_in": datasets.List(datasets.Value("string")),
            "prompt_source": datasets.Value("string"),
            "response_policy": datasets.Value("string"),
            "translation_languages": datasets.Value("null"),
            "metadata": {
                "uuid": datasets.Value("string"),
                "source": datasets.Value("string"),
                "sdg_model": datasets.Value("string"),
                "source_l1": datasets.Value("string"),
                "source_id": datasets.Value("string"),
                "source_l2": datasets.Value("null"),
                "categories": datasets.Value("string"),
                "subcategories": datasets.Value("null"),
                "adversarial_persona": datasets.Value("null"),
                "prompt_label": datasets.Value("string"),
                "response_policy": datasets.Value("string"),
                "date_created": datasets.Value("string"),
            },
        }
    )
    with pytest.raises(TypeError):
        list(
            load_dataset(
                "json",
                data_files=str(data_path),
                split="train",
                streaming=True,
                features=inferred_from_early_rows,
            )
        )

    features = _load_dataset_features_from_spec(NEMOTRON_SFT_SAFETY_V2_LOAD_DATASET_FEATURES)
    streamed_rows = list(
        load_dataset("json", data_files=str(data_path), split="train", streaming=True, features=features)
    )

    assert [row["translation_languages"] for row in streamed_rows] == [None, "en_fr"]
    assert streamed_rows[1]["metadata"]["generated_reasoning"] == (
        "The request asks for interpersonal harm, so the response should refuse and redirect."
    )


def test_nemotron_sft_safety_v2_step_transforms_chat_rows_without_reasoning_content():
    step = get_instruction_dataset(NEMOTRON_SFT_SAFETY_V2_HF_ID)
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)

    assert unwrap_versioned_value(cfg.source) == NEMOTRON_SFT_SAFETY_V2_HF_ID
    assert unwrap_versioned_value(cfg.revision) == NEMOTRON_SFT_SAFETY_V2_REVISION
    assert unwrap_versioned_value(cfg.subsets) == ["default"]
    assert unwrap_versioned_value(cfg.splits) == ["train"]
    assert unwrap_versioned_value(cfg.metadata_columns) == NEMOTRON_SFT_SAFETY_V2_METADATA_COLUMNS
    assert unwrap_versioned_value(cfg.load_dataset_features) == NEMOTRON_SFT_SAFETY_V2_LOAD_DATASET_FEATURES
    assert adapter.dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN

    result = transform_row(NEMOTRON_SFT_SAFETY_V2_TRANSLATED_SAMPLE, cfg, adapter)

    assert result is not None
    assert result.source == NEMOTRON_SFT_SAFETY_V2_HF_ID
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert result.messages[0].content == "Can you help me be mean to a classmate?"
    assert result.messages[1].content == (
        "I can't help with harassing someone. I can help you resolve the conflict respectfully."
    )
    assert "interpersonal harm" not in result.messages[1].content
    assert result.metadata == {
        "uuid": "safety-row-1-fr",
        "used_in": ["super_v3", "ultra_v3"],
        "prompt_source": "Aegis v2",
        "response_policy": "refusal_with_explanation_redirection",
        "translation_languages": "en_fr",
        "metadata": NEMOTRON_SFT_SAFETY_V2_TRANSLATED_SAMPLE["metadata"],
    }
