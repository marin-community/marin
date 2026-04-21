# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.transform.conversation.adapters import InputDatasetFormat

from experiments.posttrain.instruction_datasets import INSTRUCTION_DATASET_NAME_TO_CONFIG


def test_numinamath_cot_instruction_dataset_config():
    config = INSTRUCTION_DATASET_NAME_TO_CONFIG["AI-MO/NuminaMath-CoT"]

    assert config.hf_dataset_id == "AI-MO/NuminaMath-CoT"
    assert config.revision == "9d8d210"
    assert config.adapter.dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN
    assert config.metadata_columns == ["source"]
    assert config.splits == ["train"]


def test_numinamath_tir_instruction_dataset_config():
    config = INSTRUCTION_DATASET_NAME_TO_CONFIG["AI-MO/NuminaMath-TIR"]

    assert config.hf_dataset_id == "AI-MO/NuminaMath-TIR"
    assert config.revision == "77a91d7"
    assert config.adapter.dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN
    assert config.metadata_columns == []
    assert config.splits == ["train"]
