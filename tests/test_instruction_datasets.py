# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.executor import unwrap_versioned_value
from marin.transform.conversation.adapters import InputDatasetFormat
from marin.transform.conversation.transform_conversation import transform_row

from experiments.posttrain.instruction_datasets import (
    FINEPROOFS_SFT_METADATA_COLUMNS,
    FINEPROOFS_SFT_REVISION,
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
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
