# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.executor import unwrap_versioned_value
from marin.transform.conversation.adapters import InputDatasetFormat
from marin.transform.conversation.transform_conversation import transform_row

from experiments.posttrain.instruction_datasets import (
    ACE_REASON_1_1_SFT_HF_ID,
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

ACE_REASON_MATH_SAMPLE = {
    "category": "math",
    "source": "OpenMathReasoning",
    "input": "Find all prime numbers p such that p^3 + 1 = k^2.",
    "output": "<think>Factor p^3 + 1 and compare factors.</think>\nThe only prime is p = 2.",
}

ACE_REASON_CODE_SAMPLE = {
    "category": "code",
    "source": "OpenCodeReasoning",
    "input": "Write a function to detect a directed cycle.",
    "output": "<think>Use DFS colors.</think>\n```python\ndef has_cycle(graph): ...\n```",
}

ACE_REASON_MATH_ASSISTANT = "\n".join(
    [
        "<|start_think|>Factor p^3 + 1 and compare factors.<|end_think|>",
        "The only prime is p = 2.",
    ]
)

ACE_REASON_CODE_ASSISTANT = "\n".join(
    [
        "<|start_think|>Use DFS colors.<|end_think|>",
        "```python\ndef has_cycle(graph): ...\n```",
    ]
)


def _assert_acereason_transform_result(result, source_row, expected_assistant):
    assert result is not None
    assert result.source == ACE_REASON_1_1_SFT_HF_ID
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert result.messages[0].content == source_row["input"]
    assert result.messages[1].content == expected_assistant
    assert result.metadata == {
        "category": source_row["category"],
        "source": source_row["source"],
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


def test_acereason_sft_full_view_transforms_math_and_code_rows():
    step = get_instruction_dataset(ACE_REASON_1_1_SFT_HF_ID)
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)

    math_result = transform_row(ACE_REASON_MATH_SAMPLE, cfg, adapter)
    code_result = transform_row(ACE_REASON_CODE_SAMPLE, cfg, adapter)

    _assert_acereason_transform_result(math_result, ACE_REASON_MATH_SAMPLE, ACE_REASON_MATH_ASSISTANT)
    _assert_acereason_transform_result(code_result, ACE_REASON_CODE_SAMPLE, ACE_REASON_CODE_ASSISTANT)


def test_acereason_sft_math_view_transforms_only_math_rows():
    step = get_instruction_dataset(f"{ACE_REASON_1_1_SFT_HF_ID}/math")
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)

    code_row_with_math_source = {**ACE_REASON_CODE_SAMPLE, "source": ACE_REASON_MATH_SAMPLE["source"]}
    result = transform_row(ACE_REASON_MATH_SAMPLE, cfg, adapter)
    skipped = transform_row(code_row_with_math_source, cfg, adapter)

    _assert_acereason_transform_result(result, ACE_REASON_MATH_SAMPLE, ACE_REASON_MATH_ASSISTANT)
    assert skipped is None


def test_acereason_sft_code_view_transforms_only_code_rows():
    step = get_instruction_dataset(f"{ACE_REASON_1_1_SFT_HF_ID}/code")
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)

    math_row_with_code_source = {**ACE_REASON_MATH_SAMPLE, "source": ACE_REASON_CODE_SAMPLE["source"]}
    result = transform_row(ACE_REASON_CODE_SAMPLE, cfg, adapter)
    skipped = transform_row(math_row_with_code_source, cfg, adapter)

    _assert_acereason_transform_result(result, ACE_REASON_CODE_SAMPLE, ACE_REASON_CODE_ASSISTANT)
    assert skipped is None
