# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
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
SYNTHETIC2_SFT_VERIFIED_OUTPUT_PATH = (
    f"documents/PrimeIntellect--SYNTHETIC-2-SFT-verified-{SYNTHETIC2_SFT_VERIFIED_REVISION}-409fa9"
)

NEMOTRON_SFT_SCIENCE_V2_TOOL = {
    "type": "function",
    "function": {
        "name": "evaluate_code",
        "description": "Evaluates Python code.",
        "parameters": {"type": "object", "properties": {"source_code": {"type": "string"}}},
    },
}

NEMOTRON_SFT_SCIENCE_V2_SAMPLE = {
    "uuid": "science-v2-row-1",
    "license": "CC BY-SA 4.0",
    "used_in": ["nano_v3"],
    "metadata": {
        "topic": "Physics",
        "subtopic": "Classical Mechanics",
        "question_format": "MCQ",
        "generation_model": "DeepSeek-V3.2",
    },
    "messages": [
        {
            "role": "user",
            "content": "What is 2 + 2? Use the available tool and select the correct option.",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "index": -1,
                    "function": {"arguments": '{"source_code": "print(2 + 2)"}', "name": "evaluate_code"},
                    "id": "call_calc",
                    "type": "function",
                }
            ],
            "function_call": None,
            "reasoning_content": "I should compute the value exactly before selecting an option.",
        },
        {
            "role": "tool",
            "name": "evaluate_code",
            "tool_call_id": "call_calc",
            "content": "4",
        },
        {
            "role": "assistant",
            "content": "The correct option is 4.",
            "tool_calls": None,
            "function_call": None,
            "reasoning_content": "The tool returned 4, so the arithmetic result is 4.",
        },
    ],
    "tools": [NEMOTRON_SFT_SCIENCE_V2_TOOL],
}

EXPECTED_NEMOTRON_SFT_SCIENCE_V2_HF_ID = "nvidia/Nemotron-SFT-Science-v2"
EXPECTED_NEMOTRON_SFT_SCIENCE_V2_REVISION = "6536a5021222a94126968e8c92f29ee47fc8a7df"
EXPECTED_NEMOTRON_SFT_SCIENCE_V2_SUBSETS = ("rqa", "so", "syn_mcq", "vendor")
EXPECTED_NEMOTRON_SFT_SCIENCE_V2_METADATA_COLUMNS = ["uuid", "license", "used_in", "metadata", "tools"]
EXPECTED_NEMOTRON_SFT_SCIENCE_V2_MESSAGE_PASSTHROUGH_KEYS = (
    "reasoning_content",
    "tool_calls",
    "function_call",
    "tool_call_id",
    "name",
)

EXPECTED_OPENSCIENCE_REASONING_2_HF_ID = "nvidia/OpenScienceReasoning-2"
EXPECTED_OPENSCIENCE_REASONING_2_REVISION = "174b02c9cdf231f220765b2a1d5ece4550921894"
EXPECTED_OPENSCIENCE_REASONING_2_METADATA_COLUMNS = ["expected_answer"]

OPENSCIENCE_REASONING_2_SAMPLE = {
    "expected_answer": "C",
    "input": "Choose the answer. What does 2 + 2 equal?\n\nA: 3\nB: 5\nC: 4",
    "output": "<think>\nAdd the two integers.\n</think>\nThe answer is \\boxed{C}.",
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


def test_empty_message_passthrough_keys_preserve_existing_sft_output_path():
    step = get_instruction_dataset(SYNTHETIC2_SFT_VERIFIED_HF_ID)

    assert step.override_output_path == SYNTHETIC2_SFT_VERIFIED_OUTPUT_PATH


@pytest.mark.parametrize("subset", EXPECTED_NEMOTRON_SFT_SCIENCE_V2_SUBSETS)
def test_nemotron_sft_science_v2_lanes_are_registered(subset):
    dataset_key = f"{EXPECTED_NEMOTRON_SFT_SCIENCE_V2_HF_ID}/{subset}"
    step = get_instruction_dataset(dataset_key)
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)

    assert step.name == f"documents/{dataset_key}"
    assert step.override_output_path is not None
    assert step.override_output_path.startswith(
        f"documents/nvidia--Nemotron-SFT-Science-v2--{subset}-{EXPECTED_NEMOTRON_SFT_SCIENCE_V2_REVISION}-"
    )
    assert unwrap_versioned_value(cfg.source) == EXPECTED_NEMOTRON_SFT_SCIENCE_V2_HF_ID
    assert unwrap_versioned_value(cfg.revision) == EXPECTED_NEMOTRON_SFT_SCIENCE_V2_REVISION
    assert unwrap_versioned_value(cfg.subsets) == [subset]
    assert unwrap_versioned_value(cfg.splits) == ["train"]
    assert unwrap_versioned_value(cfg.metadata_columns) == EXPECTED_NEMOTRON_SFT_SCIENCE_V2_METADATA_COLUMNS
    assert adapter.dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN
    assert adapter.message_passthrough_keys == EXPECTED_NEMOTRON_SFT_SCIENCE_V2_MESSAGE_PASSTHROUGH_KEYS


def test_nemotron_sft_science_v2_tool_trace_preserves_training_fields():
    step = get_instruction_dataset(f"{EXPECTED_NEMOTRON_SFT_SCIENCE_V2_HF_ID}/rqa")
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)

    result = transform_row(NEMOTRON_SFT_SCIENCE_V2_SAMPLE, cfg, adapter)

    assert result is not None
    assert result.source == EXPECTED_NEMOTRON_SFT_SCIENCE_V2_HF_ID
    assert result.messages[0].model_dump(exclude_none=True) == {
        "role": "user",
        "content": "What is 2 + 2? Use the available tool and select the correct option.",
    }
    assert result.messages[1].model_dump(exclude_none=True) == {
        "role": "assistant",
        "tool_calls": [
            {
                "index": -1,
                "function": {"arguments": {"source_code": "print(2 + 2)"}, "name": "evaluate_code"},
                "id": "call_calc",
                "type": "function",
            }
        ],
        "reasoning_content": "I should compute the value exactly before selecting an option.",
    }
    assert result.messages[2].model_dump(exclude_none=True) == {
        "role": "tool",
        "content": "4",
        "name": "evaluate_code",
        "tool_call_id": "call_calc",
    }
    assert result.messages[3].model_dump(exclude_none=True) == {
        "role": "assistant",
        "content": "The correct option is 4.",
        "reasoning_content": "The tool returned 4, so the arithmetic result is 4.",
    }
    assert result.metadata == {
        "uuid": "science-v2-row-1",
        "license": "CC BY-SA 4.0",
        "used_in": ["nano_v3"],
        "metadata": {
            "topic": "Physics",
            "subtopic": "Classical Mechanics",
            "question_format": "MCQ",
            "generation_model": "DeepSeek-V3.2",
        },
        "tools": [NEMOTRON_SFT_SCIENCE_V2_TOOL],
    }
    assert result.model_dump()["chat_template_kwargs"] == {"tools": [NEMOTRON_SFT_SCIENCE_V2_TOOL]}


def test_nemotron_sft_science_v2_string_tools_column_stays_metadata_only():
    step = get_instruction_dataset(f"{EXPECTED_NEMOTRON_SFT_SCIENCE_V2_HF_ID}/vendor")
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)
    vendor_sample = {**NEMOTRON_SFT_SCIENCE_V2_SAMPLE, "tools": ""}

    result = transform_row(vendor_sample, cfg, adapter)

    assert result is not None
    assert result.metadata["tools"] == ""
    assert "chat_template_kwargs" not in result.model_dump()


def test_nemotron_science_v1_lanes_are_replaced_by_science_v2():
    assert all(
        not dataset_key.startswith("nvidia/Nemotron-Science-v1/") for dataset_key in INSTRUCTION_DATASET_NAME_TO_CONFIG
    )


def test_openscience_reasoning_2_step_transforms_instruction_response_rows():
    step = get_instruction_dataset(EXPECTED_OPENSCIENCE_REASONING_2_HF_ID)
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)

    assert step.name == f"documents/{EXPECTED_OPENSCIENCE_REASONING_2_HF_ID}"
    assert step.override_output_path is not None
    assert step.override_output_path.startswith(
        f"documents/nvidia--OpenScienceReasoning-2-{EXPECTED_OPENSCIENCE_REASONING_2_REVISION}-"
    )
    assert unwrap_versioned_value(cfg.source) == EXPECTED_OPENSCIENCE_REASONING_2_HF_ID
    assert unwrap_versioned_value(cfg.revision) == EXPECTED_OPENSCIENCE_REASONING_2_REVISION
    assert unwrap_versioned_value(cfg.subsets) == ["default"]
    assert unwrap_versioned_value(cfg.splits) == ["train"]
    assert unwrap_versioned_value(cfg.metadata_columns) == EXPECTED_OPENSCIENCE_REASONING_2_METADATA_COLUMNS
    assert adapter.dataset_format == InputDatasetFormat.INSTRUCTION_RESPONSE
    assert adapter.instruction_column == "input"
    assert adapter.response_column == "output"

    result = transform_row(OPENSCIENCE_REASONING_2_SAMPLE, cfg, adapter)

    assert result is not None
    assert result.source == EXPECTED_OPENSCIENCE_REASONING_2_HF_ID
    assert result.messages[0].model_dump(exclude_none=True) == {
        "role": "user",
        "content": "Choose the answer. What does 2 + 2 equal?\n\nA: 3\nB: 5\nC: 4",
    }
    assert result.messages[1].model_dump(exclude_none=True) == {
        "role": "assistant",
        "content": "<|start_think|>\nAdd the two integers.\n<|end_think|>\nThe answer is \\boxed{C}.",
    }
    assert result.metadata == {"expected_answer": "C"}
