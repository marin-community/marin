# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from marin.execution.executor import unwrap_versioned_value
from marin.transform.conversation.adapters import InputDatasetFormat
from marin.transform.conversation.transform_conversation import transform_row

from experiments.posttrain.instruction_datasets import (
    FINEPROOFS_SFT_METADATA_COLUMNS,
    FINEPROOFS_SFT_REVISION,
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    NEMOTRON_OPENCODE_V1_SPLITS,
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

NEMOTRON_CUDA_SAMPLE = {
    "uuid": "cuda-row-1",
    "license": "cc-by-4.0",
    "used_in": ["cuda_sft"],
    "metadata": {"topic": "cuda"},
    "tools": [{"type": "function", "function": {"name": "bash"}}],
    "messages": [
        {"role": "user", "content": "Compile this CUDA kernel."},
        {
            "role": "assistant",
            "content": "I'll compile it.",
            "reasoning_content": "Compilation output is needed.",
            "tool_calls": json.dumps(
                [
                    {
                        "id": "call_cuda",
                        "type": "function",
                        "function": {"name": "bash", "arguments": json.dumps({"cmd": "nvcc kernel.cu"})},
                    }
                ]
            ),
        },
    ],
}

NEMOTRON_SWE_V3_SAMPLE = {
    "uuid": "swe-v3-row-1",
    "license": "cc-by-4.0",
    "messages": [
        {"role": "system", "content": "You are working inside a repository."},
        {"role": "user", "content": "Fix the failing issue."},
        {
            "role": "assistant",
            "content": "I'll inspect the tests.",
            "reasoning_content": "Need to inspect the failing test output.",
            "tool_calls": json.dumps(
                [
                    {
                        "id": "call_swe",
                        "type": "function",
                        "function": {"name": "bash", "arguments": json.dumps({"cmd": "pytest -q"})},
                    }
                ]
            ),
        },
        {"role": "tool", "content": "AssertionError", "tool_call_id": "call_swe"},
    ],
}

NEMOTRON_CASCADE_SAMPLE = {
    "category": "swe",
    "source": "cascade",
    "generator": "nemotron",
    "thinking": True,
    "messages": [
        {"role": "user", "content": "Fix the failing test."},
        {"role": "assistant", "content": "The issue is an off-by-one error."},
    ],
}

NEMOTRON_OPENCODE_SAMPLE = {
    "question_category": "repo_debug",
    "complexity_level": "hard",
    "uuid": "opencode-row-1",
    "enabled_tools": ["bash"],
    "skills_path": "/skills/bash.md",
    "hf_split": "bash_only_tool",
    "question": "Why does the test fail?",
    "agent_prompt": "You are a coding assistant.",
    "metadata": {"source": "opencode"},
    "tools": [{"type": "function", "function": {"name": "bash"}}],
    "messages": [
        {"role": "user", "content": "Why does the test fail?"},
        {
            "role": "assistant",
            "content": "I'll run the test.",
            "tool_calls": [
                {
                    "id": "call_test",
                    "type": "function",
                    "function": {"name": "bash", "arguments": json.dumps({"cmd": "pytest"})},
                }
            ],
        },
        {"role": "tool", "content": "AssertionError", "tool_call_id": "call_test"},
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


def test_nemotron_cuda_step_transforms_rich_tool_rows():
    step = get_instruction_dataset("nvidia/Nemotron-SFT-CUDA-v1")
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)

    result = transform_row(NEMOTRON_CUDA_SAMPLE, cfg, adapter)

    assert result is not None
    output = result.model_dump()
    assert result.source == "nvidia/Nemotron-SFT-CUDA-v1"
    assert output["metadata"] == {
        "uuid": "cuda-row-1",
        "license": "cc-by-4.0",
        "used_in": ["cuda_sft"],
    }
    assert output["tools"] == NEMOTRON_CUDA_SAMPLE["tools"]
    assert output["messages"][1]["reasoning_content"] == "Compilation output is needed."
    assert output["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"cmd": "nvcc kernel.cu"}


def test_nemotron_swe_v3_step_transforms_rich_tool_rows():
    step = get_instruction_dataset("nvidia/Nemotron-SFT-SWE-v3")
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)

    result = transform_row(NEMOTRON_SWE_V3_SAMPLE, cfg, adapter)

    assert result is not None
    output = result.model_dump()
    assert result.source == "nvidia/Nemotron-SFT-SWE-v3"
    assert output["metadata"] == {
        "uuid": "swe-v3-row-1",
        "license": "cc-by-4.0",
    }
    assert [message["role"] for message in output["messages"]] == ["system", "user", "assistant", "tool"]
    assert output["messages"][2]["reasoning_content"] == "Need to inspect the failing test output."
    assert output["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"cmd": "pytest -q"}
    assert output["messages"][3]["tool_call_id"] == "call_swe"


def test_nemotron_cascade_step_transforms_plain_chat_rows():
    step = get_instruction_dataset("nvidia/Nemotron-Cascade-SFT-SWE")
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)

    result = transform_row(NEMOTRON_CASCADE_SAMPLE, cfg, adapter)

    assert result is not None
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert result.messages[1].content == "The issue is an off-by-one error."
    assert result.metadata == {
        "category": "swe",
        "source": "cascade",
        "generator": "nemotron",
        "thinking": True,
    }


def test_nemotron_opencode_split_steps_transform_rows_with_top_level_context():
    for split_name in NEMOTRON_OPENCODE_V1_SPLITS:
        step = get_instruction_dataset(f"nvidia/Nemotron-SFT-OpenCode-v1/{split_name}")
        cfg = step.config
        adapter = unwrap_versioned_value(cfg.adapter)
        row = {**NEMOTRON_OPENCODE_SAMPLE, "hf_split": split_name}

        result = transform_row(row, cfg, adapter)

        assert result is not None
        output = result.model_dump()
        assert output["metadata"] == {
            "question_category": "repo_debug",
            "complexity_level": "hard",
            "uuid": "opencode-row-1",
            "enabled_tools": ["bash"],
            "skills_path": "/skills/bash.md",
            "hf_split": split_name,
        }
        assert output["agent_prompt"] == "You are a coding assistant."
        assert output["source_metadata"] == {"source": "opencode"}
        assert output["tools"] == NEMOTRON_OPENCODE_SAMPLE["tools"]
        assert output["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"cmd": "pytest"}
        assert output["messages"][2]["tool_call_id"] == "call_test"
