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

OPEN_R1_VERIFIABLE_CODE_KEY = "open-r1/verifiable-coding-problems-python_decontaminated-tested"
OPEN_R1_VERIFIABLE_CODE_REVISION = "77ace849f7cc8f10d23945c748f9ff464427f83b"
OPEN_R1_OPENTHOUGHTS_CODE_KEY = "open-r1/OpenThoughts-114k-Code_decontaminated"
OPEN_R1_OPENTHOUGHTS_CODE_REVISION = "b90307fcdb68ae52d01f60cfd38d5f0b4852115a"
OPEN_R1_CODEFORCES_COTS_HF_ID = "open-r1/codeforces-cots"
OPEN_R1_CODEFORCES_COTS_REVISION = "39ac85c150806230473c70ad72c31f6232fe3f41"
OPEN_R1_CODEFORCES_COTS_VIEWS = [
    ("open-r1/codeforces-cots/solutions_decontaminated/finish_stop", "solutions_decontaminated"),
    ("open-r1/codeforces-cots/solutions_py_decontaminated/finish_stop", "solutions_py_decontaminated"),
]
OPEN_R1_IOI_COTS_HF_ID = "open-r1/ioi-cots"
OPEN_R1_IOI_COTS_REVISION = "d07e70f8ed8b8b0b1843046a2a140b9dac9e9741"
OPEN_R1_IOI_BASE_KEY = "open-r1/ioi-cots/ac-target-subtask-finish-stop"
OPEN_R1_IOI_STRICT_KEY = "open-r1/ioi-cots/all-subtasks-100-finish-stop"

OPEN_R1_VERIFIABLE_CODE_SAMPLE = {
    "source": "apps",
    "task_type": "prime_rl_code",
    "in_source_id": "apps-17",
    "problem": "Write a Python function that returns the square of an integer.",
    "gold_standard_solution": "def square(n: int) -> int:\n    return n * n",
    "problem_id": "apps-17",
    "test_reward": 1.0,
    "verification_info": {"private_tests": ["large verifier blob"]},
}

OPEN_R1_OPENTHOUGHTS_CODE_SAMPLE = {
    "problem": "Return the maximum element in a list.",
    "deepseek_reasoning": "We can scan once.",
    "deepseek_solution": "def solve(nums):\n    return max(nums)",
    "ground_truth_solution": "def solve(nums):\n    return max(nums)",
    "domain": "python",
    "source": "lcb",
    "test_cases": [{"input": "[1, 3, 2]", "output": "3"}],
    "starter_code": "",
    "messages": [
        {"role": "user", "content": "Return the maximum element in a list."},
        {"role": "assistant", "content": "<think>Scan once.</think>\ndef solve(nums):\n    return max(nums)"},
    ],
    "num_tokens": 128,
}

OPEN_R1_CODEFORCES_SAMPLE = {
    "id": "cf-1000-a",
    "contest_id": 1000,
    "contest_type": "cf",
    "contest_start_year": 2018,
    "index": "A",
    "title": "Codeforces Problem A",
    "problem_type": "programming",
    "finish_reason": "stop",
    "accepted_solutions": ["large solution blob"],
    "public_tests_ms": [{"input": "1", "output": "1"}],
    "messages": [
        {"role": "user", "content": "Solve Codeforces Problem A."},
        {"role": "assistant", "content": "<think>Derive the invariant.</think>\n```cpp\nint main(){}\n```"},
    ],
}

OPEN_R1_IOI_SAMPLE = {
    "year": 2024,
    "day": 1,
    "problem_name": "Nile",
    "problem_id": "2024-day1-nile",
    "target_subtask": "subtask_1",
    "prompt": "Implement the solution.",
    "generation": "We need a data structure.",
    "uuid": "ioi-row-1",
    "metadata": {"difficulty": "hard"},
    "finish_reason": "stop",
    "code": "#include <bits/stdc++.h>",
    "code_compiles": True,
    "target_subtask_score": 1.0,
    "target_subtask_status": "AC",
    "all_subtasks_points": 100.0,
    "all_subtasks_results": [{"status": "AC"}],
    "messages": [
        {"role": "user", "content": "Solve the IOI subtask."},
        {
            "role": "assistant",
            "content": "<think>Use the target subtask constraints.</think>" "\n```cpp\nint main(){}\n```",
        },
    ],
}


def transform_instruction_row(dataset_key: str, row: dict):
    step = get_instruction_dataset(dataset_key)
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)
    return step, cfg, adapter, transform_row(row, cfg, adapter)


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


def test_open_r1_verifiable_code_step_transforms_problem_solution_rows():
    step, cfg, adapter, result = transform_instruction_row(OPEN_R1_VERIFIABLE_CODE_KEY, OPEN_R1_VERIFIABLE_CODE_SAMPLE)

    assert step.name == f"documents/{OPEN_R1_VERIFIABLE_CODE_KEY}"
    assert unwrap_versioned_value(cfg.source) == OPEN_R1_VERIFIABLE_CODE_KEY
    assert unwrap_versioned_value(cfg.revision) == OPEN_R1_VERIFIABLE_CODE_REVISION
    assert unwrap_versioned_value(cfg.subsets) == ["default"]
    assert adapter.dataset_format == InputDatasetFormat.INSTRUCTION_RESPONSE
    assert adapter.instruction_column == "problem"
    assert adapter.response_column == "gold_standard_solution"

    assert result is not None
    assert result.source == OPEN_R1_VERIFIABLE_CODE_KEY
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert result.messages[0].content == OPEN_R1_VERIFIABLE_CODE_SAMPLE["problem"]
    assert result.messages[1].content == OPEN_R1_VERIFIABLE_CODE_SAMPLE["gold_standard_solution"]
    assert result.metadata == {
        "source": "apps",
        "task_type": "prime_rl_code",
        "in_source_id": "apps-17",
        "problem_id": "apps-17",
        "test_reward": 1.0,
    }


def test_open_r1_openthoughts_code_step_transforms_messages():
    step, cfg, adapter, result = transform_instruction_row(
        OPEN_R1_OPENTHOUGHTS_CODE_KEY,
        OPEN_R1_OPENTHOUGHTS_CODE_SAMPLE,
    )

    assert step.name == f"documents/{OPEN_R1_OPENTHOUGHTS_CODE_KEY}"
    assert unwrap_versioned_value(cfg.source) == OPEN_R1_OPENTHOUGHTS_CODE_KEY
    assert unwrap_versioned_value(cfg.revision) == OPEN_R1_OPENTHOUGHTS_CODE_REVISION
    assert unwrap_versioned_value(cfg.subsets) == ["default"]
    assert adapter.dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN

    assert result is not None
    assert result.source == OPEN_R1_OPENTHOUGHTS_CODE_KEY
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert result.messages[0].content == "Return the maximum element in a list."
    assert result.messages[1].content == "<|start_think|>Scan once.<|end_think|>\ndef solve(nums):\n    return max(nums)"
    assert result.metadata == {"domain": "python", "source": "lcb", "num_tokens": 128}


def test_open_r1_codeforces_views_transform_and_filter_finished_generations():
    for dataset_key, subset_name in OPEN_R1_CODEFORCES_COTS_VIEWS:
        step, cfg, adapter, result = transform_instruction_row(dataset_key, OPEN_R1_CODEFORCES_SAMPLE)

        assert step.name == f"documents/{dataset_key}"
        assert unwrap_versioned_value(cfg.source) == OPEN_R1_CODEFORCES_COTS_HF_ID
        assert unwrap_versioned_value(cfg.revision) == OPEN_R1_CODEFORCES_COTS_REVISION
        assert unwrap_versioned_value(cfg.subsets) == [subset_name]
        assert adapter.dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN

        assert result is not None
        assert result.source == OPEN_R1_CODEFORCES_COTS_HF_ID
        assert [message.role for message in result.messages] == ["user", "assistant"]
        assert (
            result.messages[1].content == "<|start_think|>Derive the invariant.<|end_think|>\n```cpp\nint main(){}\n```"
        )
        assert result.metadata == {
            "id": "cf-1000-a",
            "contest_id": 1000,
            "contest_type": "cf",
            "contest_start_year": 2018,
            "index": "A",
            "title": "Codeforces Problem A",
            "problem_type": "programming",
            "finish_reason": "stop",
        }

        truncated_row = {**OPEN_R1_CODEFORCES_SAMPLE, "finish_reason": "length"}
        assert transform_row(truncated_row, cfg, adapter) is None


def test_open_r1_ioi_base_view_filters_to_target_subtask_ac_rows():
    step, cfg, adapter, result = transform_instruction_row(OPEN_R1_IOI_BASE_KEY, OPEN_R1_IOI_SAMPLE)

    assert step.name == f"documents/{OPEN_R1_IOI_BASE_KEY}"
    assert unwrap_versioned_value(cfg.source) == OPEN_R1_IOI_COTS_HF_ID
    assert unwrap_versioned_value(cfg.revision) == OPEN_R1_IOI_COTS_REVISION
    assert unwrap_versioned_value(cfg.subsets) == ["default"]
    assert adapter.dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN

    assert result is not None
    assert result.source == OPEN_R1_IOI_COTS_HF_ID
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert result.messages[1].content == (
        "<|start_think|>Use the target subtask constraints.<|end_think|>\n```cpp\nint main(){}\n```"
    )
    assert result.metadata == {
        "year": 2024,
        "day": 1,
        "problem_name": "Nile",
        "problem_id": "2024-day1-nile",
        "target_subtask": "subtask_1",
        "uuid": "ioi-row-1",
        "finish_reason": "stop",
        "code_compiles": True,
        "target_subtask_score": 1.0,
        "target_subtask_status": "AC",
        "all_subtasks_points": 100.0,
    }

    for field_name, rejected_value in [
        ("finish_reason", "length"),
        ("code_compiles", False),
        ("target_subtask_status", "WA"),
        ("target_subtask_score", 0.5),
    ]:
        assert transform_row({**OPEN_R1_IOI_SAMPLE, field_name: rejected_value}, cfg, adapter) is None


def test_open_r1_ioi_strict_view_requires_full_points():
    step, cfg, adapter, result = transform_instruction_row(OPEN_R1_IOI_STRICT_KEY, OPEN_R1_IOI_SAMPLE)

    assert step.name == f"documents/{OPEN_R1_IOI_STRICT_KEY}"
    assert unwrap_versioned_value(cfg.source) == OPEN_R1_IOI_COTS_HF_ID
    assert unwrap_versioned_value(cfg.revision) == OPEN_R1_IOI_COTS_REVISION
    assert adapter.dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN
    assert result is not None

    partial_row = {**OPEN_R1_IOI_SAMPLE, "all_subtasks_points": 99.0}
    assert transform_row(partial_row, cfg, adapter) is None
