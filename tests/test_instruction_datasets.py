# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from marin.execution.executor import unwrap_versioned_value
from marin.transform.conversation.adapters import InputDatasetFormat
from marin.transform.conversation.transform_conversation import (
    RowFilter,
    RowFilterOperator,
    row_matches_filters,
    transform_row,
)

from experiments.posttrain.instruction_datasets import (
    FINEPROOFS_SFT_METADATA_COLUMNS,
    FINEPROOFS_SFT_REVISION,
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    SYNTHETIC2_SFT_VERIFIED_HF_ID,
    SYNTHETIC2_SFT_VERIFIED_METADATA_COLUMNS,
    SYNTHETIC2_SFT_VERIFIED_REVISION,
    InstructionDatasetConfig,
    get_instruction_dataset,
    structured_multi_turn_adapter,
    transform_dataset_step,
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

NEMOTRON_STRUCTURED_SAMPLE = {
    "uuid": "math-row-1",
    "expected_answer": "4",
    "problem": "Solve 2 + 2.",
    "original_expected_answer": "4",
    "changed_answer_to_majority": False,
    "data_source": "aops",
    "used_in": ["train"],
    "metadata": {
        "reason_high_with_tool": {"count": 1, "pass": 1, "accuracy": 1.0},
        "reason_medium_no_tool": {"count": 1, "pass": 1, "accuracy": 1.0},
    },
    "source": "AoPS",
    "dataset": "Nemotron-SFT-Math-v4",
    "subset": "tir",
    "license": "cc-by-4.0",
    "url": "https://example.com/problem",
    "user_name": "example-user",
    "user_url": "https://example.com/user",
    "messages": [
        {"role": "user", "content": "Solve 2 + 2."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Use Python for the arithmetic.",
            "tool_calls": [
                {
                    "id": "call_python",
                    "type": "function",
                    "function": {"name": "python", "arguments": '{"code": "2 + 2"}'},
                }
            ],
        },
        {"role": "tool", "name": "python", "tool_call_id": "call_python", "content": "4"},
        {"role": "assistant", "content": "\\boxed{4}", "reasoning_content": ""},
    ],
    "tools": [{"type": "function", "function": {"name": "python", "parameters": {}}}],
}

NEMOTRON_PROOFS_V2_SAMPLE = {
    "uuid": "proof-row-1",
    "problem": "Prove that sqrt(3) is irrational.",
    "used_in": ["ultra_v3"],
    "metadata": [],
    "source": "AoPS",
    "dataset": "Nemotron-Math-Proofs-v2",
    "subset": "proof",
    "license": "CC BY 4.0",
    "messages": [
        {"role": "user", "content": "Prove that sqrt(3) is irrational."},
        {
            "role": "assistant",
            "content": "## Solution\nAssume sqrt(3)=p/q in lowest terms and derive a contradiction.",
            "reasoning_content": "Use the standard prime divisibility argument.",
        },
    ],
    "tools": [],
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


def test_nemotron_sft_math_v4_views_filter_caps_and_transform_tool_rows():
    tir_step = get_instruction_dataset("nvidia/Nemotron-SFT-Math-v4/tir-pilot")
    tir_cfg = tir_step.config
    cot_cfg = get_instruction_dataset("nvidia/Nemotron-SFT-Math-v4/cot-pilot").config
    full_tir_cfg = get_instruction_dataset("nvidia/Nemotron-SFT-Math-v4/tir").config

    assert unwrap_versioned_value(tir_cfg.source) == "nvidia/Nemotron-SFT-Math-v4"
    assert unwrap_versioned_value(tir_cfg.revision) == "a94e56aeddcf6e75d28c8bd210f40fa62309288d"
    assert unwrap_versioned_value(tir_cfg.splits) == ["train"]
    assert row_matches_filters(NEMOTRON_STRUCTURED_SAMPLE, unwrap_versioned_value(tir_cfg.row_filters))
    assert not row_matches_filters(NEMOTRON_STRUCTURED_SAMPLE, unwrap_versioned_value(cot_cfg.row_filters))
    assert unwrap_versioned_value(tir_cfg.max_examples_per_split) == 100_000
    assert unwrap_versioned_value(full_tir_cfg.max_examples_per_split) is None

    result = transform_row(NEMOTRON_STRUCTURED_SAMPLE, tir_cfg, unwrap_versioned_value(tir_cfg.adapter))

    assert result is not None
    dumped = result.model_dump()
    assert dumped["tools"] == NEMOTRON_STRUCTURED_SAMPLE["tools"]
    assert dumped["messages"][1]["reasoning_content"] == "Use Python for the arithmetic."
    assert dumped["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"code": "2 + 2"}
    assert dumped["messages"][2]["role"] == "tool"
    assert dumped["messages"][2]["tool_call_id"] == "call_python"
    assert result.metadata == {
        "uuid": "math-row-1",
        "expected_answer": "4",
        "problem": "Solve 2 + 2.",
        "used_in": ["train"],
        "metadata": {
            "reason_high_with_tool": {"count": 1, "pass": 1, "accuracy": 1.0},
            "reason_medium_no_tool": {"count": 1, "pass": 1, "accuracy": 1.0},
        },
        "source": "AoPS",
        "dataset": "Nemotron-SFT-Math-v4",
        "subset": "tir",
        "license": "cc-by-4.0",
    }


def test_nemotron_math_proofs_v2_views_filter_caps_and_transform_proof_rows():
    proof_step = get_instruction_dataset("nvidia/Nemotron-Math-Proofs-v2/proof-pilot")
    proof_cfg = proof_step.config
    verification_cfg = get_instruction_dataset("nvidia/Nemotron-Math-Proofs-v2/verification-pilot").config
    meta_verification_cfg = get_instruction_dataset("nvidia/Nemotron-Math-Proofs-v2/meta-verification-pilot").config
    full_proof_cfg = get_instruction_dataset("nvidia/Nemotron-Math-Proofs-v2/proof").config

    assert unwrap_versioned_value(proof_cfg.source) == "nvidia/Nemotron-Math-Proofs-v2"
    assert unwrap_versioned_value(proof_cfg.revision) == "d857c6b46a63ad97cfbd7b4254e2edf53e9d1666"
    assert unwrap_versioned_value(proof_cfg.splits) == ["train"]
    assert row_matches_filters(NEMOTRON_PROOFS_V2_SAMPLE, unwrap_versioned_value(proof_cfg.row_filters))
    assert not row_matches_filters(NEMOTRON_PROOFS_V2_SAMPLE, unwrap_versioned_value(verification_cfg.row_filters))
    assert not row_matches_filters(NEMOTRON_PROOFS_V2_SAMPLE, unwrap_versioned_value(meta_verification_cfg.row_filters))
    assert unwrap_versioned_value(proof_cfg.max_examples_per_split) == 2_000
    assert unwrap_versioned_value(full_proof_cfg.max_examples_per_split) is None

    result = transform_row(NEMOTRON_PROOFS_V2_SAMPLE, proof_cfg, unwrap_versioned_value(proof_cfg.adapter))

    assert result is not None
    dumped = result.model_dump()
    assert dumped["tools"] == []
    assert dumped["messages"][0]["content"] == "Prove that sqrt(3) is irrational."
    assert dumped["messages"][1]["reasoning_content"] == "Use the standard prime divisibility argument."
    assert result.metadata == {
        "uuid": "proof-row-1",
        "problem": "Prove that sqrt(3) is irrational.",
        "used_in": ["ultra_v3"],
        "metadata": [],
        "source": "AoPS",
        "dataset": "Nemotron-Math-Proofs-v2",
        "subset": "proof",
        "license": "CC BY 4.0",
    }


def test_nemotron_math_v2_pilot_views_target_reasoning_splits_and_transform_tool_rows():
    high_step = get_instruction_dataset("nvidia/Nemotron-Math-v2/high-pilot")
    high_cfg = high_step.config
    medium_step = get_instruction_dataset("nvidia/Nemotron-Math-v2/medium-pilot")
    medium_cfg = medium_step.config

    assert unwrap_versioned_value(high_cfg.splits) == ["high_part00", "high_part01", "high_part02"]
    assert unwrap_versioned_value(high_cfg.max_examples_per_split) == 50_000
    assert unwrap_versioned_value(medium_cfg.splits) == ["medium"]
    assert unwrap_versioned_value(medium_cfg.max_examples_per_split) == 150_000
    assert high_step.override_output_path != medium_step.override_output_path

    result = transform_row(NEMOTRON_STRUCTURED_SAMPLE, high_cfg, unwrap_versioned_value(high_cfg.adapter))

    assert result is not None
    dumped = result.model_dump()
    assert dumped["tools"] == NEMOTRON_STRUCTURED_SAMPLE["tools"]
    assert dumped["messages"][1]["reasoning_content"] == "Use Python for the arithmetic."
    assert dumped["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"code": "2 + 2"}
    assert result.metadata["metadata"]["reason_high_with_tool"]["accuracy"] == 1.0
    assert result.metadata["used_in"] == ["train"]


def test_prismmath_and_openmathinstruct2_registered_adapters_transform_rows():
    prism_step = get_instruction_dataset("nvidia/Nemotron-PrismMath")
    prism_cfg = prism_step.config
    openmath_step = get_instruction_dataset("nvidia/OpenMathInstruct-2/train_1M")
    openmath_cfg = openmath_step.config

    prism_result = transform_row(
        {"id": "prism-1", "problem": "Find x if x + 1 = 3.", "solution": "Subtract 1 to get x = 2."},
        prism_cfg,
        unwrap_versioned_value(prism_cfg.adapter),
    )
    openmath_result = transform_row(
        {
            "problem": "Compute 6 * 7.",
            "generated_solution": "6 * 7 = 42.",
            "expected_answer": "42",
            "problem_source": "MATH",
        },
        openmath_cfg,
        unwrap_versioned_value(openmath_cfg.adapter),
    )

    assert prism_result is not None
    assert [message.content for message in prism_result.messages] == [
        "Find x if x + 1 = 3.",
        "Subtract 1 to get x = 2.",
    ]
    assert prism_result.metadata == {"id": "prism-1"}

    assert openmath_result is not None
    assert unwrap_versioned_value(openmath_cfg.splits) == ["train_1M"]
    assert [message.content for message in openmath_result.messages] == ["Compute 6 * 7.", "6 * 7 = 42."]
    assert openmath_result.metadata == {"expected_answer": "42", "problem_source": "MATH"}


def test_dataset_view_output_path_hash_accounts_for_filters_and_caps():
    base_cfg = InstructionDatasetConfig(
        hf_dataset_id="nvidia/Nemotron-SFT-Math-v4",
        revision="a94e56aeddcf6e75d28c8bd210f40fa62309288d",
        adapter=structured_multi_turn_adapter(metadata_remap={"tools": "tools"}),
        metadata_columns=["subset"],
        name="nvidia/Nemotron-SFT-Math-v4/shared-name",
        splits=["train"],
        row_filters=[RowFilter("subset", RowFilterOperator.EQUALS, "cot")],
        max_examples_per_split=100_000,
    )
    tir_cfg = replace(base_cfg, row_filters=[RowFilter("subset", RowFilterOperator.EQUALS, "tir")])
    uncapped_cot_cfg = replace(base_cfg, max_examples_per_split=None)

    cot_step = transform_dataset_step(base_cfg)
    tir_step = transform_dataset_step(tir_cfg)
    uncapped_cot_step = transform_dataset_step(uncapped_cot_cfg)

    assert cot_step.name == tir_step.name == uncapped_cot_step.name
    assert cot_step.override_output_path != tir_step.override_output_path
    assert cot_step.override_output_path != uncapped_cot_step.override_output_path
    assert tir_step.override_output_path != uncapped_cot_step.override_output_path


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
