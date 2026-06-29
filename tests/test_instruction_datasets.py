# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from datasets import Features
from marin.execution.executor import unwrap_versioned_value
from marin.transform.conversation.adapters import InputDatasetFormat
from marin.transform.conversation.transform_conversation import transform_row

from experiments.posttrain.instruction_datasets import (
    FINEPROOFS_SFT_METADATA_COLUMNS,
    FINEPROOFS_SFT_REVISION,
    INSTRUCTION_DATASET_NAME_TO_CONFIG,
    NEMOTRON_SFT_INSTRUCTION_FOLLOWING_CHAT_V2_FEATURES,
    SYNTHETIC2_SFT_VERIFIED_HF_ID,
    SYNTHETIC2_SFT_VERIFIED_METADATA_COLUMNS,
    SYNTHETIC2_SFT_VERIFIED_REVISION,
    get_instruction_dataset,
)

NEMOTRON_SFT_IF_CHAT_V2_HF_ID = "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2"
NEMOTRON_SFT_IF_CHAT_V2_REVISION = "1a9454ed054b8544503ab8d8c0a519d141a44c5b"
NEMOTRON_SFT_IF_CHAT_V3_HF_ID = "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3"
NEMOTRON_SFT_IF_CHAT_V3_REVISION = "be3b3e04ef605ac9d3f8f35b9d5a632f4a3a3402"

SYNTHETIC2_SFT_VERIFIED_SAMPLE = {
    "problem_id": "prime_rl_code_21747",
    "task_type": "prime_rl_code",
    "reward": 1.0,
    "messages": [
        {"role": "user", "content": "Write Python code to count the intersection of two sets."},
        {"role": "assistant", "content": "<think>Use set intersection.</think>\n```python\nprint(len(a & b))\n```"},
    ],
}

NEMOTRON_SFT_REASONING_OFF_SAMPLE = {
    "uuid": "8ebb12a4-4f9a-4dd1-bd04-3473ef2450a2",
    "license": "ODC-By",
    "used_in": ["super_v3"],
    "reasoning": "off",
    "messages": [
        {"role": "system", "content": "", "reasoning_content": None},
        {"role": "user", "content": "Polish the sentence: I look forward to meet you.", "reasoning_content": None},
        {"role": "assistant", "content": "I look forward to meeting you.", "reasoning_content": None},
    ],
}

NEMOTRON_SFT_REASONING_ON_SAMPLE = {
    "uuid": "ba6e772a-e496-4d39-b6c3-7f1205a12719",
    "license": "ODC-By",
    "used_in": ["super_v3"],
    "reasoning": "on",
    "messages": [
        {"role": "system", "content": "", "reasoning_content": None},
        {"role": "user", "content": "Polish the sentence: I look forward to meet you.", "reasoning_content": None},
        {
            "role": "assistant",
            "content": "I look forward to meeting you.",
            "reasoning_content": "The phrase should use the gerund after 'look forward to'.",
        },
    ],
}

NEMOTRON_SFT_V3_INSTRUCTION_FOLLOWING_SAMPLE = {
    "uuid": "4dd07659-302a-40c0-933b-7ed2b7661638",
    "used_in": ["ultra_v3"],
    "metadata": {
        "seed_dataset": "allenai/tulu-3-sft-personas-instruction-following",
        "seed_prompt_sha256": "d178cc7601e92234c0d84fbb0cafe64432e4821271bde8c1024fcf8ea44a89c3",
        "model": "openai/GPT-OSS-120B",
        "reward_model": None,
        "train_turns": [False, False, True],
    },
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain why useHistory is unavailable in react-router-dom v6."},
        {
            "role": "assistant",
            "reasoning_content": "React Router v6 removed useHistory and replaced it with useNavigate.",
            "content": "In react-router-dom v6, useHistory was removed. Use useNavigate instead.",
        },
    ],
}

NEMOTRON_SFT_V3_NO_REASONING_SAMPLE = {
    "uuid": "1f875358-47c4-444a-97c4-8b252a5f132a",
    "used_in": ["ultra_v3"],
    "metadata": {
        "seed_dataset": "allenai/tulu-3-sft-personas-instruction-following",
        "seed_prompt_sha256": "8e86d2834d9ebd0bc6a0ba599950a1f7d8b59cb9dc4daafca8d970feff9d77ab",
        "model": "openai/GPT-OSS-120B",
        "reward_model": None,
        "train_turns": [False, False, True],
    },
    "messages": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Name the HTTP status code for a missing page."},
        {"role": "assistant", "reasoning_content": "", "content": "404 Not Found."},
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


@pytest.mark.parametrize(
    ("split", "sample", "expected_assistant_content", "expected_thinking"),
    [
        (
            "reasoning_off",
            NEMOTRON_SFT_REASONING_OFF_SAMPLE,
            "I look forward to meeting you.",
            False,
        ),
        (
            "reasoning_on",
            NEMOTRON_SFT_REASONING_ON_SAMPLE,
            "\n".join(
                [
                    "<|start_think|>The phrase should use the gerund after 'look forward to'.<|end_think|>",
                    "I look forward to meeting you.",
                ]
            ),
            True,
        ),
    ],
)
def test_nemotron_sft_instruction_following_chat_v2_transforms_reasoning_split_rows(
    split,
    sample,
    expected_assistant_content,
    expected_thinking,
):
    dataset_key = f"{NEMOTRON_SFT_IF_CHAT_V2_HF_ID}/{split}"
    step = get_instruction_dataset(dataset_key)
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)

    assert step.name == f"documents/{dataset_key}"
    assert step.override_output_path is not None
    assert split in step.override_output_path
    assert unwrap_versioned_value(cfg.source) == NEMOTRON_SFT_IF_CHAT_V2_HF_ID
    assert unwrap_versioned_value(cfg.revision) == NEMOTRON_SFT_IF_CHAT_V2_REVISION
    assert unwrap_versioned_value(cfg.splits) == [split]
    assert unwrap_versioned_value(cfg.load_dataset_features) == NEMOTRON_SFT_INSTRUCTION_FOLLOWING_CHAT_V2_FEATURES
    assert adapter.dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN
    assert adapter.reasoning_content_key == "reasoning_content"

    result = transform_row(sample, cfg, adapter)

    assert result is not None
    assert result.source == NEMOTRON_SFT_IF_CHAT_V2_HF_ID
    assert [message.role for message in result.messages] == ["system", "user", "assistant"]
    assert result.messages[2].content == expected_assistant_content
    assert result.metadata == {
        "uuid": sample["uuid"],
        "license": "ODC-By",
        "used_in": ["super_v3"],
        "reasoning": split.removeprefix("reasoning_"),
    }
    assert result.model_dump()["chat_template_kwargs"] == {"enable_thinking": expected_thinking}


def test_nemotron_sft_instruction_following_chat_v2_features_accept_both_message_shapes():
    features = Features.from_dict(NEMOTRON_SFT_INSTRUCTION_FOLLOWING_CHAT_V2_FEATURES)

    reasoning_off = features.encode_example(NEMOTRON_SFT_REASONING_OFF_SAMPLE)
    reasoning_on = features.encode_example(NEMOTRON_SFT_REASONING_ON_SAMPLE)

    assert reasoning_off["messages"][2]["reasoning_content"] is None
    expected_reasoning = "The phrase should use the gerund after 'look forward to'."
    assert reasoning_on["messages"][2]["reasoning_content"] == expected_reasoning


@pytest.mark.parametrize(
    ("sample", "expected_assistant_content", "expected_thinking"),
    [
        (
            NEMOTRON_SFT_V3_INSTRUCTION_FOLLOWING_SAMPLE,
            "\n".join(
                [
                    "<|start_think|>React Router v6 removed useHistory and replaced it with useNavigate.<|end_think|>",
                    "In react-router-dom v6, useHistory was removed. Use useNavigate instead.",
                ]
            ),
            True,
        ),
        (
            NEMOTRON_SFT_V3_NO_REASONING_SAMPLE,
            "404 Not Found.",
            False,
        ),
    ],
)
def test_nemotron_sft_instruction_following_chat_v3_transforms_instruction_following_rows(
    sample,
    expected_assistant_content,
    expected_thinking,
):
    dataset_key = f"{NEMOTRON_SFT_IF_CHAT_V3_HF_ID}/instruction_following"
    step = get_instruction_dataset(dataset_key)
    cfg = step.config
    adapter = unwrap_versioned_value(cfg.adapter)

    assert step.name == f"documents/{dataset_key}"
    assert step.override_output_path is not None
    assert "instruction_following" in step.override_output_path
    assert f"{NEMOTRON_SFT_IF_CHAT_V3_HF_ID}/chat" not in INSTRUCTION_DATASET_NAME_TO_CONFIG
    assert unwrap_versioned_value(cfg.source) == NEMOTRON_SFT_IF_CHAT_V3_HF_ID
    assert unwrap_versioned_value(cfg.revision) == NEMOTRON_SFT_IF_CHAT_V3_REVISION
    assert unwrap_versioned_value(cfg.splits) == ["instruction_following"]
    assert unwrap_versioned_value(cfg.load_dataset_features) is None
    assert adapter.dataset_format == InputDatasetFormat.SINGLE_COLUMN_MULTI_TURN
    assert adapter.reasoning_content_key == "reasoning_content"

    result = transform_row(sample, cfg, adapter)

    assert result is not None
    assert result.source == NEMOTRON_SFT_IF_CHAT_V3_HF_ID
    assert [message.role for message in result.messages] == ["system", "user", "assistant"]
    assert result.messages[2].content == expected_assistant_content
    assert result.metadata == {
        "uuid": sample["uuid"],
        "used_in": ["ultra_v3"],
        "metadata": sample["metadata"],
    }
    assert result.model_dump()["chat_template_kwargs"] == {"enable_thinking": expected_thinking}


def test_default_reasoning_content_key_does_not_rehash_existing_instruction_datasets():
    step = get_instruction_dataset(SYNTHETIC2_SFT_VERIFIED_HF_ID)

    assert (
        step.override_output_path
        == f"documents/PrimeIntellect--SYNTHETIC-2-SFT-verified-{SYNTHETIC2_SFT_VERIFIED_REVISION}-409fa9"
    )


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
