# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.lazy import materialized_config
from marin.transform.conversation.transform_conversation import transform_row

from experiments.datasets.instruction import (
    SYNTHETIC2_SFT_VERIFIED_HF_ID,
    get_instruction_dataset,
)

_TEST_PREFIX = "gs://test"

SYNTHETIC2_SFT_VERIFIED_SAMPLE = {
    "problem_id": "prime_rl_code_21747",
    "task_type": "prime_rl_code",
    "reward": 1.0,
    "messages": [
        {"role": "user", "content": "Write Python code to count the intersection of two sets."},
        {"role": "assistant", "content": "<think>Use set intersection.</think>\n```python\nprint(len(a & b))\n```"},
    ],
}


def test_synthetic2_sft_verified_step_transforms_chat_rows():
    step = get_instruction_dataset(SYNTHETIC2_SFT_VERIFIED_HF_ID)
    cfg = materialized_config(step, _TEST_PREFIX)
    adapter = cfg.adapter

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
