# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.davinci_dev import env_row_to_chat_doc
from marin.datakit.download.gpt_oss_rollouts import row_to_chat_doc as gpt_oss_row_to_chat_doc


def test_gpt_oss_does_not_double_wrap_existing_reasoning():
    [document] = gpt_oss_row_to_chat_doc(
        {
            "user_content": "Question",
            "assistant_thinking": "<think>Plan</think>",
            "assistant_content": "Answer",
        }
    )

    assert document["messages"][1]["content"] == "<|start_think|>Plan<|end_think|>\n\nAnswer"


def test_chat_source_drops_unrecoverable_reasoning_delimiters():
    assert (
        env_row_to_chat_doc(
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "<think>Unclosed reasoning"},
                ]
            }
        )
        == []
    )
