# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from marin.datakit.chat_render import render_chat_record

from experiments.post_training.curriculum_sft.generation import GenerateCurriculumSFTConfig, parse_batch
from experiments.post_training.curriculum_sft.grug_pipeline import prepare_chat_record


def _response(index: int, *, task: str, continuation: list[dict[str, str]]) -> dict:
    return {
        "custom_id": f"conversation-{index:05d}",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "submit_conversation",
                                        "arguments": json.dumps({"task": task, "continuation": continuation}),
                                    }
                                }
                            ]
                        },
                    }
                ]
            },
        },
    }


def test_parse_batch_keeps_distinct_complete_conversations():
    task = "A fictional shop sold three apples. How many did it sell?"
    answer = [{"role": "assistant", "content": "It sold three apples."}]
    responses = [
        _response(0, task=task, continuation=answer),
        _response(1, task=task, continuation=answer),
        _response(2, task="Ask a different question.", continuation=[{"role": "user", "content": "More detail?"}]),
    ]
    config = GenerateCurriculumSFTConfig(
        catalog_path="unused",
        output_path="unused",
        capability_id="d00.example",
        requested_examples=3,
        accepted_examples=1,
        seed=17,
        max_completion_tokens=4096,
        task_specification="Use fictional shop questions.",
        relay_job="unused",
    )

    task_records, chat_documents = parse_batch("\n".join(json.dumps(response) for response in responses), config)

    assert [record["rejection_reason"] for record in task_records] == [None, "duplicate_task", "invalid_conversation"]
    assert [record["accepted"] for record in task_records] == [True, False, False]
    assert len(chat_documents) == 1
    assert chat_documents[0]["source_id"] == "conversation-00000"

    rendered = render_chat_record(chat_documents[0])["text"]
    assert task in rendered
    assert "It sold three apples." in rendered
    assert rendered.index(task) < rendered.index("It sold three apples.")
    prepared = prepare_chat_record(chat_documents[0])
    assert prepared["messages"] == [
        {"role": "user", "content": task},
        {"role": "assistant", "content": "It sold three apples."},
    ]
