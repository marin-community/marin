# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.raw_capability_eval_sets import (
    CapabilityEvalDatasetConfig,
    CapabilityEvalRenderer,
    _project_chat_row_to_raw_text,
    _render_capability_chat_row,
)


def test_render_wildchat_row_writes_oai_chat_and_marin_projection():
    config = CapabilityEvalDatasetConfig(
        dataset_id="allenai/WildChat",
        revision="rev",
        split="train",
        renderer=CapabilityEvalRenderer.WILDCHAT,
    )
    row = {
        "conversation_id": "conv-1",
        "conversation": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ],
        "language": "English",
        "model": "gpt-4",
        "redacted": False,
        "timestamp": "2024-01-01T00:00:00+00:00",
        "toxic": False,
        "turn": 1,
    }

    rendered = _render_capability_chat_row(config, row)
    raw_text = _project_chat_row_to_raw_text(config, rendered)

    assert rendered is not None
    assert raw_text is not None
    assert rendered["messages"] == row["conversation"]
    assert rendered["metadata"]["language"] == "English"
    assert rendered["metadata"]["model"] == "gpt-4"
    assert (
        raw_text["text"] == "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        "hello<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        "hi there<|eot_id|>"
    )


def test_render_openhands_row_keeps_full_oai_trace_but_scores_only_model_targets():
    config = CapabilityEvalDatasetConfig(
        dataset_id="nebius/SWE-rebench-openhands-trajectories",
        revision="rev",
        split="train",
        renderer=CapabilityEvalRenderer.OPENHANDS,
    )
    row = {
        "trajectory_id": "traj-1",
        "instance_id": "repo__issue-1",
        "repo": "example/repo",
        "trajectory": [
            {
                "role": "system",
                "content": "You are OpenHands agent.",
            },
            {
                "role": "user",
                "content": "Fix the failing test in example/repo.",
            },
            {
                "role": "assistant",
                "content": "Inspecting the repo.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command":"ls -la","timeout":30,"schema":{"id":"inner-id","type":"object"}}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "bash",
                "tool_call_id": "call-1",
                "content": "README.md\nsrc\n",
            },
        ],
        "tools": [{"function": {"name": "bash", "parameters": {"type": "object"}}}],
        "model_patch": "diff --git a/foo.py b/foo.py",
        "exit_status": "submit",
        "resolved": 1,
        "gen_tests_correct": 1.0,
        "pred_passes_gen_tests": 0.75,
    }

    rendered = _render_capability_chat_row(config, row)
    raw_text = _project_chat_row_to_raw_text(config, rendered)

    assert rendered is not None
    assert raw_text is not None
    assert rendered["messages"][0]["role"] == "system"
    assert rendered["messages"][1]["content"] == "Fix the failing test in example/repo."
    assert rendered["messages"][2]["tool_calls"][0]["id"] == "call-1"
    assert rendered["messages"][2]["tool_calls"][0]["function"]["arguments"] == {
        "command": "ls -la",
        "timeout": 30,
        "schema": {"id": "inner-id", "type": "object"},
    }
    assert rendered["messages"][3]["role"] == "tool"
    assert rendered["messages"][3]["tool_call_id"] == "call-1"
    assert rendered["metadata"]["instance_id"] == "repo__issue-1"
    assert rendered["metadata"]["repo"] == "example/repo"
    assert rendered["metadata"]["resolved"] == 1
    assert rendered["metadata"]["tools"] == row["tools"]
    assert rendered["chat_template_kwargs"] == {"tools": row["tools"]}

    assert '"command": "ls -la"' in raw_text["text"]
    assert "Final Patch:\ndiff --git a/foo.py b/foo.py" in raw_text["text"]
    assert "You are OpenHands agent" not in raw_text["text"]
    assert "Fix the failing test" not in raw_text["text"]
    assert "README.md\nsrc" not in raw_text["text"]
    assert "Available Tools" not in raw_text["text"]
    assert "Resolved: 1" not in raw_text["text"]
    assert "call-1" not in raw_text["text"]
    assert '"type": "function"' not in raw_text["text"]
    assert '"id": "inner-id"' in raw_text["text"]
    assert '"type": "object"' in raw_text["text"]
    assert "repo__issue-1" not in raw_text["text"]
    assert "traj-1" not in raw_text["text"]


def test_openhands_null_model_patch_does_not_score_none_literal():
    config = CapabilityEvalDatasetConfig(
        dataset_id="nebius/SWE-rebench-openhands-trajectories",
        revision="rev",
        split="train",
        renderer=CapabilityEvalRenderer.OPENHANDS,
    )
    row = {
        "trajectory_id": "traj-1",
        "trajectory": [
            {
                "role": "assistant",
                "content": "I cannot produce a patch.",
            },
        ],
        "model_patch": None,
    }

    rendered = _render_capability_chat_row(config, row)
    raw_text = _project_chat_row_to_raw_text(config, rendered)

    assert rendered is not None
    assert raw_text is not None
    assert "I cannot produce a patch." in raw_text["text"]
    assert "Final Patch" not in raw_text["text"]
    assert "None" not in raw_text["text"]


def test_render_global_mgsm_row_preserves_answer_prefix_in_chat_projection():
    config = CapabilityEvalDatasetConfig(
        dataset_id="CohereLabs/global-mgsm",
        dataset_name="en",
        revision="rev",
        split="test",
        renderer=CapabilityEvalRenderer.GLOBAL_MGSM,
    )
    row = {
        "instruction": 'Solve the problem. End with "Answer:" and the final number.',
        "question": "What is 6 * 7?",
        "answer_prefix": "Answer",
        "answer": "42",
    }

    rendered = _render_capability_chat_row(config, row)
    raw_text = _project_chat_row_to_raw_text(config, rendered)

    assert rendered is not None
    assert raw_text is not None
    assert rendered["messages"] == [
        {
            "role": "user",
            "content": (
                'Instruction:\nSolve the problem. End with "Answer:" and the final number.\n\n'
                "Question:\nWhat is 6 * 7?"
            ),
        },
        {"role": "assistant", "content": "Answer Prefix:\nAnswer\n\nAnswer:\n42"},
    ]
    assert "Answer Prefix:\nAnswer\n\nAnswer:\n42" in raw_text["text"]
