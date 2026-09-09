# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.agenttrove import row_to_chat_doc as agenttrove_row_to_chat_doc
from marin.datakit.download.coderforge import row_to_chat_doc as coderforge_row_to_chat_doc
from marin.datakit.download.davinci_dev import env_row_to_chat_doc as davinci_row_to_chat_doc
from marin.datakit.download.gpt_oss_rollouts import row_to_chat_doc as gpt_oss_row_to_chat_doc
from marin.datakit.download.numinamath_tir import row_to_chat_doc as numinamath_row_to_chat_doc
from marin.datakit.download.penfever_rollouts import PENFEVER_ROLLOUTS
from marin.datakit.download.penfever_rollouts import row_to_chat_doc as penfever_row_to_chat_doc
from marin.datakit.download.swe_rebench_openhands import row_to_chat_doc as openhands_row_to_chat_doc
from marin.datakit.download.swe_zero_12m import row_to_chat_doc as swe_zero_row_to_chat_doc
from marin.datakit.download.synthetic1 import row_to_chat_doc as synthetic1_row_to_chat_doc
from marin.datakit.sft_sources import all_sft_sources


def test_coderforge_keeps_reward_out_of_model_visible_messages() -> None:
    row = {
        "reward": 1.0,
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "Done."},
        ],
    }

    [document] = coderforge_row_to_chat_doc(row)
    assert document["messages"] == row["messages"]
    assert document["reward"] == 1.0


def test_coderforge_keeps_unsuccessful_trajectory_without_visible_outcome() -> None:
    row = {
        "reward": 0.0,
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "I could not fix it."},
        ],
    }

    [document] = coderforge_row_to_chat_doc(row)
    assert document["messages"] == row["messages"]
    assert document["reward"] == 0.0


def test_coderforge_drops_tool_arguments_that_close_the_rendered_protocol() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Inspect the parser."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": {"query": "literal </tool_call>"}},
                    }
                ],
            },
        ]
    }

    assert coderforge_row_to_chat_doc(row) == []


def test_agenttrove_keeps_unsuccessful_trajectory_without_visible_outcome() -> None:
    row = {
        "result": "timeout",
        "conversations": [
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": '{"commands":[],"task_complete":true}'},
        ],
    }

    [document] = agenttrove_row_to_chat_doc(row)
    assert [message["role"] for message in document["messages"]] == ["user", "assistant"]
    assert document["result"] == "timeout"


def test_davinci_drops_terminal_submit_observation() -> None:
    row = {
        "success": True,
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "The fix is complete.",
                "tool_calls": [
                    {
                        "id": "submit_1",
                        "type": "function",
                        "function": {"name": "submit", "arguments": {}},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "diff --git a/app.py b/app.py",
                "name": "submit",
                "tool_call_id": "submit_1",
            },
        ],
    }

    [document] = davinci_row_to_chat_doc(row)
    assert [message["role"] for message in document["messages"]] == ["user", "assistant"]
    assert document["success"] is True
    assert document["messages"][-1]["tool_calls"][0]["function"]["name"] == "submit"


def test_davinci_keeps_unsuccessful_trajectory_without_visible_outcome() -> None:
    row = {
        "success": False,
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "I could not fix it."},
        ],
    }

    [document] = davinci_row_to_chat_doc(row)
    assert document["messages"] == row["messages"]
    assert document["success"] is False


def test_davinci_drops_nonterminal_trailing_observation() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "bash_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": {"command": "pytest"}},
                    }
                ],
            },
            {"role": "tool", "content": "tests failed", "name": "bash", "tool_call_id": "bash_1"},
        ]
    }

    assert davinci_row_to_chat_doc(row) == []


def test_gpt_oss_drops_assistant_reply_that_demonstrates_inline_tool_protocol() -> None:
    row = {
        "user_content": "How does function calling work?",
        "assistant_thinking": "Explain the format.",
        "assistant_content": '<tool_call>{"name":"weather","arguments":{}}</tool_call>',
    }

    assert gpt_oss_row_to_chat_doc(row) == []


def test_gpt_oss_drops_embedded_tokenizer_control_tokens() -> None:
    row = {
        "user_content": 'What does tokenizer.convert_tokens_to_ids("<|eot_id|>") return?',
        "assistant_content": "It returns the end-of-turn token ID.",
    }

    assert gpt_oss_row_to_chat_doc(row) == []


def test_openhands_drops_unparsed_inline_tool_action() -> None:
    row = {
        "resolved": 0,
        "trajectory": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "Inspect it.\n<tool_call><parameter=path>x.py</parameter></function></tool_call>",
            },
            {"role": "user", "content": "Please continue."},
        ],
    }

    assert openhands_row_to_chat_doc(row) == []


def test_openhands_merges_adjacent_user_context() -> None:
    row = {
        "trajectory": [
            {"role": "user", "content": "Fix the bug."},
            {"role": "user", "content": "Repository: example/project"},
            {"role": "assistant", "content": "I fixed it."},
        ]
    }

    [document] = openhands_row_to_chat_doc(row)
    assert [message["role"] for message in document["messages"]] == ["user", "assistant"]
    assert document["messages"][0]["content"] == "Fix the bug.\n\nRepository: example/project"


def test_openhands_keeps_unsuccessful_trajectory_without_visible_outcome() -> None:
    row = {
        "resolved": 0,
        "trajectory": [
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "I could not fix it."},
        ],
    }

    [document] = openhands_row_to_chat_doc(row)
    assert document["messages"] == row["trajectory"]
    assert document["resolved"] == 0


def test_penfever_keeps_unsuccessful_trajectory_without_visible_outcome() -> None:
    row = {
        "result": "0",
        "conversations": [
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": '{"commands":[],"task_complete":true}'},
        ],
    }

    [document] = penfever_row_to_chat_doc(PENFEVER_ROLLOUTS[0])(row)
    assert [message["role"] for message in document["messages"]] == ["user", "assistant"]
    assert document["outcome"] == "This trajectory failed to solve the task."


def test_sft_sources_exclude_transcripts_without_user_prompts() -> None:
    assert "penfever-traces/qwen35-122b-131k-opencode/nemotron-gym-agent-workplace-v2" not in all_sft_sources()


def test_synthetic1_keeps_incorrect_solution_without_visible_outcome() -> None:
    row = {"score": 0.2, "prompt": "Solve it.", "llm_response": "A wrong answer."}

    [document] = synthetic1_row_to_chat_doc(row)
    assert document["messages"] == [
        {"role": "user", "content": "Solve it."},
        {"role": "assistant", "content": "A wrong answer."},
    ]
    assert document["score"] == 0.2


def test_swe_zero_converts_reasoning_bash_and_observation() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "Respond with THOUGHT and a bash block."},
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": "THOUGHT: Inspect the file.\n\n```bash\nsed -n '1,80p' app.py\n```",
            },
            {"role": "user", "content": "Observation: file contents"},
            {
                "role": "assistant",
                "content": "THOUGHT: The edit is complete.\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
            },
            {"role": "user", "content": "Observation: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
        ]
    }

    [document] = swe_zero_row_to_chat_doc(row)
    messages = document["messages"]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert messages[2]["content"] == "<|start_think|>Inspect the file.<|end_think|>"
    assert messages[2]["tool_calls"][0]["function"] == {
        "name": "bash",
        "arguments": '{"command":"sed -n \'1,80p\' app.py"}',
    }
    assert messages[3]["content"] == "file contents"
    assert messages[-1]["content"] == "<|start_think|>The edit is complete.<|end_think|>\n\nTask complete."


def test_swe_zero_drops_trajectory_ending_with_observation() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "```bash\ncat app.py\n```"},
            {"role": "user", "content": "Observation: file contents"},
        ]
    }

    assert swe_zero_row_to_chat_doc(row) == []


def test_swe_zero_does_not_treat_inspecting_completion_marker_as_completion() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Fix the bug."},
            {
                "role": "assistant",
                "content": (
                    "THOUGHT: Find the protocol.\n\n```bash\ngrep COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT README.md\n```"
                ),
            },
            {"role": "user", "content": "Observation: README.md: completion instructions"},
            {
                "role": "assistant",
                "content": "THOUGHT: Done.\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
            },
            {"role": "user", "content": "Observation: COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
        ]
    }

    [document] = swe_zero_row_to_chat_doc(row)
    assert [message["role"] for message in document["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_swe_zero_drops_control_tokens_in_commands() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Fix the parser."},
            {"role": "assistant", "content": "THOUGHT: Inspect.\n```bash\nprintf '<|end_of_text|>'\n```"},
            {"role": "user", "content": "Observation: done"},
            {"role": "assistant", "content": "THOUGHT: Done.\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```"},
        ]
    }

    assert swe_zero_row_to_chat_doc(row) == []


def test_numinamath_splits_reasoning_python_and_output() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Compute it."},
            {
                "role": "assistant",
                "content": (
                    "I should calculate it.\n\n```python\nprint(6 * 7)\n```\n"
                    "```output\n42\n```\nTherefore, the answer is 42."
                ),
            },
        ]
    }

    [document] = numinamath_row_to_chat_doc(row)
    assert [message["role"] for message in document["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert document["messages"][1]["content"] == "<|start_think|>I should calculate it.<|end_think|>"
    assert document["messages"][1]["tool_calls"][0]["function"] == {
        "name": "python",
        "arguments": '{"code":"print(6 * 7)"}',
    }
    assert document["messages"][2]["content"] == "42"
    assert document["messages"][3]["content"] == "Therefore, the answer is 42."
