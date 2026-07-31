# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import msgspec
import pytest
from marin.datakit.download.nemotron_sft_v3 import doc_to_text, load_jsonl_with_skips, row_to_doc


def test_doc_to_text_preserves_training_fields_without_row_metadata():
    row = {
        "uuid": "row-1",
        "metadata": {"z": 2, "a": 1},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a command",
                    "parameters": {"type": "object"},
                },
            }
        ],
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I need a directory listing.",
                "tool_calls": json.dumps(
                    [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"path":"."}'},
                        }
                    ]
                ),
            },
            {
                "role": "tool",
                "name": "bash",
                "tool_call_id": "call-1",
                "content": [{"type": "text", "value": "README.md"}],
            },
        ],
    }

    text = doc_to_text(row)

    assert (
        text
        == """<tools>
[{"function":{"description":"Run a command","name":"bash","parameters":{"type":"object"}},"type":"function"}]
</tools>

<assistant>
<reasoning>
I need a directory listing.
</reasoning>
<tool_call>
{"function":{"arguments":"{\\"path\\":\\".\\"}","name":"bash"},"id":"call-1","type":"function"}
</tool_call>
</assistant>

<tool name="bash" tool_call_id="call-1">
<content>
[{"type":"text","value":"README.md"}]
</content>
</tool>"""
    )


def test_doc_to_text_omits_empty_fields_and_sanitizes_backspace():
    text = doc_to_text(
        {
            "messages": [
                {"role": "system", "content": "", "reasoning_content": None},
                {
                    "role": "user",
                    "name": "",
                    "tool_call_id": "",
                    "content": "Put the answer in \x08oxed{}.",
                },
            ],
            "uuid": "row-1",
        }
    )

    assert text == "<system>\n</system>\n\n<user>\n<content>\nPut the answer in \\boxed{}.\n</content>\n</user>"


def test_row_to_doc_drops_pathological_line_repetition():
    row = {
        "messages": [
            {"role": "user", "content": "Translate this."},
            {"role": "assistant", "content": "<translation>\n" * 257},
        ]
    }

    assert row_to_doc(row, family="multilingual_v1", partition_name="stem_it") == []


def test_load_jsonl_with_skips_only_ignores_pinned_bad_lines(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"id": 1}\n{bad json}\n{"id": 2}\n')

    assert list(load_jsonl_with_skips(str(path), frozenset({2}))) == [{"id": 1}, {"id": 2}]
    with pytest.raises(msgspec.DecodeError):
        list(load_jsonl_with_skips(str(path), frozenset()))


def test_row_to_doc_drops_chat_rows_with_withheld_user_content():
    messages = [
        {"role": "user", "content": None},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow-up"},
    ]
    assert (
        row_to_doc(
            {"messages": messages},
            family="instruction_following_chat_v3",
            partition_name="chat",
        )
        == []
    )
