# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.pivotrl.sft import EXAMPLES_FILENAME, write_sft_examples
from experiments.post_training.pivotrl.sft_templates import GRUG_FINAL_TURN_TEMPLATE, QWEN3_FINAL_TURN_TEMPLATE

TOOLS = [{"type": "function", "name": "bash", "parameters": {"type": "object", "properties": {}}}]


def _call(command: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": command, "type": "function", "function": {"name": "bash", "arguments": command}}],
    }


def _swe_row(trajectory_id: int, prompt: list[dict], command: str) -> dict:
    record = {"expected_action": {"type": "function_call", "name": "bash", "arguments": {"command": command}}}
    request = {"input": [], "tools": TOOLS, "max_output_tokens": 512}
    return {
        "prompt": prompt,
        "extra_info": {
            "source_id": f"swe:{trajectory_id}",
            "trajectory_id": trajectory_id,
            "index": len(prompt),
            "nemotron_ultra": {"record_json": json.dumps(record), "request_json": json.dumps(request)},
        },
    }


@pytest.mark.parametrize("template", [GRUG_FINAL_TURN_TEMPLATE, QWEN3_FINAL_TURN_TEMPLATE])
def test_final_turn_template_supervises_only_the_last_turn(gpt2_tokenizer, template):
    messages = [
        {"role": "user", "content": "Fix the bug."},
        _call("EARLIER_ACTION"),
        {"role": "tool", "tool_call_id": "EARLIER_ACTION", "content": "a.py"},
        _call("FINAL_ACTION"),
    ]

    encoded = gpt2_tokenizer.apply_chat_template_with_masks([messages], chat_template=template, tools=None)
    ids, mask = encoded["input_ids"][0], encoded["assistant_masks"][0]
    supervised = gpt2_tokenizer.decode([token for token, kept in zip(ids, mask, strict=True) if kept])

    assert "FINAL_ACTION" in supervised
    assert "EARLIER_ACTION" not in supervised
    assert "EARLIER_ACTION" in gpt2_tokenizer.decode(ids)


def test_write_sft_examples_appends_the_expert_action_and_drops_long_rows(tmp_path, gpt2_tokenizer_path):
    short = [{"role": "user", "content": "Fix the bug."}]
    long = [{"role": "user", "content": "Fix the bug. " + "context " * 2000}]
    rows = [_swe_row(1, short, "ls"), _swe_row(2, long, "cat a.py")]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "train.parquet")
    output = tmp_path / "out"

    write_sft_examples(
        rows_path=str(tmp_path),
        rows_filename="train.parquet",
        output_path=str(output),
        task="experiments.post_training.pivotrl.task:PIVOT_TOOL_CALL",
        tokenizer=gpt2_tokenizer_path,
        chat_template=GRUG_FINAL_TURN_TEMPLATE,
        enable_thinking=False,
        max_tokens=1024,
    )

    with gzip.open(output / EXAMPLES_FILENAME, "rt") as stream:
        (example,) = [json.loads(line) for line in stream]
    assert example["messages"][:-1] == short
    (call,) = example["messages"][-1]["tool_calls"]
    assert call["function"] == {"name": "bash", "arguments": '{"command": "ls"}'}
    # Tools are the chat-completions form the policy is served with.
    assert example["chat_template_kwargs"]["tools"][0]["function"]["name"] == "bash"
    assert example["chat_template_kwargs"]["enable_thinking"] is False
    manifest = json.loads((output / "manifest.json").read_text())
    assert (manifest["examples"], manifest["too_long"]) == (1, 1)
