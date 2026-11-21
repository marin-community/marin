# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.chat_normalize import normalize_chat_to_parquet, validate_chat_messages, validate_tool_definitions
from marin.datakit.chat_render import render_chat_record
from marin.datakit.download.nanoswe_lightning import (
    SOURCE_CHAT_SCHEMA,
    Split,
    row_to_chat_doc,
    row_to_doc,
    transform,
    transform_chat,
)
from marin.datakit.normalize import normalize_to_parquet
from openai_harmony import Message


def _call(name: object, arguments: object) -> dict:
    return {"type": "python", "text": json.dumps({"name": name, "arguments": arguments})}


def _assistant(*parts: dict) -> dict:
    return {"role": "assistant", "content": "", "parts": list(parts)}


def _row(
    messages: list[dict], *, instance_id: object = "owner__repo.abc123.func_pm__x", verified: object = True
) -> dict:
    return {
        "messages": messages,
        "patch": "",
        "model": "nemotron-3.5-lightning",
        "instance_id": instance_id,
        "verified": verified,
        "origin": "swe-smith-v3-lightning",
        "traj_hash": "0f",
        "exit_status": "Submitted",
        "n_toolcalls": 2,
    }


ISSUE = {"role": "user", "content": "ISSUE_SENTINEL: the parser drops the last token.", "parts": None}
OBSERVATION = {
    "role": "user",
    "content": "<returncode>0</returncode>\n<output>\nOBSERVATION_SENTINEL\n</output>",
    "parts": None,
}
EDIT = {"command": "str_replace", "path": "/testbed/EDIT_SENTINEL.py", "old_str": "a\n  new_str: b", "new_str": "c"}


def _trajectory(edit: dict = EDIT) -> list[dict]:
    return [
        ISSUE,
        _assistant(
            {"type": "text", "text": "REASONING_SENTINEL"},
            _call("bash", {"command": "grep -rn parse /testbed", "timeout": 30}),
        ),
        OBSERVATION,
        _assistant(_call("file_editor", edit)),
    ]


def test_chat_doc_declares_harness_tools_and_keeps_grading_metadata():
    [doc] = row_to_chat_doc(_row(_trajectory()))
    messages = [Message.from_dict(message) for message in doc["messages"]]
    tools = json.loads(doc["chat_template_kwargs"])["tools"]
    validate_chat_messages(messages)
    validate_tool_definitions(tools, messages)
    rendered = render_chat_record(doc)["text"]

    assert [tool["function"]["name"] for tool in tools] == ["bash", "file_editor"]
    assert doc["instance_id"] == "owner__repo.abc123.func_pm__x"
    assert doc["verified"] is True
    assert rendered.index("ISSUE_SENTINEL") < rendered.index("REASONING_SENTINEL") < rendered.index("grep -rn parse")
    assert rendered.index("grep -rn parse") < rendered.index("OBSERVATION_SENTINEL") < rendered.index("EDIT_SENTINEL")
    assert "timeout" not in rendered


def test_pretraining_text_hides_the_outcome_and_keeps_edit_argument_boundaries():
    [solved] = row_to_doc(_row(_trajectory(), verified=True))
    [failed] = row_to_doc(_row(_trajectory(), verified=False))
    shifted_edit = {**EDIT, "old_str": "a", "new_str": "b\n  new_str: c"}
    [shifted] = row_to_doc(_row(_trajectory(shifted_edit)))

    assert solved["text"] == failed["text"]
    assert (solved["verified"], failed["verified"]) == (True, False)
    assert json.dumps(EDIT) in solved["text"]
    assert shifted["text"] != solved["text"]
    assert '{"command": "grep -rn parse /testbed"}' in solved["text"]


@pytest.mark.parametrize(
    "row",
    [
        _row(_trajectory(), instance_id=None),
        _row(_trajectory(), verified="false"),
        _row([ISSUE, _assistant(_call(["bash"], {"command": "ls"}))]),
        _row([ISSUE, _assistant(_call("submit", {}))]),
        _row([ISSUE, _assistant(_call("file_editor", {**EDIT, "old_func": "f"}))]),
        _row([ISSUE, _assistant(_call("file_editor", {**EDIT, "old_str": 42}))]),
        _row([ISSUE, _assistant(_call("file_editor", {**EDIT, "command": "insert", "insert_line": -1}))]),
        _row([ISSUE, _assistant(_call("file_editor", {"command": "view", "path": "/testbed", "view_range": [1]}))]),
        _row([ISSUE, _assistant(_call("bash", {"command": "ls"}), _call("bash", {"command": "pwd"}))]),
        _row(_trajectory()[:3]),
    ],
    ids=[
        "missing-task",
        "string-grade",
        "list-tool-name",
        "unknown-tool",
        "argument-outside-harness-schema",
        "argument-of-the-wrong-type",
        "argument-below-its-minimum",
        "range-of-the-wrong-length",
        "two-calls-in-one-turn",
        "ends-on-tool-result",
    ],
)
def test_rows_the_harness_or_the_experiment_cannot_use_are_dropped(row: dict):
    assert row_to_doc(row) == []
    assert row_to_chat_doc(row) == []


def test_json_string_arguments_are_decoded():
    row = _row([ISSUE, _assistant(_call("bash", json.dumps({"command": "ls /testbed"})))])

    [doc] = row_to_doc(row)
    [chat] = row_to_chat_doc(row)

    assert '{"command": "ls /testbed"}' in doc["text"]
    assert "ls /testbed" in render_chat_record(chat)["text"]


def test_split_transforms_keep_tasks_apart_and_persist_grading_metadata(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "val").mkdir(parents=True)
    other_task = [{**ISSUE, "content": "A different issue."}, *_trajectory()[1:]]
    train_rows = [
        _row(_trajectory(), instance_id="o__r.c.train_a"),
        _row(other_task, instance_id="o__r.c.train_b", verified=False),
    ]
    pq.write_table(pa.Table.from_pylist(train_rows), raw / "train-00000-of-00001.parquet")
    pq.write_table(
        pa.Table.from_pylist([_row(_trajectory(), instance_id="o__r.c.val_a")]), raw / "val" / "val-00000.parquet"
    )

    persisted: dict[tuple[str, Split], list[dict]] = {}
    for split in Split:
        text_dir, chat_dir = tmp_path / f"text-{split}", tmp_path / f"chat-{split}"
        transform(str(raw), str(text_dir), split)
        normalize_to_parquet(input_path=str(text_dir), output_path=str(tmp_path / f"text-norm-{split}"))
        transform_chat(str(raw), str(chat_dir), split)
        normalize_chat_to_parquet(
            input_path=str(chat_dir), output_path=str(tmp_path / f"chat-norm-{split}"), output_schema=SOURCE_CHAT_SCHEMA
        )
        for kind in ("text", "chat"):
            outputs = sorted((tmp_path / f"{kind}-norm-{split}" / "outputs" / "main").glob("*.parquet"))
            persisted[(kind, split)] = [row for path in outputs for row in pq.read_table(path).to_pylist()]

    for kind in ("text", "chat"):
        train = {row["instance_id"]: row["verified"] for row in persisted[(kind, Split.TRAIN)]}
        validation = {row["instance_id"]: row["verified"] for row in persisted[(kind, Split.VALIDATION)]}
        assert train == {"o__r.c.train_a": True, "o__r.c.train_b": False}
        assert validation == {"o__r.c.val_a": True}
    tools = json.loads(persisted[("chat", Split.TRAIN)][0]["chat_template_kwargs"])["tools"]
    assert tools[0]["function"]["parameters"]["properties"] == {"command": {"type": "string"}}
