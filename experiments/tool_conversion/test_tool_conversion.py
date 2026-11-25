# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
import re
import subprocess
import sys
import tarfile

import numpy as np
import pytest
from rolloutengine.chat_template import MARIN_CHAT_TEMPLATE
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers.utils.chat_template_utils import render_jinja_template

from experiments.tool_conversion.archive import archive_bytes, commit_batch, read_json, write_immutable
from experiments.tool_conversion.convert import CampaignScopeError, Disposition, convert_trace, parse_trace
from experiments.tool_conversion.export import digest, export_conversion
from experiments.tool_conversion.run import run


def trace(
    calls, results, *, name="stateful_python_code_exec", description="Execute Python in a stateful Jupyter notebook."
):
    tool = {"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object"}}}
    turns = [("system", "<tools>\n" + json.dumps(tool) + "\n</tools>"), ("user", "Compute the result.")]
    for i, call in enumerate(calls):
        turns.append(("assistant", "<tool_call>" + json.dumps(call) + "</tool_call>"))
        if i < len(results):
            turns.append(("tool", f'<tool_response name="{call["name"]}">{results[i]}</tool_response>'))
    turns.append(("assistant", "Done."))
    return "<|begin_of_text|>" + "".join(
        f"<|start_header_id|>{role}<|end_header_id|>\n{body}<|eot_id|>" for role, body in turns
    )


def test_notebook_conversion_preserves_cells_observations_and_pairing():
    original = trace(
        [
            {"id": "first", "name": "stateful_python_code_exec", "arguments": {"code": "x = [1, 2]\nx"}},
            {"id": "second", "name": "stateful_python_code_exec", "arguments": {"code": "x.append(3)\nx"}},
        ],
        ["[1, 2]", "[1, 2, 3]"],
    )
    result = convert_trace("math", original, "fixture")
    assert result.disposition == Disposition.VERIFIED
    assert [m["content"] for m in result.messages if m["role"] == "tool"] == ["[1, 2]", "[1, 2, 3]"]
    assert [m["tool_call_id"] for m in result.messages if m["role"] == "tool"] == ["first", "second"]
    assert [m["tool_calls"][0]["id"] for m in result.messages if m.get("tool_calls")] == ["first", "second"]
    assert result.text.count("<|start_header_id|>") == original.count("<|start_header_id|>")
    commands = [c.after["arguments"]["command"] for c in result.changes if c.after is not None]
    cells = []
    for command in commands:
        match = re.search(r"<<'CELL_[^']+'\n(.*)\nCELL_[^\n]+$", command, re.S)
        assert match is not None
        cells.append(match[1])
    assert cells == ["x = [1, 2]\nx", "x.append(3)\nx"]


def test_python_stdout_command_executes_exact_quoted_code(tmp_path):
    code = "print(\"literal $HOME; `uname`; 'quoted'\")"
    original = trace(
        [{"name": "evaluate_code", "arguments": {"source_code": code}}],
        ["literal $HOME; `uname`; 'quoted'\n"],
        name="evaluate_code",
        description="Evaluates the input string as Python code and returns the stdout.",
    )
    result = convert_trace("science", original, "quote")
    assert result.disposition == Disposition.VERIFIED
    assert result.changes[0].after is not None
    command = result.changes[0].after["arguments"]["command"]
    actual = subprocess.run(["bash", "-c", command], cwd=tmp_path, capture_output=True, text=True, check=True)
    expected = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert actual.stdout == expected.stdout


@pytest.mark.parametrize(
    "source,body,disposition",
    [
        ("nemotron_sft_v3/agentic_v1/tool_calling", "not parsed", Disposition.EXCLUDED),
        ("penfever-traces/glm52-terminus2/tasks", "not parsed", Disposition.EXCLUDED),
        ("plain", "<|start_header_id|>assistant<|end_header_id|>hello<|eot_id|>", Disposition.NO_CALLS),
        (
            "agenttrove",
            "excluded even without terminal syntax",
            Disposition.EXCLUDED,
        ),
    ],
)
def test_excluded_and_tool_free_traces_are_byte_preserved(source, body, disposition):
    converted = convert_trace(source, body, "skip")
    assert (converted.disposition, converted.text, converted.changes) == (disposition, body, [])


def test_reasoning_tool_examples_are_not_actions_and_editor_is_unchanged():
    original = trace(
        [{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/tmp/x"}}],
        ["text"],
        name="str_replace_editor",
    )
    example = '<|start_think|>Example <tool_call>{"name":"fake"}</tool_call><|end_think|>'
    original = original.replace("Done.", example + "Done.")
    result = convert_trace("swe", original, "editor")
    assert (result.disposition, result.text) == (Disposition.UNCHANGED, original)


def test_timeout_and_malformed_calls_never_export_partial_conversion():
    original = trace(
        [
            {"name": "stateful_python_code_exec", "arguments": {"code": "x = 1"}},
            {"name": "stateful_python_code_exec", "arguments": {"code": "while True: pass"}},
        ],
        ["", "Timed out"],
    )
    result = convert_trace("math", original, "timeout")
    assert (result.disposition, result.text, result.changes) == (Disposition.RESIDUAL, original, [])
    malformed = original.replace('"code": "x = 1"', '"wrong_argument": "x = 1"')
    assert convert_trace("math", malformed, "bad").disposition == Disposition.RESIDUAL


@pytest.mark.parametrize(
    "text",
    [
        "<|im_start|>assistant\nNo tools here.<|im_end|>",
        trace([{"name": "unreviewed_runtime", "arguments": {}}], ["result"], name="unreviewed_runtime"),
    ],
)
def test_unreviewed_formats_and_tools_fail_instead_of_becoming_skips(text):
    with pytest.raises(CampaignScopeError, match=r"README\.md: Scope and limits"):
        convert_trace("math", text, "outside-campaign")


def test_unreviewed_selection_is_rejected_before_creating_outputs(tmp_path):
    tokenizer = Tokenizer(models.BPE())
    with pytest.raises(CampaignScopeError, match=r"selection schema.*README\.md"):
        run(
            {"sources": []},
            tokenizer,
            "unreviewed",
            str(tmp_path / "output"),
            shard_start=0,
            shard_stop=1,
            batch_size=128,
            row_limit=None,
        )
    assert not (tmp_path / "output").exists()


def test_parallel_notebook_calls_do_not_imply_cell_execution_order():
    original = trace(
        [
            {"name": "stateful_python_code_exec", "arguments": {"code": "x = 1"}},
            {"name": "stateful_python_code_exec", "arguments": {"code": "x + 1"}},
        ],
        ["", "2"],
    )
    prefix, turns = parse_trace(original)
    turns[2].body += turns[4].body
    parallel = prefix + "".join(turns[i].header + turns[i].body + turns[i].ending for i in [0, 1, 2, 3, 5, 6])
    result = convert_trace("math", parallel, "parallel")
    assert (result.disposition, result.text, result.changes) == (Disposition.RESIDUAL, parallel, [])


def test_think_becomes_reasoning_and_archives_acknowledgement():
    original = trace(
        [{"name": "think", "arguments": {"thought": "Check the sign."}}], ["Your thought has been logged."], name="think"
    )
    result = convert_trace("swe", original, "thought")
    assert result.disposition == Disposition.VERIFIED
    assert result.messages[2]["reasoning_content"] == "Check the sign."
    assert not result.messages[2]["tool_calls"]
    assert result.changes[0].removed_acknowledgement == 3
    assert not any(m["role"] == "tool" for m in result.messages)


def test_think_reasoning_precedes_existing_visible_content():
    original = trace(
        [{"name": "think", "arguments": {"thought": "Check the sign."}}], ["Your thought has been logged."], name="think"
    ).replace("<tool_call>", "Let me think.<tool_call>")
    result = convert_trace("swe", original, "think-prefix")
    assert result.disposition == Disposition.VERIFIED
    turn = parse_trace(result.text)[1][2]
    message = result.messages[2]
    rendered, _ = render_jinja_template(
        conversations=[[message]], chat_template=MARIN_CHAT_TEMPLATE, bos_token="", add_generation_prompt=False
    )
    assert rendered[0] == turn.header + turn.body + turn.ending
    assert message["content"].strip() == "Let me think."


def test_final_finish_and_submit_become_native_completion():
    for name, arguments, expected in [
        ("finish", {"message": "Unable to finish: tests failed."}, "Unable to finish: tests failed."),
        ("submit", {}, ""),
    ]:
        original = trace([{"name": name, "arguments": arguments}], [], name=name)
        original = original.removesuffix("<|start_header_id|>assistant<|end_header_id|>\nDone.<|eot_id|>")
        result = convert_trace("swe", original, name)
        assert result.disposition == Disposition.VERIFIED
        assert result.messages[-1]["content"].strip() == expected
        assert not result.messages[-1]["tool_calls"]


def test_shell_rename_preserves_command_but_holds_session_mutation():
    original = trace(
        [{"name": "execute_bash", "arguments": {"command": "printf hello"}}], ["hello"], name="execute_bash"
    )
    result = convert_trace("swe", original, "shell")
    assert result.disposition == Disposition.VERIFIED
    assert result.changes[0].after is not None
    assert result.changes[0].after["arguments"] == {"command": "printf hello"}
    for command in ["cd /workspace", "x=1", "export X=1", 'eval "cd /workspace"', 'python3 -c "print(1)" &']:
        source = original.replace("printf hello", command.replace('"', '\\"'))
        held = convert_trace("swe", source, command)
        assert (held.disposition, held.text) == (Disposition.RESIDUAL, source)


def test_conversion_preserves_uncalled_tool_declarations():
    original = trace(
        [{"name": "execute_bash", "arguments": {"command": "printf hello"}}], ["hello"], name="execute_bash"
    )
    editor = {"type": "function", "function": {"name": "str_replace_editor", "parameters": {"type": "object"}}}
    other = {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}
    original = original.replace("</tools>", json.dumps(editor) + "\n" + json.dumps(other) + "\n</tools>")
    result = convert_trace("swe", original, "unused-tools")
    assert result.disposition == Disposition.VERIFIED
    assert result.tools == [other, editor, result.tools[-1]]
    assert result.tools[-1]["function"]["name"] == "bash"
    assert json.dumps(editor) in result.text and json.dumps(other) in result.text


def test_archive_receipt_binds_payload_and_refuses_overwrite(tmp_path):

    files = {"batch/dispositions.json": b"[]", "provenance.json": b"{}"}
    packed = archive_bytes(files)
    assert packed == archive_bytes(files)
    receipt = commit_batch(str(tmp_path) + "/", "batch", files, "catalog", {"shard": 0, "row": 2}, {"files_sha256": {}})
    actual = (tmp_path / "batches/batch.tar.gz").read_bytes()
    assert receipt["archive_sha256"] == digest(actual)
    assert receipt["archive_bytes"] == len(actual)
    assert read_json(str(tmp_path / "cursor.json")) == receipt["post_commit_cursor"]
    assert read_json(str(tmp_path / "receipts/batch.json")) == receipt
    with tarfile.open(fileobj=io.BytesIO(actual), mode="r:gz") as archive:
        assert archive.extractfile("batch/dispositions.json").read() == b"[]"
    with pytest.raises(ValueError, match="Existing immutable artifact differs"):
        write_immutable(str(tmp_path / "batches/batch.tar.gz"), b"changed")
    assert (tmp_path / "batches/batch.tar.gz").read_bytes() == actual


def test_export_roundtrip_masks_only_assistant_content():
    tokenizer = Tokenizer(
        models.BPE(vocab={char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}, merges=[])
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(
        ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<tool_call>", "</tool_call>"]
    )
    original = trace(
        [{"name": "execute_bash", "arguments": {"command": "echo ASSISTANT_ONLY"}}], ["RESULT_ONLY"], name="execute_bash"
    )
    converted = convert_trace("swe", original, "mask")
    exported = export_conversion(converted, tokenizer, "fixture-tokenizer")
    ids = np.frombuffer(exported.files["input_ids.i32"], dtype="<i4")
    mask = np.frombuffer(exported.files["loss_mask.u8"], dtype="u1")
    assert tokenizer.decode(ids.tolist(), skip_special_tokens=False) == converted.text
    supervised = tokenizer.decode(ids[mask.astype(bool)].tolist(), skip_special_tokens=False)
    assert "ASSISTANT_ONLY" in supervised and "Done." in supervised
    assert "RESULT_ONLY" not in supervised and "Compute the result." not in supervised
    assert "<|start_header_id|>" not in supervised
    assert len(ids) == len(mask) == exported.manifest["tokens"]
