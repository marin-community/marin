# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from rolloutengine.structure_closure import (
    PrefixStatus,
    delimited_json_closure_state,
    json_closure_state,
    paired_marker_closure_state,
    required_code_fence_state,
)


@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, -0.125e-12, [], {}, {"a": [1, {"b": 'quoted " } ] \\ \n café'}]}, ["<tool_call>", "```"]],
)
def test_all_prefixes_of_valid_json_are_extendable_and_repairs_parse(value):
    text = json.dumps(value, ensure_ascii=True)
    for length in range(len(text) + 1):
        prefix = text[:length]
        state = json_closure_state(prefix)
        assert state.status != PrefixStatus.INVALID, (prefix, state)
        if state.closing_suffix is not None:
            json.loads(prefix + state.closing_suffix)
    assert json_closure_state(text).status == PrefixStatus.COMPLETE


@pytest.mark.parametrize("prefix", ['{"a":', '{"a": [1,', '{"key', "[tru", "[1e-", '["abc\\', '["\\u12'])
def test_missing_content_never_becomes_a_fabricated_closure_target(prefix):
    state = json_closure_state(prefix)
    assert state.status == PrefixStatus.EXTENDABLE
    assert state.closing_suffix is None


@pytest.mark.parametrize(
    "prefix,suffix,opened",
    [('{"a":[{"b":1', "}]}", ("{", "[", "{")), ('["literal } ]', '"]', ("[", '"')), ('{"a":[]', "}", ("{",))],
)
def test_nested_closure_order_and_literal_punctuation(prefix, suffix, opened):
    state = json_closure_state(prefix)
    assert state.status == PrefixStatus.EXTENDABLE
    assert state.open_structures == opened
    assert state.closing_suffix == suffix
    assert json_closure_state(prefix + suffix).status == PrefixStatus.COMPLETE


@pytest.mark.parametrize(
    "prefix,offset",
    [(" {]", 2), ("[1}", 2), ("[1,]", 3), ('{"a":1,}', 7), ("01", 1), ("1.e", 2), ('["\\x', 3), ("true false", 5)],
)
def test_first_illegal_transition_is_localized(prefix, offset):
    state = json_closure_state(prefix)
    assert state.status == PrefixStatus.INVALID
    assert state.error_offset == offset
    assert json_closure_state(prefix[:offset]).status != PrefixStatus.INVALID
    assert state.closing_suffix is None


def test_duplicate_keys_include_escaped_equivalents():
    prefix = '{"a":1,"\\u0061"'
    state = json_closure_state(prefix)
    assert state.status == PrefixStatus.INVALID
    assert json_closure_state('{"a":1,"b":2}').status == PrefixStatus.COMPLETE


def test_tool_markers_inside_json_strings_do_not_close_or_reopen_the_call():
    text = '<tool_call>{"name":"shell","arguments":{"command":"echo </tool_call> <tool_call>"}}</tool_call>'
    for length in range(len(text) + 1):
        prefix = text[:length]
        state = delimited_json_closure_state(prefix, opener="<tool_call>", closer="</tool_call>")
        assert state.status != PrefixStatus.INVALID, (prefix, state)
        if state.closing_suffix is not None:
            repaired = prefix + state.closing_suffix
            assert (
                delimited_json_closure_state(repaired, opener="<tool_call>", closer="</tool_call>").status
                == PrefixStatus.COMPLETE
            )
    assert (
        delimited_json_closure_state(text, opener="<tool_call>", closer="</tool_call>").status == PrefixStatus.COMPLETE
    )


@pytest.mark.parametrize("payload", ['{"a":', '["\\u12', '{"a":1,'])
def test_tool_wrapper_does_not_repair_missing_json_content(payload):
    state = delimited_json_closure_state("<tool_call>" + payload, opener="<tool_call>", closer="</tool_call>")
    assert state.status == PrefixStatus.EXTENDABLE
    assert state.closing_suffix is None


def test_tool_wrapper_rejects_wrong_closer_and_nested_opener():
    for text in ["<tool_call>{}</wrong>", "<tool_call><tool_call>", "<tool_call>{}</tool_call>  x"]:
        state = delimited_json_closure_state(text, opener="<tool_call>", closer="</tool_call>")
        assert state.status == PrefixStatus.INVALID
        if text.endswith("  x"):
            assert state.error_offset == len(text) - 1
        assert (
            delimited_json_closure_state(text[: state.error_offset], opener="<tool_call>", closer="</tool_call>").status
            != PrefixStatus.INVALID
        )


def test_reasoning_marker_partial_closure_and_repeated_opener():
    opener, closer = "<|start_think|>", "<|end_think|>"
    prefix = opener + "Reasoning with literal { braces and ``` punctuation. <|end_th"
    state = paired_marker_closure_state(prefix, opener=opener, closer=closer)
    assert state.closing_suffix == "ink|>"
    assert (
        paired_marker_closure_state(prefix + state.closing_suffix, opener=opener, closer=closer).status
        == PrefixStatus.COMPLETE
    )
    assert (
        paired_marker_closure_state(opener + "text" + opener, opener=opener, closer=closer).status
        == PrefixStatus.INVALID
    )


@pytest.mark.parametrize("text", ['```python\nx = "```"\n```', "~~~~\n```\n~~~~~", "  ```json\n{}\n ```\n"])
def test_required_code_fences_respect_lines_and_opening_length(text):
    assert required_code_fence_state(text).status == PrefixStatus.COMPLETE


@pytest.mark.parametrize("prefix", ["```python\nprint(1)", "````\n```", "~~~\ntext\n~~", "```\n"])
def test_required_fence_repairs_only_append_closing_syntax(prefix):
    state = required_code_fence_state(prefix)
    assert state.status == PrefixStatus.EXTENDABLE
    assert state.closing_suffix is not None
    assert required_code_fence_state(prefix + state.closing_suffix).status == PrefixStatus.COMPLETE
