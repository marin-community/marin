# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from rolloutengine.closure_scoring import (
    ClosureFormat,
    ContinuationOutcome,
    closure_response_content,
    continuation_outcome,
    score_closure,
    score_closure_region,
)


@pytest.mark.parametrize(
    "structure_format,expected,early,changed",
    [
        (ClosureFormat.JSON, '{"task":"example_1234"}', '{"task":"example"}', '{"task":"wrong"}'),
        (ClosureFormat.JSON, '["example",2]', '["example"]', '["example",3]'),
        (ClosureFormat.JSON, '"example_1234"', '"example"', '"example_wrong"'),
        (
            ClosureFormat.TOOL,
            '<tool_call>{"name":"example"}</tool_call>',
            '<tool_call>{"name":"exam"}</tool_call>',
            '<tool_call>{"name":"wrong"}</tool_call>',
        ),
        (
            ClosureFormat.REASONING,
            "<|start_think|>The task is example.<|end_think|>",
            "<|start_think|>The task is <|end_think|>",
            "<|start_think|>The task is wrong.<|end_think|>",
        ),
        (ClosureFormat.CODE_FENCE, "```json\n[1,2]\n```", "```json\n[1\n```", "```json\nwrong\n```"),
    ],
)
def test_continuation_controls_distinguish_verified_early_closure_from_other_content_changes(
    structure_format, expected, early, changed
):
    assert continuation_outcome(expected, expected, structure_format) == ContinuationOutcome.REFERENCE_COMPLETED
    assert (
        continuation_outcome(early + "\nextra output", expected, structure_format)
        == ContinuationOutcome.REFERENCE_PREFIX_CLOSED_EARLY
    )
    assert continuation_outcome(changed, expected, structure_format) == ContinuationOutcome.OTHER_CONTENT_CHANGE


def test_continuation_control_does_not_call_an_invalid_escape_or_unfinished_value_premature_closure():
    expected = '{"task":"example"}'
    for candidate in ['{"task":"exam', '{"task":', '{"task":"\\x"}']:
        assert continuation_outcome(candidate, expected, ClosureFormat.JSON) == ContinuationOutcome.UNCLOSED_OR_INVALID
    assert (
        continuation_outcome('{ "task": "exam"}', expected, ClosureFormat.JSON)
        == ContinuationOutcome.OTHER_CONTENT_CHANGE
    )


@pytest.mark.parametrize(
    "structure_format,value",
    [
        (ClosureFormat.JSON, '{"a":[{"b":"} ]"}]}'),
        (ClosureFormat.JSON, '"escaped \\" and \\\\"'),
        (ClosureFormat.TOOL, '<tool_call>{"value":"</tool_call>"}</tool_call>'),
        (ClosureFormat.REASONING, "<|start_think|>done<|end_think|>"),
        (ClosureFormat.CODE_FENCE, '```json\n{"a":1}\n```'),
    ],
)
def test_region_closure_separates_later_output_from_outer_completion(structure_format, value):
    for tail in ("", "\n", "\nMore output", "\n" + value):
        score = score_closure_region(value + tail, value, structure_format)
        assert score.success
        assert score.syntax_complete
        assert score.content_preserved
        assert score.trailing_content == bool(tail.strip())
        assert score.whole_response_success == (not tail.strip())


@pytest.mark.parametrize(
    "structure_format,value,expected,syntax",
    [
        (ClosureFormat.JSON, '{"a":[]}', '{"a":[1]}', True),
        (ClosureFormat.JSON, '{"a":[1}', '{"a":[1]}', False),
        (ClosureFormat.JSON, '{"a":[1}\n{"a":[1]}', '{"a":[1]}', False),
        (ClosureFormat.JSON, '"bad\\q"', '"good"', False),
        (ClosureFormat.JSON, '"unfinished\\', '"finished"', False),
        (ClosureFormat.TOOL, '<tool_call>{"a":1}</tool_', '<tool_call>{"a":1}</tool_call>', False),
        (
            ClosureFormat.TOOL,
            '<tool_call><tool_call>{"a":1}</tool_call>',
            '<tool_call>{"a":1}</tool_call>',
            False,
        ),
        (
            ClosureFormat.REASONING,
            "<|start_think|><|start_think|>x<|end_think|>",
            "<|start_think|>x<|end_think|>",
            False,
        ),
        (ClosureFormat.CODE_FENCE, "```json\n{}\n```not-a-closing-line", "```json\n{}\n```", False),
    ],
)
def test_region_closure_does_not_repair_invalid_or_premature_structure(structure_format, value, expected, syntax):
    score = score_closure_region(value, expected, structure_format)
    assert not score.success
    assert score.syntax_complete == syntax
    assert not score.content_preserved


@pytest.mark.parametrize("marker", ["<|eot_id|>", "<|end_of_text|>"])
@pytest.mark.parametrize(
    "value,structure_format",
    [
        ('{"a":[1]}', ClosureFormat.JSON),
        ('<tool_call>{"name":"shell","arguments":{}}</tool_call>', ClosureFormat.TOOL),
        ("<|start_think|>done<|end_think|>", ClosureFormat.REASONING),
        ("```json\n{}\n```", ClosureFormat.CODE_FENCE),
    ],
)
def test_completed_structure_accepts_one_normal_turn_terminator(value, structure_format, marker):
    content = closure_response_content(value + marker, stop_reason="stop")
    assert score_closure(content, value, structure_format).success
    repeated = closure_response_content(value + marker + marker, stop_reason="stop")
    assert not score_closure(repeated, value, structure_format).success
    truncated = closure_response_content(value + marker, stop_reason="length")
    assert not score_closure(truncated, value, structure_format).success


@pytest.mark.parametrize("response", ['{"a":[1<|eot_id|>', '{"a":"<|eot_id|>"}<|eot_id|>'])
def test_turn_terminator_does_not_supply_missing_closers_or_erase_content(response):
    content = closure_response_content(response, stop_reason="stop")
    assert not score_closure(content, '{"a":[1]}', ClosureFormat.JSON).success
    if '"<|eot_id|>"' in response:
        assert score_closure(content, '{"a":"<|eot_id|>"}', ClosureFormat.JSON).success
