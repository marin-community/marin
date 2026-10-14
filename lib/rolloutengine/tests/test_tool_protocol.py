# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from rolloutengine.tool_protocol import ToolCallStatus, classify_tool_call_continuation, tool_call_regions


@pytest.mark.parametrize(
    ("continuation", "status"),
    [
        (' {"name":"read","arguments":{"path":"a"}}\n</tool_call>', ToolCallStatus.VALID),
        (
            '{"name":"read","arguments":{"text":"<tool_call></tool_call><|start_think|>"}}</tool_call>',
            ToolCallStatus.VALID,
        ),
        ("<tool_call><tool_call>", ToolCallStatus.REPEATED_OPENER),
        (" \n<tool_call>", ToolCallStatus.REPEATED_OPENER),
        ('{"name":"read","arguments":', ToolCallStatus.MALFORMED_JSON),
        ('{"name":"read","arguments":{"n":NaN}}</tool_call>', ToolCallStatus.MALFORMED_JSON),
        ('{"name":"read","arguments":{},"name":"write"}</tool_call>', ToolCallStatus.MALFORMED_JSON),
        ('{"name":"read","arguments":"{}"}</tool_call>', ToolCallStatus.INVALID_ENVELOPE),
        ('{"name":"read","arguments":{}}', ToolCallStatus.MISSING_CLOSER),
        ('{"name":"read","arguments":{}}<tool_call>', ToolCallStatus.MISSING_CLOSER),
        ("", ToolCallStatus.MALFORMED_JSON),
    ],
)
def test_completed_continuation_distinguishes_protocol_failures(continuation, status):
    assert classify_tool_call_continuation(continuation) == status


def test_scan_ignores_thinking_and_quoted_markers_then_stops_at_first_failure():
    text = (
        "<|start_think|><tool_call><tool_call><|end_think|>"
        '<tool_call>{"name":"read","arguments":{"x":"<|start_think|><tool_call>"}}</tool_call>'
        "<tool_call><tool_call><tool_call>"
    )
    regions = tool_call_regions(text, initial_thinking=False)
    assert [region.status for region in regions] == [ToolCallStatus.VALID, ToolCallStatus.REPEATED_OPENER]
    assert text[regions[0].start : regions[0].payload_start] == "<tool_call>"
    assert text[regions[0].end : regions[1].payload_start] == "<tool_call>"


def test_scan_respects_thinking_state_inherited_from_prompt():
    text = '<tool_call><tool_call><|end_think|><tool_call>{"name":"x","arguments":{}}</tool_call>'
    regions = tool_call_regions(text, initial_thinking=True)
    assert len(regions) == 1
    assert regions[0].status == ToolCallStatus.VALID
    assert text[: regions[0].start].endswith("<|end_think|>")
