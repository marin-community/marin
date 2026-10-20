# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structure continuation grading preserves the historical region reward."""

import pytest
from verifyit.candidate import grade_candidate
from verifyit.closure_scoring import ClosureFormat
from verifyit.grade import Status
from verifyit.spec import StructureClosureSpec, parse_spec, render_spec


@pytest.mark.parametrize(
    "structure_format,prefix,expected,correct,changed",
    [
        (ClosureFormat.JSON, '{"a":', '{"a":1}', "1}", "2}"),
        (
            ClosureFormat.TOOL,
            '<tool_call>\n{"a":',
            '<tool_call>\n{"a":1}</tool_call>',
            "1}</tool_call>",
            "2}</tool_call>",
        ),
        (
            ClosureFormat.REASONING,
            "<|start_think|>A",
            "<|start_think|>Answer.<|end_think|>",
            "nswer.<|end_think|>",
            "nother.<|end_think|>",
        ),
        (ClosureFormat.CODE_FENCE, '```json\n{"a":', '```json\n{"a":1}\n```', "1}\n```", "2}\n```"),
    ],
)
def test_structure_closure_roundtrip_preserves_content_and_allows_trailing_output(
    structure_format, prefix, expected, correct, changed
):
    spec = StructureClosureSpec(prefix, expected, structure_format, ("<|eot_id|>", "<|end_of_text|>"))
    restored = parse_spec(render_spec(spec))
    good = grade_candidate(restored, correct, {})
    trailing = grade_candidate(restored, correct + "\nextra text", {})
    assert (good.status, good.reward, good.detail["whole_response_success"]) == (Status.SCORED, 1.0, True)
    assert (trailing.reward, trailing.detail["trailing_content"], trailing.detail["whole_response_success"]) == (
        1.0,
        True,
        False,
    )
    assert grade_candidate(restored, changed, {}).reward == 0.0
    assert grade_candidate(restored, "", {}).reward == 0.0


def test_structure_closure_rejects_premature_completion_and_invalid_references():
    spec = StructureClosureSpec('{"values":[1,', '{"values":[1,2]}', ClosureFormat.JSON, ())
    assert grade_candidate(spec, "2]}", {}).reward == 1.0
    assert grade_candidate(spec, "]}", {}).reward == 0.0
    malformed = StructureClosureSpec('{"values":[', '{"values":[', ClosureFormat.JSON, ())
    assert grade_candidate(malformed, "1]}", {}).status == Status.INVALID_TASK
