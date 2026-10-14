# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import BaseModel
from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.stage40_build.context import BuildFailure
from taskforge.stage40_build.outputs import WebCall
from taskforge.stage40_build.sdk import (
    ATTEMPTS,
    DIAGNOSTICS_CHARS,
    accept,
    grade_diagnostics,
    research,
    structured_until,
)
from tests.stage40_build.conftest import web_tool


class Answer(BaseModel):
    value: int


async def even(answer: Answer) -> str | None:
    return None if answer.value % 2 == 0 else f"{answer.value} is odd"


async def test_structured_until_sends_each_problem_back_until_a_submission_passes(context, fake_glm):
    fake_glm.stream(content=json.dumps({"value": 41}))
    fake_glm.stream(content=json.dumps({"value": 42}))

    answer = await structured_until(context(), [{"role": "user", "content": "Pick an even number."}], Answer, even)

    assert answer == Answer(value=42)
    retry = fake_glm.requests[1]["messages"]
    assert retry[-3] == {"role": "assistant", "content": '{"value":41}'}
    assert "41 is odd" in retry[-2]["content"]


async def delegated(answer: Answer) -> str | None:
    return accept(answer)  # type: ignore[return-value] - the coroutine a program hands back without awaiting it


async def test_a_check_that_returns_another_checks_coroutine_is_awaited(context, fake_glm):
    fake_glm.stream(content=json.dumps({"value": 41}))

    answer = await structured_until(context(), [{"role": "user", "content": "Pick a number."}], Answer, delegated)

    assert answer == Answer(value=41)
    assert len(fake_glm.requests) == 1


async def test_structured_until_fails_the_build_after_its_attempts(context, fake_glm):
    for value in range(1, 2 * ATTEMPTS, 2):
        fake_glm.stream(content=json.dumps({"value": value}))

    with pytest.raises(BuildFailure, match=f"after {ATTEMPTS} requests; last problem: {2 * ATTEMPTS - 1} is odd"):
        await structured_until(context(), [{"role": "user", "content": "Pick an even number."}], Answer, even)


async def test_research_without_a_web_call_fails_the_build(context, proposal, fake_glm):
    fake_glm.stream(content="6 x 7 is 42, as everyone knows.")

    with pytest.raises(BuildFailure, match=r"research\[0\]: no web_search or web_fetch call"):
        await research(context(web_tools=(web_tool([]),)), proposal, 0)


async def test_research_with_only_error_results_fails_the_build(context, proposal, fake_glm):
    fake_glm.stream(tool_calls=(("web_search", '{"query": "6 times 7"}'),), finish="tool_calls")
    fake_glm.stream(content="The search failed; 6 x 7 is 42.")
    failing = web_tool([], "error: Parallel rejected the request (HTTP 422): bad query")

    with pytest.raises(BuildFailure, match=r"research\[0\]: no web_search or web_fetch call"):
        await research(context(web_tools=(failing,)), proposal, 0)


async def test_research_returns_the_finding_under_the_agent_limits(context, proposal, fake_glm):
    output = "6 x 7 = 42 (https://example.org/table)"
    fake_glm.stream(tool_calls=(("web_search", '{"query": "6 times 7"}'),), finish="tool_calls")
    fake_glm.stream(content="Confirmed 6 x 7 = 42 at https://example.org/table.")
    researching = context(web_tools=(web_tool([], output),))

    finding = await research(researching, proposal, 0)

    assert finding.web_calls == (
        WebCall(tool="web_search", arguments='{"query": "6 times 7"}', output_chars=len(output)),
    )
    assert finding.notes == "Confirmed 6 x 7 = 42 at https://example.org/table."
    assert "confirm the multiplication table" in fake_glm.requests[0]["messages"][0]["content"]
    assert {request["max_tokens"] for request in fake_glm.requests} == {researching.services.agent_limits.max_tokens}


def test_grade_diagnostics_shows_the_checks_a_grader_wrote_in_its_detail():
    detail = {"checks": [{"name": "reference query", "passed": False, "weight": 1.0}], "reason": "not authorized"}
    grade = GradeResult(Outcome.GRADED, 0.0, detail=detail, diagnostics={"stdout": "", "stderr": "", "exit_code": 0})

    assert json.loads(grade_diagnostics(grade)) == {
        "error": None,
        "detail": detail,
        "stdout": "",
        "stderr": "",
        "exit_code": 0,
    }


def test_grade_diagnostics_cuts_a_long_text_in_the_middle():
    detail = {"checks": [{"name": "total", "passed": False, "weight": 1.0}]}
    diagnostics = {"stdout": "x" * (3 * DIAGNOSTICS_CHARS), "stderr": "ValueError: bad row 7", "exit_code": 1}

    text = grade_diagnostics(GradeResult(Outcome.GRADED, 0.0, detail=detail, diagnostics=diagnostics))

    assert '"name": "total", "passed": false' in text
    assert "ValueError: bad row 7" in text
    assert "characters cut]" in text
    assert len(text) < DIAGNOSTICS_CHARS + 50
