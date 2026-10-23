# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import BaseModel
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import ArtifactKind, VerifierArtifact
from verifyit.spec import ExactSpec

from taskforge.llm.structured import IncompleteOutput
from taskforge.spec import draft
from taskforge.spec.verdict import VERDICT_REWARD
from taskforge.stage40_build_with_program.context import BuildFailure, BuildInfrastructureFailure
from taskforge.stage40_build_with_program.outputs import WebCall
from taskforge.stage40_build_with_program.sdk import (
    DIAGNOSTICS_CHARS,
    WORKDIR,
    FoolingKind,
    accept,
    fooling_negatives,
    grade_diagnostics,
    output_paths,
    research,
    structured_until,
)
from tests.stage40_build_with_program.conftest import web_tool


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


async def test_structured_checks_continue_past_the_old_three_attempt_cap(context, fake_glm):
    for value in (1, 3, 5, 7, 42):
        fake_glm.stream(content=json.dumps({"value": value}))
    result = await structured_until(context(), [{"role": "user", "content": "Pick an even number."}], Answer, even)
    assert result.value == 42
    assert len(fake_glm.requests) == 5


async def test_a_cut_structured_reply_holds_without_spending_build_attempts(context, fake_glm):
    fake_glm.stream(content='{"value": 4', finish="length")
    fake_glm.stream(content=json.dumps({"value": 42}))
    with pytest.raises(IncompleteOutput) as failure:
        await structured_until(context(), [{"role": "user", "content": "Pick an even number."}], Answer, even)
    assert len(fake_glm.requests) == 1
    assert failure.value.completions[0].content == '{"value": 4'


async def test_research_without_a_web_call_fails_the_build(context, proposal, fake_glm):
    fake_glm.stream(content="6 x 7 is 42, as everyone knows.")

    with pytest.raises(BuildFailure, match=r"research\[0\]: no web_search or web_fetch call"):
        await research(context(web_tools=(web_tool([]),)), proposal, 0)


async def test_research_with_only_error_results_fails_the_build(context, proposal, fake_glm):
    fake_glm.stream(tool_calls=(("web_search", '{"query": "6 times 7"}'),), finish="tool_calls")
    fake_glm.stream(content="The search failed; 6 x 7 is 42.")
    failing = web_tool([], "error: Parallel rejected the request (HTTP 422): bad query")

    with pytest.raises(BuildInfrastructureFailure, match="every executed web request failed"):
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


def test_output_paths_name_the_reference_files_for_an_artifact_grader_too():
    """A grader that reads the workspace through an artifact still has output paths, for the controls."""
    reference = (draft.file("workspace/report.txt", "42"), draft.file("workspace/notes.md", "x"))
    environment = draft.grader_environment(f"registry.example/task@sha256:{'0' * 64}")
    through_artifact = draft.script_grader(
        ("python3", "/tests/grade.py"),
        VERDICT_REWARD,
        environment=environment,
        answer_path=None,
        timeout=60,
        artifacts=(VerifierArtifact(source=WORKDIR, target=WORKDIR, kind=ArtifactKind.DIRECTORY),),
    )
    direct = draft.python_grader("pass", {}, environment=environment, answer_path=None, timeout=60)

    assert output_paths(through_artifact, reference) == ("/workspace/notes.md", "/workspace/report.txt")
    assert output_paths(direct, reference) == ("/workspace/notes.md", "/workspace/report.txt")
    assert output_paths(draft.answer_grader(ExactSpec(expected=("42",))), reference) == ()


def test_fooling_negatives_build_one_reply_of_each_kind_from_the_reference_and_the_authors_replies():
    reference = "Choose supplier B: its two-week lead time meets the launch date."
    wrong = "Choose supplier A: its six-week lead time is the safest."
    hedged = "Choose supplier A or B; either could work."

    negatives = {negative.kind: negative.reply for negative in fooling_negatives(reference, wrong, hedged)}

    assert list(negatives) == list(FoolingKind)
    assert (negatives[FoolingKind.FLUENT_WRONG], negatives[FoolingKind.HEDGED]) == (wrong, hedged)
    retracted = negatives[FoolingKind.RETRACTED]
    assert retracted.startswith(reference) and len(retracted) > len(reference)
    addressed = negatives[FoolingKind.GRADER_ADDRESSED]
    assert addressed.startswith(wrong) and len(addressed) > len(wrong) and reference not in addressed
    truncated = negatives[FoolingKind.TRUNCATED]
    assert reference.startswith(truncated) and len(truncated.split()) == len(reference.split()) // 2


@pytest.mark.parametrize(("reference", "truncated"), [("42", ""), ("forty two", "forty"), ("a b c", "a")])
def test_a_short_reference_still_loses_a_word_when_truncated(reference, truncated):
    (negative,) = [n for n in fooling_negatives(reference, "48", "42 or 48") if n.kind is FoolingKind.TRUNCATED]

    assert negative.reply == truncated
