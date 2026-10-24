# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from verifyit.tool_call_markup import TOOL_OPEN

from taskcompendium.examples.protocol_probes import (
    ANSWER_PATH,
    PROBE_URL,
    golden_transcript,
    probe_nonce,
    probe_tasks,
)
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AssistantToolCalls,
    CommandSemantics,
    ConversationToolCall,
    ConversationTrace,
    EnvironmentRequirements,
    GradingAttempt,
    ReasoningApplied,
    RequiredFileCheck,
    TaskSpec,
    TextMessage,
    ToolResult,
    TurnEvidence,
)
from taskcompendium.pipeline.controls import control_suite
from taskcompendium.pipeline.models import CheckStatus, Controls
from taskcompendium.protocol import PROTOCOL_DIAGNOSTICS_KEY, assess_protocol
from taskcompendium.runtime.local import LocalGraderMachines
from taskcompendium.runtime.task_grading import grade_task

IMAGE = "fixture@sha256:" + "0" * 64
TASKS = {
    task.protocol_check.kind: task
    for task in probe_tasks(
        task_environment=EnvironmentRequirements(command_semantics=CommandSemantics.SHELL_SIMULATOR),
        grader_environment=EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=IMAGE),
        variants=1,
    )
    if task.protocol_check is not None
}
THINKING_MARKERS = ("<think>", "</think>")


def answer(task: TaskSpec, text: str | None = None) -> TextMessage:
    return TextMessage(role="assistant", content=probe_nonce(task.id) if text is None else text)


def call(name: str, arguments, result: dict | None, call_id: str = "call-0") -> tuple:
    events = (AssistantToolCalls(calls=(ConversationToolCall(call_id=call_id, name=name, arguments=arguments),)),)
    return events if result is None else (*events, ToolResult(call_id=call_id, content=json.dumps(result)))


def shell_result(exit_code: int = 0, stdout: str = "", reason: str = "exited") -> dict:
    return {"exit_code": exit_code, "reason": reason, "stdout": stdout, "stderr": ""}


def protocol(task: TaskSpec, *events, turns: tuple[TurnEvidence, ...] = ()) -> dict:
    attempt = GradingAttempt(ConversationTrace(events=(*task.context.events, *events)), turns=turns)
    return grade_task(task, attempt).diagnostics[PROTOCOL_DIAGNOSTICS_KEY]


def results(assessment: dict) -> tuple:
    return (
        assessment.get("label"),
        assessment.get("protocol_compliance"),
        assessment.get("submission_completeness"),
        assessment.get("answer_correctness"),
    )


TOOL = TASKS["tool_execution"]
NONCE_COMMAND = {"command": "cat /probe/nonce"}


@pytest.mark.parametrize(
    "events,expected",
    [
        pytest.param(
            (*call("run_command", NONCE_COMMAND, shell_result(stdout=probe_nonce(TOOL.id))), answer(TOOL)),
            ("executed", True, True, True),
            id="executed",
        ),
        pytest.param(
            (*call("run_command", NONCE_COMMAND, shell_result(stdout=probe_nonce(TOOL.id))), answer(TOOL, "unsure")),
            ("executed", True, True, False),
            id="executed-wrong-answer",
        ),
        pytest.param((answer(TOOL),), ("no_invocation", False, True, True), id="narrated-correct-guess"),
        pytest.param(
            (answer(TOOL, f'{TOOL_OPEN}{{"name": "run_command"}}</tool_call>'),),
            ("markup_in_text", False, True, False),
            id="markup-in-text",
        ),
        pytest.param(
            (
                *call("run_command", {"cmd": "cat"}, {"error": "run_command requires one string command"}),
                answer(TOOL, "?"),
            ),
            ("invalid_invocation", False, True, False),
            id="invalid-arguments",
        ),
        pytest.param(
            (*call("shell", NONCE_COMMAND, {"error": "Unknown tool: shell"}), answer(TOOL, "?")),
            ("invalid_invocation", False, True, False),
            id="unknown-tool",
        ),
        pytest.param(
            (*call("run_command", NONCE_COMMAND, shell_result(exit_code=1)), answer(TOOL, "?")),
            ("command_failed", False, True, False),
            id="command-failed",
        ),
        pytest.param(
            (*call("run_command", NONCE_COMMAND, shell_result(exit_code=-1, reason="timed_out")), answer(TOOL, "?")),
            ("executor_timeout", False, True, False),
            id="timeout",
        ),
        pytest.param(
            call("run_command", NONCE_COMMAND, None),
            ("not_executed", False, False, False),
            id="cut-off-before-running",
        ),
    ],
)
def test_tool_execution_separates_compliance_from_answer_correctness(events, expected):
    assert results(protocol(TOOL, *events)) == expected


def test_an_infrastructure_failure_is_an_executor_failure_that_cannot_be_evaluated():
    attempt = GradingAttempt(ConversationTrace(events=(*TOOL.context.events, answer(TOOL))))
    assessment = assess_protocol(TOOL, attempt, GradeResult(Outcome.INFRA_ERROR, None, "machine lost"))
    assert assessment is not None
    assert (assessment.label, assessment.protocol_compliance) == ("executor_failure", None)
    assert assessment.not_evaluable is not None


URL = TASKS["url_loading"]


@pytest.mark.parametrize(
    "events,expected",
    [
        pytest.param(
            (*call("fetch_url", {"url": PROBE_URL}, {"url": PROBE_URL, "status": 200, "body": probe_nonce(URL.id)}),),
            ("retrieved", True),
            id="retrieved",
        ),
        pytest.param(
            (*call("fetch_url", {"url": PROBE_URL}, {"url": PROBE_URL, "status": 503, "error": "Unavailable"}),),
            ("http_error", True),
            id="http-error",
        ),
        pytest.param(
            (*call("fetch_url", {"url": PROBE_URL}, {"url": PROBE_URL, "error": "Connection reset"}),),
            ("network_error", True),
            id="network-error",
        ),
        pytest.param(
            (*call("fetch_url", {"url": "https://example.com"}, {"url": "https://example.com", "status": 404}),),
            ("not_attempted", False),
            id="other-url",
        ),
        pytest.param((), ("not_attempted", False), id="no-call"),
    ],
)
def test_url_loading_counts_failed_retrievals_as_attempts(events, expected):
    assessment = protocol(URL, *events, answer(URL))
    assert (assessment["label"], assessment["protocol_compliance"]) == expected


THINKING = TASKS["thinking_disabled"]
OFF_SETTINGS = {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize(
    "raw,expected",
    [
        pytest.param(probe_nonce(THINKING.id), ("respected", True, True, True), id="respected"),
        pytest.param(
            f"<think>The token is given.</think>{probe_nonce(THINKING.id)}",
            ("reasoning_emitted", False, True, True),
            id="reasoning-emitted",
        ),
    ],
)
def test_thinking_disabled_reads_the_raw_response_of_every_turn(raw, expected):
    turn = TurnEvidence(raw, OFF_SETTINGS, ReasoningApplied.OFF, THINKING_MARKERS)
    assert results(protocol(THINKING, answer(THINKING), turns=(turn,))) == expected


@pytest.mark.parametrize(
    "turns",
    [
        pytest.param((), id="no-evidence"),
        pytest.param(
            (TurnEvidence("token", OFF_SETTINGS, ReasoningApplied.UNVERIFIED, THINKING_MARKERS),), id="unverified"
        ),
        pytest.param((TurnEvidence("token", OFF_SETTINGS, ReasoningApplied.OFF, ()),), id="no-markers"),
    ],
)
def test_thinking_disabled_is_not_evaluable_without_serving_evidence(turns):
    assessment = protocol(THINKING, answer(THINKING), turns=turns)
    assert "not_evaluable" in assessment
    assert "protocol_compliance" not in assessment


REQUIRED_FILE = TASKS["required_file"]


@pytest.mark.parametrize(
    "files,grade,expected",
    [
        pytest.param({ANSWER_PATH: b"nonce"}, GradeResult(Outcome.GRADED, 1.0), ("submitted", True, True), id="ok"),
        pytest.param({}, GradeResult(Outcome.SUBMISSION_FAILURE, None), ("missing_file", False, False), id="missing"),
        pytest.param({ANSWER_PATH: b" \n"}, GradeResult(Outcome.GRADED, 0.0), ("empty_file", False, False), id="empty"),
    ],
)
def test_required_file_reports_submission_completeness_only(files, grade, expected):
    attempt = GradingAttempt(ConversationTrace(events=(*REQUIRED_FILE.context.events, answer(REQUIRED_FILE))), files)
    assessment = assess_protocol(REQUIRED_FILE, attempt, grade)
    assert assessment is not None
    assert (assessment.label, assessment.submission_completeness, assessment.answer_correctness) == expected
    assert assessment.protocol_compliance is None


@pytest.mark.parametrize("check", ["tool_execution", "thinking_disabled", "url_loading"])
def test_probe_golden_transcripts_pass_their_answer_and_protocol(check):
    report = control_suite(Controls(golden=golden_transcript), None).run(TASKS[check])
    assert [(result.check, result.status) for result in report.checks] == [("golden", CheckStatus.PASS)]


def test_required_file_golden_passes_in_a_grading_machine(local_machine_environment):
    environment, _factory = local_machine_environment
    [task] = (
        task
        for task in probe_tasks(task_environment=environment, grader_environment=environment, variants=1)
        if isinstance(task.protocol_check, RequiredFileCheck)
    )
    report = control_suite(Controls(golden=golden_transcript), LocalGraderMachines()).run(task)
    assert [(result.check, result.status) for result in report.checks] == [("golden", CheckStatus.PASS)]


def test_a_golden_control_that_breaks_the_protocol_fails():
    def without_tool(task: TaskSpec):
        transcript = golden_transcript(task)
        assert transcript is not None
        return type(transcript)((transcript.events[-1],))

    report = control_suite(Controls(golden=without_tool), None).run(TOOL)
    [result] = report.checks
    assert (result.check, result.status) == ("golden", CheckStatus.FAIL)
    assert "protocol_compliance" in (result.detail or "")
