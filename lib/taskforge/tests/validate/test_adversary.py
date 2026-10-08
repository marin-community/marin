# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adversary trials as agent loops: the brief as the one system turn, a submit tool that grades candidates on a fresh
machine through the task's real verifier under a submission budget, the verdict line, and evidence per trial.

The model is ``fake_glm`` (a scripted GLM router); machines are ShellSim and the file task's grader runs for real.
"""

import json

import pytest
from rolloutengine.contracts import TOTAL_TURN_TIMEOUT_STOP_REASON
from rolloutengine.task_session import session_start
from shellbox.machine import Backend
from taskcompendium.models import AnswerType, AssistantToolCalls, ConversationInput, FunctionDefinition, TextMessage
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.submission import ANSWER_CALL_NAME, ANSWER_FIELD, AnswerCall, FinalAction, PlainText
from verifyit.spec import FunctionCall, PredictedActionSpec

from taskforge.ledger.jsonl import read_entries
from taskforge.ledger.records import EntryKind
from taskforge.sandbox.factories import SHELLSIM
from taskforge.spec.draft import answer_verifier, assemble
from taskforge.validate.adversary import (
    CONTEXT_HEADER,
    PREAMBLE_SEPARATOR,
    SUBMIT_TOOL_NAME,
    AdversaryRole,
    ClaimKind,
    adversary_brief,
    parse_claim,
    run_adversaries,
)
from taskforge.validate.attempts import ADVERSARY_KEY, load_adversary_attempt, trial_files
from taskforge.validate.outcome import Cause, Graded, TrialKind, Ungraded
from taskforge.validate.submissions import trial_claim
from taskforge.validate.trials import CLEANUP_ERROR_COUNT, EngineSettings

PLAIN = PlainText(id="plain")
SHELLSIM_BACKEND = Backend.SHELLSIM.value
SUM = "/workspace/sum.txt"
WRITE_SUM = ("shell", f"echo 60 > {SUM}")
SHORTCUT = AdversaryRole.SHORTCUT


def settings(factory, capabilities=None, max_turns: int = 6) -> EngineSettings:
    return EngineSettings(
        factories={SHELLSIM_BACKEND: factory},
        capabilities={SHELLSIM_BACKEND: SHELLSIM} if capabilities is None else capabilities,
        max_turns=max_turns,
        command_timeout=10,
        tool_turn_timeout=20,
        model_turn_timeout=30,
        cleanup_timeout=10,
        conventions=(PLAIN,),
    )


async def one_trial(
    tmp_path, task, rounds, fakes, client, policy=None, factory=None, context="", convention=PLAIN, **engine
):
    """Run one adversary trial of ``task``; return it and its evidence directory."""
    site = rounds.site(tmp_path)
    factory = factory or fakes.flaky_factory(0, RuntimeError)
    trials = await run_adversaries(
        rounds.draft(task, (), convention),
        policy or rounds.policy(adversary_k=1),
        site,
        settings(factory, **engine),
        client,
        context,
    )
    (trial,) = trials[SHORTCUT]
    return trial, site.evidence_dir / "adversary" / "shortcut" / "0"


def tool_results(fake_glm) -> list[dict]:
    """The tool messages of the last request, decoded where they are JSON."""
    messages = [m for m in fake_glm.requests[-1]["messages"] if m["role"] == "tool"]
    return [json.loads(m["content"]) if m["content"].startswith("{") else m["content"] for m in messages]


async def test_the_brief_is_the_one_system_turn_and_the_task_follows_it(
    tmp_path, file_task, math_task, rounds, fakes, fake_glm, glm_client, turns
):
    turns(fake_glm, "NO_SHORTCUT")
    trial, directory = await one_trial(tmp_path / "file", file_task, rounds, fakes, glm_client)

    first = fake_glm.requests[0]
    assert first["messages"][0] == {"role": "system", "content": adversary_brief(4, "")}
    assert first["messages"][1:] == [dict(m) for m in session_start(file_task.task, PLAIN).messages]
    assert [tool["function"]["name"] for tool in first["tools"]] == ["shell", SUBMIT_TOOL_NAME]
    assert "files" in first["tools"][1]["function"]["parameters"]["properties"]
    record = json.loads((directory / "attempt-0.json").read_text())
    assert record[ADVERSARY_KEY]["system"] == adversary_brief(4, "") == trial.system
    assert record["rollout"]["messages"][0]["content"] == trial.system

    turns(fake_glm, "NO_SHORTCUT")
    await one_trial(tmp_path / "math", math_task, rounds, fakes, glm_client)

    (submit,) = fake_glm.requests[1]["tools"]
    assert submit["function"]["name"] == SUBMIT_TOOL_NAME
    assert set(submit["function"]["parameters"]["properties"]) == {"reply"}


async def test_a_task_system_prompt_follows_the_brief_in_the_same_turn(
    tmp_path, file_task, relower, rounds, fakes, fake_glm, glm_client, turns
):
    events = (TextMessage(role="system", content="You are careful."), *file_task.task.context.events)
    task = relower(file_task.task.model_copy(update={"context": ConversationInput(events=events)}))
    turns(fake_glm, "NO_SHORTCUT")

    trial, _ = await one_trial(tmp_path, task, rounds, fakes, glm_client)

    expected = f"{adversary_brief(4, '')}{PREAMBLE_SEPARATOR}You are careful."
    assert fake_glm.requests[0]["messages"][0] == {"role": "system", "content": expected}
    assert trial.system == expected


async def test_a_consumer_context_is_its_own_paragraph_before_the_budget(
    tmp_path, file_task, rounds, fakes, fake_glm, glm_client, turns
):
    turns(fake_glm, "NO_SHORTCUT")

    trial, _ = await one_trial(tmp_path, file_task, rounds, fakes, glm_client, context="Exercise capability X.")

    paragraphs = trial.system.split(PREAMBLE_SEPARATOR)
    assert paragraphs[1] == f"{CONTEXT_HEADER}\nExercise capability X."
    assert paragraphs[0].endswith("is accepted.") and paragraphs[2].startswith("You have 4 verifier submissions")
    assert trial.system == adversary_brief(4, "Exercise capability X.")


async def test_submit_grades_a_fresh_machine_through_the_verifier(
    tmp_path, file_task, rounds, fakes, fake_glm, glm_client, turns
):
    turns(fake_glm, WRITE_SUM, ("submit", "Done.", (SUM,)), "Looks right.\nNO_SHORTCUT")

    trial, _ = await one_trial(tmp_path, file_task, rounds, fakes, glm_client)

    (observation,) = tool_results(fake_glm)[1:]
    assert observation == {
        "submission": 1,
        "remaining": 3,
        "status": "graded",
        "reward": 1.0,
        "passed": True,
        "score_min": 0.0,
        "score_max": 1.0,
    }
    assert isinstance(trial.outcome, Graded) and trial.outcome.reward == 1.0
    (submission,) = trial.submissions
    assert submission.grade.detail["got"] == "60"
    assert submission.turn == 1 and submission.candidate.paths == (SUM,)
    assert resource_bytes(submission.candidate.files[0]) == b"60\n"
    assert trial_claim(trial.outcome).kind is ClaimKind.NO_SHORTCUT


async def test_the_verifier_never_sees_the_adversary_shell_state(
    tmp_path, file_task, rounds, fakes, fake_glm, glm_client, turns
):
    turns(fake_glm, WRITE_SUM, ("submit", "Done.", ()), ("submit", "Done.", (SUM,)), "NO_SHORTCUT")

    trial, _ = await one_trial(tmp_path, file_task, rounds, fakes, glm_client)

    assert [(s.ordinal, s.grade.reward, s.passed) for s in trial.submissions] == [(1, 0.0, False), (2, 1.0, True)]


async def test_a_planted_grader_is_not_the_grader_that_runs(
    tmp_path, file_task, rounds, fakes, fake_glm, glm_client, turns
):
    planted = "/tests/grader.py"
    plant = ("shell", f"mkdir -p /tests && echo 'print(1)' > {planted}")
    turns(fake_glm, plant, ("submit", "Done.", (planted,)), "NO_SHORTCUT")

    trial, _ = await one_trial(tmp_path, file_task, rounds, fakes, glm_client)

    (submission,) = trial.submissions
    assert submission.candidate.paths == (planted,) and submission.grade.reward == 0.0


async def test_the_submission_budget_is_enforced_by_the_tool(
    tmp_path, file_task, rounds, fakes, fake_glm, glm_client, turns
):
    factory = fakes.flaky_factory(0, RuntimeError)
    submit = ("submit", "Done.", ())
    turns(fake_glm, submit, submit, submit, "SHORTCUT: none found but saying so anyway")
    policy = rounds.policy(adversary_k=1, adversary_submissions=2, adversary_repair_submissions=1)

    trial, _ = await one_trial(tmp_path, file_task, rounds, fakes, glm_client, policy=policy, factory=factory)

    assert len(trial.submissions) == 2
    assert factory.creates == 3  # the adversary's machine, then one grading machine per graded submission
    assert tool_results(fake_glm)[-1] == {"error": "submission budget spent: 2 of 2; give your verdict now"}
    assert trial_claim(trial.outcome).kind is ClaimKind.SHORTCUT


async def test_a_missing_file_is_an_error_observation_not_a_grading(
    tmp_path, file_task, rounds, fakes, fake_glm, glm_client, turns
):
    factory = fakes.flaky_factory(0, RuntimeError)
    turns(fake_glm, ("submit", "Done.", (SUM,)), "NO_SHORTCUT")

    trial, _ = await one_trial(tmp_path, file_task, rounds, fakes, glm_client, factory=factory)

    assert tool_results(fake_glm) == [{"error": f"no file at {SUM} in your workspace"}]
    assert trial.submissions == () and factory.creates == 1


async def test_grading_diagnostics_are_withheld_from_the_model(
    tmp_path, file_task, rounds, fakes, fake_glm, glm_client, turns
):
    turns(fake_glm, ("shell", f"echo 59 > {SUM}"), ("submit", "Done.", (SUM,)), "NO_SHORTCUT")

    trial, _ = await one_trial(tmp_path, file_task, rounds, fakes, glm_client)

    (observation,) = tool_results(fake_glm)[1:]
    assert set(observation) == {
        "submission",
        "remaining",
        "status",
        "reward",
        "passed",
        "score_min",
        "score_max",
    }
    (submission,) = trial.submissions
    assert submission.grade.detail["got"] == "59" and not submission.passed


@pytest.mark.parametrize(
    ("reply", "kind", "why"),
    [
        ("NO_SHORTCUT", ClaimKind.NO_SHORTCUT, ""),
        ("I probed the grader twice.\n\nNO_SHORTCUT\n", ClaimKind.NO_SHORTCUT, ""),
        (
            "Found one.\nSHORTCUT: lenient regex accepts 395",
            ClaimKind.SHORTCUT,
            "lenient regex accepts 395",
        ),
        ("The SHORTCUT: is that the grader is lenient.", ClaimKind.NONE, ""),
        ("SHORTCUT: lenient\nmore prose after the verdict", ClaimKind.NONE, ""),
        ("", ClaimKind.NONE, ""),
        (None, ClaimKind.NONE, ""),
    ],
)
def test_claim_parsing(reply, kind, why):
    claim = parse_claim(reply)

    assert (claim.kind, claim.why) == (kind, why)


async def test_a_run_without_a_final_text_verdict_has_no_claim(
    tmp_path, file_task, rounds, fakes, fake_glm, glm_client, turns
):
    turns(fake_glm, *[("shell", "ls /workspace")] * 2)
    on_turns, _ = await one_trial(tmp_path / "turns", file_task, rounds, fakes, glm_client, max_turns=2)
    fake_glm.stream(content="SHORTCUT: cut mid-sentence", finish="length")
    cut, _ = await one_trial(tmp_path / "cut", file_task, rounds, fakes, glm_client)

    assert isinstance(on_turns.outcome, Graded) and on_turns.outcome.rollout.stop_reason == "max_turns"
    assert isinstance(cut.outcome, Graded) and cut.outcome.rollout.stop_reason == "length"
    assert trial_claim(on_turns.outcome).kind is ClaimKind.NONE and trial_claim(cut.outcome).kind is ClaimKind.NONE


async def test_a_total_turn_deadline_keeps_the_submissions_and_grades_the_trial(
    tmp_path, file_task, rounds, fakes, fake_glm, glm_client, turns
):
    turns(fake_glm, WRITE_SUM, ("submit", "Done.", (SUM,)))
    fake_glm.stream(content="never finished", stall_after_first=True)
    policy = rounds.policy(adversary_k=1, total_turn_timeout=0.5)

    trial, _ = await one_trial(tmp_path, file_task, rounds, fakes, glm_client, policy=policy)

    outcome = trial.outcome
    assert isinstance(outcome, Graded) and outcome.timed_out and outcome.reward == 1.0
    assert outcome.rollout.stop_reason == TOTAL_TURN_TIMEOUT_STOP_REASON and outcome.rollout.steps == ()
    (submission,) = trial.submissions
    assert submission.turn is None and submission.passed


async def test_a_workspace_machine_that_fails_to_close_is_counted_and_the_trial_stays_graded(
    tmp_path, file_task, rounds, fakes, fake_glm, glm_client, turns
):
    turns(fake_glm, WRITE_SUM, ("submit", "Done.", (SUM,)), "NO_SHORTCUT")

    trial, directory = await one_trial(
        tmp_path, file_task, rounds, fakes, glm_client, factory=fakes.faulty_factory(close_error=True)
    )

    assert isinstance(trial.outcome, Graded) and trial.outcome.reward == 1.0
    assert trial.outcome.rollout.metrics[CLEANUP_ERROR_COUNT] == 1
    assert json.loads((directory / "attempt-0.json").read_text())["cleanup_errors"] == 1


async def test_a_drained_router_is_retried_with_a_fresh_budget(
    tmp_path, file_task, rounds, fakes, fake_glm, glm_client, turns
):
    turns(fake_glm, ("submit", "Done.", ()))
    fake_glm.status(500, "router drained")
    turns(fake_glm, WRITE_SUM, ("submit", "Done.", (SUM,)), "NO_SHORTCUT")
    policy = rounds.policy(adversary_k=1, max_retries=1)

    trial, directory = await one_trial(tmp_path, file_task, rounds, fakes, glm_client, policy=policy)

    first = load_adversary_attempt(directory / "attempt-0.json")
    assert isinstance(first.outcome, Ungraded) and first.outcome.cause is Cause.MODEL_UNAVAILABLE
    assert [(s.ordinal, s.passed, s.turn) for s in first.submissions] == [(1, False, None)]
    assert isinstance(trial.outcome, Graded) and trial.outcome.reward == 1.0
    assert [(s.ordinal, s.passed, s.turn) for s in trial.submissions] == [(1, True, 1)]
    assert load_adversary_attempt(directory / "attempt-1.json") == trial


async def test_evidence_lands_per_role_and_index_with_submissions(
    tmp_path, file_task, rounds, fakes, fake_glm, glm_client, turns
):
    site = rounds.site(tmp_path)
    turns(fake_glm, *[("submit", "Done.", ()), "NO_SHORTCUT"] * 2)

    trials = await run_adversaries(
        rounds.draft(file_task, (), PLAIN),
        rounds.policy(adversary_k=2),
        site,
        settings(fakes.flaky_factory(0, RuntimeError)),
        glm_client,
        "",
    )

    files = sorted(p.relative_to(site.evidence_dir).as_posix() for p in site.evidence_dir.rglob("attempt-*.json"))
    assert files == ["adversary/shortcut/0/attempt-0.json", "adversary/shortcut/1/attempt-0.json"]
    entries = list(read_entries(tmp_path / "ledger" / "item.jsonl"))
    assert {e.step for e in entries if e.kind == EntryKind.TRIAL} == {"adversary/shortcut/0/0", "adversary/shortcut/1/0"}
    assert {e.step for e in entries if e.kind == EntryKind.LLM_CALL} == {"adversary/shortcut/0", "adversary/shortcut/1"}
    submits = [e for e in entries if e.kind == EntryKind.STEP and e.attrs["tool"] == SUBMIT_TOOL_NAME]
    assert len(submits) == sum(len(t.submissions) for t in trials[SHORTCUT]) == 2
    trial_spans = [e for e in entries if e.kind == EntryKind.TRIAL]
    assert sum(int(e.attrs["submissions"]) for e in trial_spans) == 2
    assert {e.attrs["claim"] for e in trial_spans} == {"no_shortcut"}


async def test_a_settled_trial_is_loaded_not_run_again(tmp_path, file_task, rounds, fakes, fake_glm, glm_client, turns):
    turns(fake_glm, WRITE_SUM, ("submit", "Done.", (SUM,)), "NO_SHORTCUT")
    first, _ = await one_trial(tmp_path, file_task, rounds, fakes, glm_client)
    requests = len(fake_glm.requests)

    again, _ = await one_trial(tmp_path, file_task, rounds, fakes, glm_client)

    assert again == first and len(fake_glm.requests) == requests


async def test_a_refused_machine_is_one_unsupported_attempt(tmp_path, file_task, rounds, fakes, fake_glm, glm_client):
    factory = fakes.flaky_factory(0, RuntimeError)

    trial, directory = await one_trial(tmp_path, file_task, rounds, fakes, glm_client, factory=factory, capabilities={})

    assert isinstance(trial.outcome, Ungraded) and trial.outcome.cause is Cause.MACHINE_UNSUPPORTED
    assert factory.creates == 0 and fake_glm.requests == []
    assert trial_files(directory.parents[2], TrialKind.ADVERSARY)["shortcut/0"].attempts == 1
    assert load_adversary_attempt(directory / "attempt-0.json") == trial


async def test_an_answer_call_candidate_is_submitted_through_the_answer_call(
    tmp_path, math_task, rounds, fakes, fake_glm, glm_client, turns
):
    turns(fake_glm, ("submit", "391"), ("submit", "395"), "NO_SHORTCUT")

    trial, _ = await one_trial(tmp_path, math_task, rounds, fakes, glm_client, convention=AnswerCall(id="answer-call"))

    assert [(s.grade.reward, s.passed) for s in trial.submissions] == [(0.0, False), (1.0, True)]
    candidate = trial.submissions[1].candidate
    assert isinstance(candidate.turn, AssistantToolCalls)
    assert [(c.name, c.arguments) for c in candidate.turn.calls] == [(ANSWER_CALL_NAME, {ANSWER_FIELD: "395"})]
    assert candidate.reply == "395"


@pytest.fixture
def action_task(math_task, relower):
    """A machine-less task whose answer is one ``lookup`` call for Paris."""
    lookup = FunctionDefinition(name="lookup", parameters={"type": "object", "properties": {"city": {"type": "string"}}})
    spec = PredictedActionSpec(expected_calls=(FunctionCall("lookup", {"city": "Paris"}),))
    return relower(
        assemble(
            "validate-action",
            "Look up the capital of France.",
            AnswerType.NATIVE_ACTION,
            answer_verifier(spec),
            math_task.task.source,
            environment=None,
            final_tools=(lookup,),
        )
    )


async def test_a_final_action_candidate_makes_the_calls_it_lists(
    tmp_path, action_task, rounds, fakes, fake_glm, glm_client, turns
):
    for calls in (
        [{"name": "search", "arguments": {}}],
        [{"name": "lookup", "arguments": {"city": "Rome"}}],
        [{"name": "lookup", "arguments": {"city": "Paris"}}],
    ):
        arguments = json.dumps({"reply": "", "calls": calls})
        fake_glm.stream(tool_calls=((SUBMIT_TOOL_NAME, arguments),), finish="tool_calls")
    turns(fake_glm, "NO_SHORTCUT")

    trial, _ = await one_trial(tmp_path, action_task, rounds, fakes, glm_client, convention=FinalAction(id="action"))

    (submit,) = fake_glm.requests[0]["tools"]
    assert "calls" in submit["function"]["parameters"]["properties"]
    assert tool_results(fake_glm)[0] == {"error": "search is not a final tool of the task; calls may name ['lookup']"}
    assert [(s.ordinal, s.grade.reward) for s in trial.submissions] == [(1, 0.0), (2, 1.0)]
    candidate = trial.submissions[1].candidate
    assert isinstance(candidate.turn, AssistantToolCalls) and candidate.reply is None


def test_the_budget_and_threshold_are_in_the_policy_digest(rounds):
    base = rounds.policy()

    assert base.digest != rounds.policy(adversary_submissions=5).digest
    assert base.digest != rounds.policy(adversary_repair_submissions=1).digest
