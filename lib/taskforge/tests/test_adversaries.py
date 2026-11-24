# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The default adversary roster: two briefs as policy data, one report schema, one reproduction check."""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import override

import pytest
from lib.taskforge.tests.fake_resources import resources as stage_resources
from lib.taskforge.tests.resource_doubles import CleanReviewer, FakeEnv, FakeInference
from lib.taskforge.tests.stage_fakes import CandidateEngine, Environments
from lib.taskforge.tests.test_end_to_end import (
    SETTINGS,
    Envs,
    call,
    resources,
    script,
)
from rolloutengine.test_support import arithmetic_task
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import EnvironmentRequirements

from taskforge.coordinator import run_workpiece
from taskforge.effects.context import Context
from taskforge.effects.events import Fingerprint, ReplayPolicy
from taskforge.effects.issues import ContextExceeded, Hold, OutputBudgetExhausted
from taskforge.effects.log import EventLog
from taskforge.llm.policy import replay_arguments
from taskforge.model.content_hash import digest
from taskforge.model.controls import Candidate, Control, ControlKind, RewardBand
from taskforge.model.llm import Completion, Message, Usage
from taskforge.model.llm import Role as Speaker
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import (
    Accept,
    AdversaryStatus,
    Cause,
    Continuation,
    Defect,
    DefectClaim,
    Incomplete,
    Modify,
    Observation,
    ObservedStatus,
    SearchEnd,
    SearchLimit,
)
from taskforge.model.workpiece import WORKPIECE, Conversation, Submission, Terminal, Workpiece
from taskforge.policy import Policy, Prompt, load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.grading import GradeCandidate, GradingResource
from taskforge.resources.inference import InferenceClient, InferenceRequest
from taskforge.stages.rules import default_review
from taskforge.stages.validate import validate
from taskforge.wiring import build_config, install


@pytest.fixture
def workpiece(proposal: TaskProposal) -> Workpiece:
    """The arithmetic task, whose answer is 12, with a crediting and a refusing control."""
    controls = (
        Control(Candidate("12"), RewardBand(Outcome.GRADED, reward_min=1), ControlKind.POSITIVE),
        Control(Candidate("9"), RewardBand(Outcome.GRADED, reward_max=0), ControlKind.NEGATIVE),
    )
    return Workpiece("test", proposal, submission=Submission(arithmetic_task(), controls, "authored"))


BRIEFED = {"false_accept": "hunt false accepts", "false_reject": "hunt false rejects"}
"""Each default brief by a phrase its first user message carries."""


def finding(defect: Defect, reply: str, observed: float, deserved: float) -> dict[str, object]:
    return {
        "defect": defect.value,
        "why_the_grade_is_wrong": f"{reply} is graded {observed}, deserves {deserved}",
        "candidate": {"reply": reply},
        "grader_observed_status": "graded",
        "grader_observed_reward": observed,
        "deserved_reward": deserved,
    }


class Briefed(InferenceClient):
    """Each adversary reports the findings ``reports[<its brief>]`` on its first turn, or raises it, and finishes on
    any later turn."""

    def __init__(self, reports: Mapping[str, Sequence[Mapping[str, object]] | Exception]) -> None:
        self.reports = reports
        self.requests: list[InferenceRequest] = []

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("inference", "briefed", ReplayPolicy.CACHE)

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        self.requests.append(request)
        if len(request.conversation) > 2:
            return call("finish", {"reason": "no second report"}, "finish")
        brief = next(name for name, phrase in BRIEFED.items() if phrase in request.conversation[1].content)
        report = self.reports[brief]
        if isinstance(report, Exception):
            raise report
        return call("report", {"findings": list(report)}, brief)


async def test_both_default_briefs_report_observed_and_deserved_and_a_false_reject_reproduces_fresh(
    workpiece: Workpiece,
) -> None:
    """The fake grader credits only the exact reply "12", so the equivalent "12.0" is a real false reject."""
    setup = build_config(load_policy(DEFAULTS_DIR), {})
    assert setup.policy.setting("adversary.roster", tuple) == ("false_accept", "false_reject")
    client = Briefed(
        {
            "false_accept": [finding(Defect.FALSE_ACCEPT, "9", 1.0, 0.0)],  # the grader does not credit 9
            "false_reject": [finding(Defect.FALSE_REJECT, "12.0", 0.0, 1.0)],
        }
    )
    log = EventLog()
    ctx = Context(log=log)
    bundle = replace(stage_resources(ctx), role_inference={"solve_review": CleanReviewer(), "adversary": client})
    result = (await validate(ctx, workpiece, setup, bundle)).result
    accept, reject = result.adversary
    assert (accept.name, reject.name) == ("false_accept", "false_reject")
    assert accept.status is reject.status is AdversaryStatus.REPORTED
    (unshown,) = accept.findings
    assert not unshown.met and unshown.grade is not None and unshown.grade.reward == 0
    (shown,) = reject.findings
    assert shown.defect == DefectClaim(Defect.FALSE_REJECT, Observation(ObservedStatus.GRADED, 0.0), 1.0)
    assert shown.met and shown.grade is not None and shown.grade.diagnostics["verifier_fresh"] is True
    grades = [r.event for r in log.records if isinstance(r.event, GradeCandidate)]
    assert [g.fresh_verifier for g in grades if g.candidate.reply == "12.0"] == [False, True]
    assert not any(g.fresh_verifier for g in grades if g.candidate.reply == "9")  # an unshown claim stops at one grade
    schema = next(tool for tool in client.requests[0].tools if tool.name == "report").parameters
    assert {"grader_observed_reward", "deserved_reward"} <= set(json.dumps(schema).replace('"', " ").split())
    decision = default_review(replace(workpiece, evidence=(result,)), setup.band, setup.budget)
    assert isinstance(decision, Modify) and decision.findings == ("false_reject: 12.0 is graded 0.0, deserves 1.0",)


@pytest.mark.parametrize(
    "args",
    [
        finding(Defect.FALSE_ACCEPT, "9", 0.0, 0.0),  # observed equals deserved: a correct grade
        finding(Defect.FALSE_REJECT, "12", 1.0, 0.0),  # observed above deserved: the other direction
        {**finding(Defect.FALSE_ACCEPT, "9", 1.0, 0.0), "grader_observed_status": "submission_failure"},
    ],
    ids=["correct grade", "wrong direction", "reward without a grade"],
)
async def test_a_report_whose_own_grades_show_no_defect_is_refused_back_to_the_adversary(
    workpiece: Workpiece, args: dict[str, object]
) -> None:
    setup = build_config(load_policy(DEFAULTS_DIR).overlaid({"adversary.roster": ["false_accept"]}), {})
    client = Briefed({"false_accept": [args]})
    ctx = Context(log=EventLog())
    bundle = replace(stage_resources(ctx), role_inference={"solve_review": CleanReviewer(), "adversary": client})
    (result,) = (await validate(ctx, workpiece, setup, bundle)).result.adversary
    assert client.requests[1].conversation[-1].content.startswith("Invalid report: finding 0:")
    assert result.status is AdversaryStatus.FAILED and not result.findings


class Scripted(InferenceClient):
    """Each adversary serves ``turns[<its brief>]`` in order, one completion per assistant turn, then finishes."""

    def __init__(self, turns: Mapping[str, Sequence[Completion]]) -> None:
        self.turns = turns
        self.requests: list[InferenceRequest] = []

    @override
    def fingerprint(self) -> Fingerprint:
        return Fingerprint("inference", "scripted", ReplayPolicy.CACHE)

    @override
    async def complete(self, request: InferenceRequest) -> Completion:
        self.requests.append(request)
        brief = next(name for name, phrase in BRIEFED.items() if phrase in request.conversation[1].content)
        turn = sum(message.role == Speaker.ASSISTANT for message in request.conversation)
        served = self.turns[brief]
        return served[turn] if turn < len(served) else call("finish", {"reason": "done"}, "finish")


def graded_then_reported(brief: str, reply: str, report: dict[str, object]) -> Scripted:
    """``brief`` grades ``reply`` itself, reports ``report``, and corrects a refused report with an empty one."""
    turns = (
        call("grade", {"reply": reply, "files": []}, "own-grade"),
        call("report", {"findings": [report]}, "report"),
        call("report", {"findings": []}, "correction"),
    )
    return Scripted({brief: turns})


# Regression: a report whose grader_observed_* contradicted the adversary's own grade of that exact candidate
# reproduced on the real grader in the claimed direction and turned a correct grader into Modify.
@pytest.mark.parametrize(
    ("brief", "reply", "report", "refused"),
    [
        ("false_accept", "12", finding(Defect.FALSE_ACCEPT, "12", 0.75, 0.0), True),  # graded 1.0 itself
        ("false_reject", "9", finding(Defect.FALSE_REJECT, "9", 0.25, 1.0), True),  # graded 0.0 itself
        ("false_reject", "12.0", finding(Defect.FALSE_REJECT, "12.0", 0.0, 1.0), False),  # a real false reject
    ],
    ids=["false accept of the right answer", "false reject of a wrong answer", "consistent false reject"],
)
async def test_a_report_must_copy_the_adversarys_own_grade_of_that_candidate(
    workpiece: Workpiece, brief: str, reply: str, report: dict[str, object], refused: bool
) -> None:
    setup = build_config(load_policy(DEFAULTS_DIR).overlaid({"adversary.roster": [brief]}), {})
    client = graded_then_reported(brief, reply, report)
    log = EventLog()
    ctx = Context(log=log)
    bundle = replace(stage_resources(ctx), role_inference={"solve_review": CleanReviewer(), "adversary": client})
    result = (await validate(ctx, workpiece, setup, bundle)).result
    decision = default_review(replace(workpiece, evidence=(result,)), setup.band, setup.budget)
    fresh = [r.event for r in log.records if isinstance(r.event, GradeCandidate) and r.event.fresh_verifier]
    answers = [request.conversation[-1].content for request in client.requests[2:]]
    if refused:
        (answer,) = answers  # the refusal, answered by the empty correction
        assert answer.startswith("Invalid report: finding 0: your grade call on this exact candidate returned graded")
        assert isinstance(decision, Accept) and not fresh
        assert result.adversary[0].status is AdversaryStatus.REPORTED and not result.adversary[0].findings
    else:
        assert not answers  # the report ended the session
        assert isinstance(decision, Modify) and [grade.candidate.reply for grade in fresh] == ["12.0"]


# Regression: a nonempty report whose claim failed reproduction ended the search, so a real false reject the
# adversary would have reported on its one revision was recorded as a search that ran out of budget.
async def test_a_claim_that_fails_reproduction_gets_one_revision_in_the_same_session(workpiece: Workpiece) -> None:
    setup = build_config(load_policy(DEFAULTS_DIR), {})
    client = Scripted(
        {
            "false_accept": (call("report", {"findings": []}, "empty"),),
            "false_reject": (
                call("report", {"findings": [finding(Defect.FALSE_REJECT, "12", 0.0, 1.0)]}, "first"),
                call("report", {"findings": [finding(Defect.FALSE_REJECT, "12.0", 0.0, 1.0)]}, "revised"),
            ),
        }
    )
    ctx = Context(log=EventLog())
    bundle = replace(stage_resources(ctx), role_inference={"solve_review": CleanReviewer(), "adversary": client})
    result = (await validate(ctx, workpiece, setup, bundle)).result
    calls = {
        brief: [r for r in client.requests if phrase in r.conversation[1].content] for brief, phrase in BRIEFED.items()
    }
    assert {brief: len(requests) for brief, requests in calls.items()} == {"false_accept": 1, "false_reject": 2}
    answered = calls["false_reject"][1].conversation[-1]
    assert answered.tool_call_id == "first" and answered.content.startswith("Validation replayed your report")
    assert '"reward": 1.0' in answered.content  # the first replay's grade, returned on the pending report call
    _, reject = result.adversary
    assert [(f.candidate.reply, f.met) for f in reject.findings] == [("12", False), ("12.0", True)]
    assert not any("ran out of search budget" in reason for reason in reject.reasons)
    decision = default_review(replace(workpiece, evidence=(result,)), setup.band, setup.budget)
    assert isinstance(decision, Modify) and decision.findings == ("false_reject: 12.0 is graded 0.0, deserves 1.0",)


async def test_a_search_is_out_of_budget_only_after_its_revision_ends_without_a_finding(workpiece: Workpiece) -> None:
    setup = build_config(load_policy(DEFAULTS_DIR).overlaid({"adversary.roster": ["false_reject"]}), {})
    first = call("report", {"findings": [finding(Defect.FALSE_REJECT, "12", 0.0, 1.0)]}, "first")
    client = Scripted({"false_reject": (first,)})  # its revision finishes without a report
    ctx = Context(log=EventLog())
    bundle = replace(stage_resources(ctx), role_inference={"solve_review": CleanReviewer(), "adversary": client})
    ((reject,),) = ((await validate(ctx, workpiece, setup, bundle)).result.adversary,)
    assert len(client.requests) == 2 and reject.status is AdversaryStatus.REPORTED
    assert [(f.candidate.reply, f.met) for f in reject.findings] == [("12", False)]
    assert reject.reasons == (
        "ran out of search budget without a finding (its revision 1 ended without a report); "
        "this is not a validation of the task",
    )


class Exhausting(Environments):
    """Every machine's ``cat log`` exceeds the sandbox output budget."""

    @override
    async def create(self, requirements: EnvironmentRequirements) -> FakeEnv:
        env = FakeEnv(requirements.model_dump_json(), {"cat log": OutputBudgetExhausted("output budget exhausted")})
        self.created.append(env)
        return env


# Regression: a revision that ran a shell command and then ran out of output kept the search's end at "reported"
# and recorded the one-turn conversation of its first report. The exception consumes the remaining opportunity, and
# the last report stands.
async def test_a_revision_that_exhausts_its_output_ends_out_of_budget_at_the_conversation_it_ended_with(
    workpiece: Workpiece,
) -> None:
    setup = build_config(load_policy(DEFAULTS_DIR).overlaid({"adversary.roster": ["false_reject"]}), {})
    shell = call("shell", {"command": "cat log", "timeout": 5}, "look")
    client = Scripted(
        {
            "false_reject": (
                call("report", {"findings": [finding(Defect.FALSE_REJECT, "12", 0.0, 1.0)]}, "first"),
                shell,
                call("report", {"findings": [finding(Defect.FALSE_REJECT, "12.0", 0.0, 1.0)]}, "available"),
            )
        }
    )
    ctx = Context(log=EventLog())
    bundle = replace(
        stage_resources(ctx), envs=Exhausting(), role_inference={"solve_review": CleanReviewer(), "adversary": client}
    )
    result = (await validate(ctx, workpiece, setup, bundle)).result
    (reject,) = result.adversary
    assert len(client.requests) == 2 and reject.status is AdversaryStatus.REPORTED
    assert [(f.candidate.reply, f.met) for f in reject.findings] == [("12", False)]
    assert reject.reasons == (
        "ran out of search budget without a finding (its output budget ran out during revision 1); "
        "this is not a validation of the task",
    )
    (look,) = shell.tool_calls
    ended = Message(Speaker.ASSISTANT, "", (replace(look, arguments=replay_arguments(look.arguments)),))
    final = Conversation((*client.requests[1].conversation, ended), client.requests[1].tools)
    owner = "validate/0/0/adversary:false_reject/0/0"
    assert reject.continuation == Continuation(owner, SearchEnd.BUDGET_EXHAUSTED, digest(final), 2)
    decision = default_review(replace(workpiece, evidence=(result,)), setup.band, setup.budget)
    assert isinstance(decision, Accept)


# Regression: a revision that ran into the context window, through a provider rejection or the
# harness's own prompt-window cutoff, kept the last report but typed the search end as reported with the generic note.
# A voluntary finish is the control.
@pytest.mark.parametrize(
    ("revision", "stopped", "how", "turns"),
    [
        (
            ContextExceeded("model window exceeded"),
            SearchEnd.BUDGET_EXHAUSTED,
            "its context window ran out during revision 1",
            1,
        ),
        (
            Completion("No report yet", usage=Usage(prompt_tokens=262144)),
            SearchEnd.BUDGET_EXHAUSTED,
            "its context window ran out during revision 1",
            2,
        ),
        (
            call("finish", {"reason": "nothing more"}, "finish"),
            SearchEnd.REPORTED,
            "its revision 1 ended without a report",
            2,
        ),
    ],
    ids=["provider rejection", "local cutoff", "voluntary finish"],
)
async def test_a_revision_that_runs_into_the_context_window_ends_out_of_budget_with_its_last_report(
    workpiece: Workpiece, revision: Completion | ContextExceeded, stopped: SearchEnd, how: str, turns: int
) -> None:
    setup = build_config(load_policy(DEFAULTS_DIR).overlaid({"adversary.roster": ["false_reject"]}), {})
    first = call("report", {"findings": [finding(Defect.FALSE_REJECT, "12", 0.0, 1.0)]}, "first")
    client = FakeInference((first, revision))
    ctx = Context(log=EventLog())
    bundle = replace(stage_resources(ctx), role_inference={"solve_review": CleanReviewer(), "adversary": client})
    result = (await validate(ctx, workpiece, setup, bundle)).result
    (reject,) = result.adversary
    assert len(client.requests) == 2 and reject.status is AdversaryStatus.REPORTED
    assert [(f.candidate.reply, f.met) for f in reject.findings] == [("12", False)]
    (reason,) = reject.reasons
    assert reason == f"ran out of search budget without a finding ({how}); this is not a validation of the task"
    assert reject.continuation is not None and reject.continuation.stopped is stopped
    assert reject.continuation.turns == turns and reject.continuation.conversation is not None
    if isinstance(revision, ContextExceeded):  # the conversation the rejected request carried
        held = Conversation(client.requests[1].conversation, client.requests[1].tools)
        assert reject.continuation.conversation == digest(held)
    assert isinstance(default_review(replace(workpiece, evidence=(result,)), setup.band, setup.budget), Accept)


@pytest.mark.parametrize(
    ("replies", "decided"),
    [(("12.0", "12"), Modify), (("12",), Incomplete)],
    ids=["a finding reproduced", "nothing reproduced"],
)
async def test_a_held_revision_keeps_a_finding_that_already_reproduced(
    workpiece: Workpiece, replies: tuple[str, ...], decided: type[Modify | Incomplete]
) -> None:
    setup = build_config(load_policy(DEFAULTS_DIR).overlaid({"adversary.roster": ["false_reject"]}), {})
    report = call("report", {"findings": [finding(Defect.FALSE_REJECT, r, 0.0, 1.0) for r in replies]}, "first")
    client = FakeInference((report, Hold("provider unavailable")))
    ctx = Context(log=EventLog())
    bundle = replace(stage_resources(ctx), role_inference={"solve_review": CleanReviewer(), "adversary": client})
    result = (await validate(ctx, workpiece, setup, bundle)).result
    assert len(client.requests) == 2
    decision = default_review(replace(workpiece, evidence=(result,)), setup.band, setup.budget)
    assert isinstance(decision, decided)
    if isinstance(decision, Modify):
        assert decision.findings == ("false_reject: 12.0 is graded 0.0, deserves 1.0",)


# Regression: a report alleging a wrong score whose fresh verifier crashed correctly led to Modify, but the review
# repeated the unverified score as the finding and the search said it ran out of budget without one.
async def test_a_claim_whose_fresh_verifier_crashes_is_reported_by_its_graded_cause(workpiece: Workpiece) -> None:
    setup = build_config(load_policy(DEFAULTS_DIR).overlaid({"adversary.roster": ["false_reject"]}), {})
    crash = GradeResult(
        Outcome.INFRA_ERROR,
        None,
        "grader exited 1",
        failure=GradingFailure.GRADER_EXIT,
        diagnostics={"exit_code": 1, "stderr": "ZeroDivisionError"},
    )
    client = Briefed({"false_reject": [finding(Defect.FALSE_REJECT, "12.0", 0.0, 1.0)]})
    ctx = Context(log=EventLog())
    bundle = replace(
        stage_resources(Context(log=EventLog())),
        grading=GradingResource(CandidateEngine(), "grader", CandidateEngine(crash)),
        role_inference={"solve_review": CleanReviewer(), "adversary": client},
    )
    install(ctx, bundle)
    result = (await validate(ctx, workpiece, setup, bundle)).result
    ((crashed,),) = (result.adversary[0].findings,)
    assert (crashed.met, crashed.cause) == (False, Cause.GRADER_EXECUTION)
    assert result.adversary[0].reasons == ()
    decision = default_review(replace(workpiece, evidence=(result,)), setup.band, setup.budget)
    assert isinstance(decision, Modify) and decision.findings == (
        "1 ungraded trial(s): grader_execution",
        "grader exited 1\nZeroDivisionError",
        "false_reject: its replay showed grader_execution, not the claimed defect",
    )


# Regression: with adversary.max_revisions=-1 the revision loop also skipped reproducing the report itself, so a
# genuine false reject was accepted after zero fresh grades and saved no finding.
async def test_a_zero_revision_budget_still_reproduces_the_report_and_a_negative_one_fails_at_wiring(
    workpiece: Workpiece,
) -> None:
    policy = load_policy(DEFAULTS_DIR).overlaid({"adversary.roster": ["false_reject"]})
    with pytest.raises(ValueError, match=re.escape("adversary.max_revisions must be at least 0, not -1")):
        build_config(policy.overlaid({"adversary.max_revisions": -1}), {})
    setup = build_config(policy.overlaid({"adversary.max_revisions": 0}), {})
    client = Briefed({"false_reject": [finding(Defect.FALSE_REJECT, "12.0", 0.0, 1.0)]})
    log = EventLog()
    ctx = Context(log=log)
    bundle = replace(stage_resources(ctx), role_inference={"solve_review": CleanReviewer(), "adversary": client})
    result = (await validate(ctx, workpiece, setup, bundle)).result
    fresh = [r.event for r in log.records if isinstance(r.event, GradeCandidate) and r.event.fresh_verifier]
    assert len(client.requests) == 1 and [grade.candidate.reply for grade in fresh] == ["12.0"]
    assert [(f.candidate.reply, f.met) for f in result.adversary[0].findings] == [("12.0", True)]
    decision = default_review(replace(workpiece, evidence=(result,)), setup.band, setup.budget)
    assert isinstance(decision, Modify) and decision.findings == ("false_reject: 12.0 is graded 0.0, deserves 1.0",)


def test_a_roster_name_without_a_brief_fails_at_wiring() -> None:
    """The default user prompt renders ``{brief}``, so every roster name must resolve to a brief."""
    with pytest.raises(
        ValueError, match=re.escape("adversary.roster names finder without a prompt adversary.brief.<name>")
    ):
        build_config(load_policy(DEFAULTS_DIR).overlaid({"adversary.roster": ["false_accept", "finder"]}), {})


# Regression: an indexed or attribute {brief} field escaped the check and indexed the roster name instead.
@pytest.mark.parametrize("field", ["{brief[0]}", "{brief.upper}"])
def test_an_indexed_brief_field_still_needs_a_brief_at_wiring(field: str) -> None:
    policy = load_policy(DEFAULTS_DIR).overlaid({"adversary.user": f"{field} {{task}}", "adversary.roster": ["finder"]})
    with pytest.raises(ValueError, match=re.escape("adversary.roster names finder without a prompt")):
        build_config(policy, {})


NAMED_STRATEGY = "Review strategy: {name}\n\nTask manifest:\n{task}\n\nProposal:\n{proposal}\n\nWorkspace: {workdir}"
"""The earlier user prompt, whose strategy line is the roster name itself."""


def named_strategy_policy(roster: Sequence[str], *, briefs: bool) -> Policy:
    """The defaults with the earlier ``{name}`` user prompt and ``roster``; without ``briefs``, no brief entries."""
    entries = {
        key: value
        for key, value in load_policy(DEFAULTS_DIR).entries.items()
        if briefs or not key.startswith("adversary.brief.")
    }
    return Policy({**entries, "adversary.user": Prompt(NAMED_STRATEGY)}).overlaid({"adversary.roster": list(roster)})


# Regression: an earlier policy's single `adversary` or custom roster name failed at wiring, and the documented
# user-prompt override `Review strategy: {name}` raised KeyError('name') before a sandbox opened.
@pytest.mark.parametrize(
    ("roster", "briefs"),
    [(["adversary"], False), (["finder"], False), (["false_accept"], True)],
    ids=["single adversary", "custom name", "name override with the default briefs"],
)
async def test_a_named_strategy_prompt_runs_any_roster_name(
    workpiece: Workpiece, roster: list[str], briefs: bool
) -> None:
    setup = build_config(named_strategy_policy(roster, briefs=briefs), {})
    client = FakeInference((call("report", {"findings": []}, "report"),))
    ctx = Context(log=EventLog())
    bundle = replace(stage_resources(ctx), role_inference={"solve_review": CleanReviewer(), "adversary": client})
    (result,) = (await validate(ctx, workpiece, setup, bundle)).result.adversary
    assert result.status is AdversaryStatus.REPORTED and len(client.requests) == 1
    assert client.requests[0].conversation[1].content.startswith(f"Review strategy: {roster[0]}\n\nTask manifest:")


N6_POLICY = Path(__file__).parent / "fixtures" / "n6-policy"
"""The complete default policy directory of 40d35b8bcc, which predates ``adversary.max_revisions``."""


# Regression: a complete policy directory written before the revision budget, loaded on its own rather than
# overlaid on the defaults, rendered the adversary prompt and spent one inference before raising
# KeyError('adversary.max_revisions').
@pytest.mark.parametrize("roster", [["adversary"], ["custom"]])
async def test_a_complete_policy_without_a_revision_budget_fails_at_wiring_and_its_prompt_overlaid_on_defaults_runs(
    workpiece: Workpiece, roster: list[str]
) -> None:
    historical = load_policy(N6_POLICY).overlaid({"adversary.roster": roster})
    log, client = EventLog(), FakeInference((call("report", {"findings": []}, "report"),))
    ctx = Context(log=log)
    bundle = replace(stage_resources(ctx), role_inference={"solve_review": CleanReviewer(), "adversary": client})
    with pytest.raises(KeyError, match=re.escape("adversary.max_revisions")):
        await validate(ctx, workpiece, build_config(historical, {}), bundle)
    assert not log.records and not client.requests
    overlay = {"adversary.user": historical.prompt("adversary.user").template, "adversary.roster": roster}
    (result,) = (
        await validate(ctx, workpiece, build_config(load_policy(DEFAULTS_DIR).overlaid(overlay), {}), bundle)
    ).result.adversary
    assert result.status is AdversaryStatus.REPORTED and len(client.requests) == 1


# A search that ends without a reproduced finding counts for a strong task, but every record
# says it ran out of search budget without a finding, never that it validated the task.
async def test_searches_without_a_finding_accept_and_say_the_budget_ran_out_with_a_continuation(
    proposal: TaskProposal,
) -> None:
    policy = load_policy(DEFAULTS_DIR).overlaid({**SETTINGS, "adversary.roster": ["false_accept", "false_reject"]})
    client = Briefed({"false_accept": ContextExceeded("model window exceeded"), "false_reject": []})
    bundle = replace(resources(Envs(), script()), role_inference={"solve_review": CleanReviewer(), "adversary": client})
    log = EventLog()
    finished = await run_workpiece(proposal, build_config(policy, {}), bundle, log, install=install)
    workpiece = WORKPIECE.validate_json(WORKPIECE.dump_json(finished))  # as published
    assert workpiece.terminal is Terminal.ACCEPTED
    exhausted, reported = workpiece.evidence[-1].adversary
    assert (exhausted.status, exhausted.limit) == (AdversaryStatus.EXHAUSTED, SearchLimit.CONTEXT_WINDOW)
    assert reported.status is AdversaryStatus.REPORTED and not reported.findings
    (first,) = (request for request in client.requests if BRIEFED["false_accept"] in request.conversation[1].content)
    assert exhausted.continuation == Continuation(
        "validate/0/0/adversary:false_accept/0/0",
        SearchEnd.BUDGET_EXHAUSTED,
        digest(Conversation(first.conversation, first.tools)),  # the conversation the stopped session held
        0,
    )
    assert reported.continuation is not None and reported.continuation.stopped is SearchEnd.REPORTED
    assert reported.continuation.owner == "validate/0/0/adversary:false_reject/0/0"
    decision = workpiece.decisions[-1]
    assert isinstance(decision, Accept)
    notes = [reason for reason in decision.reasons if reason.startswith(("false_accept: ", "false_reject: "))]
    assert len(notes) == 2 and all("ran out of search budget without a finding" in note for note in notes)
    assert all("not a validation of the task" in note for note in notes)
    assert not any("validated" in text for text in (*decision.reasons, *exhausted.reasons, *reported.reasons))
