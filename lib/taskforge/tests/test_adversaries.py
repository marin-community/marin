# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The default adversary roster: two briefs as policy data, one report schema, one reproduction check."""

import gzip
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import override

import pytest
from lib.taskforge.tests.fake_resources import resources as stage_resources
from lib.taskforge.tests.test_end_to_end import SETTINGS, Envs, ServedOnce, call, executions, resources
from lib.taskforge.tests.test_stages import workpiece as workpiece  # the arithmetic workpiece fixture

from taskforge.coordinator import run_workpiece
from taskforge.effects.context import Context
from taskforge.effects.events import Fingerprint, ReplayPolicy, Resolved, StageCompleted
from taskforge.effects.log import EventLog, decode
from taskforge.model.content_hash import digest
from taskforge.model.llm import Completion
from taskforge.model.proposal import TaskProposal
from taskforge.model.review import AdversaryStatus, Defect, DefectClaim, Modify, Observation, ObservedStatus
from taskforge.model.workpiece import Terminal, Workpiece
from taskforge.policy import load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.grading import GradeCandidate
from taskforge.resources.inference import InferenceClient, InferenceRequest
from taskforge.stages.classify import replayable_grade
from taskforge.stages.rules import default_review
from taskforge.stages.validate import validate
from taskforge.wiring import build_config, install

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
    """Each adversary reports ``reports[<its brief>]`` on its first turn and finishes on any later one."""

    def __init__(self, reports: Mapping[str, Mapping[str, object]]) -> None:
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
        return call("report", {"findings": [self.reports[brief]]}, brief)


async def test_both_default_briefs_report_observed_and_deserved_and_a_false_reject_reproduces_fresh(
    workpiece: Workpiece,
) -> None:
    """The fake grader credits only the exact reply "12", so the equivalent "12.0" is a real false reject."""
    setup = build_config(load_policy(DEFAULTS_DIR), {})
    assert setup.policy.setting("adversary.roster", tuple) == ("false_accept", "false_reject")
    client = Briefed(
        {
            "false_accept": finding(Defect.FALSE_ACCEPT, "9", 1.0, 0.0),  # the grader does not credit 9
            "false_reject": finding(Defect.FALSE_REJECT, "12.0", 0.0, 1.0),
        }
    )
    log = EventLog()
    ctx = Context(log=log)
    bundle = replace(stage_resources(ctx), role_inference={"adversary": client})
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
    decision = default_review(workpiece.with_(evidence=(result,)), setup.band, setup.budget)
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
    client = Briefed({"false_accept": args})
    ctx = Context(log=EventLog())
    bundle = replace(stage_resources(ctx), role_inference={"adversary": client})
    (result,) = (await validate(ctx, workpiece, setup, bundle)).result.adversary
    assert client.requests[1].conversation[-1].content.startswith("Invalid report: finding 0:")
    assert result.status is AdversaryStatus.FAILED and not result.findings


def test_a_roster_name_without_a_brief_fails_at_wiring() -> None:
    with pytest.raises(ValueError, match="adversary.roster names finder without a prompt adversary.brief.<name>"):
        build_config(load_policy(DEFAULTS_DIR).overlaid({"adversary.roster": ["false_accept", "finder"]}), {})


N6_SINGLE_ADVERSARY = Path(__file__).parent / "fixtures" / "n6-single-adversary-held.jsonl.gz"


async def test_a_single_adversary_policy_resumes_its_journal_from_before_the_briefs(proposal: TaskProposal) -> None:
    """The fixture is the journal the role-runner revision (before the briefs) published for a driver launch of the
    ordinary author script under ``{"adversary.roster": ["adversary"]}``, held on the adversary's first model call.

    The legacy name keeps its strategy line through the ``adversary`` brief. Author turns, solver trials, control
    grades and the adversary's hydration replay; only the adversary's model turns are new."""
    recorded = decode(gzip.decompress(N6_SINGLE_ADVERSARY.read_bytes()))
    served = [
        r.event.request
        for r in recorded
        if isinstance(r.event, Resolved) and isinstance(r.event.request, InferenceRequest)
    ]
    policy = load_policy(DEFAULTS_DIR).overlaid(
        {"proposals.environments": ["container"], **SETTINGS, "adversary.roster": ["adversary"]}
    )
    adversary = ServedOnce((call("shell", {"command": "inspect", "timeout": 5}, "look"), call("report", {"findings": []}, "r")), ())
    adversary.identity = "adversary"
    resumed = replace(resources(Envs(), ()), inference=ServedOnce((), served), role_inference={"adversary": adversary})
    log = EventLog(replay_from=recorded, replayable=replayable_grade)
    workpiece = await run_workpiece(proposal, build_config(policy, {}), resumed, log, install=install)
    assert workpiece.terminal is Terminal.ACCEPTED
    assert isinstance(resumed.inference, ServedOnce) and not resumed.inference.requests
    assert executions(resumed) == (0, 0) and len(adversary.requests) == 2
    assert adversary.requests[0].conversation[1].content.startswith("Review strategy: adversary\n\n")
    assert [result.name for result in workpiece.evidence[-1].adversary] == ["adversary"]
    was = next(r.event.config_digest for r in recorded if isinstance(r.event, StageCompleted))
    assert was != digest(policy)  # the briefs are new policy data under every overlay
