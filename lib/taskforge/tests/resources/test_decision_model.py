# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import cast, override

import httpx
import pytest
from lib.taskforge.tests.resource_doubles import FakeWeb
from lib.taskforge.tests.resources.inference_fakes import ScriptedProvider, segment
from rolloutengine.models import GradedRollout, TaskRolloutSpec, TraceMode
from rolloutengine.test_support import arithmetic_task
from taskcompendium.decision_model import DecisionModelRequest
from taskcompendium.grading_result import Outcome
from taskcompendium.models import DECISION_LEVELS, DECISION_QUESTION, DecisionModelGrader

from taskforge.effects.context import Context
from taskforge.effects.events import Resolved, Retrying
from taskforge.effects.issues import DecisionModelUnavailable
from taskforge.effects.log import EventLog, Record
from taskforge.llm.client import Progress, Segment
from taskforge.llm.glm_provider import FixedEndpoint, GlmEndpoint, Pool
from taskforge.lowering import InvalidTaskSpec, lower_with_settings, lowering_settings
from taskforge.model.content_hash import digest
from taskforge.model.controls import Candidate, Control, ControlKind, RewardBand
from taskforge.model.proposal import TaskProposal
from taskforge.model.workpiece import Submission, Workpiece
from taskforge.policy import Policy, load_policy
from taskforge.policy.defaults import DEFAULTS_DIR
from taskforge.resources.bundle import Resources
from taskforge.resources.decision_model import (
    ClefReader,
    DecisionModelDown,
    DecisionModelHealth,
    DecisionModelRead,
    DecisionModelReader,
)
from taskforge.resources.grading import GradeCandidate
from taskforge.resources.rollouts import SolverRollout
from taskforge.stages.author import system_prompt, task_schema
from taskforge.stages.classify import replayable_grade
from taskforge.stages.exits import Validated
from taskforge.stages.validate import solver_batch, validate
from taskforge.wiring import build_config, build_resources, decision_model_policy, install, submit_gate

MODEL = "clef@fixture-1"
ROLES = ("author", "solver", "adversary")
GRADER = DecisionModelGrader(deliverable="Six plus six.", rubric="The sum, as a number.", reference="12", cut=3.0)


def overlaid(model: str = MODEL) -> Policy:
    roles = {f"{role}.model": "test-model" for role in ROLES}
    overlay = {**roles, "decision_model.model": model, "solver.k": 3, "adversary.roster": []}
    return load_policy(DEFAULTS_DIR).overlaid(overlay)


def decision_spec(policy: Policy) -> TaskRolloutSpec:
    task = arithmetic_task(grader=GRADER)
    return lower_with_settings(task, lowering_settings(policy), {}).model_copy(update={"trace": TraceMode.MESSAGES})


def request(candidate: str, model: str = MODEL) -> DecisionModelRequest:
    return DecisionModelRequest(
        model, DECISION_QUESTION, DECISION_LEVELS, GRADER.deliverable, GRADER.rubric, GRADER.reference, candidate
    )


@dataclass
class FakeReader(DecisionModelReader):
    """Reads ``levels[candidate]``; a candidate in ``down`` is unavailable while it stays in the set."""

    levels: Mapping[str, object] = field(default_factory=lambda: {"12": 4.0, "13": 1.0})
    down: set[str] = field(default_factory=set)
    reads: list[DecisionModelRequest] = field(default_factory=list)

    @override
    async def read(self, request: DecisionModelRequest) -> float:
        self.reads.append(request)
        if request.candidate in self.down:
            raise DecisionModelDown("HTTP 503: clef_unavailable", 30.0)
        return cast(float, self.levels[request.candidate])

    @override
    async def health(self) -> DecisionModelHealth:
        raise AssertionError("grading never reads health")


@dataclass
class VirtualTime:
    """A clock that only a backoff advances; ``recovered`` runs at the first backoff, as an outage ends."""

    now: float = 0.0
    slept: list[float] = field(default_factory=list)
    recovered: set[str] | None = None

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds
        if self.recovered is not None:
            self.recovered.clear()


def installed(
    policy: Policy, reader: DecisionModelReader, solver: ScriptedProvider, ctx: Context, time: VirtualTime
) -> Resources:
    providers = {"author": ScriptedProvider(()), "adversary": ScriptedProvider(()), "solver": solver}
    built = build_resources(policy, providers, {}, FakeWeb({}, {}), decision_model=reader)
    assert built.decision_model is not None
    built = replace(built, decision_model=replace(built.decision_model, clock=time.clock, sleep=time.sleep))
    install(ctx, built)
    return built


async def decided(
    policy: Policy, rollout: bool, history: Sequence[Record] = (), *, fresh: bool = False
) -> tuple[float | None, int, list[DecisionModelRequest], Sequence[Record]]:
    """A solver rollout answering 12, or a grade of the candidate 12, replaying ``history``."""
    reader, solver = FakeReader(), ScriptedProvider((segment("12"),))
    log = EventLog(replay_from=history)
    ctx = Context(log=log)
    installed(policy, reader, solver, ctx, VirtualTime())
    spec = decision_spec(policy)
    if rollout:
        reward = cast(GradedRollout, await ctx.emit(SolverRollout(spec, 0, 0))).grade.reward
    else:
        reward = (await ctx.emit(GradeCandidate(spec, Candidate("12"), fresh_verifier=fresh))).reward
    await ctx.drain()
    return reward, len(solver.bodies), reader.reads, log.records


@pytest.mark.parametrize("rollout", [False, True], ids=["grade", "rollout"])
async def test_an_equivalent_read_replays_without_a_call_and_another_model_identity_reads_again(rollout: bool) -> None:
    reward, solved, paid, history = await decided(overlaid(), rollout)
    assert (reward, solved, paid) == (1.0, int(rollout), [request("12")])
    assert (await decided(overlaid(), rollout, history))[:3] == (1.0, 0, [])
    reward, solved, paid, _ = await decided(overlaid("clef@fixture-2"), rollout, history)
    assert (reward, solved, paid) == (1.0, int(rollout), [request("12", "clef@fixture-2")])


async def test_another_grade_of_the_same_answer_pays_no_second_read() -> None:
    """A fresh verifier changes the outer grade's identity, never the read's."""
    *_, history = await decided(overlaid(), rollout=False)
    reward, _, paid, _ = await decided(overlaid(), rollout=False, history=history, fresh=True)
    assert (reward, paid) == (1.0, [])


async def test_a_read_waits_out_an_outage_with_backoff_then_raises_the_typed_hold() -> None:
    """Never TimeoutError, which RolloutEngine would report as an expired verifier deadline."""
    policy = overlaid().overlaid({"decision_model.unavailable_wait": 100.0, "decision_model.backoff_initial": 40.0})
    reader, time, ctx = FakeReader(down={"12"}), VirtualTime(), Context(log=EventLog())
    installed(policy, reader, ScriptedProvider(()), ctx, time)
    with pytest.raises(DecisionModelUnavailable, match=f"{MODEL} answered no read for 100s; last HTTP 503"):
        await ctx.emit(GradeCandidate(decision_spec(policy), Candidate("12")))
    await ctx.drain()
    assert (time.slept, len(reader.reads)) == ([40.0, 60.0], 3)  # retry-after 30 never shortens the backoff
    events = [record.event for record in ctx._log.records]
    assert [event.attempt for event in events if isinstance(event, Retrying)] == [1, 2]
    assert not [event for event in events if isinstance(event, Resolved)]  # nothing deferred is recorded paid


async def test_a_read_that_recovers_inside_its_wait_resolves_as_one_read() -> None:
    reader, policy, ctx = FakeReader(down={"12"}), overlaid(), Context(log=EventLog())
    installed(policy, reader, ScriptedProvider(()), ctx, VirtualTime(recovered=reader.down))
    grade = await ctx.emit(GradeCandidate(decision_spec(policy), Candidate("12")))
    await ctx.drain()
    assert (grade.reward, len(reader.reads)) == (1.0, 2)
    assert len([r for r in ctx._log.records if isinstance(r.event, Resolved)]) == 2  # one read, one grade


class BarrierReader(FakeReader):
    """Counts reads in flight; each waits until ``release`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.active = self.peak = 0
        self.release, self.full = asyncio.Event(), asyncio.Event()

    @override
    async def read(self, request: DecisionModelRequest) -> float:
        self.active += 1
        self.peak = max(self.peak, self.active)
        if self.active == 2:
            self.full.set()
        await self.release.wait()
        self.active -= 1
        return 4.0


async def test_reads_in_flight_never_exceed_the_configured_concurrency() -> None:
    policy = overlaid().overlaid({"decision_model.concurrency": 2})
    reader, ctx = BarrierReader(), Context(log=EventLog())
    installed(policy, reader, ScriptedProvider(()), ctx, VirtualTime())
    spec = decision_spec(policy)
    grades = [asyncio.create_task(ctx.emit(GradeCandidate(spec, Candidate(str(n))))) for n in range(5)]
    await reader.full.wait()
    for _ in range(20):  # every other grade gets as far as it can
        await asyncio.sleep(0)
    assert (reader.active, reader.peak) == (2, 2)
    reader.release.set()
    assert [grade.reward for grade in await asyncio.gather(*grades)] == [1.0] * 5
    await ctx.drain()
    assert reader.peak == 2


async def test_an_off_scale_level_is_an_infrastructure_error_that_is_read_again_on_resume() -> None:
    policy = overlaid()
    reader, ctx = FakeReader(levels={"12": 7}), Context(log=EventLog())
    installed(policy, reader, ScriptedProvider(()), ctx, VirtualTime())
    grade = await ctx.emit(GradeCandidate(decision_spec(policy), Candidate("12")))
    await ctx.drain()
    assert (grade.status, grade.reward) == (Outcome.INFRA_ERROR, None)
    assert grade.error == f"Decision model {MODEL!r} returned 7, not a level in 0..4"
    resolved = [r.event for r in ctx._log.records if isinstance(r.event, Resolved)]
    read = next(event for event in resolved if isinstance(event.request, DecisionModelRead))
    assert not replayable_grade(read)
    assert replayable_grade(replace(read, value=3.5))


async def test_without_a_decision_model_its_grader_is_returned_to_the_author_before_any_validation() -> None:
    policy = overlaid("")
    task = arithmetic_task(grader=GRADER)
    with pytest.raises(InvalidTaskSpec, match="Decision-model grading is disabled"):
        lower_with_settings(task, lowering_settings(policy), {})
    submission = Submission(
        task, (Control(Candidate("12"), RewardBand(Outcome.GRADED, 1, 1), ControlKind.POSITIVE),), ""
    )
    settings = lowering_settings(policy)
    assert submit_gate(submission, settings=settings, factories={})[0].startswith("Decision-model grading is disabled")


def test_the_author_schema_offers_the_decision_model_kind_only_where_one_is_configured() -> None:
    assert "DecisionModelGrader" not in json.dumps(task_schema(False))
    assert "DecisionModelGrader" in json.dumps(task_schema(True))
    without, with_model = system_prompt(overlaid("")), system_prompt(overlaid())
    assert "## Decision-model graders" not in without and "DecisionModelGrader" not in without
    assert "## Decision-model graders" in with_model and "DecisionModelGrader" in with_model
    assert digest(overlaid("")) != digest(overlaid())  # the stage completions record which mode ran


@pytest.mark.parametrize(
    "overlay, error",
    [
        ({"decision_model.concurrency": 0}, "decision_model.concurrency must be positive and finite"),
        (
            {"decision_model.request_timeout": float("inf"), "decision_model.backoff_initial": -1.0},
            "decision_model.request_timeout, decision_model.backoff_initial must be positive and finite",
        ),
        ({"decision_model.unavailable_wait": 4000.0}, "must end before solver.turn_timeout"),
    ],
)
def test_decision_model_settings_are_checked_at_wiring(overlay: Mapping[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        decision_model_policy(overlaid().overlaid(overlay))


def test_a_missing_decision_model_setting_fails_at_wiring_naming_it() -> None:
    entries = {key: value for key, value in overlaid().entries.items() if key != "decision_model.backoff_maximum"}
    with pytest.raises(KeyError, match=re.escape("decision_model.backoff_maximum")):
        decision_model_policy(Policy(entries))


EXACTLY = re.escape("supplied exactly when policy decision_model.model is set")


def test_a_reader_is_supplied_exactly_when_a_decision_model_is_configured() -> None:
    providers = {role: ScriptedProvider(()) for role in ROLES}
    with pytest.raises(ValueError, match=EXACTLY):
        build_resources(overlaid(""), providers, {}, FakeWeb({}, {}), decision_model=FakeReader())
    with pytest.raises(ValueError, match=EXACTLY):
        build_resources(overlaid(), providers, {}, FakeWeb({}, {}))


def controls() -> tuple[Control, ...]:
    return (
        Control(Candidate("12"), RewardBand(Outcome.GRADED, 1, 1), ControlKind.POSITIVE),
        Control(Candidate("13"), RewardBand(Outcome.GRADED, reward_max=0), ControlKind.NEGATIVE),
    )


async def late_twelve(_progress: Progress) -> Segment:
    """A solver turn still in flight while a sibling's decision-model wait runs out: an outage cancelling its siblings
    would cancel this paid turn."""
    for _ in range(200):
        await asyncio.sleep(0)
    return segment("12")


async def validated(
    proposal: TaskProposal, reader: FakeReader, history: Sequence[Record] = ()
) -> tuple[Validated | BaseExceptionGroup, int, Sequence[Record]]:
    """Validate a decision-model task: three solver trials answering 12, and the controls 12 and 13."""
    policy = overlaid().overlaid({"decision_model.unavailable_wait": 10.0})
    solver = ScriptedProvider((late_twelve,) * 3)
    log = EventLog(replay_from=history)
    ctx = Context(log=log)
    workpiece = Workpiece("w", proposal, submission=Submission(arithmetic_task(grader=GRADER), controls(), ""))
    outcome: Validated | BaseExceptionGroup
    try:
        async with ctx.scope("workpiece"):
            built = installed(policy, reader, solver, ctx, VirtualTime())
            outcome = await validate(ctx, workpiece, build_config(policy, {}), built)
    except* DecisionModelUnavailable as error:
        outcome = error
    await ctx.drain()
    return outcome, len(solver.bodies), log.records


# The decision model is down for the negative control's answer only, so its grade waits while the solver wave and
# the positive control finish; resumed, it pays the deferred read alone and reaches the uninterrupted result.
async def test_an_outage_mid_wave_keeps_finished_siblings_and_resumes_paying_only_the_deferred_read(
    proposal: TaskProposal,
) -> None:
    clean, _, _ = await validated(proposal, FakeReader())
    reader = FakeReader(down={"13"})
    held, solved, history = await validated(proposal, reader)
    assert isinstance(held, BaseExceptionGroup) and held.split(DecisionModelUnavailable)[1] is None
    assert solved == 3
    healthy = FakeReader()
    resumed, solved, _ = await validated(proposal, healthy, history)
    assert isinstance(resumed, Validated) and isinstance(clean, Validated)
    assert (solved, healthy.reads) == (0, [request("13")])
    assert resumed == clean and all(control.met for control in resumed.result.controls)
    assert [trial.grade.reward for trial in resumed.result.solver if trial.grade] == [1.0] * 3


async def test_an_outage_in_a_solver_slot_starts_no_waiting_slot_and_resumes_only_unfinished_slots(
    proposal: TaskProposal,
) -> None:
    """With one slot at a time, the first slot's outage leaves the other two unstarted: no rollout is bought to wait."""
    reader = FakeReader(down={"12"})
    policy = overlaid().overlaid({"decision_model.unavailable_wait": 10.0, "solver.concurrency": 1})
    solver, ctx = ScriptedProvider((segment("12"),) * 3), Context(log=EventLog())
    spec = decision_spec(policy)
    async with ctx.scope("workpiece"):
        installed(policy, reader, solver, ctx, VirtualTime())
        with pytest.raises(BaseExceptionGroup) as raised:
            await solver_batch(ctx, spec, "validation", 0, (), 0, 3, concurrency=1, heartbeat=3600.0)
    await ctx.drain()
    assert raised.group_contains(DecisionModelUnavailable) and len(solver.bodies) == 1


def gateway(handler: Callable[[httpx.Request], httpx.Response]) -> ClefReader:
    source = FixedEndpoint(GlmEndpoint("http://gateway/v1", "secret", Pool("high")))
    return ClefReader(source, httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_the_default_reader_asks_clef_the_whole_request_and_returns_the_expected_level() -> None:
    sent: list[httpx.Request] = []

    def answer(call: httpx.Request) -> httpx.Response:
        sent.append(call)
        probabilities = {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.5, "4": 0.5}
        return httpx.Response(200, json={"answers": {"whole": {"probabilities": probabilities}}})

    assert await gateway(answer).read(request("12")) == 3.5
    (call,) = sent
    assert (str(call.url), call.headers["authorization"]) == ("http://gateway/v1/systemone", "Bearer secret")
    assert json.loads(call.content) == {
        "model": "clef",
        "state": {
            "task_deliverable": GRADER.deliverable,
            "rubric": GRADER.rubric,
            "y_reference_answer": "12",
            "z_candidate_answer": "12",
        },
        "questions": {"whole": {"type": "score", "instructions": DECISION_QUESTION, "criteria": list(DECISION_LEVELS)}},
    }


async def test_the_default_reader_reads_unavailability_as_down_and_other_refusals_as_bugs() -> None:
    with pytest.raises(DecisionModelDown, match="HTTP 503: clef_unavailable") as raised:
        await gateway(lambda _: httpx.Response(503, text="clef_unavailable", headers={"retry-after": "30"})).read(
            request("12")
        )
    assert raised.value.retry_after == 30.0
    with pytest.raises(DecisionModelDown, match="unreachable: ConnectError"):
        await gateway(lambda call: (_ for _ in ()).throw(httpx.ConnectError("refused", request=call))).read(
            request("12")
        )
    with pytest.raises(RuntimeError, match="refused the read with HTTP 400"):
        await gateway(lambda _: httpx.Response(400, text="bad state")).read(request("12"))


@pytest.mark.parametrize(
    "status, health",
    [
        (200, DecisionModelHealth.HEALTHY),
        (404, DecisionModelHealth.UNSUPPORTED),
        (405, DecisionModelHealth.UNSUPPORTED),
        (503, DecisionModelHealth.DOWN),
    ],
)
async def test_health_is_the_gateways_own_route_and_never_a_read(status: int, health: DecisionModelHealth) -> None:
    paths: list[str] = []

    def answer(call: httpx.Request) -> httpx.Response:
        paths.append(call.url.path)
        return httpx.Response(status)

    assert await gateway(answer).health() is health
    assert paths == ["/v1/systemone/health"]
