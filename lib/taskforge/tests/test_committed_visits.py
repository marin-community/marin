# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage boundaries through ``driver.one``: a cold launch resumes at the last stage a continuing review or the author
reached and buys nothing a published row holds, while the open frontier executes."""

import asyncio
import hashlib
import json
import math
from argparse import Namespace
from collections.abc import AsyncIterator, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath
from typing import Any, cast, override

import pytest
import shellsim
from lib.taskforge.tests.resource_doubles import CleanReviewer, FakeInference, FakeWeb
from lib.taskforge.tests.resources.inference_fakes import ScriptedProvider
from lib.taskforge.tests.stage_fakes import CandidateEngine, transcript
from lib.taskforge.tests.test_end_to_end import (
    CONFIRMING,
    OUTAGE,
    SHIPPED_K,
    Envs,
    ServedOnce,
    call,
    history,
    published_workpiece,
    resources,
    script,
    tree,
    unexpected_runtime,
)
from lib.taskforge.tests.test_solver_batch import rejected
from rolloutengine.models import (
    GradedRollout,
    Model,
    Rollout,
    RolloutInterrupted,
    RolloutOperation,
    TaskRolloutSpec,
)
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineOutputLimitExceeded
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from taskforge import driver
from taskforge.checkpoint import HEAD
from taskforge.config import TaskforgeConfig, WorkspaceHydration
from taskforge.effects.context import Context
from taskforge.effects.events import Resolved, ScopeEntered, StageBoundary
from taskforge.effects.issues import Hold, LogFailure
from taskforge.effects.log import BufferSink, EventLog, Record
from taskforge.harness.loop import AgenticHarness
from taskforge.harness.protocol import Tool
from taskforge.llm.glm_provider import EndpointSource
from taskforge.llm.policy import GLM_5_3, LLMPolicy, ModelProfile
from taskforge.model.content_hash import digest
from taskforge.model.proposal import TASK_PROPOSAL, TaskProposal
from taskforge.model.review import (
    VALIDATION_RESULT,
    Accept,
    Cause,
    EvidenceKind,
    Incomplete,
    MissingEvidence,
    Retry,
    TrialOutcome,
)
from taskforge.model.stage import Exit, Stage
from taskforge.model.workpiece import Conversation, Terminal, Workpiece
from taskforge.policy import Policy
from taskforge.report import trial_outcomes
from taskforge.resources.bundle import Resources
from taskforge.resources.grading import GradingResource
from taskforge.resources.inference import InferenceClient
from taskforge.resources.rollouts import Engine, SolverRollout
from taskforge.stages.rules import default_review
from taskforge.wiring import build_config, build_resources

RUN_WORKPIECE = driver.run_workpiece

type Slot = tuple[int, int]
"""A solver slot by (trial, attempt)."""

K = SHIPPED_K
ROLES = ("author", "solver", "adversary")
AUTHOR = tuple(completion for completion in script() if completion.tool_calls[0].name != "grade")
"""The author turns without a grade call, so the first grade the grader gives is a control's."""
REVISION = (*AUTHOR[:2], call("submit", {"notes": "revised", "declined": []}, "submit"))
"""A modification round's author turns: the same task, new notes, no finding declined."""


def even(trial: int, attempt: int) -> GradeResult:
    return GradeResult(Outcome.GRADED, float(trial % 2 == 0))


class Paid(Engine):
    """Grades each slot with ``grade``; ``runs`` is every paid execution, in order. An ``outage`` slot meets a
    decision-model outage while graded, an ``exhausted`` slot its output budget, a ``refused`` slot a model-server
    refusal and a ``limited`` slot's generation ``solver.time_limit`` (then it is graded)."""

    def __init__(
        self,
        grade: Callable[[int, int], GradeResult] = even,
        *,
        outage: Iterable[Slot] = (),
        exhausted: Iterable[Slot] = (),
        refused: Iterable[Slot] = (),
        limited: Iterable[Slot] = (),
    ) -> None:
        self.grade, self.outage, self.exhausted, self.refused = grade, set(outage), set(exhausted), set(refused)
        self.limited = set(limited)
        self.runs: list[Slot] = []

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        assert run_id is not None
        slot = trial, attempt = (int(run_id.split(":")[-2]), int(run_id.split(":")[-1]))
        self.runs.append(slot)
        stopped = Rollout(run_id, (), "interrupted")
        if slot in self.outage:
            raise RolloutInterrupted(stopped, RolloutOperation.GRADE) from OUTAGE
        if slot in self.exhausted:
            limit = MachineOutputLimitExceeded("output budget exhausted")
            raise RolloutInterrupted(stopped, RolloutOperation.ADVANCE) from limit
        if slot in self.refused:
            raise rejected()
        said = (transcript(f"trial {trial} attempt {attempt} answers 12"),)
        stop = "time_limit" if slot in self.limited else "stop"
        return GradedRollout(Rollout(run_id, said, stop), self.grade(trial, attempt))


@dataclass(slots=True)
class Grader(Engine):
    """Candidate grades; the first ``unavailable`` of them are unavailable. ``calls`` counts every paid grade."""

    unavailable: int = 0
    calls: int = 0

    @override
    async def run_task(self, spec: TaskRolloutSpec, model: Model, *, run_id: str | None = None) -> GradedRollout:
        self.calls += 1
        if self.calls <= self.unavailable:
            return GradedRollout(Rollout(run_id or "grade", (), "stop"), GradeResult(Outcome.UNAVAILABLE, None))
        return await CandidateEngine().run_task(spec, model, run_id=run_id)


@dataclass(slots=True)
class Launch:
    """One process's resources and what it paid for."""

    bundle: Resources
    solver: Paid
    grader: Grader
    reviewer: CleanReviewer

    @property
    def round_reviews(self) -> int:
        """Round solve-reviews this launch paid for; a final review's opening names the earlier reports."""
        return sum("Earlier solve-review reports" not in str(r.conversation[:2]) for r in self.reviewer.requests)


def launch(
    solver: Paid,
    grader: Grader | None = None,
    inference: FakeInference | None = None,
) -> Launch:
    grader, reviewer = grader or Grader(), CleanReviewer()
    live = resources(Envs(), AUTHOR)
    bundle = replace(
        live,
        inference=inference or live.inference,
        rollouts=replace(live.rollouts, engine=solver),
        grading=GradingResource(grader, "grader"),
        role_inference={"solve_review": reviewer},
    )
    return Launch(bundle, solver, grader, reviewer)


@dataclass(slots=True)
class Driver:
    """Runs ``driver.one`` over one proposal root, each launch on its own resources, configuration and work
    directory; ``entries`` holds each launch's stage-entry workpiece digests."""

    args: Namespace
    monkeypatch: pytest.MonkeyPatch
    entries: list[list[str]]

    @property
    def root(self) -> Path:
        return Path(self.args.output) / driver.proposal_key(self.args.proposals)

    async def run(
        self,
        attempt: Launch,
        settings: Mapping[str, object] = {},
        profiles: Mapping[str, ModelProfile] = {},
        configure: Callable[[TaskforgeConfig], TaskforgeConfig] = lambda config: config,
        cut: int | None = None,
        wired: bool = False,
    ) -> int:
        """One launch; ``cut`` cancels it as it enters its stage of that index, as a killed process stops, and
        ``wired`` keys its solver slots as the public wiring keys them under this launch's policy."""
        overlay = {"proposals.environments": ["container"], **CONFIRMING, **settings}
        Path(self.args.policy).write_text(json.dumps(overlay))

        @asynccontextmanager
        async def runtime(
            policy: Policy, _source: EndpointSource, _key: str, **_wiring: object
        ) -> AsyncIterator[tuple[TaskforgeConfig, Resources]]:
            bundle = attempt.bundle
            if wired:
                providers = {role: ScriptedProvider((), policy.setting(f"{role}.model", str)) for role in ROLES}
                key = build_resources(policy, providers, {}, FakeWeb({}, {})).rollouts.config_digest
                bundle = replace(bundle, rollouts=replace(bundle.rollouts, config_digest=key))
            yield configure(build_config(policy, {}, profiles)), bundle

        seen: list[str] = []
        self.entries.append(seen)
        original = RUN_WORKPIECE

        async def run_workpiece(*args: Any, reached: Callable[[Workpiece], None], **kwargs: Any) -> Workpiece:
            def entered(wp: Workpiece) -> None:
                if len(seen) == cut:
                    raise asyncio.CancelledError
                seen.append(digest(wp))
                reached(wp)

            return await original(*args, reached=entered, **kwargs)

        self.monkeypatch.setattr(driver, "runtime", runtime)
        self.monkeypatch.setattr(driver, "run_workpiece", run_workpiece)
        return await driver.one(self.args)

    def rows(self) -> tuple[Record, ...]:
        return history(self.root)


@pytest.fixture
def launches(proposal: TaskProposal, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Driver:
    monkeypatch.setenv("GLM_API_KEY", "fake")
    source = tmp_path / "test.proposal.json"
    source.write_bytes(TASK_PROPOSAL.dump_json(proposal))
    args = Namespace(
        proposals=str(source),
        policy=str(tmp_path / "policy.json"),
        output=str(tmp_path / "out"),
        resume_held=False,
        concurrency=2,
        buckets=4096,
        bucket_range=None,
        ranks=(),
        glm_base_url="http://fake/v1",
        glm_relay_job=None,
    )
    return Driver(args, monkeypatch, [])


def kinds(wp: Workpiece) -> list[str]:
    return [decision.kind for decision in wp.decisions]


def scopes(rows: Sequence[Record], name: str) -> int:
    return sum(isinstance(r.event, ScopeEntered) and r.event.name == name for r in rows)


def confirmation(attempt: int, *bought: Iterable[Slot]) -> set[Slot]:
    """The confirmation slots of ``attempt`` that no earlier launch finished."""
    return {(trial, attempt) for trial in range(2, 2 + K)} - {slot for runs in bought for slot in runs}


def finished(launch_: Launch) -> set[Slot]:
    return set(launch_.solver.runs) - launch_.solver.outage


def republish(root: Path, events: bytes) -> None:
    """Replace the root's history with ``events`` under its head, as one chunk."""
    head = json.loads((root / HEAD).read_bytes())
    name = hashlib.sha256(events).hexdigest() + ".jsonl"
    (root / "chunks" / name).write_bytes(events)
    (root / HEAD).write_text(json.dumps({**head, "chunks": [name], "size": len(events)}))


def boundaries(rows: Sequence[Record]) -> list[Stage]:
    """The stage each recorded boundary starts, in journal order."""
    return [r.event.following for r in rows if isinstance(r.event, StageBoundary)]


def published_events(root: Path) -> bytes:
    head = json.loads((root / HEAD).read_bytes())
    return b"".join((root / "chunks" / name).read_bytes() for name in head["chunks"])


async def test_three_cold_launches_enter_identical_visits_and_only_the_first_buys_them(launches: Driver) -> None:
    first, second, third = launch(Paid(outage={(2, 1)})), launch(Paid(outage={(3, 1)})), launch(Paid())
    assert await launches.run(first) == driver.DEFERRED
    assert await launches.run(second) == driver.DEFERRED
    assert await launches.run(third) == 0
    wp = published_workpiece(launches.root)
    assert wp.terminal is Terminal.ACCEPTED and kinds(wp) == ["accept", "accept"]
    one, two, three = launches.entries  # the later launches resume at the confirmation, which held
    assert len(one) == 4 and two == one[3:] == three[:1] and len(three) == 2
    assert sorted(first.solver.runs)[:2] == [(0, 0), (1, 0)]
    assert [slot for slot in (*second.solver.runs, *third.solver.runs) if slot[1] == 0] == []
    assert not finished(first) & set(second.solver.runs)
    assert not (finished(first) | finished(second)) & set(third.solver.runs)
    assert set(third.solver.runs) == confirmation(1, finished(first), finished(second))
    assert (first.grader.calls > 0, second.grader.calls, third.grader.calls) == (True, 0, 0)
    assert (first.round_reviews, second.round_reviews, third.round_reviews) == (1, 0, 0)


async def test_a_cold_launch_keeps_the_retry_a_control_outage_bought(launches: Driver) -> None:
    """Round 0's control grade was unavailable, so its review retried. A cold launch whose grader would now answer
    keeps that Retry: it resumes at the confirmation and only the confirmation frontier executes."""
    first = launch(Paid(outage={(2, 2)}), Grader(unavailable=1))
    assert await launches.run(first) == driver.DEFERRED
    held = len(launches.rows())
    second = launch(Paid())
    assert await launches.run(second) == 0
    wp = published_workpiece(launches.root)
    assert kinds(wp) == ["retry", "accept", "accept"] and wp.terminal is Terminal.ACCEPTED
    assert sorted(first.solver.runs)[:4] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert sorted(second.solver.runs) == sorted(confirmation(2, finished(first)))
    assert (second.grader.calls, second.round_reviews, len(second.reviewer.requests)) == (0, 0, 1)
    new = launches.rows()[held:]
    assert scopes(new, "validate") == 0 and scopes(new, "confirm") == 1  # no duration sample for a committed visit
    assert boundaries(launches.rows()) == [Stage.AUTHOR, Stage.VALIDATE, Stage.VALIDATE, Stage.CONFIRM]


async def test_a_trial_without_a_transcript_stays_absent_beside_a_newer_resolution_of_its_slot(launches: Driver) -> None:
    """Slot (1, 0) ran out of output twice: a scored failure with no transcript. A later resolution of the same key,
    planted after the hold, must not become the transcript the final review reads."""
    first = launch(Paid(exhausted={(1, 0)}, outage={(2, 1)}))
    assert await launches.run(first) == driver.DEFERRED
    rows = launches.rows()
    [solved] = [
        r.event
        for r in rows
        if isinstance(r.event, Resolved)
        and isinstance(r.event.request, SolverRollout)
        and (r.event.request.trial, r.event.request.attempt) == (0, 0)
    ]
    assert isinstance(solved.request, SolverRollout)
    log = EventLog(replay_from=rows)
    intruder = replace(solved.request, trial=1)
    said = GradedRollout(Rollout("x", (transcript("INTRUDER answers 12"),), "stop"), GradeResult(Outcome.GRADED, 1.0))
    planted = (log.append(intruder), log.append(Resolved(intruder, solved.fingerprints, said)))
    sink, events = BufferSink(), published_events(launches.root)
    sink.start(events)
    await sink.write(planted)
    republish(launches.root, events + sink.data)
    second = launch(Paid())
    assert await launches.run(second) == 0
    wp = published_workpiece(launches.root)
    assert wp.evidence[0].solver[1].receipt is None and wp.evidence[-1].solver[1].grade == wp.evidence[0].solver[1].grade
    assert sorted(second.solver.runs) == sorted(confirmation(1, finished(first)))
    gone = MissingEvidence(EvidenceKind.TRANSCRIPTS, "solver trials [1] have no complete transcript")
    assert (final := wp.evidence[-1].review) is not None and final.missing == (gone,)  # the final review's own


@pytest.mark.parametrize("tokens, reviews", [(128, 0), (256, 1)])
async def test_only_a_changed_effective_reviewer_sampling_reopens_the_visit(
    launches: Driver, tokens: int, reviews: int
) -> None:
    """Two solve-review profiles alike in name, output cap and resource identity, differing only in sampling."""
    profile = {"solve_review": replace(GLM_5_3, sampling=LLMPolicy(max_tokens=128))}
    changed = {"solve_review": replace(GLM_5_3, sampling=LLMPolicy(max_tokens=tokens))}
    first = launch(Paid(outage={(2, 1)}))
    assert await launches.run(first, profiles=profile) == driver.DEFERRED
    held = len(launches.rows())
    second = launch(Paid())
    assert await launches.run(second, profiles=changed) == 0
    assert published_workpiece(launches.root).terminal is Terminal.ACCEPTED
    assert (second.round_reviews, scopes(launches.rows()[held:], "validate")) == (reviews, reviews)
    assert [slot for slot in second.solver.runs if slot[1] == 0] == []  # the solver slots replay from the cache


def boundary_entries(lines: Sequence[bytes]) -> list[tuple[int, dict[str, Any], list[Any]]]:
    """Each boundary with evidence: its line index, row and decoded entry, in journal order."""
    rows = [(n, json.loads(line)) for n, line in enumerate(lines)]
    return [
        (n, row, entry)
        for n, row in rows
        if row["event"].get("type") == StageBoundary.kind
        and (entry := json.loads(row["event"]["payload"]["entry"]))[0]["evidence"]
    ]


async def refuses_the_damaged_root(launches: Driver, lines: list[bytes], at: int, row: dict[str, Any]) -> None:
    """Republish ``row`` at line ``at``; the next launch must refuse before claim and leave the tree as it was."""
    lines[at] = json.dumps(row, sort_keys=True).encode() + b"\n"
    republish(launches.root, b"".join(lines))
    before = tree(launches.root)
    launches.monkeypatch.setattr(driver, "runtime", unexpected_runtime)
    with pytest.raises(LogFailure, match=r"trial \d+'s receipt names no resolution of its slot") as e:
        await driver.one(launches.args)
    assert driver.exit_code(e.value) == driver.CORRUPT
    assert tree(launches.root) == before


@pytest.mark.parametrize("damage", ["another row", "a sibling slot"])
async def test_a_malformed_boundary_receipt_leaves_the_held_head_byte_identical(launches: Driver, damage: str) -> None:
    assert await launches.run(launch(Paid(outage={(2, 1)}))) == driver.DEFERRED
    lines = published_events(launches.root).splitlines(keepends=True)
    [(at, row, entry)] = boundary_entries(lines)
    first, second = entry[0]["evidence"][-1]["solver"]
    if damage == "another row":
        first["receipt"] = 1
    else:  # a same-type resolution of the wrong slot
        first["receipt"], second["receipt"] = second["receipt"], first["receipt"]
    row["event"]["payload"]["entry"] = json.dumps(entry)
    await refuses_the_damaged_root(launches, lines, at, row)


def diagnosed(trial: int, attempt: int) -> GradeResult:
    """Round 0 is too easy; later rounds solve half. Diagnostics in nonalphabetic order, as a grader may build them."""
    reward = 1.0 if attempt == 0 else float(trial % 2 == 0)
    return GradeResult(Outcome.GRADED, reward, diagnostics={"zeta": trial, "alpha": attempt})


async def test_nonalphabetic_grade_diagnostics_hydrate_the_same_author_bytes_on_a_cold_rebuild(
    launches: Driver,
) -> None:
    """Round 0's too-easy placement asks for a modification; the round-1 author holds on an outage after hydrating
    ``.validation/0.json``. The cold launch rebuilds that author on a fresh machine from the committed evidence: the
    same bytes, so the recorded turns replay, nothing historical is bought and the open frontier finishes."""
    first_author = FakeInference((*AUTHOR, *REVISION[:2], OUTAGE))
    first = launch(Paid(diagnosed), inference=first_author)
    assert await launches.run(first) == driver.DEFERRED
    assert kinds(published_workpiece(launches.root))[:1] == ["modify"]
    resumed = ServedOnce(REVISION[2:], first_author.requests[:-1])
    second = launch(Paid(diagnosed), inference=resumed)
    held = len(launches.rows())
    assert await launches.run(second) == 0
    wp = published_workpiece(launches.root)
    assert wp.terminal is Terminal.ACCEPTED and kinds(wp) == ["modify", "accept", "accept"]
    assert not resumed.repeated and len(resumed.requests) == 1
    assert sorted(second.solver.runs) == [(0, 1), (1, 1), *sorted(confirmation(2))]
    assert scopes(launches.rows()[held:], "validate") == 1  # round 1's, never round 0's again


def nonfinite(reward: float) -> Callable[[int, int], GradeResult]:
    """Round 0's first trial reports ``reward``, so the round asks for a modification; later rounds solve half."""

    def grade(trial: int, attempt: int) -> GradeResult:
        return GradeResult(Outcome.GRADED, reward if (trial, attempt) == (0, 0) else float(trial % 2 == 0))

    return grade


def validation_files(attempt: Launch) -> list[bytes]:
    """Every ``.validation/0.json`` this launch's machines were given."""
    envs = cast(Envs, attempt.bundle.envs).created
    return [data for env in envs for path, data in env.files.items() if str(path).endswith(".validation/0.json")]


def strict(constant: str) -> object:
    raise ValueError(f"{constant} is not JSON")


@pytest.mark.parametrize("reward", [math.nan, math.inf])
async def test_a_nonfinite_reward_reaches_the_author_as_strict_json_alike_live_and_stored(
    launches: Driver, reward: float
) -> None:
    """Regression: the author's evidence file carried the literal tokens NaN or Infinity, which no strict JSON reader
    accepts, where the wire form writes null."""
    first_author = FakeInference((*AUTHOR, *REVISION[:2], OUTAGE))
    first = launch(Paid(nonfinite(reward)), inference=first_author)
    assert await launches.run(first) == driver.DEFERRED
    assert kinds(published_workpiece(launches.root))[:1] == ["modify"]
    second = launch(Paid(nonfinite(reward)), inference=ServedOnce(REVISION[2:], first_author.requests[:-1]))
    assert await launches.run(second) == 0
    [live], [stored] = validation_files(first), validation_files(second)
    assert live == stored
    evidence = published_workpiece(launches.root).evidence[0]
    wire = json.loads(VALIDATION_RESULT.dump_json(evidence, exclude={"solver": {"__all__": {"receipt"}}}))
    assert json.loads(live, parse_constant=strict) == wire and wire["solver"][0]["grade"]["reward"] is None


async def test_a_heartbeat_only_policy_edit_is_a_new_plan_and_replans_every_visit(launches: Driver) -> None:
    """The documented contract (README): a policy edit is a new plan, so every visit recomputes from the first one,
    even when only an operational setting changed. Here round 0 replays the control grade its retry bought under the
    same key and accepts, so the held launch's paid Retry round (slots (0, 1), (1, 1)) is no longer on the route."""
    first = launch(Paid(outage={(2, 2)}), Grader(unavailable=1))
    assert await launches.run(first) == driver.DEFERRED
    assert kinds(published_workpiece(launches.root))[:2] == ["retry", "accept"]
    second = launch(Paid())
    assert await launches.run(second, {"solver.heartbeat_interval": 7.0}) == 0
    wp = published_workpiece(launches.root)
    # History recomputed: no visit is reused, round 0 decides again on replayed rows and its Retry is gone.
    assert scopes(launches.rows(), "validate") == 3 and len(wp.evidence) == 2
    assert kinds(wp) == ["accept", "accept"] and wp.terminal is Terminal.ACCEPTED
    # What it buys: round 0's solve-review again, and the whole confirmation under attempt 1; no solver slot of round 0.
    assert sorted(second.solver.runs) == sorted(confirmation(1))
    assert (second.grader.calls, second.round_reviews) == (0, 1)


async def test_lowering_the_missing_trial_allowance_turns_a_tolerated_round_into_a_manual_hold(
    launches: Driver,
) -> None:
    settings = {"solver.k": 3, "budget.max_missing_trials": 1}
    first = launch(Paid(refused={(2, 0)}, outage={(3, 1)}))
    assert await launches.run(first, settings) == driver.DEFERRED
    assert kinds(published_workpiece(launches.root))[:1] == ["accept"]
    second = launch(Paid(refused={(2, 0)}))
    assert await launches.run(second, {**settings, "budget.max_missing_trials": 0}) == 0
    wp = published_workpiece(launches.root)
    assert wp.terminal is Terminal.HELD and isinstance(last := wp.decisions[-1], Incomplete)
    assert last.awaits is None and kinds(wp) == ["incomplete"]
    assert (second.solver.runs, second.grader.calls) == ([], 0)


def restated(workpiece: Workpiece, band: Any, budget: Any) -> Any:
    return default_review(workpiece, band, budget)


async def test_a_custom_review_rule_reuses_no_visit(launches: Driver) -> None:
    def custom(config: TaskforgeConfig) -> TaskforgeConfig:
        return replace(config, review=restated)

    assert await launches.run(launch(Paid(outage={(2, 1)})), configure=custom) == driver.DEFERRED
    held = len(launches.rows())
    assert await launches.run(launch(Paid()), configure=custom) == 0
    assert scopes(launches.rows()[held:], "validate") == scopes(launches.rows()[held:], "author") == 1


@dataclass(frozen=True, slots=True)
class HoldsEveryRound(AgenticHarness):
    """A reviewer session that sends every round to a person: a subclass of the stock loop that runs otherwise."""

    @override
    async def run(
        self, ctx: Context, session: Conversation, client: InferenceClient, tools: Mapping[str, Tool[object]]
    ) -> Exit:
        raise Hold("this reviewer sends every round to a person")


def holding_reviewer(config: TaskforgeConfig) -> TaskforgeConfig:
    stock = config.solve_review.harness
    assert type(stock) is AgenticHarness
    harness = HoldsEveryRound(**{item.name: getattr(stock, item.name) for item in fields(stock)})
    return replace(config, solve_review=replace(config.solve_review, harness=harness))


# Regression: a reviewer subclass shared the stock compatibility digest, so a cold launch accepted on the stock
# reviewer's committed evidence while a fresh run with that reviewer held.
async def test_a_reviewer_subclass_reuses_no_stock_visit_and_decides_as_a_fresh_run(launches: Driver) -> None:
    assert await launches.run(launch(Paid(outage={(2, 1)}))) == driver.DEFERRED
    held = len(launches.rows())
    assert await launches.run(launch(Paid()), configure=holding_reviewer) == 0
    cold = published_workpiece(launches.root)
    assert scopes(launches.rows()[held:], "validate") == 1  # the visit executes again
    launches.args.output += "-fresh"
    assert await launches.run(launch(Paid()), configure=holding_reviewer) == 0
    fresh = published_workpiece(launches.root)
    assert cold.terminal is fresh.terminal is Terminal.HELD
    assert cold.decisions[-1] == fresh.decisions[-1]


def invalid_in_confirmation(trial: int, attempt: int) -> GradeResult:
    """Round 0 solves half; its confirmation's slot 3 reports an invalid task, which buys a final revision."""
    return GradeResult(Outcome.INVALID_TASK, None) if (trial, attempt) == (3, 1) else even(trial, attempt)


@pytest.mark.parametrize("plans", [(120, 240, 120), (240, 120, 240)])
async def test_a_relaunch_under_another_plan_leaves_the_shipped_round_its_own(
    launches: Driver, plans: tuple[int, int, int]
) -> None:
    """Regression: a confirmation found the round it ships by evidence index, the latest commitment winning, and a
    relaunch under another solver tool timeout commits its own round 0. Back on the first plan, the live check compared
    the carried trials with the other plan's round and held a run that, relaunched on its own plan, accepts."""

    async def run(timeout: int, solver: Paid, inference: FakeInference | None = None) -> Launch:
        attempt = launch(solver, inference=inference)
        settings = {"budget.final_revisions": 1, "solver.tool_timeout": timeout}
        assert await launches.run(attempt, settings) in (0, driver.DEFERRED)
        return attempt

    first, other, last = plans
    await run(first, Paid(invalid_in_confirmation, outage={(2, 1)}))
    await run(other, Paid(invalid_in_confirmation, outage={(2, 1)}))
    final = await run(last, Paid(invalid_in_confirmation), FakeInference(REVISION))
    mixed = published_workpiece(launches.root)
    launches.args.output += "-control"
    await run(first, Paid(invalid_in_confirmation, outage={(2, 1)}))
    control = await run(first, Paid(invalid_in_confirmation), FakeInference(REVISION))
    alone = published_workpiece(launches.root)
    assert mixed.terminal is alone.terminal is Terminal.ACCEPTED
    assert kinds(mixed) == kinds(alone) and "revise" in kinds(mixed)
    assert sorted(final.solver.runs) == sorted(control.solver.runs)
    assert (final.grader.calls, final.round_reviews) == (control.grader.calls, control.round_reviews)
    assert [slot for slot in final.solver.runs if slot[1] == 0] == []  # round 0 is bought by no later launch


async def test_explicit_resume_retries_only_refusals_in_the_open_confirmation(launches: Driver) -> None:
    """``--resume-held`` replaces the refusals of the visit it resumes and nothing committed: validation's tolerated
    refusal (2, 0) stays at its receipt, an unflagged launch adds no row, and the flagged one buys only the
    confirmation's refused slots, with no validation scope, control grade or round review."""
    settings = {"solver.k": 3, "budget.max_missing_trials": 1}
    first = launch(Paid(refused={(2, 0), (3, 1), (4, 1)}))
    assert await launches.run(first, settings) == 0
    held = published_workpiece(launches.root)
    assert held.terminal is Terminal.HELD and kinds(held) == ["accept", "incomplete"]
    baseline = held.evidence[0].solver[2]
    assert baseline.trial == 2 and baseline.receipt is not None

    rows = len(launches.rows())
    skipped = launch(Paid())
    assert await launches.run(skipped, settings) == 0
    assert len(launches.rows()) == rows and skipped.solver.runs == []

    launches.args.resume_held = True
    resumed = launch(Paid())
    assert await launches.run(resumed, settings) == 0
    wp = published_workpiece(launches.root)
    assert wp.terminal is Terminal.ACCEPTED and kinds(wp) == ["accept", "accept"]
    assert wp.evidence[0].solver[2] == baseline
    assert sorted(resumed.solver.runs) == [(3, 1), (4, 1)]
    assert (scopes(launches.rows()[rows:], "validate"), resumed.grader.calls, resumed.round_reviews) == (0, 0, 0)


RETRYING = {"solver.k": 3, "budget.max_validation_retries": 1, "budget.max_modify_rounds": 0}
"""Three trials, no missing one tolerated, one validation retry and no modification round."""


def confirmed(attempt: int, placed: int = 3) -> list[Slot]:
    """Every confirmation slot of ``attempt`` after three validation trials, ``placed`` of them in the base."""
    return [(trial, attempt) for trial in range(3, 3 + 3 + K - placed)]


def graded(*ungraded: Slot, invalid: Iterable[Slot] = ()) -> Callable[[int, int], GradeResult]:
    """``even``, except that each ``ungraded`` slot has no grade and each ``invalid`` one reports an invalid task."""
    invalid = set(invalid)

    def grade(trial: int, attempt: int) -> GradeResult:
        if (trial, attempt) in ungraded:
            return GradeResult(Outcome.UNAVAILABLE, None)
        return GradeResult(Outcome.INVALID_TASK, None) if (trial, attempt) in invalid else even(trial, attempt)

    return grade


def cuts(*stages: int) -> pytest.MarkDecorator:
    """No launch between, or one killed as it enters a stage after resuming at the retry's validation: that validation,
    the retry's review once its slots are bought, the confirmation, or the final review once its slots are bought."""
    return pytest.mark.parametrize(
        "cut", [None, *stages], ids=lambda cut: "two launches" if cut is None else f"killed at stage {cut}"
    )


@cuts(0, 1)
async def test_a_retried_round_keeps_its_consumed_slots_and_its_replacement_fault(
    launches: Driver, cut: int | None
) -> None:
    """Round 0's slots 1 and 2 were ungraded, so its review retried. In the retry, slot 1 reported an invalid task and
    slot 2 then met a decision-model outage. Every later launch keeps round 0's slots historical though their grader
    would now answer, reuses the invalid task, finishes slot 2 once and abandons. A launch killed at any stage boundary
    on the way (``cut``) changes neither what is bought nor the decision."""
    first = launch(Paid(graded((1, 0), (2, 0), invalid={(1, 1)}), outage={(2, 1)}))
    assert await launches.run(first, RETRYING) == driver.DEFERRED
    assert kinds(published_workpiece(launches.root))[:1] == ["retry"]
    assert sorted(first.solver.runs) == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    bought: list[Slot] = []
    if cut is not None:
        killed = launch(Paid(graded(invalid={(1, 1)})))
        with pytest.raises(asyncio.CancelledError):
            await launches.run(killed, RETRYING, cut=cut)
        bought = killed.solver.runs
    last = launch(Paid(graded(invalid={(1, 1)})))
    assert await launches.run(last, RETRYING) == 0
    wp = published_workpiece(launches.root)
    assert kinds(wp) == ["retry", "abandon"] and wp.terminal is Terminal.ABANDONED
    assert [t.cause for t in wp.evidence[1].solver] == [None, Cause.INVALID_TASK, None]
    assert (bought, last.solver.runs) == (([(2, 1)], []) if cut == 1 else ([], [(2, 1)]))
    assert last.grader.calls == 0


@cuts(0, 1, 2, 3)
async def test_an_unreviewed_replacement_stays_open_for_every_later_launch(launches: Driver, cut: int | None) -> None:
    """Round 0's slot 1 was ungraded, so its review retried; in the retry slot 1 was ungraded again and slot 2 met a
    decision-model outage, and no review used either. Later launches whose grader answers buy both once and accept,
    however a launch between them was killed."""
    first = launch(Paid(graded((1, 0), (1, 1)), outage={(2, 1)}))
    assert await launches.run(first, RETRYING) == driver.DEFERRED
    assert sorted(first.solver.runs) == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    bought: list[Slot] = []
    if cut is not None:
        killed = launch(Paid())
        with pytest.raises(asyncio.CancelledError):
            await launches.run(killed, RETRYING, cut=cut)
        bought = sorted(killed.solver.runs)
    last = launch(Paid())
    assert await launches.run(last, RETRYING) == 0
    wp = published_workpiece(launches.root)
    assert kinds(wp) == ["retry", "accept", "accept"] and wp.terminal is Terminal.ACCEPTED
    retry, untouched = [(1, 1), (2, 1)], ([], [(1, 1), (2, 1), *confirmed(2)])
    expected = {None: untouched, 0: untouched, 1: (retry, confirmed(2)), 2: (retry, confirmed(2))}
    assert (bought, sorted(last.solver.runs)) == expected.get(cut, ([*retry, *confirmed(2)], []))
    assert [t.cause for t in wp.evidence[0].solver] == [None, Cause.NO_GRADE, None]


@pytest.mark.parametrize("stage", ["validation", "confirmation"])
@pytest.mark.parametrize(
    "grade", [GradeResult(Outcome.GRADED, 1.0), GradeResult(Outcome.INFRA_ERROR, None, failure=GradingFailure.TIMEOUT)]
)
async def test_a_time_limited_trial_is_a_typed_failed_trial_no_later_launch_buys_again(
    launches: Driver, stage: str, grade: GradeResult
) -> None:
    """Slot 1's generation ran out ``solver.time_limit``: it counts unsolved whatever its grade says, a grader
    infrastructure failure on its partial output included, with its own outcome. Neither the review nor a later
    launch buys it again, whether the launch that bought it deferred before its visit was reviewed or after."""

    def scored(trial: int, attempt: int) -> GradeResult:
        return grade if (trial, attempt) == (1, 0) else even(trial, attempt)

    outage = {(2, 0) if stage == "validation" else (3, 1)}
    first = launch(Paid(scored, limited={(1, 0)}, outage=outage))
    assert await launches.run(first, RETRYING) == driver.DEFERRED
    last = launch(Paid())
    assert await launches.run(last, RETRYING) == 0
    wp = published_workpiece(launches.root)
    assert kinds(wp) == ["accept", "accept"] and wp.terminal is Terminal.ACCEPTED
    trial = wp.evidence[0].solver[1]
    assert (trial.cause, trial.outcome, trial.grade) == (Cause.TIME_LIMIT, TrialOutcome.TIME_LIMIT, grade)
    assert first.solver.runs.count((1, 0)) == 1 and (1, 0) not in last.solver.runs
    assert sorted(last.solver.runs) == sorted({*outage, *confirmed(1)} - finished(first))
    assert isinstance(accepted := wp.decisions[-1], Accept)
    assert "Counted unsolved without a gradable answer: 1 time_limit" in accepted.reasons
    assert "Confirmation: 4 of 8 trials solved: in_band" in accepted.reasons
    assert trial_outcomes(wp) == {TrialOutcome.GRADED: 7, TrialOutcome.TIME_LIMIT: 1}  # each slot once


async def test_a_retry_beside_a_time_limited_trial_records_the_missing_grade_as_its_cause(launches: Driver) -> None:
    """Slot 0 ran out ``solver.time_limit`` and slot 1 has no grade: the retry is for the missing grade, and the journal
    says so. A limit is an unsolved trial, never a retry's cause, even when it comes first."""
    first = launch(Paid(graded((1, 0)), limited={(0, 0)}))
    assert await launches.run(first, RETRYING) == 0
    wp = published_workpiece(launches.root)
    assert wp.decisions[0] == Retry(Cause.NO_GRADE, 1) and kinds(wp) == ["retry", "accept", "accept"]
    assert wp.evidence[0].solver[0].outcome is TrialOutcome.TIME_LIMIT


async def test_a_changed_time_limit_is_another_plan_and_the_old_limit_reuses_its_paid_slots(launches: Driver) -> None:
    """The solver's replay identity carries ``solver.time_limit``: under another limit round 0 is bought again, and
    back at the first limit every launch starts at the author but buys only the first plan's open confirmation
    slots."""
    plans = [launch(Paid(outage={(3, 1)})), launch(Paid(outage={(3, 1)})), launch(Paid())]
    for plan, limit in zip(plans[:2], (4200.0, 3600.0), strict=True):
        assert await launches.run(plan, {**RETRYING, "solver.time_limit": limit}, wired=True) == driver.DEFERRED
    assert await launches.run(plans[2], {**RETRYING, "solver.time_limit": 4200.0}, wired=True) == 0
    assert published_workpiece(launches.root).terminal is Terminal.ACCEPTED
    other, back = plans[1].solver.runs, plans[2].solver.runs
    assert [slot for slot in other if slot[1] == 0] == [(0, 0), (1, 0), (2, 0)]
    assert sorted(back) == sorted(set(confirmed(1)) - finished(plans[0]))
    assert boundaries(launches.rows()).count(Stage.AUTHOR) == 3  # each plan change resets to the author


async def test_explicit_resume_rereads_the_live_grade_and_keeps_the_paid_fault(launches: Driver) -> None:
    """Round 0's slot 0 was refused, inside the allowance of two, and slots 1 and 2 were ungraded, so its
    review retried. The retry's slot 1 reported an invalid task and slot 2's grader failed for infrastructure, a manual
    hold. An unflagged launch adds nothing; ``--resume-held`` buys only slot 2 again, keeps round 0's refusal and the
    invalid task, and abandons."""
    settings = {**RETRYING, "budget.max_missing_trials": 2}
    infra = GradeResult(Outcome.INFRA_ERROR, None, failure=GradingFailure.TIMEOUT)

    def grade(trial: int, attempt: int) -> GradeResult:
        return infra if (trial, attempt) == (2, 1) else graded((1, 0), (2, 0), invalid={(1, 1)})(trial, attempt)

    first = launch(Paid(grade, refused={(0, 0)}))
    assert await launches.run(first, settings) == 0
    held = published_workpiece(launches.root)
    assert held.terminal is Terminal.HELD and kinds(held) == ["retry", "incomplete"]
    rows = len(launches.rows())
    skipped = launch(Paid())
    assert await launches.run(skipped, settings) == 0
    assert len(launches.rows()) == rows and skipped.solver.runs == []
    launches.args.resume_held = True
    resumed = launch(Paid(graded(invalid={(1, 1)})))
    assert await launches.run(resumed, settings) == 0
    wp = published_workpiece(launches.root)
    assert kinds(wp) == ["retry", "abandon"] and wp.terminal is Terminal.ABANDONED
    assert resumed.solver.runs == [(2, 1)] and resumed.grader.calls == 0
    assert wp.evidence[0].solver[0].cause is Cause.HELD and wp.evidence[1].solver[1].cause is Cause.INVALID_TASK


async def test_a_plan_switched_back_after_a_reset_starts_at_the_author_and_buys_only_what_no_row_holds(
    launches: Driver,
) -> None:
    """A holds in its confirmation; B, a policy edit, is killed just after its reset boundary; A's next launch starts at
    the author because B's reset is the latest boundary, and every paid effect it meets hits its own key."""
    first = launch(Paid(outage={(2, 1)}))
    assert await launches.run(first) == driver.DEFERRED
    with pytest.raises(asyncio.CancelledError):
        await launches.run(launch(Paid()), {"solver.heartbeat_interval": 7.0}, cut=0)
    again = launch(Paid())
    assert await launches.run(again) == 0
    assert published_workpiece(launches.root).terminal is Terminal.ACCEPTED
    first_plan, reset, back = [Stage.AUTHOR, Stage.VALIDATE, Stage.CONFIRM], [Stage.AUTHOR], [Stage.AUTHOR]
    assert boundaries(launches.rows()) == [*first_plan, *reset, *back, Stage.VALIDATE, Stage.CONFIRM]
    assert launches.entries[2][0] == launches.entries[0][0]  # the author, from the initial workpiece
    assert sorted(again.solver.runs) == sorted(confirmation(1, finished(first)))
    assert (again.grader.calls, again.round_reviews) == (0, 0)


@pytest.mark.parametrize(
    "workspace",
    [
        WorkspaceHydration((inline_resource("reference.txt", b"V2"),)),
        WorkspaceHydration((inline_resource("reference.txt", b"V1"),), rendered=lambda _wp, _policy: ()),
    ],
    ids=["changed static content", "custom renderer"],
)
async def test_a_changed_or_unidentified_workspace_never_resumes_past_the_author(
    launches: Driver, workspace: WorkspaceHydration
) -> None:
    """The author may have built its task from ``/reference.txt``: a launch whose workspace differs, or whose renderer
    has no identity, starts at the author on a machine holding what this launch hydrates."""
    v1 = WorkspaceHydration((inline_resource("reference.txt", b"V1"),))
    assert await launches.run(launch(Paid(outage={(2, 1)})), configure=lambda c: replace(c, workspace=v1)) == (
        driver.DEFERRED
    )
    held = len(launches.rows())
    second = launch(Paid())
    assert await launches.run(second, configure=lambda c: replace(c, workspace=workspace)) == 0
    assert scopes(launches.rows()[held:], "author") == 1
    author = cast(Envs, second.bundle.envs).created[0]
    assert author.files[PurePosixPath("/reference.txt")] == resource_bytes(workspace.static[0])


class InjectedMachines(ShellSimMachineFactory):
    """A caller's own machine factory: it may configure machines in ways no plan identity sees."""


async def test_an_injected_machine_factory_never_resumes_past_the_author(launches: Driver) -> None:
    """The same backend name as the stock factory of the held launch, so only the factory's type differs."""
    first = launch(Paid(outage={(2, 1)}))
    first.bundle = replace(first.bundle, factories={"shellsim": ShellSimMachineFactory(context=shellsim.Context())})
    assert await launches.run(first) == driver.DEFERRED
    held = len(launches.rows())
    second = launch(Paid())
    second.bundle = replace(second.bundle, factories={"shellsim": InjectedMachines(context=shellsim.Context())})
    assert await launches.run(second) == 0
    new = launches.rows()[held:]
    assert scopes(new, "author") == 1 and boundaries(new)[0] is Stage.AUTHOR


async def test_a_launch_after_the_published_author_boundary_opens_no_author_machine_or_model_call(
    launches: Driver,
) -> None:
    """Killed as validation starts, after the author's boundary; the next launch starts at validation."""
    with pytest.raises(asyncio.CancelledError):
        await launches.run(launch(Paid()), cut=1)
    assert boundaries(launches.rows()) == [Stage.AUTHOR, Stage.VALIDATE]
    held = len(launches.rows())
    again = launch(Paid(), inference=FakeInference(()))  # any author turn would raise
    assert await launches.run(again) == 0
    assert published_workpiece(launches.root).terminal is Terminal.ACCEPTED
    assert scopes(launches.rows()[held:], "author") == 0
    assert cast(FakeInference, again.bundle.inference).requests == []
    assert cast(Envs, again.bundle.envs).created == []


@pytest.mark.parametrize("change", ["author sampling", "proposal bytes"])
async def test_a_changed_author_session_or_proposal_resumes_at_the_author(
    launches: Driver, proposal: TaskProposal, change: str
) -> None:
    assert await launches.run(launch(Paid(outage={(2, 1)}))) == driver.DEFERRED
    held = len(launches.rows())
    profiles = {"author": replace(GLM_5_3, sampling=LLMPolicy(max_tokens=128))} if change == "author sampling" else {}
    if change == "proposal bytes":
        edited = replace(proposal, body=proposal.body + "\nOne more sentence.")
        Path(launches.args.proposals).write_bytes(TASK_PROPOSAL.dump_json(edited))
    assert await launches.run(launch(Paid()), profiles=profiles) == 0
    new = launches.rows()[held:]
    assert scopes(new, "author") == 1 and boundaries(new)[0] is Stage.AUTHOR
