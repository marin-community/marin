# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""validate_round: the unit graph, resuming from the unit files, files keyed by what they cover, cancellation, and
the adversary trials.

Invariants: the grader rejects incorrect answers (each FALSE_ACCEPT submission graded; finish_search refuses prose or
an uncited defect); the solver cannot read any grader file (the white-box brief holds the controls); the solve rate
rests on complete graded evidence (no early stop without a MODIFY round); the validation evidence belongs to the exact
task (trial files keyed by the lowered spec and control-set digests).
"""

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import pytest
from pydantic import BaseModel
from rolloutengine.models import Model, ModelRequest, ModelTurn, Rollout, RolloutLimits
from shellbox.machine import Backend, MachineSpec, MachineTerminated
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.grading_result import Outcome as GradeStatus
from taskcompendium.models import AnswerType, PlainText, Source
from verifyit.spec import ExactSpec

from taskforge.defaults import DEFAULT_POLICY
from taskforge.ledger.records import EntryKind, Ledger, LedgerEntry
from taskforge.llm.client import (
    Completion,
    FinishReason,
    ModelClient,
    ModelRequestRejected,
    ModelUnavailable,
    StructuredResult,
    ToolCall,
    Usage,
)
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.sandbox.factories import MachineHost
from taskforge.spec import draft
from taskforge.stage10_proposal.model import parse
from taskforge.stage30_author.loader import GRADER_STEP
from taskforge.stage40_build_with_program.outputs import GradedCandidate, Grader
from taskforge.stage40_build_with_program.run import RLEnvBuildStepOutcome, TaskDraft, WorkpieceBuildProvenance
from taskforge.stage40_build_with_program.step_store import (
    SDK_VERSION,
    CacheStatus,
    RLEnvBuildStepKey,
    RLEnvBuildStepStore,
)
from taskforge.stage50_validate.adversary import (
    FALSE_ACCEPT,
    FALSE_REJECT,
    AdversaryMode,
    AdversaryPolicy,
    Defect,
    SearchReport,
    Submission,
    SubmissionRecord,
    trial_outcome,
)
from taskforge.stage50_validate.control_drafts import CONTROLS_STEP
from taskforge.stage50_validate.controls import ControlVerdict
from taskforge.stage50_validate.outcome import Cause, Ungraded
from taskforge.stage50_validate.round import (
    EarlyDecision,
    LoweringStatus,
    TrialStatus,
    ValidationContext,
    ValidationResult,
    ValidationSettings,
    load_validation,
    validate_round,
    write_validation,
)
from taskforge.stage50_validate.trials import EngineSettings, RetryBackoff
from taskforge.stage60_review.decision import Accept, BandOutcome, Hold, Modify
from taskforge.stage60_review.rules import BandRules, default_review
from tests.sandbox.fixture_images import FixtureImageFactory, shell_grader
from tests.stage40_build_with_program.conftest import PROPOSAL
from tests.stage50_validate.conftest import ENGINE_LIMITS, FACTORIES

WORKPIECE = "workpiece"
TASK = draft.assemble(
    "workpiece",
    "What is 6 * 7? Reply with the number alone.",
    AnswerType.TEXT,
    PlainText(),
    draft.answer_grader(ExactSpec(expected=("42",))),
    Source(dataset="test", revision="r", row="0", importer_revision="test"),
    environment=draft.requirements(image=None),
)
GRADER = Grader(
    package=draft.answer_grader(ExactSpec(expected=("42",))),
    answer_contract="Reply with the number alone.",
    reference_reply="42",
    reference_files=(),
    secret_values=("42",),
    rejection_ceiling=0.0,
    prototyped=(GradedCandidate("42", (), Outcome.GRADED, 1.0), GradedCandidate("", (), Outcome.GRADED, 0.0)),
)
CONTROL_IDS = ["P-reference", "N1-shortcut", "N2-off-by-one", "N3"]


def controls_reply(wrong: str) -> str:
    """A control set ``write_controls`` accepts, whose plausible-wrong control replies ``wrong``."""
    fields = ("id", "kind", "category", "concern", "final_reply", "reward_min", "reward_max")
    rows = (
        ("P-reference", "positive", "known_correct", "reference", "42", 0.99, None),
        ("N1-shortcut", "negative", "task_specific_shortcut", "shortcut", "6 * 7", None, 0.0),
        ("N2-off-by-one", "negative", "plausible_wrong", "acceptance", wrong, None, 0.0),
        ("N3", "malformed", "empty_or_malformed", "extraction", "", None, 0.0),
    )
    return json.dumps(
        {"controls": [{**dict(zip(fields, row, strict=True)), "files": [], "rationale": "r"} for row in rows]}
    )


def built_draft(store_root: Path) -> TaskDraft:
    """``TASK`` as a build leaves it: its grader step output stored under ``store_root``."""
    key = RLEnvBuildStepKey(sdk_version=SDK_VERSION, source_digest="grader", inputs={}, model_digest="model")
    memo = RLEnvBuildStepStore(str(store_root), WORKPIECE).store(key, GRADER_STEP, GRADER)
    provenance = WorkpieceBuildProvenance(
        workpiece_id=WORKPIECE,
        proposal_digest="proposal",
        program_digest="program",
        sdk_version=SDK_VERSION,
        model="model",
        model_digest="model",
        round=0,
        steps=(RLEnvBuildStepOutcome(CacheStatus.MISS, memo),),
    )
    return TaskDraft(task=TASK, provenance=provenance)


def adversary_policy(k: int, submissions: int = 3, mode: AdversaryMode = AdversaryMode.BLACKBOX) -> AdversaryPolicy:
    return AdversaryPolicy(
        spec=FALSE_ACCEPT,
        k=k,
        submissions=submissions,
        max_turns=6,
        sampling=LLMPolicy(max_continuations=0),
        mode=mode,
        min_probes=1,
        prompts=DEFAULT_POLICY,
    )


def round_context(tmp_path: Path, modify_rounds_left: int = 0) -> ValidationContext:
    return ValidationContext(
        workpiece_id=WORKPIECE,
        round=0,
        round_dir=tmp_path / "round",
        proposal=parse(PROPOSAL),
        store_root=str(tmp_path / "store"),
        modify_rounds_left=modify_rounds_left,
    )


def round_settings(
    ledger,
    k: int = 2,
    limits: RolloutLimits | None = None,
    adversary: AdversaryPolicy | None = None,
) -> ValidationSettings:
    return ValidationSettings(
        host=MachineHost.LAPTOP,
        engine=EngineSettings(factories=FACTORIES, command_timeout=10, limits=limits or ENGINE_LIMITS),
        k=k,
        max_retries=1,
        retry_backoff=RetryBackoff(initial=0.001, maximum=0.001, factor=2.0, jitter=0.0),
        sampling=LLMPolicy(),
        adversary=(adversary,) if adversary else (),
        ledger=ledger,
    )


@dataclass
class SolverModel:
    """A solver whose every attempt replies ``42``. It records each attempt's step, sets ``called`` on its first
    request, waits on ``release`` before answering when one is given, and fails an attempt whose step is not in
    ``allowed`` when that is given."""

    allowed: frozenset[str] | None = None
    release: asyncio.Event | None = None
    called: asyncio.Event = field(default_factory=asyncio.Event)
    steps: list[str] = field(default_factory=list)

    def for_step(self, step: str) -> Model:
        if self.allowed is not None and step not in self.allowed:
            raise AssertionError(f"the round ran {step}, which had its file")
        self.steps.append(step)
        return self._answer

    async def _answer(self, request: ModelRequest) -> ModelTurn:
        self.called.set()
        if self.release is not None:
            await self.release.wait()
        message = {"role": "assistant", "content": "42"}
        return ModelTurn(message, (), (), None, "stop", text="42")


@dataclass
class FailingTrials:
    """A solver whose attempts of each trial in ``errors`` raise its error, and whose other attempts reply ``42``.

    It records each attempt's step.
    """

    errors: Mapping[str, Callable[[], Exception]]
    steps: list[str] = field(default_factory=list)

    def for_step(self, step: str) -> Model:
        self.steps.append(step)
        error = self.errors.get(step.split("/")[1])

        async def answer(request: ModelRequest) -> ModelTurn:
            if error is not None:
                raise error()
            return ModelTurn({"role": "assistant", "content": "42"}, (), (), None, "stop", text="42")

        return answer


@dataclass
class HeldClient:
    """``client`` once ``release`` is set."""

    client: ModelClient
    release: asyncio.Event

    @property
    def model(self) -> str:
        return self.client.model

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        await self.release.wait()
        return await self.client.complete(messages, policy, request_fields)

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        await self.release.wait()
        return await self.client.complete_structured(messages, output_type, policy)


class UncalledClient:
    """A model the round must not call: the controls are on disk, or no adversary trial is due."""

    @property
    def model(self) -> str:
        return "adversary"

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        raise AssertionError("the round called a model whose work was on disk")

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        raise AssertionError("the round called a model whose work was on disk")


@dataclass(frozen=True)
class Submit:
    reply: str


USAGE = Usage(prompt_tokens=10, completion_tokens=5, reasoning_tokens=0, cached_tokens=0)


@dataclass
class AdversaryScript:
    """An adversary whose n-th reply in a conversation is ``turns[n]``: a ``submit`` call or its final text.

    It sets ``called`` on its first request and ``answered`` when it gives its final text, sets ``held`` and waits on
    ``release`` before its reply numbered ``hold_at`` when that is given, and keeps every conversation it was sent.
    """

    turns: tuple[Submit | SearchReport | str | Exception, ...]
    hold_at: int | None = None
    release: asyncio.Event = field(default_factory=asyncio.Event)
    called: asyncio.Event = field(default_factory=asyncio.Event)
    answered: asyncio.Event = field(default_factory=asyncio.Event)
    held: asyncio.Event = field(default_factory=asyncio.Event)
    requests: list[list[Message]] = field(default_factory=list)

    @property
    def model(self) -> str:
        return "adversary"

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        self.requests.append(list(messages))
        self.called.set()
        index = sum(message["role"] == "assistant" for message in messages)
        if index == self.hold_at:
            self.held.set()
            await self.release.wait()
        turn = self.turns[index]
        if isinstance(turn, Exception):
            raise turn
        if isinstance(turn, Submit):
            call = ToolCall(f"call-{index}", "submit", json.dumps({"reply": turn.reply}))
            return Completion("", "", (call,), FinishReason.TOOL_CALLS, USAGE, 0.1, 0.01, 0.09, 0, ())
        self.answered.set()
        if isinstance(turn, SearchReport):
            call = ToolCall(
                f"call-{index}",
                "finish_search",
                json.dumps({"defects": [asdict(defect) for defect in turn.defects], "summary": turn.summary}),
            )
            return Completion("", "", (call,), FinishReason.TOOL_CALLS, USAGE, 0.1, 0.01, 0.09, 0, ())
        return Completion(turn, "", (), FinishReason.STOP, USAGE, 0.1, 0.01, 0.09, 0, ())

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        raise AssertionError("an adversary makes no structured call")


PROBES = (Submit("41"), Submit("42"), SearchReport((), "No findings"))


@dataclass
class StepSignal:
    """Records every entry in ``ledger`` and sets ``recorded`` once an entry for ``step`` is recorded."""

    ledger: Ledger
    step: str
    recorded: asyncio.Event = field(default_factory=asyncio.Event)

    def record(self, entry: LedgerEntry) -> None:
        self.ledger.record(entry)
        if entry.step == self.step:
            self.recorded.set()


ALL_SOLVE_RATES = BandRules(0, 1, frozenset({BandOutcome.IN_BAND}))
VIOLATED_CONTROL = "N2-off-by-one"
"""The control ``controls_reply("42")`` violates: a negative control replying the right answer."""


def trial_steps(ledger, since: int = 0) -> set[str]:
    return {entry.step for entry in ledger.entries[since:] if entry.kind is EntryKind.TRIAL}


def evidence_files(result: ValidationResult) -> list[str]:
    assert result.solver is not None
    adversary = [trial.attempt_file for trial in result.adversary]
    submissions = [submission.attempt_file for trial in result.adversary for submission in trial.submissions]
    return [
        *(trial.attempt_file for trial in result.solver.trials),
        *(c.attempt_file for c in result.controls),
        *adversary,
        *submissions,
    ]


def files_under(directory: Path) -> dict[str, bytes]:
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


async def test_solver_trials_start_before_the_controls_exist_and_replay_waits_for_them(
    glm_client, fake_glm, ledger, tmp_path
):
    fake_glm.stream(content=controls_reply("41"))
    release = asyncio.Event()
    solver = SolverModel()
    context, settings = round_context(tmp_path), round_settings(ledger)
    running = asyncio.create_task(
        validate_round(
            built_draft(tmp_path / "store"), context, settings, solver, HeldClient(glm_client, release), UncalledClient()
        )
    )

    await asyncio.wait_for(solver.called.wait(), timeout=30)
    assert not list(context.round_dir.rglob("controls.json"))
    release.set()
    result = await running

    controls_ended = max(entry.ended for entry in ledger.entries if entry.step == CONTROLS_STEP)
    trials = [entry for entry in ledger.entries if entry.kind is EntryKind.TRIAL]
    assert min(entry.started for entry in trials if entry.step.startswith("solver/")) < controls_ended
    assert all(entry.started >= controls_ended for entry in trials if entry.step.startswith("control/"))
    assert result.solver is not None and result.solver.stats.solved == 2 and result.solver.ungraded == {}
    assert [(c.id, c.verdict) for c in result.controls] == [(i, ControlVerdict.MET) for i in CONTROL_IDS]
    # validation.json holds the whole result without the attempt files it names.
    write_validation(context.round_dir, result)
    for name in evidence_files(result):
        (context.round_dir / name).unlink()
    assert load_validation(context.round_dir) == result


async def test_a_relaunch_redoes_only_the_units_whose_files_are_missing(glm_client, fake_glm, ledger, tmp_path):
    fake_glm.stream(content=controls_reply("41"))
    context, settings = round_context(tmp_path), round_settings(ledger)
    first = await validate_round(
        built_draft(tmp_path / "store"), context, settings, SolverModel(), glm_client, UncalledClient()
    )
    assert first.solver is not None
    (context.round_dir / first.solver.trials[1].attempt_file).unlink()
    (context.round_dir / first.controls[1].attempt_file).unlink()
    recorded = len(ledger.entries)

    solver = SolverModel(allowed=frozenset({"solver/1/0"}))
    again = await validate_round(
        built_draft(tmp_path / "store"), context, settings, solver, UncalledClient(), UncalledClient()
    )

    assert solver.steps == ["solver/1/0"]
    assert trial_steps(ledger, recorded) == {"solver/1/0", "control/N1-shortcut/0"}
    assert again == first


async def test_the_trials_of_another_lowered_spec_are_not_read_as_done(glm_client, fake_glm, ledger, tmp_path):
    """Other deadlines lower the same draft to another spec: its trials run in their own directory."""
    fake_glm.stream(content=controls_reply("41"))
    draft_of_round = built_draft(tmp_path / "store")
    context = round_context(tmp_path)
    first = await validate_round(
        draft_of_round, context, round_settings(ledger), SolverModel(), glm_client, UncalledClient()
    )
    trials_dir = tmp_path / "round" / Path(evidence_files(first)[0]).parents[2]
    before = files_under(trials_dir)

    solver = SolverModel()
    longer = round_settings(ledger).engine.limits.model_copy(update={"total_turn_timeout": 40, "attempt_timeout": 80})
    other = await validate_round(
        draft_of_round, context, round_settings(ledger, limits=longer), solver, UncalledClient(), UncalledClient()
    )

    assert other.lowering.task_digest != first.lowering.task_digest
    assert sorted(solver.steps) == ["solver/0/0", "solver/1/0"]
    assert files_under(trials_dir) == before
    assert not any(name.startswith(str(trials_dir.relative_to(tmp_path / "round"))) for name in evidence_files(other))


async def test_a_rewritten_control_set_is_replayed_again(glm_client, fake_glm, ledger, tmp_path):
    fake_glm.stream(content=controls_reply("41"))
    fake_glm.stream(content=controls_reply("43"))
    context, settings = round_context(tmp_path), round_settings(ledger)
    first = await validate_round(
        built_draft(tmp_path / "store"), context, settings, SolverModel(), glm_client, UncalledClient()
    )
    (next(context.round_dir.rglob("controls.json"))).unlink()
    recorded = len(ledger.entries)

    again = await validate_round(
        built_draft(tmp_path / "store"),
        context,
        settings,
        SolverModel(allowed=frozenset()),
        glm_client,
        UncalledClient(),
    )

    assert trial_steps(ledger, recorded) == {f"control/{control_id}/0" for control_id in CONTROL_IDS}
    assert again.solver == first.solver
    assert {Path(c.attempt_file).parents[2] for c in again.controls}.isdisjoint(
        {Path(c.attempt_file).parents[2] for c in first.controls}
    )


async def test_a_trial_cancelled_in_flight_leaves_no_attempt_and_is_rerun(glm_client, fake_glm, ledger, tmp_path):
    fake_glm.stream(content=controls_reply("41"))
    fake_glm.stream(content=controls_reply("41"))
    context, settings = round_context(tmp_path), round_settings(ledger, k=1)
    hanging = SolverModel(release=asyncio.Event())
    running = asyncio.create_task(
        validate_round(built_draft(tmp_path / "store"), context, settings, hanging, glm_client, UncalledClient())
    )
    await asyncio.wait_for(hanging.called.wait(), timeout=30)

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert not [path for path in context.round_dir.rglob("*") if "/solver/" in str(path) and path.is_file()]
    solver = SolverModel()
    result = await validate_round(
        built_draft(tmp_path / "store"), context, settings, solver, glm_client, UncalledClient()
    )
    assert solver.steps == ["solver/0/0"]
    assert result.solver is not None and result.solver.ungraded == {}
    assert [trial.status for trial in result.solver.trials] == [TrialStatus.GRADED]


async def test_a_rejected_lowering_runs_no_trial_and_round_trips(glm_client, fake_glm, ledger, tmp_path):
    fake_glm.stream(content=controls_reply("41"))
    context, settings = round_context(tmp_path), round_settings(ledger)
    settings = replace(settings, engine=replace(settings.engine, factories={}))

    result = await validate_round(
        built_draft(tmp_path / "store"),
        context,
        settings,
        SolverModel(allowed=frozenset()),
        glm_client,
        UncalledClient(),
    )

    assert result.lowering.status is LoweringStatus.REJECTED
    assert result.solver is None and result.controls == ()
    write_validation(context.round_dir, result)
    assert load_validation(context.round_dir) == result


async def test_adversary_trials_start_once_lowering_and_controls_exist_and_wait_for_no_solver(
    glm_client, fake_glm, ledger, tmp_path
):
    fake_glm.stream(content=controls_reply("41"))
    controls_release, solver = asyncio.Event(), SolverModel(release=asyncio.Event())
    adversary = AdversaryScript(PROBES)
    context, settings = round_context(tmp_path), round_settings(
        ledger, k=1, adversary=adversary_policy(k=1, mode=AdversaryMode.WHITEBOX)
    )
    running = asyncio.create_task(
        validate_round(
            built_draft(tmp_path / "store"),
            context,
            settings,
            solver,
            HeldClient(glm_client, controls_release),
            adversary,
        )
    )

    await asyncio.wait_for(solver.called.wait(), timeout=30)
    assert not adversary.called.is_set()
    controls_release.set()
    await asyncio.wait_for(adversary.answered.wait(), timeout=30)
    assert solver.release is not None
    solver.release.set()
    result = await running

    controls_ended = max(entry.ended for entry in ledger.entries if entry.step == CONTROLS_STEP)
    adversary_trials = [e for e in ledger.entries if e.kind is EntryKind.TRIAL and e.step.startswith("adversary/")]
    solver_trials = [e for e in ledger.entries if e.kind is EntryKind.TRIAL and e.step.startswith("solver/")]
    assert [e.step for e in adversary_trials] == ["adversary/false_accept-0/0"]
    assert adversary_trials[0].started >= controls_ended
    assert adversary_trials[0].ended < solver_trials[0].ended
    # The white-box brief holds this round's controls.
    system = str(adversary.requests[0][0]["content"])
    assert all(f"control {control_id}" in system for control_id in CONTROL_IDS)
    (trial,) = result.adversary
    assert (trial.status, trial.report) == (TrialStatus.GRADED, SearchReport((), "No findings"))
    # validation.json holds the adversary field without the attempt files it names.
    write_validation(context.round_dir, result)
    for name in evidence_files(result):
        (context.round_dir / name).unlink()
    assert load_validation(context.round_dir) == result


async def test_each_submission_within_the_budget_is_graded_as_a_control_and_recorded_with_its_reward(
    glm_client, fake_glm, ledger, tmp_path
):
    fake_glm.stream(content=controls_reply("41"))
    report = SearchReport((Defect((2,), "it accepts 42.", True),), "Found defect")
    adversary = AdversaryScript((Submit("41"), Submit("42"), Submit("43"), report))
    context, settings = round_context(tmp_path), round_settings(
        ledger, k=1, adversary=adversary_policy(k=1, submissions=2)
    )

    result = await validate_round(
        built_draft(tmp_path / "store"), context, settings, SolverModel(), glm_client, adversary
    )

    (trial,) = result.adversary
    graded = [(s.ordinal, s.status, s.reward, s.passed) for s in trial.submissions]
    assert graded == [(1, GradeStatus.GRADED, 0.0, False), (2, GradeStatus.GRADED, 1.0, True)]
    assert trial.report == report
    for submission in trial.submissions:
        grading = json.loads((context.round_dir / submission.attempt_file).read_bytes())
        assert grading["grade"]["reward"] == submission.reward
        assert Path(submission.attempt_file).parts[-3] == "control"
    observations = [json.loads(str(m["content"])) for m in adversary.requests[-1] if m["role"] == "tool"]
    assert [observation["reward"] for observation in observations[:2]] == [0.0, 1.0]
    assert "submission budget spent" in observations[2]["error"]
    graded_dir = context.round_dir / Path(trial.submissions[0].attempt_file).parents[1]
    assert sorted(path.name for path in graded_dir.iterdir()) == ["s1", "s2"]


async def test_a_relaunch_reruns_only_a_missing_adversary_attempt(glm_client, fake_glm, ledger, tmp_path):
    fake_glm.stream(content=controls_reply("41"))
    context, settings = round_context(tmp_path), round_settings(ledger, k=1, adversary=adversary_policy(k=2))
    first = await validate_round(
        built_draft(tmp_path / "store"), context, settings, SolverModel(), glm_client, AdversaryScript(PROBES)
    )
    (context.round_dir / first.adversary[1].attempt_file).unlink()
    recorded = len(ledger.entries)

    again = await validate_round(
        built_draft(tmp_path / "store"),
        context,
        settings,
        SolverModel(allowed=frozenset()),
        UncalledClient(),
        AdversaryScript(PROBES),
    )

    steps = trial_steps(ledger, recorded)
    assert {step for step in steps if not step.startswith("control/s")} == {"adversary/false_accept-1/1"}
    assert again.adversary[0] == first.adversary[0]
    rerun, before = again.adversary[1], first.adversary[1]
    assert rerun.attempt_file.endswith("adversary/false_accept-1/attempt-1.json")
    assert [(s.ordinal, s.reward) for s in rerun.submissions] == [(s.ordinal, s.reward) for s in before.submissions]
    assert (again.solver, again.controls) == (first.solver, first.controls)


async def test_a_search_under_another_policy_keeps_the_old_attempt_and_runs_with_its_full_retries(
    glm_client, fake_glm, ledger, tmp_path
):
    """Evidence binds to the exact search: a relaunch with another adversary policy reads no earlier attempt as done."""
    fake_glm.stream(content=controls_reply("41"))
    context, settings = round_context(tmp_path), round_settings(ledger, k=1, adversary=adversary_policy(k=1))
    first = await validate_round(
        built_draft(tmp_path / "store"), context, settings, SolverModel(), glm_client, AdversaryScript(PROBES)
    )
    saved = (context.round_dir / first.adversary[0].attempt_file).read_bytes()
    recorded = len(ledger.entries)

    changed = replace(settings, adversary=(adversary_policy(k=1, submissions=2),))
    again = await validate_round(
        built_draft(tmp_path / "store"),
        context,
        changed,
        SolverModel(allowed=frozenset()),
        UncalledClient(),
        AdversaryScript(PROBES),
    )

    assert {step for step in trial_steps(ledger, recorded) if not step.startswith("control/s")} == {
        "adversary/false_accept-0/0"
    }
    assert again.adversary[0].attempt_file != first.adversary[0].attempt_file
    assert (context.round_dir / first.adversary[0].attempt_file).read_bytes() == saved
    assert (again.solver, again.controls) == (first.solver, first.controls)


@pytest.mark.parametrize(
    ("turns", "hold_at"),
    [(PROBES, 0), ((Submit("41"), "unused"), 1)],
    ids=["before-any-tool-call", "after-a-graded-submission"],
)
async def test_an_adversary_attempt_deadline_retains_an_ungraded_attempt_and_retries_then_holds(
    glm_client, fake_glm, ledger, tmp_path, turns, hold_at
):
    adversary = AdversaryScript(turns, hold_at=hold_at)
    fake_glm.stream(content=controls_reply("41"))
    context, settings = round_context(tmp_path), round_settings(ledger, k=1, adversary=adversary_policy(k=1))
    settings = replace(
        settings,
        engine=replace(settings.engine, limits=settings.engine.limits.model_copy(update={"attempt_timeout": 0.5})),
    )

    result = await validate_round(
        built_draft(tmp_path / "store"), context, settings, SolverModel(), glm_client, adversary
    )

    (trial,) = result.adversary
    assert (trial.status, trial.cause) == (TrialStatus.UNGRADED, Cause.ATTEMPT_TIMEOUT)
    assert trial.attempt_file.endswith("adversary/false_accept-0/attempt-1.json")
    assert (context.round_dir / trial.attempt_file).exists()
    assert isinstance(await default_review(result, band=ALL_SOLVE_RATES), Hold)


async def test_an_adversary_attempt_cancelled_in_flight_leaves_no_attempt_file(glm_client, fake_glm, ledger, tmp_path):
    fake_glm.stream(content=controls_reply("41"))
    context, settings = round_context(tmp_path), round_settings(ledger, k=1, adversary=adversary_policy(k=1))
    hanging = AdversaryScript(PROBES, hold_at=1)
    running = asyncio.create_task(
        validate_round(built_draft(tmp_path / "store"), context, settings, SolverModel(), glm_client, hanging)
    )
    await asyncio.wait_for(hanging.held.wait(), timeout=30)

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert not list(context.round_dir.rglob("adversary/false_accept-0/attempt-*"))
    # The cancelled attempt's graded submission stays, so the rerun is numbered after it.
    assert list(context.round_dir.rglob("adversary/false_accept-0/submissions/0/control/s1/attempt-0.json"))
    result = await validate_round(
        built_draft(tmp_path / "store"), context, settings, SolverModel(), UncalledClient(), AdversaryScript(PROBES)
    )
    (trial,) = result.adversary
    assert trial.status is TrialStatus.GRADED
    assert trial.attempt_file.endswith("adversary/false_accept-0/attempt-1.json")


async def test_a_violated_control_with_a_modify_round_left_cancels_the_held_trials_and_names_them(
    glm_client, fake_glm, ledger, tmp_path
):
    fake_glm.stream(content=controls_reply("42"))
    controls_release = asyncio.Event()
    solver = SolverModel(release=asyncio.Event())
    context = round_context(tmp_path, modify_rounds_left=1)
    settings = round_settings(ledger, k=2, adversary=adversary_policy(k=1))
    running = asyncio.create_task(
        validate_round(
            built_draft(tmp_path / "store"),
            context,
            settings,
            solver,
            HeldClient(glm_client, controls_release),
            AdversaryScript(PROBES, hold_at=0),
        )
    )
    await asyncio.wait_for(solver.called.wait(), timeout=30)
    controls_release.set()

    result = await asyncio.wait_for(running, timeout=30)

    assert result.early == EarlyDecision(VIOLATED_CONTROL, ("solver/0", "solver/1", "adversary/false_accept-0"))
    assert result.solver is not None and result.solver.trials == () and result.solver.ungraded == {}
    assert result.adversary == ()
    verdicts = {c.id: c.verdict for c in result.controls}
    assert verdicts == {i: ControlVerdict.VIOLATED if i == VIOLATED_CONTROL else ControlVerdict.MET for i in CONTROL_IDS}
    trials_dir = context.round_dir / Path(result.controls[0].attempt_file).parents[3]
    assert not list(trials_dir.glob("solver/**/attempt-*.json"))
    assert not list(trials_dir.glob("adversary/**/attempt-*.json"))
    write_validation(context.round_dir, result)
    assert load_validation(context.round_dir) == result


async def test_without_a_modify_round_left_a_violated_control_waits_for_every_trial_to_end(
    glm_client, fake_glm, ledger, tmp_path
):
    fake_glm.stream(content=controls_reply("42"))
    signal = StepSignal(ledger, f"control/{VIOLATED_CONTROL}/0")
    solver = SolverModel(release=signal.recorded)
    context = round_context(tmp_path, modify_rounds_left=0)
    settings = round_settings(signal, k=2, adversary=adversary_policy(k=1))

    result = await asyncio.wait_for(
        validate_round(built_draft(tmp_path / "store"), context, settings, solver, glm_client, AdversaryScript(PROBES)),
        timeout=60,
    )

    assert result.early is None
    assert result.solver is not None and [t.status for t in result.solver.trials] == [TrialStatus.GRADED] * 2
    assert [t.status for t in result.adversary] == [TrialStatus.GRADED]
    assert {c.id: c.verdict for c in result.controls}[VIOLATED_CONTROL] is ControlVerdict.VIOLATED
    violated_ended = next(e.ended for e in ledger.entries if e.step == signal.step)
    assert all(e.ended >= violated_ended for e in ledger.entries if e.step.startswith("solver/"))


UNAVAILABLE = {"0": lambda: ModelUnavailable("the hold ran out", ())}
"""Trial 0 fails for a retryable cause on every attempt."""


async def test_a_review_retry_reruns_only_the_trial_ungraded_for_a_retryable_cause_with_a_full_budget(
    glm_client, fake_glm, ledger, tmp_path
):
    fake_glm.stream(content=controls_reply("41"))
    settings = round_settings(ledger, k=3)
    errors = {**UNAVAILABLE, "2": lambda: ModelRequestRejected(400, "invalid tools", ())}
    first = await validate_round(
        built_draft(tmp_path / "store"),
        round_context(tmp_path),
        settings,
        FailingTrials(errors),
        glm_client,
        UncalledClient(),
    )
    assert first.solver is not None
    assert [(t.status, t.cause) for t in first.solver.trials] == [
        (TrialStatus.UNGRADED, Cause.MODEL_UNAVAILABLE),
        (TrialStatus.GRADED, None),
        (TrialStatus.UNGRADED, Cause.MODEL_REJECTED),
    ]
    assert first.solver.trials[0].attempt_file.endswith("solver/0/attempt-1.json")

    retried = round_context(tmp_path)
    solver = FailingTrials(UNAVAILABLE)
    again = await validate_round(
        built_draft(tmp_path / "store"), retried, settings, solver, UncalledClient(), UncalledClient()
    )

    assert sorted(solver.steps) == ["solver/0/2", "solver/0/3", "solver/2/1"]
    assert again.solver is not None and again.solver.trials[1] == first.solver.trials[1]
    assert again.solver.trials[2].status is TrialStatus.GRADED
    assert again.solver.trials[0].attempt_file.endswith("solver/0/attempt-3.json")
    trial_dir = retried.round_dir / Path(again.solver.trials[0].attempt_file).parent
    assert sorted(path.name for path in trial_dir.iterdir()) == [f"attempt-{n}.json" for n in range(4)]
    assert again.controls == first.controls

    # Graded trials and task defects are done; another validation, a relaunch included, runs trial 0 alone again.
    relaunched = FailingTrials(UNAVAILABLE)
    third = await validate_round(
        built_draft(tmp_path / "store"), retried, settings, relaunched, UncalledClient(), UncalledClient()
    )
    assert relaunched.steps == ["solver/0/4", "solver/0/5"]
    assert third.solver is not None and third.solver.trials[1:] == again.solver.trials[1:]
    assert sorted(path.name for path in trial_dir.iterdir()) == [f"attempt-{n}.json" for n in range(6)]


class FailedVerifierFactory(FixtureImageFactory):
    async def create(self, spec: MachineSpec):
        raise MachineTerminated("verifier worker disappeared")


async def test_an_adversary_verifier_outage_holds_after_bounded_retries(glm_client, fake_glm, ledger, tmp_path):
    fake_glm.stream(content=controls_reply("41"))
    context = round_context(tmp_path)
    settings = round_settings(ledger, k=1, adversary=adversary_policy(k=1))
    settings = replace(
        settings,
        engine=replace(
            settings.engine,
            factories={
                **settings.engine.factories,
                Backend.DOCKER.value: FailedVerifierFactory(),
            },
        ),
    )
    package = shell_grader(
        "exit 0",
        {},
        environment=draft.grader_environment("registry.example/verifier@sha256:" + "0" * 64),
        answer_path=draft.ANSWER_PATH,
        timeout=30,
    )
    built = built_draft(tmp_path / "store")
    built = replace(built, task=built.task.model_copy(update={"grader": package.grader}))
    adversary = AdversaryScript((Submit("42"), SearchReport((), "No findings")))

    result = await validate_round(built, context, settings, SolverModel(), glm_client, adversary)

    (trial,) = result.adversary
    assert (trial.status, trial.cause) == (TrialStatus.UNGRADED, Cause.MACHINE_TERMINATED)
    assert trial.attempt_file.endswith("adversary/false_accept-0/attempt-1.json")
    assert all(s.status == GradeStatus.UNAVAILABLE for s in trial.submissions)
    first = json.loads((context.round_dir / Path(trial.attempt_file).with_name("attempt-0.json")).read_bytes())
    assert first["cause"] == Cause.MACHINE_TERMINATED


@pytest.mark.parametrize("spec", [FALSE_ACCEPT, FALSE_REJECT])
@pytest.mark.parametrize("actionable", [True, False])
async def test_defects_use_same_contract_without_second_review(glm_client, fake_glm, ledger, tmp_path, spec, actionable):
    fake_glm.stream(content=controls_reply("41"))
    built = built_draft(tmp_path / "store")
    settings = round_settings(ledger, adversary=replace(adversary_policy(k=1), spec=spec))
    context = round_context(tmp_path)
    report = SearchReport(
        (Defect((1,), "Candidate exposes a defect", actionable), Defect((2,), "Informational observation", False)),
        "Completed search",
    )
    checked = await validate_round(
        built,
        context,
        settings,
        SolverModel(),
        glm_client,
        AdversaryScript((Submit("41"), Submit("42"), report)),
    )
    verdict = await default_review(checked, band=ALL_SOLVE_RATES)
    assert isinstance(verdict, Modify if actionable else Accept)
    if isinstance(verdict, Modify):
        assert len(verdict.findings) == 1 and "Candidate exposes a defect" in verdict.findings[0]
    (trial,) = checked.adversary
    assert trial.name == spec.name and trial.report == report
    assert len(fake_glm.requests) == 1  # Controls only; no second-review model call.
    saved_search = (context.round_dir / trial.attempt_file).read_bytes()
    resumed = await validate_round(
        built, context, settings, SolverModel(allowed=frozenset()), UncalledClient(), UncalledClient()
    )
    assert type(await default_review(resumed, band=ALL_SOLVE_RATES)) is type(verdict)
    assert (context.round_dir / trial.attempt_file).read_bytes() == saved_search


@pytest.mark.parametrize("ending", ["No findings", SearchReport((Defect((99,), "unknown candidate", True),), "invalid")])
async def test_prose_or_invalid_claim_never_completes_a_search(glm_client, fake_glm, ledger, tmp_path, ending):
    fake_glm.stream(content=controls_reply("41"))
    settings = round_settings(ledger, adversary=replace(adversary_policy(k=1), max_turns=2))
    checked = await validate_round(
        built_draft(tmp_path / "store"),
        round_context(tmp_path),
        settings,
        SolverModel(),
        glm_client,
        AdversaryScript((Submit("41"), ending)),
    )
    (trial,) = checked.adversary
    assert trial.status is TrialStatus.UNGRADED and trial.report is None
    assert not isinstance(await default_review(checked, band=ALL_SOLVE_RATES), Accept)


async def test_each_roster_spec_runs_under_its_own_name_brief_and_budgets(glm_client, fake_glm, ledger, tmp_path):
    fake_glm.stream(content=controls_reply("41"))
    policies = (
        replace(adversary_policy(k=1, submissions=1), max_turns=2),
        replace(adversary_policy(k=2, submissions=3), spec=FALSE_REJECT),
    )
    settings = replace(round_settings(ledger), adversary=policies)
    adversary = AdversaryScript((Submit("41"), SearchReport((), "No findings")))
    checked = await validate_round(
        built_draft(tmp_path / "store"), round_context(tmp_path), settings, SolverModel(), glm_client, adversary
    )
    assert [(trial.trial, trial.name) for trial in checked.adversary] == [
        ("false_accept-0", "false_accept"),
        ("false_reject-0", "false_reject"),
        ("false_reject-1", "false_reject"),
    ]
    assert all(trial.status is TrialStatus.GRADED and len(trial.submissions) == 1 for trial in checked.adversary)
    assert len({trial.submissions[0].attempt_file for trial in checked.adversary}) == 3
    systems = {str(request[0]["content"]) for request in adversary.requests}
    assert sorted(FALSE_ACCEPT.brief in system for system in systems) == [False, True]
    assert sorted(FALSE_REJECT.brief in system for system in systems) == [False, True]


async def test_interrupted_search_retains_evidence_and_explicit_resume_starts_a_new_attempt(
    glm_client, fake_glm, ledger, tmp_path
):
    fake_glm.stream(content=controls_reply("41"))
    context = round_context(tmp_path)
    settings = replace(round_settings(ledger, adversary=adversary_policy(k=1)), max_retries=0)
    built = built_draft(tmp_path / "store")
    checked = await validate_round(
        built,
        context,
        settings,
        SolverModel(),
        glm_client,
        AdversaryScript((Submit("41"), ModelUnavailable("request interrupted", ()))),
    )
    (trial,) = checked.adversary
    path = context.round_dir / trial.attempt_file
    original = path.read_bytes()
    saved = json.loads(original)
    assert trial.status is TrialStatus.UNGRADED and trial.cause is Cause.MODEL_UNAVAILABLE
    assert saved["adversary"]["submissions"][0]["reply"] == "41"
    messages = saved["rollout"]["conversations"][0]["messages"]
    assert json.loads(next(message["content"] for message in messages if message["role"] == "tool"))["submission"] == 1
    assert isinstance(await default_review(checked, band=ALL_SOLVE_RATES), Hold)

    # Explicit resume grants another attempt, retaining the original failed search and its evidence.
    resumed = await validate_round(
        built,
        context,
        settings,
        SolverModel(allowed=frozenset()),
        UncalledClient(),
        AdversaryScript(PROBES),
    )
    assert isinstance(await default_review(resumed, band=ALL_SOLVE_RATES), Accept)
    assert resumed.adversary[0].attempt_file.endswith("attempt-1.json")
    assert path.read_bytes() == original
    assert len(list(path.parent.glob("attempt-*.json"))) == 2


@pytest.mark.parametrize(
    "causes,stop,expected",
    [
        ((Cause.BUDGET_EXHAUSTED, Cause.MODEL_UNAVAILABLE), "stop", Cause.MODEL_UNAVAILABLE),
        ((Cause.INVALID_TASK, Cause.MODEL_UNAVAILABLE), "stop", Cause.MODEL_UNAVAILABLE),
        ((Cause.MODEL_UNAVAILABLE, Cause.INVALID_TASK), "max_turns", Cause.MODEL_UNAVAILABLE),
        ((Cause.INVALID_TASK,), "max_turns", Cause.BUDGET_EXHAUSTED),
        ((Cause.INVALID_TASK,), "stop", Cause.INVALID_TASK),
    ],
)
def test_search_failure_precedence_accounts_for_every_submission(causes, stop, expected):
    rollout = Rollout("search", (), stop)
    submissions = tuple(
        Submission(
            SubmissionRecord(
                index, None, f"s{index}", "candidate", (), GradeResult(GradeStatus.UNAVAILABLE, None), False, 0.0
            ),
            Ungraded(cause, str(cause), rollout, None),
        )
        for index, cause in enumerate(causes, 1)
    )

    outcome = trial_outcome(rollout, submissions)

    assert isinstance(outcome, Ungraded) and outcome.cause is expected
