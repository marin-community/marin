# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""validate_round: the unit graph, resuming from the unit files, files keyed by what they cover, cancellation, and
the adversary trials, on decision-model tasks too."""

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from pydantic import BaseModel
from rolloutengine.models import Model, ModelRequest, ModelTurn, RolloutLimits
from shellbox.machine import Backend, MachineSpec, MachineTerminated
from taskcompendium.grading_result import Outcome
from taskcompendium.grading_result import Outcome as GradeStatus
from taskcompendium.models import AnswerType, PlainText, Source
from verifyit.spec import ExactSpec

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
from taskforge.spec.decision import QUESTION
from taskforge.spec.rubric import Rubric, RubricItem
from taskforge.stage10_proposal.model import Verification, parse
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
    SUBMISSIONS_DIR,
    AdversaryMode,
    AdversaryPolicy,
    Claim,
    ClaimKind,
    adversary_brief,
    adversary_record,
    run_adversary_trial,
)
from taskforge.stage50_validate.control_drafts import CONTROLS_STEP
from taskforge.stage50_validate.controls import ControlVerdict
from taskforge.stage50_validate.outcome import Cause, Graded, TrialKind
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
from taskforge.stage50_validate.trials import EngineSettings, RetryBackoff, TrialPlan
from tests.sandbox.fixture_images import FixtureImageFactory, PortlessShellSimFactory, shell_grader
from tests.spec.fake_decision import FakeDecisionClient
from tests.stage40_build_with_program.conftest import PROPOSAL

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

    def control(control_id: str, kind: str, category: str, concern: str, reply: str, low, high) -> dict:
        return {
            "id": control_id,
            "kind": kind,
            "category": category,
            "concern": concern,
            "final_reply": reply,
            "files": [],
            "reward_min": low,
            "reward_max": high,
            "rationale": "r",
        }

    return json.dumps(
        {
            "controls": [
                control("P-reference", "positive", "known_correct", "reference", "42", 0.99, None),
                control("N1-shortcut", "negative", "task_specific_shortcut", "shortcut", "6 * 7", None, 0.0),
                control("N2-off-by-one", "negative", "plausible_wrong", "acceptance", wrong, None, 0.0),
                control("N3", "malformed", "empty_or_malformed", "extraction", "", None, 0.0),
            ]
        }
    )


def built_draft(store_root: Path) -> TaskDraft:
    """``TASK`` as a build leaves it: its grader step output stored under ``store_root``."""
    key = RLEnvBuildStepKey(sdk_version=SDK_VERSION, source_digest="grader", inputs={}, model_digest="model")
    memo = RLEnvBuildStepStore(str(store_root), WORKPIECE).store(key, GRADER_STEP, GRADER)
    provenance = WorkpieceBuildProvenance(
        workpiece_id=WORKPIECE,
        proposal_digest="proposal",
        verification=Verification.ANSWER_IN_BOX,
        verification_change=None,
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
        k=k,
        submissions=submissions,
        repair_submissions=1,
        max_turns=6,
        sampling=LLMPolicy(max_continuations=0),
        mode=mode,
        min_probes=3,
    )


def round_context(tmp_path: Path, modify_rounds_left: int = 0, validation_pass: int = 1) -> ValidationContext:
    return ValidationContext(
        workpiece_id=WORKPIECE,
        round=0,
        round_dir=tmp_path / "round",
        proposal=parse(PROPOSAL),
        store_root=str(tmp_path / "store"),
        modify_rounds_left=modify_rounds_left,
        validation_pass=validation_pass,
    )


def round_settings(
    ledger,
    k: int = 2,
    limits: RolloutLimits | None = None,
    adversary: AdversaryPolicy | None = None,
) -> ValidationSettings:
    return ValidationSettings(
        host=MachineHost.LAPTOP,
        engine=EngineSettings(
            factories={Backend.SHELLSIM.value: PortlessShellSimFactory(), Backend.DOCKER.value: FixtureImageFactory()},
            command_timeout=10,
            limits=limits
            or RolloutLimits(
                max_turns=4,
                tool_turn_timeout=20,
                model_turn_timeout=30,
                cleanup_timeout=10,
                total_turn_timeout=30,
                attempt_timeout=60,
                verifier_timeout=None,
                max_total_tokens=None,
            ),
            decision=None,
        ),
        k=k,
        max_retries=1,
        retry_backoff=RetryBackoff(initial=0.001, maximum=0.001, factor=2.0, jitter=0.0),
        sampling=LLMPolicy(),
        adversary=adversary or adversary_policy(k=0),
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
        return "uncalled"

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

    turns: tuple[Submit | str, ...]
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
        if isinstance(turn, Submit):
            call = ToolCall(f"call-{index}", "submit", json.dumps({"reply": turn.reply}))
            return Completion("", "", (call,), FinishReason.TOOL_CALLS, USAGE, 0.1, 0.01, 0.09, 0, ())
        self.answered.set()
        return Completion(turn, "", (), FinishReason.STOP, USAGE, 0.1, 0.01, 0.09, 0, ())

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        raise AssertionError("an adversary makes no structured call")


PROBES = (Submit("41"), Submit("42"), "I submitted the honest answer.\nNO_SHORTCUT")


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
    assert [e.step for e in adversary_trials] == ["adversary/0/0"]
    assert adversary_trials[0].started >= controls_ended
    assert adversary_trials[0].ended < solver_trials[0].ended
    # The white-box brief holds this round's controls.
    system = str(adversary.requests[0][0]["content"])
    assert all(f"control {control_id}" in system for control_id in CONTROL_IDS)
    (trial,) = result.adversary
    assert (trial.status, trial.claim) == (TrialStatus.GRADED, Claim(ClaimKind.NO_SHORTCUT, "", ""))
    # validation.json holds the adversary field without the attempt files it names.
    write_validation(context.round_dir, result)
    for name in evidence_files(result):
        (context.round_dir / name).unlink()
    assert load_validation(context.round_dir) == result


async def test_each_submission_is_graded_as_a_control_and_recorded_with_its_reward(
    glm_client, fake_glm, ledger, tmp_path
):
    fake_glm.stream(content=controls_reply("41"))
    adversary = AdversaryScript((Submit("41"), Submit("42"), "SHORTCUT: it accepts 42.\ngrader line: expected 42"))
    context, settings = round_context(tmp_path), round_settings(ledger, k=1, adversary=adversary_policy(k=1))

    result = await validate_round(
        built_draft(tmp_path / "store"), context, settings, SolverModel(), glm_client, adversary
    )

    (trial,) = result.adversary
    graded = [(s.ordinal, s.status, s.reward, s.passed) for s in trial.submissions]
    assert graded == [(1, GradeStatus.GRADED, 0.0, False), (2, GradeStatus.GRADED, 1.0, True)]
    assert trial.claim == Claim(ClaimKind.SHORTCUT, "it accepts 42.", "expected 42")
    for submission in trial.submissions:
        grading = json.loads((context.round_dir / submission.attempt_file).read_bytes())
        assert grading["grade"]["reward"] == submission.reward
        assert Path(submission.attempt_file).parts[-3] == "control"
    observations = [m for m in adversary.requests[-1] if m["role"] == "tool"]
    assert [json.loads(str(m["content"]))["reward"] for m in observations] == [0.0, 1.0]


async def test_the_submission_budget_grades_no_submission_past_it(glm_client, fake_glm, ledger, tmp_path):
    fake_glm.stream(content=controls_reply("41"))
    adversary = AdversaryScript((Submit("40"), Submit("41"), Submit("42"), "NO_SHORTCUT"))
    context, settings = round_context(tmp_path), round_settings(
        ledger, k=1, adversary=adversary_policy(k=1, submissions=2)
    )

    result = await validate_round(
        built_draft(tmp_path / "store"), context, settings, SolverModel(), glm_client, adversary
    )

    (trial,) = result.adversary
    assert [s.ordinal for s in trial.submissions] == [1, 2]
    last_observation = [m for m in adversary.requests[-1] if m["role"] == "tool"][-1]
    assert "submission budget spent" in json.loads(str(last_observation["content"]))["error"]
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
    assert {step for step in steps if not step.startswith("control/s")} == {"adversary/1/1"}
    assert again.adversary[0] == first.adversary[0]
    rerun, before = again.adversary[1], first.adversary[1]
    assert rerun.attempt_file.endswith("adversary/1/attempt-1.json")
    assert [(s.ordinal, s.reward) for s in rerun.submissions] == [(s.ordinal, s.reward) for s in before.submissions]
    assert (again.solver, again.controls) == (first.solver, first.controls)


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

    assert not list(context.round_dir.rglob("adversary/0/attempt-*"))
    # The cancelled attempt's graded submission stays, so the rerun is numbered after it.
    assert list(context.round_dir.rglob("adversary/0/submissions/0/control/s1/attempt-0.json"))
    result = await validate_round(
        built_draft(tmp_path / "store"), context, settings, SolverModel(), UncalledClient(), AdversaryScript(PROBES)
    )
    (trial,) = result.adversary
    assert trial.status is TrialStatus.GRADED
    assert trial.attempt_file.endswith("adversary/0/attempt-1.json")


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

    assert result.early == EarlyDecision(VIOLATED_CONTROL, ("solver/0", "solver/1", "adversary/0"))
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

    retried = round_context(tmp_path, validation_pass=2)
    solver = FailingTrials(UNAVAILABLE)
    again = await validate_round(
        built_draft(tmp_path / "store"), retried, settings, solver, UncalledClient(), UncalledClient()
    )

    assert solver.steps == ["solver/0/2", "solver/0/3"]
    assert again.solver is not None and again.solver.trials[1:] == first.solver.trials[1:]
    assert again.solver.trials[0].attempt_file.endswith("solver/0/attempt-3.json")
    trial_dir = retried.round_dir / Path(again.solver.trials[0].attempt_file).parent
    assert sorted(path.name for path in trial_dir.iterdir()) == [f"attempt-{n}.json" for n in range(4)]
    assert again.controls == first.controls

    # The same retry relaunched finds every budget spent and runs nothing.
    unchanged = await validate_round(
        built_draft(tmp_path / "store"),
        retried,
        settings,
        SolverModel(allowed=frozenset()),
        UncalledClient(),
        UncalledClient(),
    )
    assert unchanged == again

    # A relaunch of the workpiece abandoned after that retry validates the round's third pass: trial 0 alone runs
    # again, with a full budget.
    relaunched = FailingTrials(UNAVAILABLE)
    third = await validate_round(
        built_draft(tmp_path / "store"),
        round_context(tmp_path, validation_pass=3),
        settings,
        relaunched,
        UncalledClient(),
        UncalledClient(),
    )
    assert relaunched.steps == ["solver/0/4", "solver/0/5"]
    assert third.solver is not None and third.solver.trials[1:] == first.solver.trials[1:]
    assert sorted(path.name for path in trial_dir.iterdir()) == [f"attempt-{n}.json" for n in range(6)]


DECISION_REFERENCE = "Approved total: 1234.56 USD."
DECISION_REPLY = "The approved total is 1,234.56 USD."
DECISION_CRITERION = "The approved total is 1234.56 USD, within 0.01."
DECISION_RUBRIC = Rubric(
    deliverable="The approved total of the ledger.",
    items=[
        RubricItem(
            id="total",
            intent="The answer states the approved total.",
            pass_criterion=DECISION_CRITERION,
            equivalences=["$1,234.56"],
            negation_handling="",
            weight=1.0,
            role="scored",
            cap_value=None,
        )
    ],
)


FOOLING_KIND = "keyword_stuffing"
"""The fooling answer the calibrated decision grader read, which no brief shows."""


def decision_task(relower):
    """A task graded by a calibrated decision grader, as a build leaves one."""
    grader = draft.decision_grader(DECISION_RUBRIC, DECISION_REFERENCE).model_copy(
        update={"acceptance_level": 2.0, "calibration": {"reference": 3.6}, "fooling": {FOOLING_KIND: 0.4}}
    )
    task = draft.assemble(
        "ledger",
        "What is the approved total?",
        AnswerType.TEXT,
        PlainText(),
        grader,
        Source(dataset="test", revision="r", row="0", importer_revision="test"),
        environment=None,
    )
    return relower(task)


def adversary_plan(tmp_path: Path, settings: ValidationSettings) -> TrialPlan:
    return TrialPlan(
        workpiece_id=WORKPIECE,
        round=0,
        kind=TrialKind.ADVERSARY,
        k=1,
        max_retries=0,
        retry_backoff=settings.retry_backoff,
        evidence_dir=tmp_path / "evidence",
        ledger=settings.ledger,
        first_attempt=0,
    )


async def run_decision_adversary(
    tmp_path: Path, ledger, relower, policy: AdversaryPolicy, adversary: AdversaryScript, decision
):
    """One adversary trial on the decision task, on a host whose decision client is ``decision``."""
    settings = round_settings(ledger, adversary=policy)
    engine = replace(settings.engine, decision=decision)
    spec = decision_task(relower)
    brief = adversary_brief(spec, (), policy)
    return await run_adversary_trial(spec, brief, policy, adversary_plan(tmp_path, settings), engine, adversary, "0", 0)


async def test_a_whitebox_adversary_on_a_decision_task_is_shown_the_rubric_and_reference(ledger, relower, tmp_path):
    systems = {}
    for mode in AdversaryMode:
        adversary = AdversaryScript(("NO_SHORTCUT",))
        policy = adversary_policy(k=1, mode=mode)
        await run_decision_adversary(tmp_path / mode, ledger, relower, policy, adversary, FakeDecisionClient({}))
        systems[mode] = str(adversary.requests[0][0]["content"])

    whitebox, blackbox = systems[AdversaryMode.WHITEBOX], systems[AdversaryMode.BLACKBOX]
    assert DECISION_CRITERION in whitebox and DECISION_REFERENCE in whitebox and QUESTION in whitebox
    assert FOOLING_KIND not in whitebox
    assert DECISION_CRITERION not in blackbox and DECISION_REFERENCE not in blackbox


async def test_a_host_without_a_decision_client_refuses_a_decision_adversary_before_its_first_attempt(
    ledger, relower, tmp_path
):
    adversary = AdversaryScript((Submit(DECISION_REPLY), "NO_SHORTCUT"))

    with pytest.raises(ValueError, match="no decision client"):
        await run_decision_adversary(tmp_path, ledger, relower, adversary_policy(k=1), adversary, None)

    assert adversary.requests == []
    assert not (tmp_path / "evidence").exists()
    assert ledger.entries == []


async def test_a_decision_adversarys_submissions_are_graded_on_the_host_with_their_rewards(ledger, relower, tmp_path):
    client = FakeDecisionClient({DECISION_REPLY: 3.4})
    adversary = AdversaryScript((Submit("1000 USD"), Submit(DECISION_REPLY), "NO_SHORTCUT"))

    outcome = await run_decision_adversary(tmp_path, ledger, relower, adversary_policy(k=1), adversary, client)

    assert isinstance(outcome, Graded) and outcome.reward == 1.0
    assert [request[0]["z_candidate_answer"] for request in client.requests] == ["1000 USD", DECISION_REPLY]
    trial_dir = tmp_path / "evidence" / str(TrialKind.ADVERSARY) / "0"
    record = adversary_record((trial_dir / "attempt-0.json").read_bytes())
    assert [(s.grade.reward, s.passed) for s in record.submissions] == [(0.0, False), (1.0, True)]
    grading = trial_dir / SUBMISSIONS_DIR / "0" / str(TrialKind.CONTROL) / "s2" / "attempt-0.json"
    assert json.loads(grading.read_bytes())["grade"]["detail"] == {"level": 3.4, "bar": 2.0, "accepted": True}


class FailedVerifierFactory(FixtureImageFactory):
    async def create(self, spec: MachineSpec):
        raise MachineTerminated("verifier worker disappeared")


async def test_an_adversary_verifier_outage_remains_ungraded_and_retries(glm_client, fake_glm, ledger, tmp_path):
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
    adversary = AdversaryScript((Submit("42"), "NO_SHORTCUT"))

    result = await validate_round(built, context, settings, SolverModel(), glm_client, adversary)

    (trial,) = result.adversary
    assert (trial.status, trial.cause) == (TrialStatus.UNGRADED, Cause.MACHINE_TERMINATED)
    assert trial.attempt_file.endswith("adversary/0/attempt-1.json")
    assert all(s.status == GradeStatus.UNAVAILABLE for s in trial.submissions)
    first = json.loads((context.round_dir / Path(trial.attempt_file).with_name("attempt-0.json")).read_bytes())
    assert first["cause"] == Cause.MACHINE_TERMINATED
