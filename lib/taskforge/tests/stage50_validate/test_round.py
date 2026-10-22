# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""validate_round: the unit graph, resuming from the unit files, files keyed by what they cover, cancellation."""

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from pydantic import BaseModel
from rolloutengine.models import Model, ModelRequest, ModelTurn
from shellbox.machine import Backend
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, PlainText, Source
from verifyit.spec import ExactSpec

from taskforge.ledger.records import EntryKind
from taskforge.llm.client import Completion, ModelClient, StructuredResult
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
from taskforge.stage50_validate.control_drafts import CONTROLS_STEP
from taskforge.stage50_validate.controls import ControlVerdict
from taskforge.stage50_validate.round import (
    LoweringStatus,
    TrialStatus,
    ValidationContext,
    ValidationResult,
    load_validation,
    validate_round,
)
from taskforge.stage50_validate.trials import Deadlines, EngineSettings, RetryBackoff
from tests.sandbox.fixture_images import FixtureImageFactory, PortlessShellSimFactory
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
        program_digest="program",
        sdk_version=SDK_VERSION,
        model="model",
        model_digest="model",
        round=0,
        steps=(RLEnvBuildStepOutcome(CacheStatus.MISS, memo),),
    )
    return TaskDraft(task=TASK, provenance=provenance)


def round_context(tmp_path: Path, ledger, k: int = 2, deadlines: Deadlines | None = None) -> ValidationContext:
    return ValidationContext(
        workpiece_id=WORKPIECE,
        round=0,
        round_dir=tmp_path / "round",
        proposal=parse(PROPOSAL),
        store_root=str(tmp_path / "store"),
        host=MachineHost.LAPTOP,
        engine=EngineSettings(
            factories={Backend.SHELLSIM.value: PortlessShellSimFactory(), Backend.DOCKER.value: FixtureImageFactory()},
            max_turns=4,
            command_timeout=10,
            tool_turn_timeout=20,
            model_turn_timeout=30,
            cleanup_timeout=10,
        ),
        deadlines=deadlines or Deadlines(total_turn_timeout=30, attempt_timeout=60),
        k=k,
        max_retries=1,
        retry_backoff=RetryBackoff(initial=0.001, maximum=0.001, factor=2.0, jitter=0.0),
        sampling=LLMPolicy(),
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
    """A control writer for a round whose controls are on disk: any call fails the test."""

    @property
    def model(self) -> str:
        return "uncalled"

    async def complete(
        self, messages: Sequence[Message], policy: LLMPolicy, request_fields: Mapping[str, object] | None = None
    ) -> Completion:
        raise AssertionError("the round wrote controls it had on disk")

    async def complete_structured[T: BaseModel](
        self, messages: Sequence[Message], output_type: type[T], policy: LLMPolicy
    ) -> StructuredResult[T]:
        raise AssertionError("the round wrote controls it had on disk")


def trial_steps(ledger, since: int = 0) -> set[str]:
    return {entry.step for entry in ledger.entries[since:] if entry.kind is EntryKind.TRIAL}


def evidence_files(result: ValidationResult) -> list[str]:
    assert result.solver is not None
    return [trial.attempt_file for trial in result.solver.trials] + [c.attempt_file for c in result.controls]


def files_under(directory: Path) -> dict[str, bytes]:
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


async def test_solver_trials_start_before_the_controls_exist_and_replay_waits_for_them(
    glm_client, fake_glm, ledger, tmp_path
):
    fake_glm.stream(content=controls_reply("41"))
    release = asyncio.Event()
    solver = SolverModel()
    context = round_context(tmp_path, ledger)
    running = asyncio.create_task(
        validate_round(built_draft(tmp_path / "store"), context, solver, HeldClient(glm_client, release))
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
    for name in evidence_files(result):
        (context.round_dir / name).unlink()
    assert load_validation(context.round_dir) == result


async def test_a_relaunch_redoes_only_the_units_whose_files_are_missing(glm_client, fake_glm, ledger, tmp_path):
    fake_glm.stream(content=controls_reply("41"))
    context = round_context(tmp_path, ledger)
    first = await validate_round(built_draft(tmp_path / "store"), context, SolverModel(), glm_client)
    assert first.solver is not None
    (context.round_dir / first.solver.trials[1].attempt_file).unlink()
    (context.round_dir / first.controls[1].attempt_file).unlink()
    recorded = len(ledger.entries)

    solver = SolverModel(allowed=frozenset({"solver/1/0"}))
    again = await validate_round(built_draft(tmp_path / "store"), context, solver, UncalledClient())

    assert solver.steps == ["solver/1/0"]
    assert trial_steps(ledger, recorded) == {"solver/1/0", "control/N1-shortcut/0"}
    assert again == first
    assert load_validation(context.round_dir) == first


async def test_the_trials_of_another_lowered_spec_are_not_read_as_done(glm_client, fake_glm, ledger, tmp_path):
    """Other deadlines lower the same draft to another spec: its trials run in their own directory."""
    fake_glm.stream(content=controls_reply("41"))
    draft_of_round = built_draft(tmp_path / "store")
    first = await validate_round(draft_of_round, round_context(tmp_path, ledger), SolverModel(), glm_client)
    trials_dir = tmp_path / "round" / Path(evidence_files(first)[0]).parents[2]
    before = files_under(trials_dir)

    solver = SolverModel()
    longer = Deadlines(total_turn_timeout=40, attempt_timeout=80)
    other = await validate_round(
        draft_of_round, round_context(tmp_path, ledger, deadlines=longer), solver, UncalledClient()
    )

    assert other.lowering.task_digest != first.lowering.task_digest
    assert sorted(solver.steps) == ["solver/0/0", "solver/1/0"]
    assert files_under(trials_dir) == before
    assert not any(name.startswith(str(trials_dir.relative_to(tmp_path / "round"))) for name in evidence_files(other))


async def test_a_rewritten_control_set_is_replayed_again(glm_client, fake_glm, ledger, tmp_path):
    fake_glm.stream(content=controls_reply("41"))
    fake_glm.stream(content=controls_reply("43"))
    context = round_context(tmp_path, ledger)
    first = await validate_round(built_draft(tmp_path / "store"), context, SolverModel(), glm_client)
    (next(context.round_dir.rglob("controls.json"))).unlink()
    recorded = len(ledger.entries)

    again = await validate_round(built_draft(tmp_path / "store"), context, SolverModel(allowed=frozenset()), glm_client)

    assert trial_steps(ledger, recorded) == {f"control/{control_id}/0" for control_id in CONTROL_IDS}
    assert again.solver == first.solver
    assert {Path(c.attempt_file).parents[2] for c in again.controls}.isdisjoint(
        {Path(c.attempt_file).parents[2] for c in first.controls}
    )


async def test_a_trial_cancelled_in_flight_leaves_no_attempt_and_is_rerun(glm_client, fake_glm, ledger, tmp_path):
    fake_glm.stream(content=controls_reply("41"))
    fake_glm.stream(content=controls_reply("41"))
    context = round_context(tmp_path, ledger, k=1)
    hanging = SolverModel(release=asyncio.Event())
    running = asyncio.create_task(validate_round(built_draft(tmp_path / "store"), context, hanging, glm_client))
    await asyncio.wait_for(hanging.called.wait(), timeout=30)

    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert not [path for path in context.round_dir.rglob("*") if "/solver/" in str(path) and path.is_file()]
    solver = SolverModel()
    result = await validate_round(built_draft(tmp_path / "store"), context, solver, glm_client)
    assert solver.steps == ["solver/0/0"]
    assert result.solver is not None and result.solver.ungraded == {}
    assert [trial.status for trial in result.solver.trials] == [TrialStatus.GRADED]


async def test_a_rejected_lowering_runs_no_trial_and_round_trips(glm_client, fake_glm, ledger, tmp_path):
    fake_glm.stream(content=controls_reply("41"))
    context = round_context(tmp_path, ledger)
    context = replace(context, engine=replace(context.engine, factories={}))

    result = await validate_round(built_draft(tmp_path / "store"), context, SolverModel(allowed=frozenset()), glm_client)

    assert result.lowering.status is LoweringStatus.REJECTED
    assert result.solver is None and result.controls == ()
    assert load_validation(context.round_dir) == result
