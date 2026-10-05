# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executable task evidence, reset behavior and curation replay."""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from shellbox.machine import Command, DockerImage, ExitReason, MachineSpec, Result

from taskcompendium.grader import grader_config
from taskcompendium.grading import grade_task
from taskcompendium.models import ConversationTrace, FunctionCall, Source, TaskSpec, TextMessage
from taskcompendium.pipeline.datasets import calendar, nemo_actions, shell_files
from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import CheckStatus, DatasetRecipe, GeneratedSource, IntendedUse, RawRow
from taskcompendium.pipeline.review import BatchReviewer
from taskcompendium.pipeline.verification import PLAIN, verify_task
from taskcompendium.runtime.calendar import CalendarFactory, CalendarGoal, calendar_controls
from taskcompendium.runtime.checks import check_episodes, episode_suite
from taskcompendium.runtime.controls import tool_turn
from taskcompendium.runtime.episode import ScriptedActor, run_episode
from taskcompendium.runtime.models import ActorTask, RolloutRecord, Termination
from taskcompendium.runtime.shell import ShellFactory

from .pipeline_stages import run_stages
from .test_pipeline import BatchService


@pytest.fixture
def calendar_task():
    source = Source(dataset="mock/calendar", revision="1", row="0", importer_revision="1")
    task = calendar.normalize(RawRow("calendar-0", source, next(calendar.generate_rows(1))))
    assert isinstance(task, TaskSpec)
    return task


async def test_calendar_alternative_solutions_preserve_state_and_reset(calendar_task):
    suite = episode_suite(CalendarFactory(), max_steps=4)
    # Run async here instead of nesting the synchronous pipeline entrypoint's event loop.
    report = await check_episodes(calendar_task, CalendarFactory(), max_steps=suite.parameters["max_steps"])
    assert all(check.status == CheckStatus.PASS for check in report.checks)
    reference, alternate, reset = [
        rollout for rollout in report.rollouts if rollout.control in {"reference", "alternate", "reset"}
    ]
    reference_state, alternate_state = json.loads(reference.state_json), json.loads(alternate.state_json)
    assert reference_state["events"][-1]["start"] != alternate_state["events"][-1]["start"]
    assert reference.state_json == reset.state_json
    assert reference_state["events"][:2] == calendar_task.environment.tool_providers["calendar"].initial_state["events"]
    saved = RolloutRecord.model_validate_json(reference.model_dump_json())
    assert saved.evidence() == reference.evidence()


async def test_calendar_deleting_existing_meeting_cannot_satisfy_goal(calendar_task):
    verifier = CalendarGoal.model_validate(grader_config(calendar_task))
    valid = calendar_controls(verifier)[1].responses[0]
    rollout = await run_episode(
        calendar_task,
        ScriptedActor((tool_turn("delete_event", {"id": "busy-alice"}), valid)),
        CalendarFactory(),
        max_steps=4,
        control="tampered",
    )
    result = grade_task(calendar_task, PLAIN, ConversationTrace(events=rollout.events), rollout.evidence())
    assert (result.status, result.reward) == ("graded", 0.0)


@dataclass
class ObservingActor:
    public: list[ActorTask] = field(default_factory=list)

    def respond(self, task, events):
        self.public.append(task)
        # Continue calling a read-only tool until the episode budget expires.
        return tool_turn("list_events", {})


async def test_episode_budget_keeps_tool_observations_without_private_actor_inputs(calendar_task):
    actor = ObservingActor()
    rollout = await run_episode(calendar_task, actor, CalendarFactory(), max_steps=1, control="budget")
    assert rollout.termination == Termination.STEP_LIMIT
    assert (
        json.loads(rollout.events[-1].content)["events"]
        == calendar_task.environment.tool_providers["calendar"].initial_state["events"]
    )
    assert actor.public[0].tools[0].name == "list_events"


@dataclass
class FileMachine:
    """External machine boundary with uploaded files and bounded capture reads."""

    files: dict[str, bytes] = field(default_factory=dict)
    closed: bool = False

    async def run(self, command: Command):
        if command.argv[0] == "mkdir":
            return Result(0, b"", b"", False, False, ExitReason.EXITED)
        path = command.argv[4]
        data = self.files.get(path)
        if data is None:
            return Result(44, b"", b"", False, False, ExitReason.EXITED)
        limit = command.output_limit_bytes
        return Result(0, data[:limit], b"", len(data) > limit, False, ExitReason.EXITED)

    async def upload(self, source: Path, target: str):
        self.files[target] = source.read_bytes()

    async def download(self, source: str, target: Path):
        target.write_bytes(self.files[source])

    async def close(self):
        self.closed = True


@dataclass
class FileMachines:
    machines: list[FileMachine] = field(default_factory=list)

    async def create(self, spec):
        machine = FileMachine()
        self.machines.append(machine)
        return machine


async def test_shell_uploads_public_files_without_oracle_and_captures_submission():
    source = Source(dataset="mock/shell-files", revision="1", row="0", importer_revision="1")
    task = shell_files.normalize(RawRow("shell-0", source, next(shell_files.generate_rows(1))))
    assert isinstance(task, TaskSpec)
    machines = FileMachines()
    factory = ShellFactory(machines, MachineSpec(DockerImage("test")), {"backend": "file-machine"}, 1, 1024)
    env = await factory.create(task)
    assert "/workspace/people.csv" in machines.machines[0].files
    assert shell_files.CONTROL_PATH not in machines.machines[0].files
    missing = grade_task(
        task,
        PLAIN,
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="Done."))),
        await env.evidence(),
    )
    assert missing.reward == 0.0
    machines.machines[0].files[shell_files.OUTPUT_PATH] = b"person-0-2\nperson-0-0\n"
    correct = grade_task(
        task,
        PLAIN,
        ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="Done."))),
        await env.evidence(),
    )
    assert correct.reward == 1.0
    await env.close()
    fresh = await factory.create(task)
    assert (await fresh.evidence()).files == {}
    await fresh.close()
    assert all(machine.closed for machine in machines.machines)


@dataclass(frozen=True)
class UnavailableFactory:
    @property
    def identity(self):
        return {"backend": "unavailable"}

    async def create(self, task):
        raise RuntimeError("Machine service unavailable")


async def test_environment_failure_is_recorded_without_reward(calendar_task):
    rollout = await run_episode(calendar_task, ScriptedActor(()), UnavailableFactory(), max_steps=2, control="failure")
    assert rollout.termination == Termination.INFRA_ERROR
    assert rollout.artifacts == ()
    assert rollout.detail == "Machine service unavailable"


def test_calendar_pipeline_persists_controls_and_replays_without_runtime(tmp_path):
    recipe = DatasetRecipe(
        name="calendar-fixture",
        version="1",
        source=GeneratedSource("mock/calendar", "1", "default", "train", calendar.__name__),
        pipeline=replace(calendar.pipeline(), check_suite=episode_suite(CalendarFactory(), max_steps=4)),
        intended_use=IntendedUse.TRAIN,
        inputs=RecipeInputs(SourceFiles(("*.jsonl",), SourceFormat.JSONL), ()),
    )
    reviewer = BatchReviewer(BatchService(), "fake", "1")
    first = run_stages(recipe, calendar.generate_rows(2), output_path=tmp_path, limit=2, reviewer=reviewer)
    assert first["dispositions"] == {"keep": 2}
    rollouts = [
        RolloutRecord.model_validate(row)
        for evidence in (tmp_path / "audited/evidence").glob("*/attempt-*/checks.json")
        for checks in json.loads(evidence.read_text()).values()
        for row in checks["rollouts"]
    ]
    assert len(rollouts) == 10
    unavailable = replace(
        recipe,
        pipeline=replace(
            recipe.pipeline,
            check_suite=replace(
                recipe.pipeline.check_suite, run=lambda task: pytest.fail("Runtime was called during replay")
            ),
        ),
    )
    second = run_stages(unavailable, iter(()), output_path=tmp_path, limit=2, reviewer=reviewer)
    assert first == second


def test_nemo_pipeline_normalizes_untyped_source_messages_and_checks_actions():
    path = Path(__file__).parent / "fixtures/nemo/predicted-action.json"
    data = json.loads(path.read_text())
    for item in data["responses_create_params"]["input"]:
        if item.get("type") == "message" and item["role"] in {"system", "user"}:
            del item["type"]
    source = Source(
        dataset="fixture/nemo",
        revision="1",
        row="0",
        importer_revision="1",
    )
    task = nemo_actions.normalize(RawRow("action-0", source, data))
    assert isinstance(task, TaskSpec)
    assert all(check.status == CheckStatus.PASS for check in verify_task(task))
    assert task.context.events[-1].content == data["responses_create_params"]["input"][-1]["content"]


async def test_calendar_event_ids_stay_unique_after_deletion(calendar_task):
    environment = await CalendarFactory().create(calendar_task)
    meeting = {"title": "Planning", "start": 660, "end": 690, "participants": ["Alice", "Bob"]}
    first = json.loads(await environment.step(FunctionCall(name="create_event", arguments=meeting)))
    await environment.step(FunctionCall(name="delete_event", arguments={"id": "busy-alice"}))
    second = json.loads(await environment.step(FunctionCall(name="create_event", arguments=meeting)))
    events = json.loads((await environment.evidence()).state_json)["events"]
    assert first["id"] != second["id"]
    assert len({event["id"] for event in events}) == len(events)
