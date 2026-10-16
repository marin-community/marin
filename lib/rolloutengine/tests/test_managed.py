# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independent attempts through one engine, at most a fixed number at a time."""

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, field, replace

import pytest
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, Command, Machine, PackageSpec
from taskcompendium.grading_result import Outcome
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, Source

from rolloutengine.agent import agent_harness
from rolloutengine.contracts import (
    ModelRequest,
    ModelTurn,
    Prompt,
    RolloutContractError,
    RolloutInterrupted,
    RolloutOperation,
)
from rolloutengine.managed import UNGRADED_REASON, ManagedAgent, agent_task
from rolloutengine.test_support import ReplayModel, arithmetic_task, engine, machine_runtime, rollout_spec, shell_call

SOURCE = Source(dataset="fixture", revision="1", row="0", importer_revision="1")
WORKSPACE = EnvironmentRequirements(
    command_semantics=CommandSemantics.LINUX_PROCESS,
    docker_image="fixture@sha256:" + "0" * 64,
    capabilities=("shell", "filesystem"),
)


@dataclass
class ShellSimFactory(DockerMachineFactory):
    """Run image commands on the built-in filesystem, and keep each machine."""

    backend: Backend = Backend.DOCKER
    machines: list[Machine] = field(default_factory=list)

    async def create(self, spec):
        machine = await ShellSimMachineFactory().create(
            replace(spec, source=PackageSpec(), workdir=spec.workdir or "/workspace")
        )
        self.machines.append(machine)
        return machine


def served(request: ModelRequest, message: dict, turn: int) -> ModelTurn:
    prompt = (*request.prefix_token_ids, 90, 91) if request.prefix_token_ids else (10, 11)
    return ModelTurn(message, prompt, (20 + turn,), (-0.5,), "stop")


@dataclass
class ScriptedModel:
    """Reply by task: the first message of each request selects that task's replies, in order."""

    replies: dict[str, list[dict]]

    async def complete(self, request: ModelRequest) -> ModelTurn:
        turn = sum(message["role"] == "assistant" for message in request.messages)
        return served(request, self.replies[request.messages[0]["content"]][turn], turn)


def managed(model, machine_factories=None, *, harnesses=None, concurrency: int) -> ManagedAgent:
    bound = engine(model, machine_factories, harnesses=harnesses)
    return ManagedAgent(bound.engine, bound.model, concurrency)


async def collect(agent: ManagedAgent, specs) -> dict[str, object]:
    return {outcome.spec.task.id: outcome async for outcome in agent.run(specs)}


async def test_graded_and_ungraded_specs_share_one_engine():
    model = ScriptedModel(
        {
            "What is six plus six?": [{"role": "assistant", "content": "12"}],
            "Summarize the plan.": [{"role": "assistant", "content": "A summary."}],
        }
    )
    specs = [rollout_spec(arithmetic_task()), rollout_spec(agent_task("notes", "Summarize the plan.", source=SOURCE))]
    outcomes = await collect(managed(model, concurrency=2), specs)

    graded = outcomes["arithmetic"].rollout.grade
    ungraded = outcomes["notes"].rollout.grade
    assert (graded.status, graded.reward) == (Outcome.GRADED, 1.0)
    assert (ungraded.status, ungraded.reward, ungraded.error) == (Outcome.UNAVAILABLE, None, UNGRADED_REASON)
    assert [outcome.error for outcome in outcomes.values()] == [None, None]


async def test_concurrent_attempts_keep_their_own_workspace():
    names = ("left", "right")
    model = ScriptedModel(
        {
            f"Write {name}.": [
                shell_call(f"mkdir -p /workspace/out && echo {name} > /workspace/out/{name}.txt"),
                {"role": "assistant", "content": "Done."},
            ]
            for name in names
        }
    )
    specs = [
        rollout_spec(
            agent_task(name, f"Write {name}.", source=SOURCE, environment=WORKSPACE, output_paths=("/workspace/out",)),
            machine=machine_runtime(),
        )
        for name in names
    ]
    factory = ShellSimFactory()
    outcomes = await collect(managed(model, {"local": factory}, concurrency=2), specs)

    assert {name: outcome.rollout.outputs for name, outcome in outcomes.items()} == {
        name: {f"/workspace/out/{name}.txt": f"{name}\n".encode()} for name in names
    }
    assert len(factory.machines) == 2


async def test_admission_is_bounded_and_reads_specs_lazily():
    release = asyncio.Event()
    both_waiting = asyncio.Event()
    produced = 0
    waiting = 0
    peak = 0

    class BlockingModel:
        async def complete(self, request):
            nonlocal waiting, peak
            waiting += 1
            peak = max(peak, waiting)
            if waiting == 2:
                both_waiting.set()
            try:
                await release.wait()
            finally:
                waiting -= 1
            return served(request, {"role": "assistant", "content": "ok"}, 0)

    def specs():
        nonlocal produced
        for index in range(5):
            produced += 1
            yield rollout_spec(agent_task(f"task-{index}", "Answer.", source=SOURCE))

    running = asyncio.create_task(collect(managed(BlockingModel(), concurrency=2), specs()))
    await asyncio.wait_for(both_waiting.wait(), timeout=5)
    assert produced == 2
    release.set()
    outcomes = await asyncio.wait_for(running, timeout=5)
    assert (len(outcomes), peak) == (5, 2)


async def test_closing_the_outcomes_cancels_running_attempts():
    slow_waiting = asyncio.Event()

    class Model:
        async def complete(self, request):
            if request.messages[0]["content"] == "Slow.":
                slow_waiting.set()
                await asyncio.Event().wait()
            await slow_waiting.wait()
            return served(request, {"role": "assistant", "content": "Done."}, 0)

    specs = [
        rollout_spec(
            agent_task(name, f"{name.capitalize()}.", source=SOURCE, environment=WORKSPACE), machine=machine_runtime()
        )
        for name in ("slow", "fast")
    ]
    factory = ShellSimFactory()
    async with aclosing(managed(Model(), {"local": factory}, concurrency=2).run(specs)) as outcomes:
        async for outcome in outcomes:
            assert outcome.spec.task.id == "fast"
            break

    assert len(factory.machines) == 2
    for machine in factory.machines:
        with pytest.raises(RuntimeError):
            await machine.run(Command(("true",)))


async def test_a_failed_model_call_ends_only_its_own_attempt():
    class Model:
        async def complete(self, request):
            if request.messages[0]["content"] == "Fail.":
                raise ConnectionError("endpoint unavailable")
            return served(request, {"role": "assistant", "content": "12"}, 0)

    specs = [rollout_spec(agent_task("broken", "Fail.", source=SOURCE)), rollout_spec(arithmetic_task())]
    outcomes = await collect(managed(Model(), concurrency=2), specs)

    broken = outcomes["broken"]
    assert isinstance(broken.error, RolloutInterrupted)
    assert broken.error.operation == RolloutOperation.MODEL
    assert broken.rollout is broken.error.rollout
    assert outcomes["arithmetic"].rollout.grade.reward == 1.0


async def test_a_token_contract_violation_stops_the_run():
    class Model:
        async def complete(self, request):
            return ModelTurn({"role": "assistant", "content": "12"}, (10, 11), (), None, "stop")

    specs = [rollout_spec(arithmetic_task()), rollout_spec(agent_task("next", "Answer.", source=SOURCE))]
    outcomes = []
    with pytest.raises(RolloutContractError):
        async for outcome in managed(Model(), concurrency=1).run(specs):
            outcomes.append(outcome)
    assert outcomes == []


async def test_an_agent_harness_runs_under_the_managed_agent():
    async def run_agent(context, complete):
        prompt = Prompt(({"role": "user", "content": "What is six plus six?"},), {})
        turn = await complete(prompt)
        await complete(Prompt((*prompt.messages, turn.message, {"role": "user", "content": "Only the number."}), {}))

    model = ReplayModel([{"role": "assistant", "content": "Let me add."}, {"role": "assistant", "content": "12"}])
    agent = managed(model, harnesses={"agent": agent_harness(run_agent)}, concurrency=1)
    outcomes = await collect(agent, [rollout_spec(arithmetic_task(), harness="agent")])

    rollout = outcomes["arithmetic"].rollout
    assert (rollout.grade.status, rollout.grade.reward) == (Outcome.GRADED, 1.0)
    assert rollout.loss_mask == (1, 0, 0, 1)


async def test_repeated_tasks_get_distinct_run_keys():
    events = []

    class Observer:
        async def on_event(self, event):
            events.append(event.run_key)

    bound = engine(ReplayModel([{"role": "assistant", "content": "12"}] * 3), observer=Observer())
    agent = ManagedAgent(bound.engine, bound.model, concurrency=1)
    outcomes = [outcome async for outcome in agent.run([rollout_spec(arithmetic_task())] * 3)]

    call_ids = {outcome.run_key.split("/")[0] for outcome in outcomes}
    assert len(call_ids) == 1
    assert [outcome.run_key.split("/", 1)[1] for outcome in outcomes] == ["arithmetic/0", "arithmetic/1", "arithmetic/2"]
    assert sorted(set(events)) == sorted(outcome.run_key for outcome in outcomes)


async def test_concurrent_calls_get_distinct_run_keys():
    agent = managed(ReplayModel([{"role": "assistant", "content": "12"}] * 2), concurrency=1)

    async def keys():
        return [outcome.run_key async for outcome in agent.run([rollout_spec(arithmetic_task())])]

    first, second = await asyncio.gather(keys(), keys())
    assert first != second


async def test_a_failing_spec_stream_keeps_the_finished_outcome():
    def specs():
        yield rollout_spec(arithmetic_task())
        raise RuntimeError("The spec source failed")

    agent = managed(ReplayModel([{"role": "assistant", "content": "12"}]), concurrency=1)
    outcomes = []
    with pytest.raises(RuntimeError, match="spec source failed"):
        async for outcome in agent.run(specs()):
            outcomes.append(outcome)

    assert [outcome.rollout.grade.reward for outcome in outcomes] == [1.0]
