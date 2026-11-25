"""Native rollout harnesses preserve exact sampled work through SkyRL."""

import asyncio

import numpy as np
import pytest
from omegaconf import OmegaConf
from rolloutengine.engine import RolloutEngine
from marinskyrl.agent_harnesses import AgentHarnessSettings, TaskAgentHarnesses
from rolloutengine.harness.external import AgentHarness
from rolloutengine.harness.step import StepHarness
from rolloutengine.models import ContextWindowExceeded, Prompt, Transition
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker, _model_turn
from skyrl_train.trajectory_runners.model_clients import (
    ContextLengthExceededError,
    GenerationBudgetExceededError,
)
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TokenProvenance
from taskcompendium.grading_result import Outcome
from taskcompendium.grader import verifyit_package
from taskcompendium.models import AgentMachine, AnswerType, ScriptGrader
from verifyit.spec import NumericSpec
from tests.cpu.task_specs import SHELL_ENVIRONMENT, lowered_task, task_input, text_task


class WorkerTokenizer:
    eos_token_id = 99

    def decode(self, token_ids, **kwargs):
        return "Working"


@pytest.fixture
def worker_config():
    return OmegaConf.create(
        {
            "backend": "vllm",
            "max_input_length": 3,
            "max_model_len": 128,
            "apply_overlong_filtering": False,
            "sampling_params": {
                "max_generate_length": 8,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0,
                "logprobs": 0,
            },
        }
    )


@pytest.fixture
def agent_harnesses(monkeypatch):
    async def tokenize(_payload):
        raise AssertionError("This scripted harness only requests completions")

    def bind(agent):
        monkeypatch.setattr("marinskyrl.agent_harnesses.pi_harness", lambda _options, _tokenize: AgentHarness(agent))
        return TaskAgentHarnesses(
            AgentHarnessSettings.model_validate(
                {"pi": {"version": "0.87.0", "thinking": "high", "thinking_format": "chat-template", "setup_timeout": 10}}
            ),
            "fixture",
            tokenize,
        )

    return bind


def sampled_reply(content="Working"):
    return {
        "token_provenance": TokenProvenance.ENGINE,
        "assistant_messages": [{"role": "assistant", "content": content}],
        "prompt_ids": [[1, 2]],
        "response_ids": [[3]],
        "response_logprobs": [[-0.1]],
        "stop_reasons": ["stop"],
    }


class ScriptedClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.mark.asyncio
async def test_native_step_harness_keeps_length_stopped_rollout():
    class Episode:
        async def prepare(self):
            return Prompt(({"role": "user", "content": "Start"},), {})

        async def advance(self, _turn):
            return Transition(done=False, observations=({"role": "user", "content": "Continue"},))

        async def close(self):
            pass

    reply = sampled_reply()
    del reply["response_logprobs"]
    client = ScriptedClient([reply, GenerationBudgetExceededError([1, 2, 3, 4])])

    async def model(request):
        return await _model_turn(
            client,
            request,
            sampling_params={"max_tokens": 8},
            max_context_length=128,
            max_prompt_length=128,
            chat_template_kwargs={},
            session_id="native",
        )

    runner = RolloutEngine({}, {"native": StepHarness(lambda _context: Episode())})
    result = await runner.run_task(lowered_task(text_task(), session_name="native"), model)
    assert result.rollout.stop_reason == "length"
    assert len(result.rollout.conversations[0].steps) == 1
    assert result.rollout.conversations[0].response_token_ids == (3,)


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_first", [False, True], ids=["first-budget", "later-serving"])
async def test_native_worker_recovers_without_spending_a_turn(agent_harnesses, worker_config, sample_first):
    rejection = (
        ContextLengthExceededError("context_overflow", "fixture", 400)
        if sample_first
        else GenerationBudgetExceededError([1, 2, 3, 4])
    )
    replies = ([sampled_reply()] if sample_first else []) + [rejection, sampled_reply("12")]

    async def agent(_context, complete):
        if sample_first:
            await complete(Prompt(({"role": "user", "content": "Start"},), {}))
        with pytest.raises(ContextWindowExceeded):
            await complete(Prompt(({"role": "user", "content": "Oversized"},), {}))
        await complete(Prompt(({"role": "user", "content": "Compacted: return 12"},), {}))

    task = text_task().model_copy(update={"grader": verifyit_package(NumericSpec("12", 0, 0)).grader})
    request = TaskRequestBatch(
        rows=(task_input(lowered_task(task, "pi", max_turns=2 if sample_first else 1)),),
        sampling_params={"seed": 20261125},
    )
    client = ScriptedClient(replies)
    worker = TaskRolloutWorker(
        worker_config,
        WholeTaskProjection(worker_config, WorkerTokenizer()),
        client,
        {},
        agents=agent_harnesses(agent),
        shutdown_timeout=5,
    )
    try:
        (result,) = await worker.generate(request)
        batch = await worker.training_batch(request, [result])
    finally:
        await worker.shutdown()
    assert result.failure is None
    assert result.grade.status == Outcome.GRADED
    assert batch["response_ids"] == [[3]]
    assert batch["loss_masks"] == [[1]]
    assert batch["rollout_logprobs"][0] == pytest.approx([-0.1])
    assert np.sum(batch["rewards"]) == 1.0
    messages = batch["evidence_messages"][0]
    assert [message["content"] for message in messages if message["role"] == "assistant"] == (
        ["Working", "12"] if sample_first else ["12"]
    )
    assert batch["assistant_turn_token_ids"] == ([[[3], [3]]] if sample_first else [[[3]]])
    seeds = [request["sampling_params"]["seed"] for request in client.requests]
    assert len(seeds) == len(set(seeds))


@pytest.mark.asyncio
async def test_native_worker_keeps_a_workspace_verdict_without_sampled_tokens(agent_harnesses, worker_config):
    class WaitingClient:
        async def generate(self, _request):
            await asyncio.Future()

    async def agent(context, complete):
        try:
            await complete(Prompt(({"role": "user", "content": "Start"},), {}))
        finally:
            await context.machine.run(Command(("sh", "-c", "touch /workspace/agent-stopped")))

    task = text_task().model_copy(
        update={
            "environment_requirements": SHELL_ENVIRONMENT,
            "answer_type": AnswerType.WORKSPACE_STATE,
            "grader": ScriptGrader(
                argv=("sh", "-c", "test -f /workspace/agent-stopped && echo 1"),
                cwd="/workspace",
                answer_path=None,
                placement=AgentMachine(),
            ),
        }
    )
    request = TaskRequestBatch(
        rows=(task_input(lowered_task(task, "pi", backend="shellsim", total_turn_timeout=0.1)),),
    )
    worker = TaskRolloutWorker(
        worker_config,
        WholeTaskProjection(worker_config, WorkerTokenizer()),
        WaitingClient(),
        {"shellsim": ShellSimMachineFactory()},
        agents=agent_harnesses(agent),
        shutdown_timeout=5,
    )
    try:
        (result,) = await worker.generate(request)
        batch = await worker.training_batch(request, [result])
    finally:
        await worker.shutdown()

    assert (result.grade.status, result.grade.reward) == (Outcome.GRADED, 1.0)
    assert result.failure is None
    assert result.steps == result.response_token_ids == ()
    assert batch["verification_results"][0].score == 1.0
    assert batch["loss_masks"] == [[0]]
    assert batch["rollout_logprobs"][0] == pytest.approx([0.0])


@pytest.mark.asyncio
async def test_training_machine_limit_leaves_inline_work_unblocked(agent_harnesses, worker_config):
    release_machine = asyncio.Event()
    first_creation = asyncio.Event()
    inline_started = asyncio.Event()

    class PausedFactory(ShellSimMachineFactory):
        creations = 0

        async def create(self, spec):
            self.creations += 1
            first_creation.set()
            await release_machine.wait()
            return await super().create(spec)

    async def agent(context, complete):
        await complete(Prompt(({"role": "user", "content": context.task.id},), {}))
        if context.machine is None:
            inline_started.set()

    machines = PausedFactory()
    rows = tuple(
        task_input(lowered_task(text_task(task_id).model_copy(update={"environment_requirements": SHELL_ENVIRONMENT}),
                                "pi", backend="shellsim"))
        for task_id in ("machine-a", "machine-b")
    )
    request = TaskRequestBatch(rows=(*rows, task_input(lowered_task(text_task("inline"), "pi"))))
    worker = TaskRolloutWorker(
        worker_config,
        WholeTaskProjection(worker_config, WorkerTokenizer()),
        ScriptedClient([sampled_reply()] * 3),
        {"shellsim": machines},
        agents=agent_harnesses(agent),
        concurrent_training_machine_tasks=1,
        shutdown_timeout=5,
    )
    try:
        async with asyncio.timeout(5), asyncio.TaskGroup() as group:
            pending = group.create_task(worker.generate(request))
            await inline_started.wait()
            await first_creation.wait()
            assert machines.creations == 1
            release_machine.set()
        assert all(result.failure is None for result in pending.result())
        assert machines.creations == 2
    finally:
        await worker.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(("exit_code", "retries"), [(2, 0), (137, 1)])
async def test_agent_setup_retries_signal_exit(agent_harnesses, worker_config, exit_code, retries):
    async def agent(_context, _complete):
        pytest.fail("An unsuccessful setup must stop the agent")

    async def skip_backoff(_delay):
        pass

    environment = SHELL_ENVIRONMENT.model_copy(update={"setup_commands": (f"exit {exit_code}",)})
    task = text_task().model_copy(update={"environment_requirements": environment})
    request = TaskRequestBatch(rows=(task_input(lowered_task(task, "pi", backend="shellsim")),))
    worker = TaskRolloutWorker(
        worker_config,
        WholeTaskProjection(worker_config, WorkerTokenizer()),
        ScriptedClient([]),
        {"shellsim": ShellSimMachineFactory()},
        retry_wait=skip_backoff,
        agents=agent_harnesses(agent),
        shutdown_timeout=5,
    )
    try:
        (result,) = await worker.generate(request)
    finally:
        await worker.shutdown()
    assert result.failure is not None
    assert result.failure.diagnostics["setup_exit_code"] == exit_code
    assert result.metrics["rollout_retries"] == retries
