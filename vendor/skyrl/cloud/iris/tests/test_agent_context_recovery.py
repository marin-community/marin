"""Native rollout harnesses preserve exact sampled work through SkyRL."""

import asyncio

import numpy as np
import pytest
from omegaconf import OmegaConf
from ray.exceptions import ActorDiedError
from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from rolloutengine.engine import RolloutEngine
from marinskyrl.agent_harnesses import AgentHarnessSettings, TaskAgentHarnesses
from rolloutengine.harness.external import AgentHarness
from rolloutengine.harness.step import StepHarness
from rolloutengine.models import ContextWindowExceeded, ModelRequest, Prompt, Transition
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command
from skyrl_train.rollouts.training_record import ModelGenerationUsage
from skyrl_train.telemetry import GeneratedWork
from skyrl_train.rollouts.task_projections import StepTaskProjection, WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker, _model_turn
from skyrl_train.trajectory_runners.model_clients import (
    ContextLengthExceededError,
    DirectModelClient,
    GenerationBudgetExceededError,
    ModelServerError,
)
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TokenProvenance, TrajectoryID
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

    async def generate(self, _request):
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
            usage=ModelGenerationUsage(),
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
    first = {**sampled_reply(), "stop_reasons": ["length"]}
    replies = ([first] if sample_first else []) + [rejection, sampled_reply("12")]

    async def agent(_context, complete):
        if sample_first:
            await complete(Prompt(({"role": "user", "content": "Start"},), {}))
        with pytest.raises(ContextWindowExceeded):
            await complete(Prompt(({"role": "user", "content": "Oversized"},), {}))
        await complete(Prompt(({"role": "user", "content": "Compacted: return 12"},), {}))

    task = text_task().model_copy(update={"grader": verifyit_package(NumericSpec("12", 0, 0)).grader})
    request = TaskRequestBatch(
        rows=(task_input(lowered_task(task, "pi", max_turns=2 if sample_first else 1)),),
    )
    worker = TaskRolloutWorker(
        worker_config,
        WholeTaskProjection(worker_config, WorkerTokenizer()),
        ScriptedClient(replies),
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
    work = GeneratedWork.from_batch(batch["response_ids"], batch.get("is_last_step"), batch["rollout_metrics"])
    assert work.generated_token_count == 1
    assert work.paid_tokens == 1 + int(sample_first)
    assert work.served_length_stops == work.unprojected_length_stops == int(sample_first)
    assert work.unobserved_calls == work.unobserved_token_limit == 0


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


@pytest.mark.asyncio
@pytest.mark.parametrize("projection", [WholeTaskProjection, StepTaskProjection])
async def test_pi_retry_charges_engine_replies_without_turning_them_into_policy_tokens(
    agent_harnesses, worker_config, projection
):
    first = sampled_reply("Working")
    first["response_ids"] = [[3, 4, 5]]
    first["response_logprobs"] = [[-0.1, -0.1, -0.1]]
    last = sampled_reply("12")
    client = ScriptedClient([first, ModelServerError("transport", "fixture", 503), last])
    attempts = 0

    async def agent(_context, complete):
        nonlocal attempts
        attempts += 1
        await complete(Prompt(({"role": "user", "content": "Return twelve"},), {}))
        if attempts == 1:
            await complete(Prompt(({"role": "user", "content": "Check again"},), {}))

    task = text_task()
    request = TaskRequestBatch(
        rows=(task_input(lowered_task(task, "pi", max_turns=2)),), trajectory_ids=(TrajectoryID(task.id, 0),)
    )
    worker = TaskRolloutWorker(
        worker_config,
        projection(worker_config, WorkerTokenizer()),
        client,
        {},
        agents=agent_harnesses(agent),
        retry_wait=lambda _: asyncio.sleep(0),
        shutdown_timeout=5,
    )
    try:
        (result,) = await worker.generate(request)
        batch = await worker.training_batch(request, [result])
    finally:
        await worker.shutdown()
    assert attempts == 2 and result.failure is None
    assert batch["response_ids"] == last["response_ids"]
    work = GeneratedWork.from_batch(batch["response_ids"], batch.get("is_last_step"), batch["rollout_metrics"])
    assert work.generated_token_count == len(last["response_ids"][0])
    assert work.paid_tokens == len(first["response_ids"][0]) + len(last["response_ids"][0])
    assert work.served_turns == 2
    assert work.abandoned_retry_tokens == 3
    assert work.unobserved_calls == 1
    assert work.unobserved_token_limit == worker_config.sampling_params.max_generate_length


@pytest.mark.asyncio
@pytest.mark.parametrize("projection, payload_lengths", [(WholeTaskProjection, [4]), (StepTaskProjection, [1, 1])])
async def test_agent_tool_observations_are_payload_not_paid_generation(
    agent_harnesses, worker_config, projection, payload_lengths
):
    initial = {"role": "user", "content": "Count the three items"}
    tool = {"role": "tool", "tool_call_id": "count1", "content": "three"}
    a, b = sampled_reply("count"), sampled_reply("3")
    a["assistant_messages"][0]["tool_calls"] = [
        {"id": "count1", "type": "function", "function": {"name": "count", "arguments": "{}"}}
    ]
    b["prompt_ids"] = [[1, 2, 3, 7, 8]]

    async def agent(_context, complete):
        first = await complete(Prompt((initial,), {}))
        await complete(Prompt((initial, first.message, tool), {}))

    task = text_task().model_copy(update={"grader": verifyit_package(NumericSpec("3", 0, 0)).grader})
    req = TaskRequestBatch(
        rows=(task_input(lowered_task(task, "pi", max_turns=2)),), trajectory_ids=(TrajectoryID(task.id, 0),)
    )
    worker = TaskRolloutWorker(
        worker_config,
        projection(worker_config, WorkerTokenizer()),
        ScriptedClient([a, b]),
        {},
        agents=agent_harnesses(agent),
        shutdown_timeout=5,
    )
    try:
        results = await worker.generate(req)
        batch = await worker.training_batch(req, results)
    finally:
        await worker.shutdown()
    assert results[0].failure is None
    assert list(map(len, batch["response_ids"])) == payload_lengths
    assert sum(map(sum, batch["loss_masks"])) == 2
    work = GeneratedWork.from_batch(batch["response_ids"], batch.get("is_last_step"), batch["rollout_metrics"])
    assert work.paid_tokens == len(a["response_ids"][0]) + len(b["response_ids"][0])
    assert work.generated_token_count == sum(payload_lengths)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "prompt_ids,options,expected",
    [
        ((1, 2), {}, 18),
        (None, {"extra_body": {"max_completion_tokens": 7}}, 7),
        (None, {"max_completion_tokens": 5, "extra_body": {"continue_final_message": True}}, 5),
        (None, {}, 20),
    ],
)
async def test_uncertain_model_reply_uses_serving_output_allowance_with_nullable_sampling(
    prompt_ids, options, expected
):
    usage = ModelGenerationUsage()
    error = ModelServerError("transport", "fixture", 503)
    with pytest.raises(ModelServerError):
        await _model_turn(
            ScriptedClient([error]),
            ModelRequest(({"role": "user", "content": "Count to three"},), options, prompt_token_ids=prompt_ids),
            sampling_params={"max_tokens": None},
            max_context_length=20,
            max_prompt_length=15,
            chat_template_kwargs={},
            session_id="lost-response",
            usage=usage,
        )
    assert (usage.tokens, usage.unobserved_calls, usage.unobserved_token_limit) == (0, 1, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chat-restart", "chat-overflow", "chat-tool-cap", "raw-failover", "raw-dies"])
async def test_direct_engine_partial_replies_are_charged_before_discard(mode):
    returned = []
    budgets = []

    class Tokenizer(WorkerTokenizer):
        def encode(self, _text, **_kwargs):
            return [8, 9]

    class Engine:
        async def tokenize(self, _request):
            return {"tokens": [1, 2]}

        async def chat_completion(self, request):
            body = request["json"]
            budgets.append(body["max_completion_tokens"])
            if body.get("continue_final_message"):
                if mode == "chat-overflow":
                    return {"error_category": "context_overflow", "error": {"code": 400, "message": "context full"}}
                return {"error": {"code": 400, "message": "continue_final_message rejected before sampling"}}
            ids = [5, 6] if not returned else [7]
            reason = "abort" if not returned else "stop"
            if mode == "chat-tool-cap":
                ids, reason = [5, 6, 7, 8, 9], "tool_calls"
            returned.append(ids)
            return {
                "prompt_token_ids": [1, 2],
                "choices": [
                    {"token_ids": ids, "finish_reason": reason, "message": {"role": "assistant", "content": "ok"}}
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": len(ids), "total_tokens": 2 + len(ids)},
            }

        async def generate(self, request):
            budgets.append(request["sampling_params"]["max_tokens"])
            if len(budgets) == 2:
                raise ActorDiedError()
            ids = [5, 6] if not returned else [7]
            reason = "abort" if not returned else "stop"
            returned.append(ids)
            return {"responses": ["ok"], "response_ids": [ids], "stop_reasons": [reason]}

    cfg = get_default_config()
    cfg.generator.enable_http_endpoint = False
    engines = [Engine(), Engine()] if mode == "raw-failover" else [Engine()]
    client = DirectModelClient(InferenceEngineClient(engines, Tokenizer(), cfg))
    usage = ModelGenerationUsage()
    request = ModelRequest(
        ({"role": "user", "content": "Name a color"},), {}, prompt_token_ids=(1, 2) if mode.startswith("raw") else None
    )
    call = _model_turn(
        client,
        request,
        sampling_params={"max_tokens": 5},
        max_context_length=16,
        max_prompt_length=None,
        chat_template_kwargs={},
        session_id="meter",
        usage=usage,
    )
    if mode == "chat-tool-cap":
        turn = await call
        assert turn.stop_reason == "length"
        assert usage.length_stops == 1
        assert usage.tokens == len(turn.response_token_ids) == sum(map(len, returned)) == 5
        assert usage.abandoned_retry_tokens == usage.unobserved_calls == 0
        return
    if mode == "chat-overflow":
        with pytest.raises(ContextLengthExceededError):
            await call
    elif mode == "raw-dies":
        with pytest.raises(RuntimeError, match="All inference engines have died"):
            await call
    else:
        assert (await call).response_token_ids == (7,)
    assert usage.tokens == sum(map(len, returned))
    assert usage.abandoned_retry_tokens == 2
    assert budgets == ([5, 3, 5] if mode in ("chat-restart", "raw-failover") else [5, 3])
    assert (usage.unobserved_calls, usage.unobserved_token_limit) == ((1, 3) if mode.startswith("raw") else (0, 0))
