"""Native rollout harnesses preserve exact sampled work through SkyRL."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finelog.client import FlushResult, LogClient
from finestore.rl.task_health import ATTEMPT_RECEIPTS_DIRECTORY, EVALUATION_RECEIPTS_DIRECTORY, AttemptReceipt
from iris.runtime import telemetry as runtime_telemetry
from omegaconf import OmegaConf, open_dict
from rolloutengine.engine import RolloutEngine
from rolloutengine.finelog_observer import EVENTS_NAMESPACE, MESSAGES_NAMESPACE
from marinskyrl.agent_harnesses import AgentHarnessSettings, TaskAgentHarnesses
from rolloutengine.harness.external import AgentHarness
from rolloutengine.harness.step import StepHarness
from rolloutengine.models import ContextWindowExceeded, Prompt, Transition
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import TASK_SPEC_COLUMN, SourceTaskDataset
from skyrl_train.rollouts.group_grader import GroupGraderSpec
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker, TaskRolloutWorkerSpec, _model_turn
from skyrl_train.rollouts.workers import WorkerShard
from skyrl_train.trajectory_runners.model_clients import (
    ContextLengthExceededError,
    GenerationBudgetExceededError,
)
from skyrl_train.trajectory_runners.types import BatchMetadata, TaskRequestBatch, TokenProvenance
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.grader import verifyit_package
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import SCHEMA_VERSION, UPGRADABLE_SCHEMA_VERSION, AgentMachine, AnswerType, ScriptGrader
from verifyit.spec import NumericSpec
from tests.cpu.task_specs import SHELL_ENVIRONMENT, lowered_task, task_input, text_task


class WorkerTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2]

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
async def test_agent_setup_retries_signal_exit(agent_harnesses, worker_config, exit_code, retries, tmp_path):
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
        task_health_export=str(tmp_path),
        shutdown_timeout=5,
    )
    try:
        (result,) = await worker.generate(request)
    finally:
        await worker.shutdown()
    assert result.failure is not None
    assert result.failure.diagnostics["setup_exit_code"] == exit_code
    assert result.metrics["rollout_retries"] == retries
    receipts = read_receipts(tmp_path / ATTEMPT_RECEIPTS_DIRECTORY)
    assert len(receipts) == len({receipt.rollout_id for receipt in receipts}) == retries + 1
    assert all(receipt.rollout_id for receipt in receipts)
    assert {receipt.setup_exit_code for receipt in receipts} == {exit_code}


def read_receipts(directory):
    return [
        AttemptReceipt.model_validate_json(line)
        for path in directory.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]


@pytest.mark.asyncio
async def test_native_eval_failure_joins_receipt_to_transcript_after_shutdown(tmp_path, monkeypatch):
    class BufferedTable:
        def __init__(self):
            self.pending = []
            self.rows = []

        def write(self, rows):
            self.pending.extend(rows)

        def flush(self, timeout):
            self.rows.extend(self.pending)
            self.pending.clear()
            return FlushResult.SUCCEEDED

    class UnavailableEngine:
        async def tokenize(self, _payload):
            return {"tokens": [1, 2]}

        async def chat_completion(self, _payload):
            raise ConnectionError("The serving endpoint disconnected")

    def resolve(*, run_id, execution_uid):
        return runtime_telemetry.FinelogRuntime(
            "fixture", lambda endpoint: endpoint, {"run_id": run_id, "execution_uid": execution_uid}
        )

    tables = {name: BufferedTable() for name in (EVENTS_NAMESPACE, MESSAGES_NAMESPACE)}
    client = SimpleNamespace(get_table=lambda namespace, *args, **kwargs: tables[namespace], close=lambda: None)
    monkeypatch.setattr(runtime_telemetry, "resolve", resolve)
    monkeypatch.setattr(LogClient, "connect", lambda *args, **kwargs: client)
    config = get_default_config()
    with open_dict(config), open_dict(config.environment.task_sessions):
        config.runtime = {"run_id": "validation-run", "attempt_id": "execution-2"}
        config.environment.task_sessions.max_verifier_workers = 0
    config.trainer.export_path = str(tmp_path)
    task = text_task(prompt="What is the answer?", source="upstream/data")
    stored_task = {**task.model_dump(mode="json"), "schema_version": UPGRADABLE_SCHEMA_VERSION}
    path = tmp_path / "validation.parquet"
    pq.write_table(pa.table({TASK_SPEC_COLUMN: [json.dumps(stored_task)], "source": ["curated-math"]}), path)
    tokenizer = WorkerTokenizer()
    dataset = SourceTaskDataset(
        [str(path)],
        tokenizer,
        128,
        num_workers=1,
        environment_configs=OmegaConf.to_container(config.environment.task_sessions, resolve=True),
    )
    (row,) = dataset.collate_fn(list(dataset))
    loaded = row["input"]
    assert loaded.spec.task.schema_version == SCHEMA_VERSION
    request = TaskRequestBatch(rows=(loaded,), batch_metadata=BatchMetadata(global_step=7, training_phase="eval"))
    worker = TaskRolloutWorkerSpec.from_config(config, [UnavailableEngine()]).build(tokenizer, WorkerShard(0, 1))
    try:
        (result,) = await worker.generate(request)
    finally:
        await worker.shutdown()

    assert result.failure is not None
    (receipt,) = read_receipts(tmp_path / ATTEMPT_RECEIPTS_DIRECTORY / EVALUATION_RECEIPTS_DIRECTORY)
    assert receipt.task_digest == canonical_sha256(stored_task)
    assert receipt.task_key == ("curated-math", "upstream/data", "1", "0")
    assert receipt.exception_type == "ConnectionError"
    assert receipt.rollout_id
    events = tables[EVENTS_NAMESPACE].rows
    assert {event.rollout_id for event in events} == {receipt.rollout_id}
    assert {event.event_type for event in events} >= {"attempt_started", "attempt_ended"}
    (message,) = [
        message
        for message in tables[MESSAGES_NAMESPACE].rows
        if message.role == "user" and json.loads(message.content)["content"] == "What is the answer?"
    ]
    assert (message.rollout_id, message.run_id, message.run_attempt_id) == (
        receipt.rollout_id,
        "validation-run",
        "execution-2",
    )
    assert (message.phase, message.step) == ("eval", 7)


@pytest.mark.asyncio
async def test_cancelled_batch_retains_completed_task_receipts(worker_config, tmp_path):
    second_started = asyncio.Event()

    class Client:
        async def generate(self, request):
            if any(message["content"] == "Wait" for message in request["prompts"][0]):
                second_started.set()
                await asyncio.Future()
            return sampled_reply()

    worker = TaskRolloutWorker(
        worker_config,
        WholeTaskProjection(worker_config, WorkerTokenizer()),
        Client(),
        {},
        task_health_export=str(tmp_path),
        concurrent_tasks=1,
        shutdown_timeout=5,
    )
    request = TaskRequestBatch(
        rows=tuple(
            task_input(lowered_task(text_task(task_id, prompt=prompt)))
            for task_id, prompt in (("complete", "Answer"), ("cancelled", "Wait"))
        )
    )
    pending = asyncio.create_task(worker.generate(request))
    try:
        await asyncio.wait_for(second_started.wait(), 5)
    finally:
        await worker.shutdown()
    with pytest.raises(asyncio.CancelledError):
        await pending
    rows = read_receipts(tmp_path / ATTEMPT_RECEIPTS_DIRECTORY)
    assert len(rows) == len({receipt.task_id for receipt in rows}) == 2
    receipts = {receipt.task_id: receipt for receipt in rows}
    assert receipts["complete"].exception_type is None
    assert receipts["complete"].task_key == (None, "fixture", "1", "0")
    assert receipts["cancelled"].exception_type == "CancelledError"
    assert receipts["cancelled"].outcome == Outcome.INFRA_ERROR


@pytest.mark.asyncio
async def test_task_receipt_reflects_the_final_group_grade(worker_config, tmp_path):
    def grade(_task, _specification, rollouts, _eligible, _phase):
        return [replace(rollout, grade=GradeResult(Outcome.GRADED, 0.75)) for rollout in rollouts]

    worker = TaskRolloutWorker(
        worker_config,
        WholeTaskProjection(worker_config, WorkerTokenizer()),
        ScriptedClient([sampled_reply()]),
        {},
        task_health_export=str(tmp_path),
        group_graders={"fixture": grade},
        shutdown_timeout=5,
    )
    task = task_input(lowered_task(text_task()), group_grader=GroupGraderSpec(name="fixture", parameters_json="{}"))
    try:
        await worker.generate(TaskRequestBatch(rows=(task,)))
    finally:
        await worker.shutdown()
    (receipt,) = read_receipts(tmp_path / ATTEMPT_RECEIPTS_DIRECTORY)
    assert receipt.outcome == Outcome.GRADED
