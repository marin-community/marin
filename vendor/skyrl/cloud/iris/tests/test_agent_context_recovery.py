"""Context rejection must reach the installed agent without disabling its endpoint."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.step import StepHarness
from rolloutengine.models import ContextWindowExceeded, ModelTurn, Prompt, Transition
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.harbor_trial_session import HarborTrialHarness
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.agent_backend import TRIAL_HEADER, AgentGenerationStopped, AgentInferenceBackend
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state
from skyrl_train.rollouts.task_worker import TaskRolloutWorker, _model_turn
from skyrl_train.trajectory_runners.model_clients import (
    ContextLengthExceededError,
    GenerationBudgetExceededError,
    ModelServerError,
)
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TokenProvenance, TrajectoryID
from skyrl_gym.verification import VerificationStatus
from taskcompendium.grading_result import Outcome
from taskcompendium.models import NoGrader
from taskcompendium.grader import verifyit_package
from verifyit.spec import NumericSpec
from tests.cpu.task_specs import lowered_task, task_input, text_task


class Decoder:
    def decode(self, token_ids):
        return "12"


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


class HarborTrials:
    def __init__(self, trial_error=None):
        self.trial_error = trial_error

    def original_harness(self, *args):
        async def trial(complete):
            try:
                await complete(Prompt(({"role": "user", "content": "Start"},), {"max_tokens": 8}))
                await complete(Prompt(({"role": "user", "content": "Longer history"},), {"max_tokens": 1}))
            except (ContextWindowExceeded, AgentGenerationStopped):
                pass
            return SimpleNamespace(
                trial_uri="fixture/trial",
                exception_info=None if self.trial_error is None else SimpleNamespace(exception_type=self.trial_error),
                verifier_result=SimpleNamespace(rewards={"reward": 0.75}),
            )

        return HarborTrialHarness(trial, "reward")

    async def shutdown(self):
        pass


@pytest.fixture
def open_worker(worker_config):
    @asynccontextmanager
    async def create(installed_agents, replies, *, harbor):
        worker = TaskRolloutWorker(
            worker_config,
            WholeTaskProjection(worker_config, WorkerTokenizer()),
            ScriptedClient(replies),
            {},
            harbor=HarborTaskSettings.from_config(OmegaConf.create({"harbor": harbor})),
            installed_agents=installed_agents,
            shutdown_timeout=5,
        )
        try:
            yield worker
        finally:
            await worker.shutdown()

    return create


@pytest.mark.asyncio
async def test_agent_can_retry_after_receiving_a_typed_context_error():
    calls = 0

    async def complete(prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ContextWindowExceeded("Maximum context length exceeded; compact and retry")
        return ModelTurn({"role": "assistant", "content": "12"}, (10, 11), (20,), None, "stop")

    async def tokenize(payload):
        raise AssertionError("Generation must route through the registered session")

    backend = AgentInferenceBackend("fixture", 65536, tokenize, Decoder())
    request = {"headers": {TRIAL_HEADER: "trial"}, "json": {"messages": [{"role": "user", "content": "Return 12"}]}}
    with backend.session("trial", complete):
        rejection = await backend.chat_completion(request)
        assert rejection["error"]["type"] == "context_length_exceeded"
        assert rejection["error"]["code"] == 400
        response = await backend.chat_completion(request)
    assert response["choices"][0]["message"]["content"] == "12"


def test_streaming_agent_context_error_is_http_400_before_sse_starts():
    calls = 0

    async def complete(_prompt):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ContextWindowExceeded("Maximum context length exceeded")
        return ModelTurn({"role": "assistant", "content": "12"}, (10, 11), (20,), None, "stop")

    async def tokenize(_payload):
        raise AssertionError("Generation must route through the registered session")

    backend = AgentInferenceBackend("fixture", 65536, tokenize, Decoder())
    payload = {"model": "fixture", "messages": [{"role": "user", "content": "Return 12"}], "stream": True}
    set_global_state(backend, None)
    try:
        with backend.session("trial", complete), TestClient(create_app()) as client:
            rejection = client.post("/v1/chat/completions", json=payload, headers={TRIAL_HEADER: "trial"})
            assert rejection.status_code == 400
            assert rejection.json()["error"]["type"] == "context_length_exceeded"
            response = client.post("/v1/chat/completions", json=payload, headers={TRIAL_HEADER: "trial"})
            assert response.status_code == 200
            chunks = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
            assert chunks[0]["choices"][0]["delta"]["content"] == "12"
    finally:
        set_global_state(None, None)


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
@pytest.mark.parametrize(
    "reject_call,trial_error",
    [(2, None), (1, None), (2, "ContextLengthExceededError")],
    ids=["sampled-turn", "first-request", "agent-error-policy"],
)
async def test_original_harbor_trial_keeps_sampled_tokens_and_grade_after_context_rejection(
    open_worker, reject_call, trial_error
):
    """A reduced output budget starts a new prompt; rejecting it must retain the prior turn."""

    task = text_task().model_copy(update={"tags": ("harbor",), "grader": NoGrader(reason="Harbor grades the trial")})
    request = TaskRequestBatch(
        rows=(task_input(lowered_task(task, "harbor_trial", max_turns=8)),),
        trajectory_ids=(TrajectoryID("harbor", 0),),
    )
    replies = [sampled_reply()] * (reject_call - 1) + [GenerationBudgetExceededError([1, 2, 3, 4])]
    async with open_worker(
        HarborTrials(trial_error), replies, harbor={"max_retries": 0, "enable_error_classification": True}
    ) as worker:
        (result,) = await worker.generate(request)
        batch = await worker.training_batch(request, [result])
    if reject_call == 1:
        assert result.stop_reason == "length"
        assert result.failure is None
        assert result.response_token_ids == ()
        return
    assert result.grade.status == Outcome.GRADED
    assert result.grade.reward == 0.75
    assert batch["response_ids"] == [[3]]
    assert batch["loss_masks"] == [[1]]
    assert batch["rollout_logprobs"][0] == pytest.approx([-0.1])
    assert batch["verification_results"][0].status == VerificationStatus.VERIFIED
    assert np.sum(batch["rewards"]) == (0.75 if trial_error is None else 0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_first", [False, True], ids=["first-budget", "later-serving"])
async def test_native_worker_recovers_without_spending_a_turn(open_worker, sample_first):
    rejection = (
        ContextLengthExceededError("context_overflow", "fixture", 400)
        if sample_first
        else GenerationBudgetExceededError([1, 2, 3, 4])
    )
    replies = ([sampled_reply()] if sample_first else []) + [rejection, sampled_reply("12")]

    class InstalledAgents:
        async def run(self, _machine, complete, **kwargs):
            if sample_first:
                await complete(Prompt(({"role": "user", "content": "Start"},), {}))
            with pytest.raises(ContextWindowExceeded):
                await complete(Prompt(({"role": "user", "content": "Oversized"},), {}))
            await complete(Prompt(({"role": "user", "content": "Compacted: return 12"},), {}))

        async def shutdown(self):
            pass

    task = text_task().model_copy(
        update={"tags": ("harbor",), "grader": verifyit_package(NumericSpec("12", 0, 0)).grader}
    )
    request = TaskRequestBatch(
        rows=(task_input(lowered_task(task, max_turns=2 if sample_first else 1)),),
    )
    async with open_worker(InstalledAgents(), replies, harbor={"max_retries": 0}) as worker:
        (result,) = await worker.generate(request)
        batch = await worker.training_batch(request, [result])
    assert result.failure is None
    assert result.grade.status == Outcome.GRADED
    assert batch["response_ids"] == [[3]]
    assert batch["loss_masks"] == [[1]]
    assert batch["rollout_logprobs"][0] == pytest.approx([-0.1])
    assert np.sum(batch["rewards"]) == 1.0


@pytest.mark.asyncio
async def test_harbor_worker_keeps_the_original_first_request_serving_error(open_worker):
    task = text_task().model_copy(update={"tags": ("harbor",)})
    request = TaskRequestBatch(rows=(task_input(lowered_task(task, "harbor_trial")),))
    async with open_worker(
        HarborTrials(), [ContextLengthExceededError("context_overflow", "fixture", 400)], harbor={"max_retries": 0}
    ) as worker:
        (result,) = await worker.generate(request)
    assert result.response_token_ids == ()
    assert result.stop_reason == "error"
    assert result.failure.exception_type == "ContextLengthExceededError"
    assert result.failure.diagnostics["request_id"] == "fixture"


@pytest.mark.asyncio
async def test_harbor_worker_resets_recovery_state_before_retrying(open_worker):
    task = text_task().model_copy(update={"tags": ("harbor",)})
    request = TaskRequestBatch(rows=(task_input(lowered_task(task, "harbor_trial")),))
    replies = [
        sampled_reply(),
        ModelServerError("server_error", "fixture", 503),
        GenerationBudgetExceededError([1, 2, 3, 4]),
    ]
    async with open_worker(
        HarborTrials(), replies, harbor={"max_retries": 1, "min_wait_sec": 0, "max_wait_sec": 0}
    ) as worker:
        (result,) = await worker.generate(request)
    assert result.response_token_ids == ()
    assert result.stop_reason == "length"
    assert result.failure is None
    assert result.metrics["rollout_retries"] == 1
