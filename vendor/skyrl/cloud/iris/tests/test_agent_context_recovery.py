"""Context rejection must reach the installed agent without disabling its endpoint."""

import json

import pytest
from fastapi.testclient import TestClient
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.step import StepHarness
from rolloutengine.models import ContextWindowExceeded, ModelTurn, Prompt, Transition
from skyrl_train.rollouts.agent_backend import TRIAL_HEADER, AgentInferenceBackend
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state
from skyrl_train.rollouts.task_worker import _model_turn
from skyrl_train.trajectory_runners.model_clients import GenerationBudgetExceededError
from skyrl_train.trajectory_runners.types import TokenProvenance
from tests.cpu.task_specs import lowered_task, text_task


class Decoder:
    def decode(self, token_ids):
        return "12"


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

    class Client:
        calls = 0

        async def generate(self, _request):
            self.calls += 1
            if self.calls == 2:
                raise GenerationBudgetExceededError([1, 2, 3, 4])
            return {
                "token_provenance": TokenProvenance.ENGINE,
                "assistant_messages": [{"role": "assistant", "content": "Working"}],
                "prompt_ids": [[1, 2]],
                "response_ids": [[3]],
                "stop_reasons": ["stop"],
            }

    client = Client()

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
