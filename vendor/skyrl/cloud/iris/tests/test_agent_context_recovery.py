"""Context rejection must reach the installed agent without disabling its endpoint."""

import pytest
from rolloutengine.models import ContextWindowExceeded, ModelTurn
from skyrl_train.rollouts.agent_backend import TRIAL_HEADER, AgentInferenceBackend


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
