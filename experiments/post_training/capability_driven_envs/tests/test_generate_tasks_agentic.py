# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

# These tests run in the marin-taskforge package environment; environments without it skip them.
pytest.importorskip("taskforge")

import httpx
import openai
from rolloutengine.models import Model, ModelRequest, ModelTurn, TokenUsage
from taskforge.llm.glm_client import GlmEndpoint, Pool

from experiments.post_training.capability_driven_envs.generate_tasks_agentic import RelayModel

OLD = GlmEndpoint("http://old-relay/v1", "token", Pool.BULK)
NEW = GlmEndpoint("http://new-relay/v1", "token", Pool.BULK)
REQUEST = ModelRequest(messages=({"role": "user", "content": "hi"},), options={})


class MovingRelay:
    """A relay at ``OLD`` that moved to ``NEW`` after the first lookup."""

    def __init__(self):
        self.lookups = 0

    def current(self) -> GlmEndpoint:
        self.lookups += 1
        return OLD if self.lookups == 1 else NEW


def connect(endpoint: GlmEndpoint) -> Model:
    async def model(request: ModelRequest) -> ModelTurn:
        if endpoint == OLD:
            raise openai.APIConnectionError(request=httpx.Request("POST", endpoint.base_url))
        message = {"role": "assistant", "content": endpoint.base_url}
        return ModelTurn(message, (), (), None, "stop", usage=TokenUsage(1, 1))

    return model


@pytest.mark.asyncio
async def test_a_relay_model_follows_a_relay_that_moved():
    relay = MovingRelay()
    model = RelayModel(relay, connect)

    first, second = await model(REQUEST), await model(REQUEST)

    assert first.message["content"] == second.message["content"] == NEW.base_url
    # The model looks the relay up again only after the connection error.
    assert relay.lookups == 2
