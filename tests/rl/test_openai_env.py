# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests ChatEchoEnv with openai_responses mock."""

import asyncio
from collections import deque

import pytest

try:
    import openai_responses
except ImportError:  # pragma: no cover
    openai = None  # type: ignore
    openai_responses = None  # type: ignore


from marin.rl.envs.openai_echo import ChatEchoEnv
from marin.rl.types import InferenceEndpoint, RolloutGroup


@pytest.mark.skipif(openai_responses is None, reason="openai_responses not installed")
@openai_responses.mock()  # type: ignore[arg-type]
def test_chat_echo_env(openai_mock):  # type: ignore[valid-type]
    # Prepare mock response
    openai_mock.chat.completions.create.response = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": "Hello! How can I help?", "role": "assistant"},
            }
        ]
    }

    # Collect rollouts emitted by the env
    collected: deque[RolloutGroup] = deque()

    def sink(groups: list[RolloutGroup]):
        collected.extend(groups)

    env = ChatEchoEnv(
        inference=InferenceEndpoint("https://api.openai.com/v1"),
        rollout_sink=sink,  # type: ignore[arg-type]
        prompt="Hello!",
        max_iters=1,  # run exactly once then stop
        api_key="sk-fake",
    )

    asyncio.run(env.run())

    # Ensure sink received one rollout group with expected content
    assert len(collected) == 1
    rg = collected.pop()
    assert rg.rollouts[0].turns[0].message == "Hello! How can I help?"
    # Mock should have been hit exactly once
    assert openai_mock.chat.completions.create.route.call_count == 1
