# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from contextlib import nullcontext

import httpx
import pytest
from rolloutengine.contracts import ModelTurn

from experiments.post_training.russell_rsi.token_preflight import run_token_preflight


@pytest.mark.parametrize("uses_tool", [False, True])
def test_preflight_requires_real_shell_execution_even_when_the_answer_passes(tmp_path, uses_tool):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"tokens": [1]}))
        ) as client:
            calls = 0

            async def turn(request):
                nonlocal calls
                await client.post("https://model.test/v1/completions", json={"prompt": list(request.prefix_token_ids)})
                calls += 1
                if calls == 1 and uses_tool:
                    assert "48213" not in request.messages[0]["content"]
                    message = {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "shell-1",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": '{"command":"cat /workspace/preflight-value.txt"}',
                                },
                            }
                        ],
                    }
                    return ModelTurn(message, (1,), (2,), None, "stop")
                if uses_tool:
                    observations = [message for message in request.messages if message.get("role") == "tool"]
                    assert "48213" in observations[0]["content"]
                prompt = (1, 2, 3) if uses_tool else (1,)
                return ModelTurn({"role": "assistant", "content": "48213"}, prompt, (4,), None, "stop")

            await run_token_preflight(turn, client, str(tmp_path))

    with nullcontext() if uses_tool else pytest.raises(ValueError, match="two-turn shell tool exchange"):
        asyncio.run(run())
    evidence = json.loads((tmp_path / "token-preflight.json").read_text())
    assert evidence["status"] == ("passed" if uses_tool else "failed")
    assert len(evidence["requests"]) == (2 if uses_tool else 1)
    assert len(evidence["turns"]) == (2 if uses_tool else 1)
    assert evidence["rollout"]["grade"]["reward"] == 1
