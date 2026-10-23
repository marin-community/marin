from unittest.mock import AsyncMock

import pytest
from harbor.agents.installed.aider import Aider
from harbor.agents.installed.openhands import OpenHands
from harbor.agents.installed.openhands_sdk import OpenHandsSDK
from harbor.agents.installed.swe_agent import SweAgent


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_class", [Aider, SweAgent, OpenHands, OpenHandsSDK])
async def test_litellm_harness_install_fails_before_sandbox_changes(
    tmp_path, agent_class
):
    agent = agent_class(logs_dir=tmp_path, model_name="openai/test-model")
    environment = AsyncMock()
    with pytest.raises(RuntimeError, match="blocked dependency LiteLLM"):
        await agent.install(environment)
    environment.exec.assert_not_awaited()
