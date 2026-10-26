from pathlib import Path

import pytest
from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.agent.name import AgentName
from harbor.models.trial.config import AgentConfig


class CaptureAgent(BaseAgent):
    captured_kwargs: dict = {}
    captured_model_name: str | None = None

    def __init__(self, logs_dir: Path, model_name: str | None = None, **kwargs):
        type(self).captured_kwargs = kwargs
        type(self).captured_model_name = model_name
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)

    @staticmethod
    def name() -> str:
        return "capture-agent"

    def version(self) -> str:
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        pass


def test_create_agent_from_config_passes_agent_kwargs(monkeypatch, tmp_path):
    monkeypatch.setitem(AgentFactory._AGENT_MAP, AgentName.PI, CaptureAgent)

    AgentFactory.create_agent_from_config(
        AgentConfig(
            name=AgentName.PI.value,
            model_name="fake/model",
            kwargs={"session_id": "test-session-12345"},
        ),
        logs_dir=tmp_path,
    )

    assert CaptureAgent.captured_kwargs["session_id"] == "test-session-12345"


@pytest.mark.parametrize(
    "config",
    [
        AgentConfig(name="acp:opencode@1.3.9"),
        AgentConfig(name="claude-code"),
        AgentConfig(name="oracle"),
        AgentConfig(name="nop"),
        AgentConfig(name="claude-code", mode="local"),
        AgentConfig(import_path="harbor.agents.installed.claude_code:ClaudeCode"),
        AgentConfig(import_path="shellbox.agent:BashAgent"),
    ],
)
def test_unsupported_agents_are_rejected_before_initialization(tmp_path, config):
    with pytest.raises(ValueError, match="not supported by Marin"):
        AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)


def test_create_agent_from_config_routes_alias_and_records_model_name(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(AgentFactory._AGENT_MAP, AgentName.PI, CaptureAgent)

    agent = AgentFactory.create_agent_from_config(
        AgentConfig(
            name=AgentName.PI.value,
            model_name="hosted_vllm/org/canonical-model",
            model_alias="hosted_vllm/resumed-endpoint",
        ),
        logs_dir=tmp_path,
    )

    assert CaptureAgent.captured_model_name == "hosted_vllm/org/canonical-model"
    assert agent.model_name == "hosted_vllm/org/canonical-model"
    assert agent.model_alias == "hosted_vllm/resumed-endpoint"
    assert agent.to_agent_info().model_info is not None
    assert agent.to_agent_info().model_info.name == "org/canonical-model"
