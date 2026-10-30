import json
import shlex
import tomllib
from unittest.mock import AsyncMock

import pytest

from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex
from harbor.models.agent.context import AgentContext


@pytest.mark.parametrize("agent_type", [ClaudeCode, Codex])
def test_native_agent_captured_tokens_survive_missing_cli_session(tmp_path, agent_type):
    agent = agent_type(
        logs_dir=tmp_path,
        model_name="hosted_vllm/bfcl-test",
        api_base="http://literal/v1",
        model_info={"max_input_tokens": 32768, "max_output_tokens": 8192},
        collect_rollout_details=True,
    )
    (tmp_path / "literal.jsonl").write_text(
        json.dumps(
            {
                "literal": {
                    "prompt_token_ids": [7, 8],
                    "completion_token_ids": [9],
                    "logprobs": [-0.25],
                }
            }
        )
        + "\n"
    )
    context = AgentContext()
    agent.populate_context_post_run(context)
    assert context.rollout_details == [
        {
            "prompt_token_ids": [[7, 8]],
            "completion_token_ids": [[9]],
            "logprobs": [[-0.25]],
        }
    ]
    assert context.metadata is not None
    assert context.metadata["rollout_correlation_id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_type", [ClaudeCode, Codex])
async def test_native_agent_cli_wire_config_matches_capture_identity(
    tmp_path, monkeypatch, agent_type
):
    monkeypatch.delenv("CLAUDE_CODE_USE_BEDROCK", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    agent = agent_type(
        logs_dir=tmp_path,
        model_name="hosted_vllm/bfcl-test",
        api_base="http://literal/v1",
        model_info={"max_input_tokens": 32768, "max_output_tokens": 8192},
        collect_rollout_details=True,
    )
    environment = AsyncMock()
    environment.default_user = "agent"
    environment.agent_timeout_sec = None
    environment.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
    context = AgentContext()
    await agent.run("Print hello", environment, context)
    agent.populate_context_post_run(context)
    calls = environment.exec.call_args_list
    assert context.metadata is not None
    identity = context.metadata["rollout_correlation_id"]
    if agent_type is ClaudeCode:
        env = calls[-1].kwargs["env"]
        assert env["ANTHROPIC_BASE_URL"] == "http://literal"
        assert env["ANTHROPIC_MODEL"] == "bfcl-test"
        assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "bfcl-test"
        assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "bfcl-test"
        assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "bfcl-test"
        assert env["ANTHROPIC_CUSTOM_HEADERS"] == f"x-ot-trial-id: {identity}"
        assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "8192"
    else:
        setup = next(call for call in calls if "printf %s" in call.kwargs["command"])
        words = shlex.split(setup.kwargs["command"])
        config = tomllib.loads(words[words.index("printf") + 2])
        assert config["model_context_window"] == 32768
        provider = config["model_providers"][config["model_provider"]]
        assert provider["base_url"] == "http://literal/v1"
        assert provider["wire_api"] == "responses"
        assert provider["supports_websockets"] is False
        assert provider["http_headers"]["x-ot-trial-id"] == identity
        assert setup.kwargs["env"]["OPENAI_API_KEY"] == "EMPTY"
