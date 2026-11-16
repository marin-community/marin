import logging

import pytest

from harbor.agents.terminus_2.terminus_2 import EpisodeLoggingPaths
from harbor.agents.terminus_kira import TerminusKira
from harbor.llms.base import LLMResponse


def _bare_agent() -> TerminusKira:
    agent = object.__new__(TerminusKira)
    agent.logger = logging.getLogger("test-terminus-kira")
    return agent


@pytest.mark.asyncio
async def test_native_tool_response_becomes_shared_loop_interaction(monkeypatch):
    agent = _bare_agent()
    response = LLMResponse(
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "execute_commands",
                    "arguments": (
                        '{"analysis":"inspect","plan":"list files",'
                        '"commands":[{"keystrokes":"ls\\n","duration":90}]}'
                    ),
                },
            }
        ],
    )

    async def query(*args, **kwargs):
        return response

    monkeypatch.setattr(agent, "_query_llm", query)
    result = await agent._handle_llm_interaction(
        chat=object(),
        prompt="task",
        logging_paths=EpisodeLoggingPaths(None, None, None),
    )

    assert [
        (command.keystrokes, command.duration_sec) for command in result.commands
    ] == [("ls\n", 60)]
    assert result.analysis == "inspect"
    assert result.plan == "list files"
    assert result.native_tool_calls == response.tool_calls


@pytest.mark.asyncio
async def test_invalid_native_tool_arguments_request_a_retry(monkeypatch):
    agent = _bare_agent()

    async def query(*args, **kwargs):
        return LLMResponse(
            content="",
            tool_calls=[
                {
                    "id": "call-bad",
                    "type": "function",
                    "function": {"name": "execute_commands", "arguments": "{"},
                }
            ],
        )

    monkeypatch.setattr(agent, "_query_llm", query)
    result = await agent._handle_llm_interaction(
        chat=object(),
        prompt="task",
        logging_paths=EpisodeLoggingPaths(None, None, None),
    )

    assert result.commands == []
    assert result.feedback.startswith("ERROR: Invalid arguments for execute_commands:")
