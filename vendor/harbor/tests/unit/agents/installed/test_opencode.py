"""Unit tests for OpenCode agent ATIF trajectory mapping."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harbor.agents.installed.base import (
    AgentKilledBySignalError,
    ModelAuthenticationError,
    NonZeroAgentExitCodeError,
)
from harbor.agents.installed.opencode import OpenCode
from harbor.llms.base import ContextLengthExceededError
from harbor.models.agent.context import AgentContext
from harbor.trial.errors import AgentTimeoutError
from harbor.utils.artifact_writer import get_artifact_writer


def _write_events(logs_dir, events):
    """Write a list of event dicts as JSON lines to opencode.txt."""
    lines = [json.dumps(e) for e in events]
    (logs_dir / "opencode.txt").write_text("\n".join(lines) + "\n")


def _make_step_start(session_id, message_id, timestamp=1700000000000, snapshot="abc"):
    return {
        "type": "step_start",
        "timestamp": timestamp,
        "sessionID": session_id,
        "part": {
            "id": f"prt_start_{message_id}",
            "sessionID": session_id,
            "messageID": message_id,
            "type": "step-start",
            "snapshot": snapshot,
        },
    }


def _make_text(session_id, message_id, text, timestamp=1700000001000):
    return {
        "type": "text",
        "timestamp": timestamp,
        "sessionID": session_id,
        "part": {
            "id": f"prt_text_{message_id}",
            "sessionID": session_id,
            "messageID": message_id,
            "type": "text",
            "text": text,
            "time": {"start": timestamp, "end": timestamp},
        },
    }


def _make_tool_use(
    session_id,
    message_id,
    tool_name,
    tool_input,
    tool_output,
    call_id="call_1",
    timestamp=1700000002000,
):
    return {
        "type": "tool_use",
        "timestamp": timestamp,
        "sessionID": session_id,
        "part": {
            "id": f"prt_tool_{message_id}",
            "sessionID": session_id,
            "messageID": message_id,
            "type": "tool",
            "callID": call_id,
            "tool": tool_name,
            "state": {
                "status": "completed",
                "input": tool_input,
                "output": tool_output,
            },
        },
    }


def _make_reasoning(session_id, message_id, text, timestamp=1700000001500):
    return {
        "type": "reasoning",
        "timestamp": timestamp,
        "sessionID": session_id,
        "part": {
            "id": f"prt_reasoning_{message_id}",
            "sessionID": session_id,
            "messageID": message_id,
            "type": "reasoning",
            "text": text,
            "time": {"start": timestamp, "end": timestamp},
        },
    }


def _make_step_finish(
    session_id,
    message_id,
    cost=0.01,
    input_tok=100,
    output_tok=50,
    reasoning_tok=0,
    cache_read=0,
    cache_write=0,
    timestamp=1700000003000,
):
    return {
        "type": "step_finish",
        "timestamp": timestamp,
        "sessionID": session_id,
        "part": {
            "id": f"prt_finish_{message_id}",
            "sessionID": session_id,
            "messageID": message_id,
            "type": "step-finish",
            "reason": "stop",
            "cost": cost,
            "tokens": {
                "total": input_tok + output_tok + cache_read + cache_write,
                "input": input_tok,
                "output": output_tok,
                "reasoning": reasoning_tok,
                "cache": {"read": cache_read, "write": cache_write},
            },
        },
    }


def _make_user(session_id, text, timestamp=1699999999000):
    return {
        "type": "user",
        "timestamp": timestamp,
        "sessionID": session_id,
        "parts": [{"type": "text", "text": text}],
    }


class TestOpenCodeSupportsAtif:
    def test_supports_atif_flag(self):
        assert OpenCode.SUPPORTS_ATIF is True


class TestOpenCodeMillisToIso:
    def test_converts_millis_timestamp(self):
        result = OpenCode._millis_to_iso(1700000000000)
        assert result is not None
        assert "2023-11-14" in result

    def test_returns_none_for_none(self):
        assert OpenCode._millis_to_iso(None) is None

    def test_handles_invalid_timestamp(self):
        assert OpenCode._millis_to_iso(float("inf")) is None


class TestOpenCodeParseStdout:
    def test_parses_json_lines(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        events = [
            _make_step_start("s1", "m1"),
            _make_text("s1", "m1", "Hello"),
            _make_step_finish("s1", "m1"),
        ]
        _write_events(temp_dir, events)

        parsed = agent._parse_stdout()
        assert len(parsed) == 3
        assert parsed[0]["type"] == "step_start"
        assert parsed[1]["type"] == "text"
        assert parsed[2]["type"] == "step_finish"

    def test_returns_empty_when_no_file(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        assert agent._parse_stdout() == []

    def test_skips_non_json_lines(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        content = 'not json\n{"type":"step_start","sessionID":"s1"}\nalso not json\n'
        (temp_dir / "opencode.txt").write_text(content)

        parsed = agent._parse_stdout()
        assert len(parsed) == 1


class TestOpenCodeConvertEvents:
    def test_text_only_turn(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        events = [
            _make_step_start("s1", "m1"),
            _make_text("s1", "m1", "Hello, I will help you."),
            _make_step_finish("s1", "m1", cost=0.015, input_tok=100, output_tok=50),
        ]

        trajectory = agent._convert_events_to_trajectory(events)

        assert trajectory is not None
        assert trajectory.schema_version == "ATIF-v1.6"
        assert trajectory.session_id == "s1"
        assert trajectory.agent.name == "opencode"
        assert len(trajectory.steps) == 1
        assert trajectory.steps[0].source == "agent"
        assert trajectory.steps[0].message == "Hello, I will help you."
        assert trajectory.steps[0].metrics.prompt_tokens == 100
        assert trajectory.steps[0].metrics.completion_tokens == 50
        assert trajectory.steps[0].metrics.cost_usd == 0.015

    def test_tool_call_turn(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        events = [
            _make_step_start("s1", "m1"),
            _make_text("s1", "m1", "Let me create that file."),
            _make_tool_use(
                "s1",
                "m1",
                "write",
                {"filePath": "/app/hello.txt", "content": "Hello!"},
                "Wrote file successfully.",
                call_id="toolu_abc",
            ),
            _make_step_finish("s1", "m1", cost=0.02, input_tok=200, output_tok=80),
        ]

        trajectory = agent._convert_events_to_trajectory(events)

        assert trajectory is not None
        assert len(trajectory.steps) == 1
        step = trajectory.steps[0]
        assert step.message == "Let me create that file."
        assert len(step.tool_calls) == 1
        assert step.tool_calls[0].function_name == "write"
        assert step.tool_calls[0].tool_call_id == "toolu_abc"
        assert step.tool_calls[0].arguments == {
            "filePath": "/app/hello.txt",
            "content": "Hello!",
        }
        assert step.observation is not None
        assert step.observation.results[0].content == "Wrote file successfully."

    def test_reasoning_content_is_captured(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        events = [
            _make_step_start("s1", "m1"),
            _make_reasoning("s1", "m1", "I should inspect README first."),
            _make_text("s1", "m1", "I'll read the README and summarize it."),
            _make_step_finish("s1", "m1", cost=0.01, input_tok=120, output_tok=40),
        ]

        trajectory = agent._convert_events_to_trajectory(events)

        assert trajectory is not None
        step = trajectory.steps[0]
        assert step.reasoning_content == "I should inspect README first."
        assert step.message == "I'll read the README and summarize it."

    def test_multiple_reasoning_events_are_joined(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        events = [
            _make_step_start("s1", "m1"),
            _make_reasoning("s1", "m1", "First thought."),
            _make_reasoning("s1", "m1", "Second thought."),
            _make_tool_use(
                "s1",
                "m1",
                "glob",
                {"pattern": "README*"},
                "/app/README.md",
            ),
            _make_step_finish("s1", "m1", cost=0.01, input_tok=10, output_tok=5),
        ]

        trajectory = agent._convert_events_to_trajectory(events)

        assert trajectory is not None
        step = trajectory.steps[0]
        assert step.reasoning_content == "First thought.\n\nSecond thought."
        assert step.message == "(tool use)"
        assert step.tool_calls is not None

    def test_empty_reasoning_is_ignored(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        events = [
            _make_step_start("s1", "m1"),
            _make_reasoning("s1", "m1", ""),
            _make_tool_use("s1", "m1", "pwd", {}, "/app"),
            _make_step_finish("s1", "m1", cost=0.001, input_tok=1, output_tok=1),
        ]

        trajectory = agent._convert_events_to_trajectory(events)

        assert trajectory is not None
        step = trajectory.steps[0]
        assert step.reasoning_content is None
        assert step.message == "(tool use)"

    def test_multiple_turns(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        events = [
            _make_step_start("s1", "m1", timestamp=1700000000000),
            _make_text("s1", "m1", "I'll write the file."),
            _make_tool_use(
                "s1",
                "m1",
                "write",
                {"filePath": "test.txt", "content": "hi"},
                "Done.",
            ),
            _make_step_finish("s1", "m1", cost=0.01, input_tok=100, output_tok=50),
            _make_step_start("s1", "m2", timestamp=1700000004000),
            _make_text("s1", "m2", "File created successfully."),
            _make_step_finish(
                "s1",
                "m2",
                cost=0.005,
                input_tok=150,
                output_tok=30,
                cache_read=100,
            ),
        ]

        trajectory = agent._convert_events_to_trajectory(events)

        assert trajectory is not None
        assert len(trajectory.steps) == 2
        assert trajectory.steps[0].tool_calls is not None
        assert trajectory.steps[1].tool_calls is None
        assert trajectory.steps[1].message == "File created successfully."

        # Check final metrics are aggregated
        fm = trajectory.final_metrics
        assert fm.total_cost_usd == 0.015
        assert fm.total_completion_tokens == 80  # 50 + 30
        assert fm.total_cached_tokens == 100
        assert fm.total_steps == 2

    def test_cache_tokens_in_metrics(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        events = [
            _make_step_start("s1", "m1"),
            _make_text("s1", "m1", "Thinking..."),
            _make_step_finish(
                "s1",
                "m1",
                cost=0.001,
                input_tok=5,
                output_tok=40,
                cache_read=500,
                cache_write=200,
            ),
        ]

        trajectory = agent._convert_events_to_trajectory(events)

        assert trajectory is not None
        step = trajectory.steps[0]
        # prompt_tokens = input + cache_read
        assert step.metrics.prompt_tokens == 505
        assert step.metrics.cached_tokens == 500
        assert step.metrics.extra == {"cache_write_tokens": 200}

    def test_empty_events(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        assert agent._convert_events_to_trajectory([]) is None

    def test_no_step_finish_events_only(self, temp_dir):
        """Events without step_start/step_finish produce no steps."""
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        events = [{"type": "error", "sessionID": "s1", "error": {"name": "Err"}}]
        assert agent._convert_events_to_trajectory(events) is None

    def test_session_id_extracted(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        events = [
            _make_step_start("ses_abc123", "m1"),
            _make_text("ses_abc123", "m1", "hi"),
            _make_step_finish("ses_abc123", "m1"),
        ]

        trajectory = agent._convert_events_to_trajectory(events)
        assert trajectory.session_id == "ses_abc123"


class TestOpenCodePopulateContextPostRun:
    @pytest.mark.asyncio
    async def test_run_persists_instruction_as_initial_user_step(self, temp_dir):
        instruction = "Create a hello-world file."
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        mock_env = AsyncMock()

        async def exec_side_effect(**kwargs):
            if "tee /logs/agent/opencode.txt" in kwargs["command"]:
                _write_events(
                    temp_dir,
                    [
                        _make_step_start("s1", "m1"),
                        _make_text("s1", "m1", "Done!"),
                        _make_step_finish("s1", "m1"),
                    ],
                )
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        mock_env.exec.side_effect = exec_side_effect

        context = AgentContext()
        await agent.run(instruction, mock_env, context)
        agent.populate_context_post_run(context)
        get_artifact_writer().flush(timeout=30)

        data = json.loads((temp_dir / "trajectory.json").read_text())
        assert [(step["source"], step["message"]) for step in data["steps"]] == [
            ("user", instruction),
            ("agent", "Done!"),
        ]

    def test_persists_user_event_as_initial_user_step(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        agent._instruction = "Ignore this fallback instruction."
        _write_events(
            temp_dir,
            [
                _make_user("s1", "Use the prompt emitted by OpenCode."),
                _make_step_start("s1", "m1"),
                _make_text("s1", "m1", "Done!"),
                _make_step_finish("s1", "m1"),
            ],
        )

        agent.populate_context_post_run(AgentContext())
        get_artifact_writer().flush(timeout=30)

        data = json.loads((temp_dir / "trajectory.json").read_text())
        assert [(step["source"], step["message"]) for step in data["steps"]] == [
            ("user", "Use the prompt emitted by OpenCode."),
            ("agent", "Done!"),
        ]

    def test_populates_context_and_writes_trajectory(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        events = [
            _make_step_start("s1", "m1"),
            _make_text("s1", "m1", "Done!"),
            _make_step_finish(
                "s1",
                "m1",
                cost=0.05,
                input_tok=1000,
                output_tok=500,
                cache_read=200,
            ),
        ]
        _write_events(temp_dir, events)

        context = AgentContext()
        agent.populate_context_post_run(context)
        get_artifact_writer().flush(timeout=30)

        # Check trajectory file was written
        trajectory_path = temp_dir / "trajectory.json"
        assert trajectory_path.exists()
        data = json.loads(trajectory_path.read_text())
        assert data["schema_version"] == "ATIF-v1.6"
        assert data["session_id"] == "s1"
        assert len(data["steps"]) == 1

        # Check context was populated
        assert context.cost_usd == 0.05
        assert context.n_input_tokens == 1200  # 1000 + 200 cache_read
        assert context.n_output_tokens == 500
        assert context.n_cache_tokens == 200

    def test_writes_partial_trajectory_without_step_finish(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        _write_events(
            temp_dir,
            [
                _make_step_start("s1", "m1"),
                _make_text("s1", "m1", "The process was interrupted."),
                _make_tool_use(
                    "s1",
                    "m1",
                    "bash",
                    {"command": "python server.py &"},
                    "server started",
                ),
            ],
        )

        agent.populate_context_post_run(AgentContext())
        get_artifact_writer().flush(timeout=30)

        data = json.loads((temp_dir / "trajectory.json").read_text())
        assert len(data["steps"]) == 1
        step = data["steps"][0]
        assert step["message"] == "The process was interrupted."
        assert step["tool_calls"][0]["arguments"] == {"command": "python server.py &"}
        assert step["observation"]["results"][0]["content"] == "server started"
        assert step.get("metrics") is None

    def test_noop_when_no_output_file(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        context = AgentContext()
        agent.populate_context_post_run(context)
        get_artifact_writer().flush(timeout=30)

        assert not (temp_dir / "trajectory.json").exists()
        assert context.cost_usd is None
        assert context.n_input_tokens is None

    def test_noop_when_output_has_no_valid_events(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        (temp_dir / "opencode.txt").write_text("not json at all\n")

        context = AgentContext()
        agent.populate_context_post_run(context)
        get_artifact_writer().flush(timeout=30)
        assert not (temp_dir / "trajectory.json").exists()


class TestOpenCodeInstall:
    @pytest.mark.asyncio
    async def test_reuses_exact_pinned_version(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, version="1.18.2")
        environment = AsyncMock()
        environment.default_user = "agent"
        environment.exec.return_value = SimpleNamespace(
            return_code=0,
            stdout="1.18.2\n",
            stderr="",
        )

        await agent.install(environment)

        environment.exec.assert_awaited_once()
        assert "opencode --version" in environment.exec.call_args.kwargs["command"]
        assert "user" not in environment.exec.call_args.kwargs

    @pytest.mark.asyncio
    async def test_reinstalls_mismatched_pinned_version(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, version="1.18.2")
        environment = AsyncMock()
        environment.default_user = "agent"
        environment.exec.side_effect = [
            SimpleNamespace(return_code=0, stdout="1.18.1\n", stderr=""),
            SimpleNamespace(return_code=0, stdout="", stderr=""),
            SimpleNamespace(return_code=0, stdout="1.18.2\n", stderr=""),
        ]

        await agent.install(environment)

        assert environment.exec.await_count == 3
        assert environment.exec.call_args_list[1].kwargs["user"] == "root"
        assert (
            "opencode-ai@1.18.2" in environment.exec.call_args_list[2].kwargs["command"]
        )


class TestOpenCodeRunCommands:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model_name", "variable"),
        [
            ("anthropic/claude-sonnet-4-5", "ANTHROPIC_BASE_URL"),
            ("google/gemini-2.5-pro", "GOOGLE_BASE_URL"),
        ],
    )
    async def test_run_forwards_provider_base_url(
        self, temp_dir, monkeypatch, model_name, variable
    ):
        endpoint = "https://provider.example.test/v1"
        monkeypatch.setenv(variable, endpoint)
        agent = OpenCode(logs_dir=temp_dir, model_name=model_name)
        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")

        await agent.run("do something", mock_env, AsyncMock())

        assert mock_env.exec.call_args_list[-1].kwargs["env"][variable] == endpoint

    @pytest.mark.asyncio
    async def test_run_command_structure(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        await agent.run("do something", mock_env, AsyncMock())
        exec_calls = mock_env.exec.call_args_list
        assert len(exec_calls) == 2
        assert "opencode.json" in exec_calls[0].kwargs["command"]
        assert "opencode" in exec_calls[-1].kwargs["command"]
        assert "tee /logs/agent/opencode.txt" in exec_calls[-1].kwargs["command"]

    @pytest.mark.asyncio
    async def test_no_opencode_data_dir_in_env(self, temp_dir):
        """OPENCODE_DATA_DIR is not needed since we parse stdout."""
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        await agent.run("do something", mock_env, AsyncMock())
        exec_calls = mock_env.exec.call_args_list
        assert "OPENCODE_DATA_DIR" not in exec_calls[-1].kwargs["env"]

    @pytest.mark.asyncio
    async def test_fake_vcs_present(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        await agent.run("do something", mock_env, AsyncMock())
        exec_calls = mock_env.exec.call_args_list
        assert exec_calls[0].kwargs["env"]["OPENCODE_FAKE_VCS"] == "git"

    @pytest.mark.asyncio
    async def test_file_read_feature_flag_disabled(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")

        await agent.run("do something", mock_env, AsyncMock())

        opencode_call = mock_env.exec.call_args_list[-1]
        assert opencode_call.kwargs["env"]["OPENCODE_DISABLE_FFF"] == "1"

    @pytest.mark.asyncio
    async def test_variant_flag_is_included(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="openai/gpt-5.3-codex",
            variant="xhigh",
        )
        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        await agent.run("do something", mock_env, AsyncMock())
        exec_calls = mock_env.exec.call_args_list
        assert "--variant xhigh" in exec_calls[-1].kwargs["command"]

    @pytest.mark.asyncio
    async def test_model_flag_is_included(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="my-provider/my-model",
        )
        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        await agent.run("do something", mock_env, AsyncMock())
        exec_calls = mock_env.exec.call_args_list
        assert "--model=my-provider/my-model" in exec_calls[-1].kwargs["command"]

    @pytest.mark.asyncio
    async def test_raises_when_json_error_event_is_emitted(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="openai/gpt-5.3-codex")
        mock_env = AsyncMock()

        async def exec_side_effect(**kwargs):
            if "tee /logs/agent/opencode.txt" in kwargs["command"]:
                _write_events(
                    temp_dir,
                    [
                        {
                            "type": "error",
                            "error": {
                                "name": "ProviderError",
                                "data": {"message": "provider unavailable"},
                            },
                        }
                    ],
                )
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        mock_env.exec.side_effect = exec_side_effect

        with pytest.raises(NonZeroAgentExitCodeError) as exc_info:
            await agent.run("do something", mock_env, AgentContext())

        assert str(exc_info.value) == (
            "OpenCode emitted error event(s): provider unavailable"
        )


_OVERFLOW_MESSAGE = (
    "This model's maximum context length is 32768 tokens. However, you "
    "requested 4096 output tokens and your prompt contains at least 28673 "
    "input tokens, for a total of at least 32769 tokens. Please reduce the "
    "length of the input prompt or the number of requested output tokens."
)


def _context_overflow_event():
    """A ContextOverflowError event exactly as opencode teed it in the grug
    32k-overflow trials (upstream vLLM 400 BadRequestError)."""
    return {
        "type": "error",
        "timestamp": 1784658090120,
        "sessionID": "ses_overflow",
        "error": {
            "name": "ContextOverflowError",
            "data": {
                "message": _OVERFLOW_MESSAGE,
                "responseBody": json.dumps(
                    {
                        "error": {
                            "message": _OVERFLOW_MESSAGE,
                            "type": "BadRequestError",
                            "param": "input_tokens",
                            "code": 400,
                        }
                    }
                ),
            },
        },
    }


class TestOpenCodeContextOverflow:
    """A served-context overflow must surface as the clean, classified
    ContextLengthExceededError (passthrough-scored), NOT a generic
    NonZeroAgentExitCodeError that tanks the trial with retries."""

    @pytest.mark.asyncio
    async def test_overflow_with_nonzero_exit_raises_context_length_error(
        self, temp_dir
    ):
        """opencode exits 1 after emitting ContextOverflowError -> the exec
        raises NonZeroAgentExitCodeError, which we reclassify."""
        agent = OpenCode(logs_dir=temp_dir, model_name="vllm/some/model")
        mock_env = AsyncMock()

        async def exec_side_effect(**kwargs):
            command = kwargs["command"]
            if "tee /logs/agent/opencode.txt" in command:
                _write_events(temp_dir, [_context_overflow_event()])
                # opencode exits 1 -> base._exec raises NonZeroAgentExitCodeError.
                return SimpleNamespace(return_code=1, stdout="", stderr="")
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        mock_env.exec.side_effect = exec_side_effect

        with pytest.raises(ContextLengthExceededError) as exc_info:
            await agent.run("do something", mock_env, AgentContext())
        assert "context window" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_overflow_uses_failed_command_stdout_before_logs_are_collected(
        self, temp_dir
    ):
        """Remote environments return command stdout before copying agent logs
        into the local trial directory."""
        agent = OpenCode(logs_dir=temp_dir, model_name="vllm/some/model")
        mock_env = AsyncMock()

        async def exec_side_effect(**kwargs):
            if "tee /logs/agent/opencode.txt" in kwargs["command"]:
                return SimpleNamespace(
                    return_code=1,
                    stdout=json.dumps(_context_overflow_event()),
                    stderr="",
                )
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        mock_env.exec.side_effect = exec_side_effect

        with pytest.raises(ContextLengthExceededError):
            await agent.run("do something", mock_env, AgentContext())

    @pytest.mark.asyncio
    async def test_overflow_with_zero_exit_raises_context_length_error(self, temp_dir):
        """opencode exits 0 but emits a ContextOverflowError event -> the
        error-event path must also reclassify."""
        agent = OpenCode(logs_dir=temp_dir, model_name="vllm/some/model")
        mock_env = AsyncMock()

        async def exec_side_effect(**kwargs):
            if "tee /logs/agent/opencode.txt" in kwargs["command"]:
                _write_events(temp_dir, [_context_overflow_event()])
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        mock_env.exec.side_effect = exec_side_effect

        with pytest.raises(ContextLengthExceededError):
            await agent.run("do something", mock_env, AgentContext())

    @pytest.mark.asyncio
    async def test_non_overflow_nonzero_exit_still_raises_nonzero(self, temp_dir):
        """A genuine (non-overflow) failure is NOT swallowed: a plain non-zero
        exit still surfaces as NonZeroAgentExitCodeError."""
        agent = OpenCode(logs_dir=temp_dir, model_name="vllm/some/model")
        mock_env = AsyncMock()

        async def exec_side_effect(**kwargs):
            command = kwargs["command"]
            if "tee /logs/agent/opencode.txt" in command:
                _write_events(
                    temp_dir,
                    [
                        {
                            "type": "error",
                            "error": {
                                "name": "SomeOtherError",
                                "data": {"message": "disk full"},
                            },
                        }
                    ],
                )
                return SimpleNamespace(return_code=1, stdout="", stderr="")
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        mock_env.exec.side_effect = exec_side_effect

        with pytest.raises(NonZeroAgentExitCodeError) as exc_info:
            await agent.run("do something", mock_env, AgentContext())
        assert not isinstance(exc_info.value, ContextLengthExceededError)

    @pytest.mark.asyncio
    async def test_signal_kill_raises_agent_killed_by_signal(self, temp_dir):
        """A signal kill of the opencode CLI (exit 129 = 128 + SIGHUP, e.g. the
        agent's own `kill -HUP` hitting its process group) is classified as
        AgentKilledBySignalError, not a bare non-zero exit."""
        agent = OpenCode(logs_dir=temp_dir, model_name="vllm/some/model")
        mock_env = AsyncMock()

        async def exec_side_effect(**kwargs):
            if "tee /logs/agent/opencode.txt" in kwargs["command"]:
                return SimpleNamespace(return_code=129, stdout="", stderr="Hangup")
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        mock_env.exec.side_effect = exec_side_effect

        with pytest.raises(AgentKilledBySignalError) as exc_info:
            await agent.run("do something", mock_env, AgentContext())
        assert exc_info.value.signal_number == 1


_AUTHENTICATION_MESSAGE = "The supplied model-endpoint credential was rejected."


def _authentication_error_event():
    return {
        "type": "error",
        "error": {
            "name": "AuthenticationError",
            "data": {
                "message": _AUTHENTICATION_MESSAGE,
                "responseBody": json.dumps(
                    {
                        "error": {
                            "message": _AUTHENTICATION_MESSAGE,
                            "type": "AuthenticationError",
                            "code": 401,
                        }
                    }
                ),
            },
        },
    }


class TestOpenCodeAuthenticationFailure:
    @pytest.mark.asyncio
    async def test_authentication_event_with_nonzero_exit_is_classified(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="vllm/some/model")
        mock_env = AsyncMock()

        async def exec_side_effect(**kwargs):
            if "tee /logs/agent/opencode.txt" in kwargs["command"]:
                _write_events(temp_dir, [_authentication_error_event()])
                return SimpleNamespace(return_code=1, stdout="", stderr="")
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        mock_env.exec.side_effect = exec_side_effect

        with pytest.raises(ModelAuthenticationError) as exc_info:
            await agent.run("do something", mock_env, AgentContext())

        assert "model endpoint" in str(exc_info.value)
        assert "expired capability URL" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_authentication_event_with_zero_exit_is_classified(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="vllm/some/model")
        mock_env = AsyncMock()

        async def exec_side_effect(**kwargs):
            if "tee /logs/agent/opencode.txt" in kwargs["command"]:
                _write_events(temp_dir, [_authentication_error_event()])
            return SimpleNamespace(return_code=0, stdout="", stderr="")

        mock_env.exec.side_effect = exec_side_effect

        with pytest.raises(ModelAuthenticationError):
            await agent.run("do something", mock_env, AgentContext())


class TestOpenCodeLiteralTraces:
    """Tests for OpenCode's literal token-level trace support."""

    def test_supports_literal_traces_flag(self):
        assert OpenCode.SUPPORTS_LITERAL_TRACES is True

    def test_parse_literal_proxy_log_returns_none_when_absent(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="openai/gpt-5.3-codex")
        assert agent._parse_literal_proxy_log() is None

    def test_parse_literal_proxy_log_reads_pairs(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="openai/gpt-5.3-codex")
        lines = [
            json.dumps(
                {
                    "literal": {
                        "prompt_token_ids": [1, 2],
                        "completion_token_ids": [3],
                        "logprobs": [-0.1],
                    }
                }
            ),
            json.dumps(
                {
                    "literal": {
                        "prompt_token_ids": [4],
                        "completion_token_ids": [5, 6],
                        "logprobs": [-0.2, -0.3],
                    }
                }
            ),
        ]
        (temp_dir / "literal.jsonl").write_text("\n".join(lines) + "\n")
        pairs = agent._parse_literal_proxy_log()
        assert pairs is not None
        assert len(pairs) == 2
        assert pairs[0]["completion_token_ids"] == [3]
        assert pairs[1]["prompt_token_ids"] == [4]

    def test_parse_literal_proxy_log_skips_malformed_lines(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="openai/gpt-5.3-codex")
        content = (
            "not json\n"
            + json.dumps({"literal": {"completion_token_ids": [9]}})
            + "\n\n"
        )
        (temp_dir / "literal.jsonl").write_text(content)
        pairs = agent._parse_literal_proxy_log()
        assert pairs is not None
        assert len(pairs) == 1
        assert pairs[0]["completion_token_ids"] == [9]

    def test_merge_literal_into_trajectory_happy_path(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="openai/gpt-5.3-codex")
        events = [
            _make_step_start("s1", "m1"),
            _make_text("s1", "m1", "Hello"),
            _make_step_finish("s1", "m1", cost=0.01, input_tok=10, output_tok=5),
        ]
        _write_events(temp_dir, events)
        pairs = [
            {
                "prompt_token_ids": [1, 2, 3],
                "completion_token_ids": [4, 5],
                "logprobs": [-0.1, -0.2],
            }
        ]
        trajectory = agent._convert_events_to_trajectory(agent._parse_stdout())
        assert trajectory is not None
        agent._merge_literal_into_trajectory(trajectory, pairs)
        step = trajectory.steps[0]
        assert step.metrics.prompt_token_ids == [1, 2, 3]
        assert step.metrics.completion_token_ids == [4, 5]
        assert step.metrics.logprobs == [-0.1, -0.2]
        # Pre-existing token counts preserved.
        assert step.metrics.prompt_tokens == 10
        assert step.metrics.completion_tokens == 5

    def test_merge_literal_handles_count_mismatch(self, temp_dir):
        """More pairs than steps (or vice versa) merges defensively."""
        agent = OpenCode(logs_dir=temp_dir, model_name="openai/gpt-5.3-codex")
        events = [
            _make_step_start("s1", "m1"),
            _make_text("s1", "m1", "Hello"),
            _make_step_finish("s1", "m1"),
            _make_step_start("s1", "m2"),
            _make_text("s1", "m2", "World"),
            _make_step_finish("s1", "m2"),
        ]
        _write_events(temp_dir, events)
        trajectory = agent._convert_events_to_trajectory(agent._parse_stdout())
        assert len([s for s in trajectory.steps if s.source == "agent"]) == 2
        # Only one pair → first step merged, second untouched.
        agent._merge_literal_into_trajectory(
            trajectory, [{"completion_token_ids": [99]}]
        )
        assert trajectory.steps[0].metrics.completion_token_ids == [99]
        assert trajectory.steps[1].metrics.completion_token_ids is None

    def test_merge_literal_creates_metrics_when_absent(self, temp_dir):
        """Steps without usage metrics still receive literal data."""
        agent = OpenCode(logs_dir=temp_dir, model_name="openai/gpt-5.3-codex")
        # step_finish with no token usage → metrics would be None pre-merge.
        events = [
            _make_step_start("s1", "m1"),
            _make_text("s1", "m1", "Hello"),
            _make_step_finish("s1", "m1", cost=0.0, input_tok=0, output_tok=0),
        ]
        _write_events(temp_dir, events)
        trajectory = agent._convert_events_to_trajectory(agent._parse_stdout())
        assert trajectory.steps[0].metrics is None
        agent._merge_literal_into_trajectory(
            trajectory, [{"prompt_token_ids": [1], "completion_token_ids": [2]}]
        )
        assert trajectory.steps[0].metrics is not None
        assert trajectory.steps[0].metrics.prompt_token_ids == [1]
        assert trajectory.steps[0].metrics.completion_token_ids == [2]

    def test_populate_context_merges_proxy_log_into_trajectory(self, temp_dir):
        """End-to-end: populate_context_post_run folds literal.jsonl into trajectory.json."""
        agent = OpenCode(logs_dir=temp_dir, model_name="openai/gpt-5.3-codex")
        _write_events(
            temp_dir,
            [
                _make_step_start("s1", "m1"),
                _make_text("s1", "m1", "Hello"),
                _make_step_finish("s1", "m1", cost=0.01, input_tok=10, output_tok=5),
            ],
        )
        (temp_dir / "literal.jsonl").write_text(
            json.dumps(
                {
                    "literal": {
                        "prompt_token_ids": [1, 2],
                        "completion_token_ids": [3],
                        "logprobs": [-0.5],
                    }
                }
            )
            + "\n"
        )
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        get_artifact_writer().flush(timeout=30)

        traj = json.loads((temp_dir / "trajectory.json").read_text())
        step = traj["steps"][0]
        assert step["metrics"]["prompt_token_ids"] == [1, 2]
        assert step["metrics"]["completion_token_ids"] == [3]
        assert step["metrics"]["logprobs"] == [-0.5]

    def test_populate_context_leaves_rollout_details_none_by_default(self, temp_dir):
        """S0 repro / S1 flag-off byte-identical baseline: a non-empty literal.jsonl
        is merged into trajectory.json but ``context.rollout_details`` stays None.

        This is the core RL gap: opencode is a CLI agent that bypasses harbor
        ``Chat`` so ``context.rollout_details`` is never populated, and the
        existing literal merge only lands in ``trajectory.json`` / step metrics.
        MarinSkyRL reads ``result.agent_result.rollout_details`` → empty → TIS
        logprob extraction breaks. Without ``collect_rollout_details`` opted in,
        behavior must be byte-identical to this (rollout_details stays None).
        """
        agent = OpenCode(logs_dir=temp_dir, model_name="openai/gpt-5.3-codex")
        _write_events(
            temp_dir,
            [
                _make_step_start("s1", "m1"),
                _make_text("s1", "m1", "Hello"),
                _make_step_finish("s1", "m1", cost=0.01, input_tok=10, output_tok=5),
                _make_step_start("s1", "m2"),
                _make_text("s1", "m2", "World"),
                _make_step_finish("s1", "m2", cost=0.02, input_tok=20, output_tok=7),
            ],
        )
        # Two-turn literal log incl. a mismatched-length (logprobs != ids) turn.
        (temp_dir / "literal.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "literal": {
                                "prompt_token_ids": [1, 2],
                                "completion_token_ids": [3, 4],
                                "logprobs": [-0.5, -0.6],
                            }
                        }
                    ),
                    json.dumps(
                        {
                            "literal": {
                                "prompt_token_ids": [5, 6, 7],
                                "completion_token_ids": [8, 9],
                                "logprobs": [-0.7],  # length mismatch (1 vs 2)
                            }
                        }
                    ),
                ]
            )
            + "\n"
        )
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        get_artifact_writer().flush(timeout=30)

        # The literal data DID land in the trajectory (existing behavior)...
        traj = json.loads((temp_dir / "trajectory.json").read_text())
        assert traj["steps"][0]["metrics"]["completion_token_ids"] == [3, 4]
        # ...but rollout_details — what the RL loop consumes — is still None.
        assert ctx.rollout_details is None

    def test_build_rollout_details_from_literal_parity(self, temp_dir):
        """S1: builder yields one RolloutDetail w/ per-turn lists; the mismatched
        turn records [] logprobs (never mis-paired), completion ids preserved."""
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="openai/gpt-5.3-codex",
            collect_rollout_details=True,
        )
        pairs = [
            {
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3, 4],
                "logprobs": [-0.5, -0.6],
            },
            {
                "prompt_token_ids": [5, 6, 7],
                "completion_token_ids": [8, 9],
                "logprobs": [-0.7],  # length mismatch (1 vs 2)
            },
        ]
        details = agent._build_rollout_details_from_literal(pairs)
        assert len(details) == 1
        rd = details[0]
        assert rd["prompt_token_ids"] == [[1, 2], [5, 6, 7]]
        assert rd["completion_token_ids"] == [[3, 4], [8, 9]]
        # Turn 0 parity holds; turn 1 mismatch → empty logprobs, stays aligned.
        assert rd["logprobs"] == [[-0.5, -0.6], []]
        assert len(rd["logprobs"]) == len(rd["completion_token_ids"])
        for lp, ids in zip(rd["logprobs"], rd["completion_token_ids"]):
            assert len(lp) == 0 or len(lp) == len(ids)

    def test_build_rollout_details_missing_logprobs_records_empty(self, temp_dir):
        """A turn with completion ids but no logprobs records [] (index-aligned)."""
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="openai/gpt-5.3-codex",
            collect_rollout_details=True,
        )
        details = agent._build_rollout_details_from_literal(
            [{"prompt_token_ids": [1], "completion_token_ids": [2, 3]}]
        )
        assert details[0]["completion_token_ids"] == [[2, 3]]
        assert details[0]["logprobs"] == [[]]

    def test_build_rollout_details_empty_when_no_token_data(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="openai/gpt-5.3-codex",
            collect_rollout_details=True,
        )
        assert agent._build_rollout_details_from_literal([{}]) == []

    def test_populate_context_sets_rollout_details_when_flag_on(self, temp_dir):
        """S1 flag-ON end-to-end: same S0 fixture, but with
        collect_rollout_details=True context.rollout_details is populated with
        per-turn token ids + parity-guarded logprobs."""
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="openai/gpt-5.3-codex",
            collect_rollout_details=True,
        )
        _write_events(
            temp_dir,
            [
                _make_step_start("s1", "m1"),
                _make_text("s1", "m1", "Hello"),
                _make_step_finish("s1", "m1", cost=0.01, input_tok=10, output_tok=5),
                _make_step_start("s1", "m2"),
                _make_text("s1", "m2", "World"),
                _make_step_finish("s1", "m2", cost=0.02, input_tok=20, output_tok=7),
            ],
        )
        (temp_dir / "literal.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "literal": {
                                "prompt_token_ids": [1, 2],
                                "completion_token_ids": [3, 4],
                                "logprobs": [-0.5, -0.6],
                            }
                        }
                    ),
                    json.dumps(
                        {
                            "literal": {
                                "prompt_token_ids": [5, 6, 7],
                                "completion_token_ids": [8, 9],
                                "logprobs": [-0.7],  # length mismatch
                            }
                        }
                    ),
                ]
            )
            + "\n"
        )
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        get_artifact_writer().flush(timeout=30)

        assert ctx.rollout_details is not None
        assert len(ctx.rollout_details) == 1
        rd = ctx.rollout_details[0]
        assert rd["completion_token_ids"] == [[3, 4], [8, 9]]
        assert rd["prompt_token_ids"] == [[1, 2], [5, 6, 7]]
        assert rd["logprobs"] == [[-0.5, -0.6], []]
        # Trajectory merge still happens too (unchanged behavior).
        traj = json.loads((temp_dir / "trajectory.json").read_text())
        assert traj["steps"][0]["metrics"]["completion_token_ids"] == [3, 4]

    def test_config_injects_correlation_header_when_flag_on(self, temp_dir):
        """S2a: collect_rollout_details=True stamps a per-trial x-ot-trial-id into
        the vllm provider options.headers (the confirmed opencode-1.18.2 channel)."""
        import shlex

        from harbor.literal.proxy import CORRELATION_ID_HEADER

        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="vllm/my-model",
            collect_rollout_details=True,
        )
        assert agent.rollout_correlation_id  # minted
        cmd = agent._build_register_config_command()
        tokens = shlex.split(cmd)
        cfg = json.loads(tokens[tokens.index("echo") + 1])
        headers = cfg["provider"]["vllm"]["options"]["headers"]
        assert headers[CORRELATION_ID_HEADER] == agent.rollout_correlation_id

    def test_config_no_correlation_header_when_flag_off(self, temp_dir):
        """Flag off → no id minted, no headers key (byte-identical config)."""
        import shlex

        agent = OpenCode(logs_dir=temp_dir, model_name="vllm/my-model")
        assert agent.rollout_correlation_id is None
        cmd = agent._build_register_config_command()
        tokens = shlex.split(cmd)
        cfg = json.loads(tokens[tokens.index("echo") + 1])
        assert "headers" not in cfg["provider"]["vllm"].get("options", {})

    def test_config_openai_compatible_and_header_under_hosted_vllm(
        self, temp_dir, monkeypatch
    ):
        """GAP C real-path: the RL rollout generator forces model_name=
        ``hosted_vllm/<alias>`` (terminal_bench_generator), so the openai-compatible
        provider block (npm + baseURL + apiKey) AND the x-ot-trial-id correlation header
        must fire for the ``hosted_vllm`` provider id, not only for bare ``vllm``. The
        prior header test used ``vllm/...`` and so passed while the real path no-op'd."""
        import shlex

        from harbor.literal.proxy import CORRELATION_ID_HEADER

        monkeypatch.setenv(
            "OPENAI_BASE_URL", "https://ingress.example/proxy/t/tok/ep/v1"
        )
        monkeypatch.setenv("OPENCODE_DUMMY_KEY", "dummy-inert-key")
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="hosted_vllm/my-model",
            collect_rollout_details=True,
        )
        assert agent.rollout_correlation_id  # minted
        cmd = agent._build_register_config_command()
        tokens = shlex.split(cmd)
        cfg = json.loads(tokens[tokens.index("echo") + 1])
        provider_cfg = cfg["provider"]["hosted_vllm"]
        # openai-compatible routing (else opencode calls the Responses API vLLM 404s)
        assert provider_cfg["npm"] == "@ai-sdk/openai-compatible"
        options = provider_cfg["options"]
        assert options["baseURL"] == "https://ingress.example/proxy/t/tok/ep/v1"
        assert options["apiKey"] == "dummy-inert-key"
        # the per-trial correlation header the shared RecordProxy attributes by
        assert options["headers"][CORRELATION_ID_HEADER] == agent.rollout_correlation_id

    def test_config_prefers_harbor_model_endpoint_over_openai_base_url(
        self, temp_dir, monkeypatch
    ):
        """The self-hosted vLLM baseURL comes from the harbor-specific HARBOR_MODEL_ENDPOINT,
        NOT OPENAI_BASE_URL. OPENAI_BASE_URL is reserved for genuine OpenAI traffic (the
        LLM-judge verifiers read it); the RL runner publishes only HARBOR_MODEL_ENDPOINT, so
        a real OPENAI_BASE_URL must NOT win here (and must not be misrouted to vLLM)."""
        import shlex

        monkeypatch.setenv(
            "HARBOR_MODEL_ENDPOINT", "https://iris.oa.dev/proxy/t/tok/ep/v1"
        )
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("OPENCODE_DUMMY_KEY", "dummy-inert-key")
        agent = OpenCode(logs_dir=temp_dir, model_name="hosted_vllm/my-model")
        cmd = agent._build_register_config_command()
        tokens = shlex.split(cmd)
        cfg = json.loads(tokens[tokens.index("echo") + 1])
        options = cfg["provider"]["hosted_vllm"]["options"]
        assert options["baseURL"] == "https://iris.oa.dev/proxy/t/tok/ep/v1"

    def test_config_falls_back_to_openai_base_url_when_no_harbor_endpoint(
        self, temp_dir, monkeypatch
    ):
        """Back-compat: with no HARBOR_MODEL_ENDPOINT, the baseURL still resolves from
        OPENAI_BASE_URL (nothing that relied on the old behavior regresses)."""
        import shlex

        monkeypatch.delenv("HARBOR_MODEL_ENDPOINT", raising=False)
        monkeypatch.setenv(
            "OPENAI_BASE_URL", "https://ingress.example/proxy/t/tok/ep/v1"
        )
        monkeypatch.setenv("OPENCODE_DUMMY_KEY", "dummy-inert-key")
        agent = OpenCode(logs_dir=temp_dir, model_name="hosted_vllm/my-model")
        cmd = agent._build_register_config_command()
        tokens = shlex.split(cmd)
        cfg = json.loads(tokens[tokens.index("echo") + 1])
        options = cfg["provider"]["hosted_vllm"]["options"]
        assert options["baseURL"] == "https://ingress.example/proxy/t/tok/ep/v1"

    def test_config_baseurl_from_api_base_kwarg_with_env_unset(
        self, temp_dir, monkeypatch
    ):
        """DATA-FLOW guard (the fullgate1-3 fix): the baseURL comes from the api_base
        CTOR KWARG the RL generator passes — with BOTH env vars UNSET. The RL trial's
        opencode config-build runs in a process that does not inherit run_rl's late-set
        HARBOR_MODEL_ENDPOINT/OPENAI_BASE_URL, so the env-only path produced an
        `undefined` baseURL; the kwarg travels with the trial dispatch as data and works."""
        import shlex

        monkeypatch.delenv("HARBOR_MODEL_ENDPOINT", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("OPENCODE_DUMMY_KEY", "dummy-inert-key")
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="hosted_vllm/my-model",
            api_base="https://iris.oa.dev/proxy/t/tok/otagent-x/v1",
        )
        cmd = agent._build_register_config_command()
        tokens = shlex.split(cmd)
        cfg = json.loads(tokens[tokens.index("echo") + 1])
        options = cfg["provider"]["hosted_vllm"]["options"]
        assert options["baseURL"] == "https://iris.oa.dev/proxy/t/tok/otagent-x/v1"

    def test_config_api_base_kwarg_wins_over_env(self, temp_dir, monkeypatch):
        """The api_base kwarg is PRIMARY over the env vars (env stays a fallback for the
        datagen/non-RL paths that set it)."""
        import shlex

        monkeypatch.setenv("HARBOR_MODEL_ENDPOINT", "https://env.example/harbor/v1")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("OPENCODE_DUMMY_KEY", "dummy-inert-key")
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="hosted_vllm/my-model",
            api_base="https://iris.oa.dev/proxy/t/tok/otagent-x/v1",
        )
        cmd = agent._build_register_config_command()
        tokens = shlex.split(cmd)
        cfg = json.loads(tokens[tokens.index("echo") + 1])
        options = cfg["provider"]["hosted_vllm"]["options"]
        assert options["baseURL"] == "https://iris.oa.dev/proxy/t/tok/otagent-x/v1"

    def test_config_hosted_vllm_no_header_when_flag_off(self, temp_dir, monkeypatch):
        """Flag off under hosted_vllm → still openai-compatible routed, but NO header
        (byte-identical to a non-RL hosted_vllm run)."""
        import shlex

        monkeypatch.setenv("OPENAI_BASE_URL", "https://ingress.example/v1")
        agent = OpenCode(logs_dir=temp_dir, model_name="hosted_vllm/my-model")
        assert agent.rollout_correlation_id is None
        cmd = agent._build_register_config_command()
        tokens = shlex.split(cmd)
        cfg = json.loads(tokens[tokens.index("echo") + 1])
        provider_cfg = cfg["provider"]["hosted_vllm"]
        assert provider_cfg["npm"] == "@ai-sdk/openai-compatible"
        assert "headers" not in provider_cfg.get("options", {})

    def test_version_and_opencode_config_reach_ctor(self, temp_dir):
        """GAP B (agent side): the ``version`` pin + ``opencode_config`` passthrough must
        land on the OpenCode instance (self._version drives the install pin; opencode_config
        deep-merges into opencode.json). These are the kwargs harbor_config now forwards."""
        import shlex

        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="hosted_vllm/my-model",
            version="1.18.2",
            opencode_config={"compaction": {"auto": True, "reserved": 16384}},
        )
        assert agent._version == "1.18.2"
        assert agent._opencode_config == {
            "compaction": {"auto": True, "reserved": 16384}
        }
        # the passthrough is deep-merged into the written opencode.json
        cmd = agent._build_register_config_command()
        tokens = shlex.split(cmd)
        cfg = json.loads(tokens[tokens.index("echo") + 1])
        assert cfg["compaction"] == {"auto": True, "reserved": 16384}

    def test_populate_context_sets_correlation_id_metadata_when_flag_on(self, temp_dir):
        """S2a: the id is surfaced on context.metadata so the RL glue can correlate."""
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="vllm/my-model",
            collect_rollout_details=True,
        )
        _write_events(
            temp_dir,
            [
                _make_step_start("s1", "m1"),
                _make_text("s1", "m1", "Hello"),
                _make_step_finish("s1", "m1", cost=0.01, input_tok=10, output_tok=5),
            ],
        )
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        get_artifact_writer().flush(timeout=30)
        assert ctx.metadata is not None
        assert ctx.metadata["rollout_correlation_id"] == agent.rollout_correlation_id

    def test_populate_context_no_correlation_metadata_when_flag_off(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="vllm/my-model")
        _write_events(
            temp_dir,
            [
                _make_step_start("s1", "m1"),
                _make_text("s1", "m1", "Hello"),
                _make_step_finish("s1", "m1", cost=0.01, input_tok=10, output_tok=5),
            ],
        )
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        get_artifact_writer().flush(timeout=30)
        assert ctx.metadata is None

    def test_populate_context_no_rollout_details_when_flag_on_but_no_log(
        self, temp_dir
    ):
        """Flag on but no literal.jsonl → rollout_details stays None (no synthesis)."""
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="openai/gpt-5.3-codex",
            collect_rollout_details=True,
        )
        _write_events(
            temp_dir,
            [
                _make_step_start("s1", "m1"),
                _make_text("s1", "m1", "Hello"),
                _make_step_finish("s1", "m1", cost=0.01, input_tok=10, output_tok=5),
            ],
        )
        ctx = AgentContext()
        agent.populate_context_post_run(ctx)
        get_artifact_writer().flush(timeout=30)
        assert ctx.rollout_details is None

    def test_convert_reads_literal_from_stream_when_present(self, temp_dir):
        """Defensive: if opencode's stream carries token IDs, they are read directly."""
        agent = OpenCode(logs_dir=temp_dir, model_name="openai/gpt-5.3-codex")
        finish = _make_step_finish("s1", "m1", cost=0.01, input_tok=10, output_tok=5)
        finish["part"]["prompt_token_ids"] = [7, 8]
        finish["part"]["completion_token_ids"] = [9]
        finish["part"]["logprobs"] = [-0.4]
        _write_events(
            temp_dir,
            [_make_step_start("s1", "m1"), _make_text("s1", "m1", "Hi"), finish],
        )
        trajectory = agent._convert_events_to_trajectory(agent._parse_stdout())
        assert trajectory.steps[0].metrics.prompt_token_ids == [7, 8]
        assert trajectory.steps[0].metrics.completion_token_ids == [9]
        assert trajectory.steps[0].metrics.logprobs == [-0.4]


class TestOpenCodeAutoFlag:
    """Tests that the run command uses OpenCode's documented --auto bypass flag."""

    @pytest.mark.asyncio
    async def test_run_uses_auto_not_dangerously_skip_permissions(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5-20250929"
        )
        mock_env = AsyncMock()
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        await agent.run("do something", mock_env, AsyncMock())
        command = mock_env.exec.call_args_list[-1].kwargs["command"]
        assert "--auto" in command
        assert "--dangerously-skip-permissions" not in command


class TestOpenCodeModelLimit:
    """Tests for the bounded per-model ``limit`` written into opencode.json.

    Regression guard for opencode #1735 (custom-provider ``max_tokens`` defaults to
    ~32000): without a bounded ``limit.output`` opencode over-requests output tokens
    and the served model 400s on every real prompt ("maximum context length ... total
    ... tokens"), failing 100% of trials.
    """

    @staticmethod
    def _config_from_command(command: str) -> dict:
        """Extract the opencode.json dict from a _build_register_config_command string."""
        # Command shape: mkdir -p ... && echo <shlex-quoted-json> > .../opencode.json
        marker = "echo "
        start = command.index(marker) + len(marker)
        end = command.rindex(" > ")
        quoted = command[start:end]
        # shlex.quote wraps in single quotes; strip them (json has no single quotes).
        if quoted.startswith("'") and quoted.endswith("'"):
            quoted = quoted[1:-1]
        return json.loads(quoted)

    def _limit_for(self, temp_dir, model_name, model_info=None):
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name=model_name,
            **({"model_info": model_info} if model_info is not None else {}),
        )
        command = agent._build_register_config_command()
        assert command is not None
        config = self._config_from_command(command)
        provider = model_name.split("/", 1)[0]
        model_id = model_name.split("/", 1)[1]
        return config["provider"][provider]["models"][model_id]["limit"]

    def test_limit_uses_model_info_and_reserves_output(self, temp_dir):
        # window=32000, output=4096 -> context reserves output AND a 1024 safety
        # margin: 32000-4096-1024=26880 (margin keeps prompt+output strictly < window).
        limit = self._limit_for(
            temp_dir,
            "hosted_vllm/Qwen3-8B",
            model_info={"max_input_tokens": 32000, "max_output_tokens": 4096},
        )
        assert limit == {"context": 26880, "output": 4096}
        # invariant: a maximal prompt (<=context) + output stays within the window.
        assert limit["context"] + limit["output"] <= 32000

    def test_limit_falls_back_to_safe_defaults_when_model_info_absent(self, temp_dir):
        # RL path may not plumb model_info; opencode must STILL be bounded.
        limit = self._limit_for(temp_dir, "hosted_vllm/Qwen3-8B")
        # window default 32768, output default 4096, 1024 margin -> context 27648.
        assert limit == {"context": 27648, "output": 4096}
        assert limit["context"] + limit["output"] <= 32768

    def test_limit_clamps_oversized_output_below_window(self, temp_dir):
        # A caller that over-requests output must be clamped so context stays valid.
        limit = self._limit_for(
            temp_dir,
            "hosted_vllm/Qwen3-8B",
            model_info={"max_input_tokens": 8192, "max_output_tokens": 999999},
        )
        assert limit["output"] <= 8191
        assert limit["context"] >= 1
        assert limit["context"] + limit["output"] <= 8192

    def test_limit_ignores_non_numeric_model_info(self, temp_dir):
        limit = self._limit_for(
            temp_dir,
            "hosted_vllm/Qwen3-8B",
            model_info={"max_input_tokens": None, "max_output_tokens": "n/a"},
        )
        assert limit == {"context": 27648, "output": 4096}

    def test_limit_applied_on_eval_openai_provider_path(self, temp_dir):
        # The overflow also hits the (non-RL) openai-compatible eval path.
        limit = self._limit_for(
            temp_dir,
            "openai/some-model",
            model_info={"max_input_tokens": 16384, "max_output_tokens": 2048},
        )
        assert limit == {"context": 13312, "output": 2048}

    def test_job_level_opencode_config_can_override_limit(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="hosted_vllm/Qwen3-8B",
            model_info={"max_input_tokens": 32000, "max_output_tokens": 4096},
            opencode_config={
                "provider": {
                    "hosted_vllm": {"models": {"Qwen3-8B": {"limit": {"output": 2048}}}}
                }
            },
        )
        config = self._config_from_command(agent._build_register_config_command())
        limit = config["provider"]["hosted_vllm"]["models"]["Qwen3-8B"]["limit"]
        # deep-merge: job override wins on output, auto context preserved as sibling.
        assert limit["output"] == 2048
        assert limit["context"] == 26880


class TestOpenCodeServerSideTimeout:
    """F1: the hard server-side ``timeout`` wrapper that forcibly reaps a
    runaway opencode.

    opencode has no native turn cap and harbor runs its ENTIRE loop as one
    detached sandbox command, so without this a looping opencode outlives its
    trial deadline and orphans the sandbox (auto_stop=0). run() wraps the
    invocation in a shell ``timeout`` keyed off the agent-phase deadline the
    trial records on the environment (``agent_timeout_sec``).
    """

    def test_prefix_empty_when_no_deadline(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="hosted_vllm/Qwen3-8B")
        assert (
            agent._server_timeout_prefix(SimpleNamespace(agent_timeout_sec=None)) == ""
        )

    def test_prefix_empty_when_non_numeric_deadline(self, temp_dir):
        # A test double / env without a numeric deadline must be a no-op.
        agent = OpenCode(logs_dir=temp_dir, model_name="hosted_vllm/Qwen3-8B")
        assert agent._server_timeout_prefix(AsyncMock()) == ""

    def test_prefix_reaps_strictly_before_app_deadline(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="hosted_vllm/Qwen3-8B")
        prefix = agent._server_timeout_prefix(SimpleNamespace(agent_timeout_sec=1800))
        assert prefix == "timeout -k 5 -s TERM 1785 "
        # TERM at deadline-15, SIGKILL 5s later — both < the app-level deadline,
        # so opencode is reaped server-side before wait_for cancels the poll.
        term = 1800 - OpenCode._SERVER_TIMEOUT_GRACE_SEC
        assert term + OpenCode._SERVER_TIMEOUT_KILL_AFTER_SEC < 1800

    def test_prefix_floors_at_one_for_tiny_deadline(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="hosted_vllm/Qwen3-8B")
        assert (
            agent._server_timeout_prefix(SimpleNamespace(agent_timeout_sec=3))
            == "timeout -k 5 -s TERM 1 "
        )

    @pytest.mark.asyncio
    async def test_run_wraps_command_with_timeout_when_deadline_set(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="my-provider/my-model")
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = 1800
        mock_env.exec.return_value = SimpleNamespace(
            return_code=0, stdout="", stderr=""
        )
        await agent.run("do something", mock_env, AgentContext())
        command = mock_env.exec.call_args_list[-1].kwargs["command"]
        assert (
            "timeout -k 5 -s TERM 1785 opencode --model=my-provider/my-model" in command
        )
        # The `| tee` stays OUTSIDE the timeout so it flushes and exits on EOF.
        assert command.index("timeout -k") < command.index("| stdbuf -oL tee")

    @pytest.mark.asyncio
    async def test_run_command_byte_identical_when_no_deadline(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="my-provider/my-model")
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = None
        mock_env.exec.return_value = SimpleNamespace(
            return_code=0, stdout="", stderr=""
        )
        await agent.run("do something", mock_env, AgentContext())
        command = mock_env.exec.call_args_list[-1].kwargs["command"]
        assert "timeout -k" not in command
        assert "opencode --model=my-provider/my-model run" in command

    @pytest.mark.asyncio
    async def test_timeout_kill_reclassified_as_agent_timeout_error(self, temp_dir):
        # 124 = shell `timeout` fired; surfaced as AgentTimeoutError so the trial
        # treats it as an agent timeout (retry-excluded), not a generic crash.
        agent = OpenCode(logs_dir=temp_dir, model_name="my-provider/my-model")
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = 1800

        async def exec_side_effect(**kwargs):
            code = 124 if "tee /logs/agent/opencode.txt" in kwargs["command"] else 0
            return SimpleNamespace(return_code=code, stdout="", stderr="")

        mock_env.exec.side_effect = exec_side_effect
        with pytest.raises(AgentTimeoutError):
            await agent.run("do something", mock_env, AgentContext())

    @pytest.mark.parametrize("return_code", [137, 143])
    @pytest.mark.asyncio
    async def test_signal_exit_preserves_agent_killed_error(
        self, temp_dir, return_code
    ):
        agent = OpenCode(logs_dir=temp_dir, model_name="my-provider/my-model")
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = 1800

        async def exec_side_effect(**kwargs):
            code = (
                return_code
                if "tee /logs/agent/opencode.txt" in kwargs["command"]
                else 0
            )
            return SimpleNamespace(return_code=code, stdout="", stderr="")

        mock_env.exec.side_effect = exec_side_effect
        with pytest.raises(AgentKilledBySignalError) as exc_info:
            await agent.run("do something", mock_env, AgentContext())
        assert exc_info.value.signal_number == return_code - 128

    @pytest.mark.asyncio
    async def test_genuine_nonzero_exit_stays_nonzero_error(self, temp_dir):
        # A real opencode crash (exit 1) must NOT be masked as a timeout.
        agent = OpenCode(logs_dir=temp_dir, model_name="my-provider/my-model")
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = 1800

        async def exec_side_effect(**kwargs):
            code = 1 if "tee /logs/agent/opencode.txt" in kwargs["command"] else 0
            return SimpleNamespace(return_code=code, stdout="", stderr="")

        mock_env.exec.side_effect = exec_side_effect
        with pytest.raises(NonZeroAgentExitCodeError):
            await agent.run("do something", mock_env, AgentContext())

    @pytest.mark.asyncio
    async def test_124_not_reclassified_without_server_timeout(self, temp_dir):
        # No deadline => no server-side timeout => a 124 is a genuine failure.
        agent = OpenCode(logs_dir=temp_dir, model_name="my-provider/my-model")
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = None

        async def exec_side_effect(**kwargs):
            code = 124 if "tee /logs/agent/opencode.txt" in kwargs["command"] else 0
            return SimpleNamespace(return_code=code, stdout="", stderr="")

        mock_env.exec.side_effect = exec_side_effect
        with pytest.raises(NonZeroAgentExitCodeError):
            await agent.run("do something", mock_env, AgentContext())


class TestOpenCodeMaxTurns:
    """Lever 2: the graceful ``steps`` iteration cap wired into opencode.json.

    opencode exposes no ``run`` CLI flag for a turn cap, but its config supports
    a per-agent ``steps`` field (legacy ``maxSteps``) that forces the agent to
    stop iterating and respond with text once reached. Written for the default
    ``build`` agent (the one ``opencode run`` uses with no --agent).
    """

    @staticmethod
    def _config_from_command(command: str) -> dict:
        marker = "echo "
        start = command.index(marker) + len(marker)
        end = command.rindex(" > ")
        quoted = command[start:end]
        if quoted.startswith("'") and quoted.endswith("'"):
            quoted = quoted[1:-1]
        return json.loads(quoted)

    def test_steps_written_for_build_agent_when_set(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir, model_name="hosted_vllm/Qwen3-8B", max_turns=30
        )
        config = self._config_from_command(agent._build_register_config_command())
        assert config["agent"]["build"]["steps"] == 30

    def test_no_agent_block_when_unset(self, temp_dir):
        agent = OpenCode(logs_dir=temp_dir, model_name="hosted_vllm/Qwen3-8B")
        config = self._config_from_command(agent._build_register_config_command())
        assert "agent" not in config

    def test_job_level_config_overrides_steps(self, temp_dir):
        agent = OpenCode(
            logs_dir=temp_dir,
            model_name="hosted_vllm/Qwen3-8B",
            max_turns=30,
            opencode_config={"agent": {"build": {"steps": 99}}},
        )
        config = self._config_from_command(agent._build_register_config_command())
        # deep-merge: job-level override wins over the auto-written cap.
        assert config["agent"]["build"]["steps"] == 99
        # Sibling provider config is preserved.
        assert "provider" in config
