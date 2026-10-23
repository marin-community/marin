"""Unit tests for the Pi installed agent."""

import asyncio
import contextlib
import json
import os
import shlex
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from upath import UPath

from harbor.agents.installed.base import (
    AgentKilledBySignalError,
    ModelAuthenticationError,
    NonZeroAgentExitCodeError,
)
from harbor.agents.installed.pi import Pi
from harbor.environments.daytona import DaytonaEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from harbor.llms.base import ContextLengthExceededError
from harbor.literal.proxy import CORRELATION_ID_HEADER
from harbor.models.agent.context import AgentContext
from harbor.trial.errors import AgentTimeoutError, PhaseCancellationError


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


class _LocalDaytonaProcess:
    """Run SDK commands locally, retaining Daytona's per-session queue."""

    def __init__(self, home, *, hold_submission=False, fail_cleanup=False):
        self.home = home
        self.sessions = {}
        self.commands = {}
        self.ready = asyncio.Event()
        self.hold_submission = hold_submission
        self.fail_cleanup = fail_cleanup

    async def create_session(self, session_id):
        self.sessions[session_id] = asyncio.Lock()

    async def delete_session(self, session_id):
        del self.sessions[session_id]

    async def execute_session_command(self, session_id, request, timeout=None):
        if self.fail_cleanup and not request.run_async:
            return SimpleNamespace(exit_code=1)
        command_id = str(len(self.commands))
        task = asyncio.create_task(self._run(session_id, request.command))
        self.commands[command_id] = task
        if request.run_async:
            if self.hold_submission and command_id == "0":
                await asyncio.Event().wait()
            return SimpleNamespace(cmd_id=command_id)
        result = await task
        return SimpleNamespace(exit_code=result.return_code)

    async def _run(self, session_id, command):
        async with self.sessions[session_id]:
            process = await asyncio.create_subprocess_exec(
                "/bin/bash",
                "-c",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={"HOME": str(self.home), "PATH": "/usr/bin:/bin"},
            )
            stderr_task = asyncio.create_task(process.stderr.read())
            lines = []
            async for line in process.stdout:
                lines.append(line)
                if line.strip() == b"READY":
                    self.ready.set()
            return SimpleNamespace(
                return_code=await process.wait(),
                stdout=b"".join(lines).decode(),
                stderr=(await stderr_task).decode(),
            )

    async def get_session_command(self, session_id, command_id):
        task = self.commands[command_id]
        return SimpleNamespace(
            id=command_id,
            exit_code=task.result().return_code if task.done() else None,
        )

    async def get_session_command_logs(self, session_id, command_id):
        return self.commands[command_id].result()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ending",
    [
        "deadline",
        "cancel",
        "cancel_submission",
        "cleanup_failure",
        "complete",
        "orphan_deadline",
        "orphan_cancel",
        "orphan_complete",
    ],
)
async def test_pi_daytona_session_remains_usable_after_agent_exit(
    tmp_path, monkeypatch, ending
):
    script = (
        '#!/bin/bash\necho $$ > "$HOME/pi.pid"\n'
        'setsid bash -c \'echo $$ > "$HOME/tool.pid"; trap "" TERM; '
        'echo READY; read -r line < "$HOME/block"; touch "$HOME/late"\' &\nwait\n'
        if ending != "complete"
        else '#!/bin/bash\nprintf "completed\\n"\n'
    )
    orphan = ending.startswith("orphan_")
    if orphan:
        ending = ending.removeprefix("orphan_")
        tool = (
            'echo $$ > "$HOME/tool.pid"; trap "" TERM; '
            'read -r line < "$HOME/block"; touch "$HOME/late"'
        )
        # The intermediate shell exits after starting a detached tool. The tool
        # closes pipeline FDs, so normal completion cannot rely on EOF to wait.
        launcher = (
            f"setsid bash -c {shlex.quote(tool)} >/dev/null 2>&1 & "
            'while [ ! -s "$HOME/tool.pid" ]; do sleep 0.01; done'
        )
        script = (
            '#!/bin/bash\necho $$ > "$HOME/pi.pid"\n'
            f"bash -c {shlex.quote(launcher)}\n"
            + (
                'printf "completed\\n"\n'
                if ending == "complete"
                else 'echo READY; read -r line < "$HOME/block"\n'
            )
        )
    shell = _local_pi_shell(tmp_path, monkeypatch, script)
    os.mkfifo(tmp_path / "block")
    process = _LocalDaytonaProcess(
        tmp_path,
        hold_submission=ending == "cancel_submission",
        fail_cleanup=ending == "cleanup_failure",
    )
    environment = DaytonaEnvironment(
        environment_dir=tmp_path,
        environment_name="pi-regression",
        session_id="pi-regression",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="ubuntu:22.04"),
    )
    environment._sandbox = SimpleNamespace(
        id="local", process=process, fs=shell.environment
    )
    environment.agent_timeout_sec = 2.0 if ending == "deadline" else None
    agent = Pi(logs_dir=tmp_path, model_name="openai/test-model")
    task = asyncio.create_task(agent.run("Run a tool", environment, AgentContext()))
    try:
        if ending in ("cancel", "cancel_submission", "cleanup_failure"):
            await asyncio.wait_for(process.ready.wait(), timeout=10)
            task.cancel()
            with pytest.raises(
                PhaseCancellationError
                if ending == "cleanup_failure"
                else asyncio.CancelledError
            ):
                await task
        elif ending == "deadline":
            with pytest.raises(AgentTimeoutError):
                await asyncio.wait_for(task, timeout=10)
        else:
            await asyncio.wait_for(task, timeout=10)
            assert (shell.artifact_dir / "pi.txt").read_text() == "completed\n"

        if ending == "cleanup_failure":
            return
        for name in ("pi.pid", "tool.pid") if ending != "complete" or orphan else ():
            pid = int((tmp_path / name).read_text())
            stat = Path(f"/proc/{pid}/stat")
            assert not stat.exists()
        assert not (tmp_path / "late").exists()
        verifier = await asyncio.wait_for(
            environment.exec("printf verifier-ready"), timeout=5
        )
        assert verifier.stdout == "verifier-ready"
        assert verifier.return_code == 0
    finally:
        # Preserve node hygiene even when the regression assertion fails.
        for name in ("tool.pid", "pi.pid"):
            path = tmp_path / name
            if path.exists():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(path.read_text()), 9)
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        await asyncio.gather(*process.commands.values())


def _local_pi_shell(temp_dir, monkeypatch, pi_script):
    monkeypatch.setattr(Pi, "_SUPERVISOR_PATH", str(temp_dir / "supervisor.py"))
    endpoint_path = temp_dir / "harbor-pi-endpoint.env"
    extension_path = temp_dir / "harbor-pi-context-budget.js"
    monkeypatch.setattr(Pi, "_ENDPOINT_ENV_PATH", str(endpoint_path))
    monkeypatch.setattr(Pi, "_CONTEXT_BUDGET_EXTENSION_PATH", str(extension_path))

    bootstrap_bin = temp_dir / "bootstrap-bin"
    fake_bin = temp_dir / "fake-bin"
    nvm_dir = temp_dir / ".nvm"
    artifact_dir = temp_dir / "pi-artifacts"
    for directory in (bootstrap_bin, fake_bin, nvm_dir, artifact_dir):
        directory.mkdir()

    node = shutil.which("node")
    assert node is not None
    (bootstrap_bin / "rm").symlink_to("/bin/rm")
    scripts = {
        fake_bin / "node": f'#!/bin/sh\nexec {shlex.quote(node)} "$@"\n',
        fake_bin / "pi": pi_script,
        fake_bin / "stdbuf": '#!/bin/sh\nshift\nexec "$@"\n',
        fake_bin / "tee": (
            "#!/bin/sh\n"
            'mkdir -p "$HOME/pi-artifacts"\n'
            'exec /usr/bin/tee "$HOME/pi-artifacts/$(basename "$1")"\n'
        ),
    }
    for path, contents in scripts.items():
        path.write_text(contents)
        path.chmod(0o755)
    # Sourcing NVM selects the task image's default Node, not Pi's install version.
    (nvm_dir / "nvm.sh").write_text(
        'export PATH="$HOME/bootstrap-bin:/usr/bin:/bin"\n'
        "nvm() {\n"
        '  [ "$1" = use ] && [ "$2" = 22 ] || return 1\n'
        '  export PATH="$HOME/fake-bin:/usr/bin:/bin"\n'
        "}\n"
    )

    class LocalShellEnvironment:
        agent_timeout_sec = None

        def scoped_exec_cancellation(self, command):
            return contextlib.nullcontext()

        async def upload_file(self, source_path, target_path):
            shutil.copyfile(source_path, target_path)

        async def exec(self, command, **kwargs):
            completed = subprocess.run(
                ["/bin/bash", "-c", command],
                capture_output=True,
                text=True,
                env={"HOME": str(temp_dir), "PATH": str(bootstrap_bin)},
            )
            return SimpleNamespace(
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )

    return SimpleNamespace(
        environment=LocalShellEnvironment(),
        artifact_dir=artifact_dir,
        endpoint_path=endpoint_path,
        extension_path=extension_path,
    )


@pytest.mark.asyncio
async def test_version_check_uses_installed_node_with_different_nvm_default(
    temp_dir, monkeypatch
):
    shell = _local_pi_shell(temp_dir, monkeypatch, '#!/bin/sh\nprintf "0.87.0\\n"\n')
    agent = Pi(logs_dir=temp_dir)

    result = await shell.environment.exec(agent.get_version_command())

    assert result.return_code == 0
    assert agent.parse_version(result.stdout) == "0.87.0"


class TestPiAgent:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("version", "expected_package"),
        [
            (None, "@earendil-works/pi-coding-agent@latest"),
            ("0.73.1", "@mariozechner/pi-coding-agent@0.73.1"),
            ("0.74.0", "@earendil-works/pi-coding-agent@0.74.0"),
        ],
    )
    async def test_install_uses_version_compatible_package(
        self, temp_dir, version, expected_package
    ):
        agent = Pi(logs_dir=temp_dir, version=version)
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = None
        mock_env.scoped_exec_cancellation = MagicMock(
            return_value=contextlib.nullcontext()
        )
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")

        await agent.install(mock_env)

        install_command = next(
            call.kwargs["command"]
            for call in mock_env.exec.call_args_list
            if "npm install -g" in call.kwargs["command"]
        )
        assert f"npm install -g --ignore-scripts {expected_package}" in install_command

    @pytest.mark.asyncio
    async def test_install_rejects_invalid_package_version(self, temp_dir):
        agent = Pi(logs_dir=temp_dir, version="latest??")
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = None
        mock_env.scoped_exec_cancellation = MagicMock(
            return_value=contextlib.nullcontext()
        )

        with pytest.raises(ValueError, match="not a valid package version"):
            await agent.install(mock_env)

    @pytest.mark.asyncio
    async def test_run_command_structure(self, temp_dir):
        agent = Pi(logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5")
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = None
        mock_env.scoped_exec_cancellation = MagicMock(
            return_value=contextlib.nullcontext()
        )
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
            await agent.run("Fix the bug", mock_env, AsyncMock())

        exec_calls = mock_env.exec.call_args_list
        run_cmd = exec_calls[-1].kwargs["command"]
        assert "--provider anthropic" in run_cmd
        assert "--model claude-sonnet-4-5" in run_cmd
        assert "--print" in run_cmd
        assert "--mode json" in run_cmd
        assert "--no-session" in run_cmd
        assert "pi.txt" in run_cmd

    @pytest.mark.asyncio
    async def test_run_no_model(self, temp_dir):
        agent = Pi(logs_dir=temp_dir)
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = None
        mock_env.scoped_exec_cancellation = MagicMock(
            return_value=contextlib.nullcontext()
        )
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        with pytest.raises(ValueError, match="provider/model_name"):
            await agent.run("Fix the bug", mock_env, AsyncMock())

    @pytest.mark.asyncio
    async def test_run_no_slash_in_model(self, temp_dir):
        agent = Pi(logs_dir=temp_dir, model_name="claude-sonnet-4-5")
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = None
        mock_env.scoped_exec_cancellation = MagicMock(
            return_value=contextlib.nullcontext()
        )
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        with pytest.raises(ValueError, match="provider/model_name"):
            await agent.run("Fix the bug", mock_env, AsyncMock())

    @pytest.mark.asyncio
    async def test_run_with_any_provider(self, temp_dir):
        agent = Pi(logs_dir=temp_dir, model_name="my-provider/my-model")
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = None
        mock_env.scoped_exec_cancellation = MagicMock(
            return_value=contextlib.nullcontext()
        )
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        await agent.run("Fix the bug", mock_env, AsyncMock())
        run_command = mock_env.exec.call_args_list[-1].kwargs["command"]
        assert "--provider my-provider --model my-model" in run_command

    @pytest.mark.asyncio
    async def test_api_key_forwarding_anthropic(self, temp_dir):
        agent = Pi(logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5")
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = None
        mock_env.scoped_exec_cancellation = MagicMock(
            return_value=contextlib.nullcontext()
        )
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        env_vars = {
            "ANTHROPIC_API_KEY": "ak-123",
            "UNRELATED_KEY": "ignored",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            await agent.run("Fix the bug", mock_env, AsyncMock())

        run_env = mock_env.exec.call_args_list[-1].kwargs["env"]
        assert run_env["ANTHROPIC_API_KEY"] == "ak-123"
        assert "UNRELATED_KEY" not in run_env

    @pytest.mark.asyncio
    async def test_api_key_forwarding_openai(self, temp_dir):
        agent = Pi(logs_dir=temp_dir, model_name="openai/gpt-4")
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = None
        mock_env.scoped_exec_cancellation = MagicMock(
            return_value=contextlib.nullcontext()
        )
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        env_vars = {
            "OPENAI_API_KEY": "sk-456",
            "UNRELATED_KEY": "ignored",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            await agent.run("Fix the bug", mock_env, AsyncMock())

        run_env = mock_env.exec.call_args_list[-1].kwargs["env"]
        assert run_env["OPENAI_API_KEY"] == "sk-456"
        assert "UNRELATED_KEY" not in run_env

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model_name", "variable"),
        [
            ("anthropic/claude-sonnet-4-5", "ANTHROPIC_BASE_URL"),
            ("openai/gpt-4", "OPENAI_BASE_URL"),
        ],
    )
    async def test_provider_base_url_forwarding(self, temp_dir, model_name, variable):
        agent = Pi(logs_dir=temp_dir, model_name=model_name)
        mock_env = AsyncMock()
        mock_env.agent_timeout_sec = None
        mock_env.scoped_exec_cancellation = MagicMock(
            return_value=contextlib.nullcontext()
        )
        mock_env.exec.return_value = AsyncMock(return_code=0, stdout="", stderr="")
        endpoint = "https://provider.example.test/v1"

        with patch.dict(os.environ, {variable: endpoint}, clear=False):
            await agent.run("Fix the bug", mock_env, AsyncMock())

        assert mock_env.exec.call_args_list[-1].kwargs["env"][variable] == endpoint

    def test_thinking_cli_flag(self, temp_dir):
        agent = Pi(logs_dir=temp_dir, thinking="high")
        flags = agent.build_cli_flags()
        assert "--thinking high" in flags

    def test_thinking_invalid_value(self, temp_dir):
        with pytest.raises(ValueError, match="Valid values"):
            Pi(logs_dir=temp_dir, thinking="ultra")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("context_window", "max_tokens", "keep_recent_tokens"),
        [
            (16384, 4096, 6144),
            (32768, 16384, 8192),
            (65536, 16384, 24576),
        ],
    )
    async def test_run_scales_compaction_to_model_limits(
        self, temp_dir, monkeypatch, context_window, max_tokens, keep_recent_tokens
    ):
        shell = _local_pi_shell(
            temp_dir,
            monkeypatch,
            (
                "#!/bin/sh\n"
                "printf '%s\\n' "
                '\'{"type":"message_end","message":{"role":"assistant",'
                '"stopReason":"stop"}}\'\n'
            ),
        )
        settings_path = temp_dir / ".pi/agent/settings.json"
        settings_path.parent.mkdir(parents=True)
        settings_path.write_text(json.dumps({"theme": "dark"}))
        agent = Pi(
            logs_dir=temp_dir,
            model_name="openai/test-model",
            model_info={
                "max_input_tokens": context_window,
                "max_output_tokens": max_tokens,
            },
        )

        await agent.run("Fix the bug", shell.environment, AgentContext())

        settings = json.loads(settings_path.read_text())
        assert settings == {
            "theme": "dark",
            "compaction": {
                "enabled": True,
                "reserveTokens": max_tokens,
                "keepRecentTokens": keep_recent_tokens,
            },
        }

    @pytest.mark.asyncio
    async def test_run_uses_explicit_compaction_reserve_inside_context_window(
        self, temp_dir, monkeypatch
    ):
        shell = _local_pi_shell(
            temp_dir,
            monkeypatch,
            (
                "#!/bin/sh\n"
                "printf '%s\\n' "
                '\'{"type":"message_end","message":{"role":"assistant",'
                '"stopReason":"stop"}}\'\n'
            ),
        )
        agent = Pi(
            logs_dir=temp_dir,
            model_name="openai/test-model",
            model_info={"max_input_tokens": 65536, "max_output_tokens": 16384},
            compaction_reserve_tokens=32768,
        )

        await agent.run("Fix the bug", shell.environment, AgentContext())

        settings = json.loads((temp_dir / ".pi/agent/settings.json").read_text())
        assert settings["compaction"] == {
            "enabled": True,
            "reserveTokens": 32768,
            "keepRecentTokens": 16384,
        }

    @pytest.mark.parametrize("compaction_reserve_tokens", [16383, 65536])
    def test_explicit_compaction_reserve_must_fit_model_limits(
        self, temp_dir, compaction_reserve_tokens
    ):
        with pytest.raises(ValueError, match="compaction_reserve_tokens"):
            Pi(
                logs_dir=temp_dir,
                model_name="openai/test-model",
                model_info={
                    "max_input_tokens": 65536,
                    "max_output_tokens": 16384,
                },
                compaction_reserve_tokens=compaction_reserve_tokens,
            )

    @pytest.mark.asyncio
    async def test_run_without_model_limits_preserves_pi_compaction_settings(
        self, temp_dir, monkeypatch
    ):
        shell = _local_pi_shell(
            temp_dir,
            monkeypatch,
            (
                "#!/bin/sh\n"
                "printf '%s\\n' "
                '\'{"type":"message_end","message":{"role":"assistant",'
                '"stopReason":"stop"}}\'\n'
            ),
        )
        settings_path = temp_dir / ".pi/agent/settings.json"
        settings_path.parent.mkdir(parents=True)
        original_settings = {
            "compaction": {"reserveTokens": 12000, "keepRecentTokens": 18000}
        }
        settings_path.write_text(json.dumps(original_settings))
        agent = Pi(logs_dir=temp_dir, model_name="openai/test-model")

        await agent.run("Fix the bug", shell.environment, AgentContext())

        assert json.loads(settings_path.read_text()) == original_settings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_name", "thinking_format"),
    [
        ("Qwen/Qwen3.6-35B-A3B", "qwen-chat-template"),
        ("google/gemma-4-26B-A4B", "chat-template"),
    ],
)
async def test_hosted_vllm_endpoint_registers_pi_provider(
    temp_dir, model_name, thinking_format
):
    endpoint = "https://iris.example/proxy/t/cluster=cw/scoped-token/serve.inference/v1"
    served_id = "campaign-model"
    agent = Pi(
        logs_dir=temp_dir,
        model_name=model_name,
        model_alias=f"hosted_vllm/{served_id}",
        api_base=endpoint,
        api_key="dummy-key",
        model_info={"max_input_tokens": 65536, "max_output_tokens": 16384},
        thinking_format=thinking_format,
    )
    mock_env = AsyncMock()
    mock_env.agent_timeout_sec = None
    mock_env.scoped_exec_cancellation = MagicMock(return_value=contextlib.nullcontext())
    mock_env.exec.return_value = SimpleNamespace(return_code=0, stdout="", stderr="")
    uploaded_files = {}

    async def capture_endpoint(source_path, target_path):
        uploaded_files[target_path] = source_path.read_text()

    mock_env.upload_file.side_effect = capture_endpoint

    registration_command = agent._build_register_provider_command(
        "hosted_vllm", served_id
    )
    assert registration_command is not None
    subprocess.run(
        ["bash", "-c", registration_command],
        check=True,
        env={
            **os.environ,
            "HOME": str(temp_dir),
            "HARBOR_PI_API_BASE": endpoint,
            "HARBOR_PI_API_KEY": "dummy-key",
        },
    )
    await agent.run("Fix the bug", mock_env, AgentContext())

    provider = json.loads((temp_dir / ".pi/agent/models.json").read_text())[
        "providers"
    ]["hosted_vllm"]
    assert provider == {
        "baseUrl": endpoint,
        "api": "openai-completions",
        "apiKey": "HARBOR_PI_API_KEY",
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "requiresReasoningContentOnAssistantMessages": True,
            "thinkingFormat": thinking_format,
            "chatTemplateKwargs": {"enable_thinking": {"$var": "thinking.enabled"}},
        },
        "models": [
            {
                "id": served_id,
                "name": served_id,
                "reasoning": True,
                "input": ["text"],
                "contextWindow": 65536,
                "maxTokens": 16384,
                "cost": {
                    "input": 0,
                    "output": 0,
                    "cacheRead": 0,
                    "cacheWrite": 0,
                },
            }
        ],
    }
    run_call = mock_env.exec.call_args_list[-1]
    run_tokens = shlex.split(shlex.split(run_call.kwargs["command"])[-1])
    assert run_tokens[run_tokens.index("--provider") + 1] == "hosted_vllm"
    assert run_tokens[run_tokens.index("--model") + 1] == served_id
    assert run_call.kwargs["env"] == {}
    assert uploaded_files[Pi._ENDPOINT_ENV_PATH] == (
        f"export HARBOR_PI_API_BASE={endpoint}\nexport HARBOR_PI_API_KEY=dummy-key\n"
    )
    assert Pi._CONTEXT_BUDGET_EXTENSION_PATH in uploaded_files
    assert all(
        endpoint not in call.kwargs["command"] for call in mock_env.exec.call_args_list
    )
    assert endpoint not in registration_command


@pytest.mark.asyncio
async def test_hosted_vllm_request_budget_uses_exact_server_token_count(
    temp_dir,
):
    endpoint = "https://iris.example/proxy/t/scoped-token/eval/v1"
    extension_path = temp_dir / "harbor-pi-context-budget.js"
    agent = Pi(
        logs_dir=temp_dir,
        model_name="open-athena/test-model",
        model_alias="hosted_vllm/served-model",
        api_base=endpoint,
        api_key="dummy-key",
        model_info={"max_input_tokens": 32768, "max_output_tokens": 16384},
        thinking_format="chat-template",
    )
    environment = AsyncMock()

    async def copy_extension(source_path, target_path):
        assert target_path == str(extension_path)
        shutil.copyfile(source_path, target_path)

    environment.upload_file.side_effect = copy_extension
    with patch.object(Pi, "_CONTEXT_BUDGET_EXTENSION_PATH", str(extension_path)):
        await agent._stage_context_budget_extension(environment)

    payload = {
        "model": "served-model",
        "messages": [{"role": "user", "content": "dense tabular data"}],
        "tools": [{"type": "function", "function": {"name": "read"}}],
        "max_tokens": 16384,
    }
    script = """
import fs from "node:fs";
const path = process.argv[1];
const payload = JSON.parse(process.argv[2]);
const source = fs.readFileSync(path, "utf8");
const module = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
let handler;
module.default({ on: (event, candidate) => { if (event === "before_provider_request") handler = candidate; } });
let tokenizeRequest;
globalThis.fetch = async (url, options) => {
  tokenizeRequest = { url, body: JSON.parse(options.body) };
  return { ok: true, json: async () => ({ tokens: Array.from({ length: 26500 }, (_, i) => i) }) };
};
const transformed = await handler({ payload });
process.stdout.write(JSON.stringify({ transformed, tokenizeRequest }));
"""
    completed = subprocess.run(
        [
            "node",
            "--input-type=module",
            "--eval",
            script,
            str(extension_path),
            json.dumps(payload),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["transformed"]["max_tokens"] == 6268
    assert result["tokenizeRequest"] == {
        "url": "https://iris.example/proxy/t/scoped-token/eval/tokenize",
        "body": {
            "model": "served-model",
            "messages": payload["messages"],
            "add_generation_prompt": True,
            "tools": payload["tools"],
        },
    }


@pytest.mark.asyncio
async def test_hosted_vllm_request_budget_truncates_token_dense_tool_output(
    temp_dir,
):
    endpoint = "https://iris.example/proxy/t/scoped-token/eval/v1"
    extension_path = temp_dir / "harbor-pi-context-budget.js"
    agent = Pi(
        logs_dir=temp_dir,
        model_name="open-athena/test-model",
        model_alias="hosted_vllm/served-model",
        api_base=endpoint,
        api_key="dummy-key",
        model_info={"max_input_tokens": 32768, "max_output_tokens": 16384},
        thinking_format="chat-template",
    )
    environment = AsyncMock()

    async def copy_extension(source_path, target_path):
        assert target_path == str(extension_path)
        shutil.copyfile(source_path, target_path)

    environment.upload_file.side_effect = copy_extension
    with patch.object(Pi, "_CONTEXT_BUDGET_EXTENSION_PATH", str(extension_path)):
        await agent._stage_context_budget_extension(environment)

    dense_output = "row,value\n" + "x,1234567890\n" * 4000
    payload = {
        "model": "served-model",
        "messages": [
            {"role": "user", "content": "Inspect the complete table"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": dense_output},
        ],
        "max_tokens": 16384,
    }
    script = """
import fs from "node:fs";
const path = process.argv[1];
const payload = JSON.parse(process.argv[2]);
const source = fs.readFileSync(path, "utf8");
const module = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
let handler;
module.default({ on: (event, candidate) => { if (event === "before_provider_request") handler = candidate; } });
const tokenCount = (request) => {
  const toolChars = request.messages
    .filter((message) => message.role === "tool")
    .reduce((total, message) => total + message.content.length, 0);
  return 9000 + Math.ceil(toolChars / 2);
};
globalThis.fetch = async (_url, options) => {
  const request = JSON.parse(options.body);
  return {
    ok: true,
    json: async () => ({ tokens: Array.from({ length: tokenCount(request) }, (_, i) => i) }),
  };
};
const transformed = await handler({ payload });
process.stdout.write(JSON.stringify({ transformed, promptTokens: tokenCount(transformed) }));
"""
    completed = subprocess.run(
        [
            "node",
            "--input-type=module",
            "--eval",
            script,
            str(extension_path),
            json.dumps(payload),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["promptTokens"] <= 32768 - 16384
    assert result["transformed"]["max_tokens"] == 16384
    truncated = result["transformed"]["messages"][-1]["content"]
    assert truncated.startswith("row,value")
    assert truncated.endswith("x,1234567890\n")
    assert "tool output truncated to fit model context" in truncated


@pytest.mark.asyncio
async def test_hosted_vllm_full_command_loads_nvm_and_preserves_capability_url(
    temp_dir, monkeypatch
):
    endpoint = "https://iris.example/proxy/t/scoped-token/eval/v1"
    shell = _local_pi_shell(
        temp_dir,
        monkeypatch,
        (
            "#!/bin/sh\n"
            "printf "
            "'"
            '{"type":"message_end","message":{"role":"assistant",'
            '"stopReason":"error","errorMessage":"401 authentication required: %s"}}\\n'
            "' "
            '"$HARBOR_PI_API_BASE"\n'
        ),
    )

    agent = Pi(
        logs_dir=temp_dir,
        model_name="Qwen/Qwen3.6-35B-A3B",
        model_alias="hosted_vllm/served-model",
        api_base=endpoint,
        api_key="dummy-key",
        thinking_format="qwen-chat-template",
    )

    with pytest.raises(ModelAuthenticationError):
        await agent.run("Fix the bug", shell.environment, AgentContext())

    captured_output = (shell.artifact_dir / "pi.txt").read_text()
    assert endpoint in captured_output
    assert not shell.endpoint_path.exists()
    assert not (temp_dir / ".pi/agent/models.json").exists()


@pytest.mark.parametrize(
    ("error_message", "expected_error"),
    [
        ("Connection error.", NonZeroAgentExitCodeError),
        (
            "400 This model's maximum context length is 32768 tokens. However, "
            "you requested 16384 output tokens and your prompt contains at least "
            "16385 input tokens, for a total of at least 32769 tokens.",
            ContextLengthExceededError,
        ),
    ],
    ids=["connection", "context-overflow"],
)
@pytest.mark.asyncio
async def test_hosted_vllm_failure_is_classified(
    temp_dir, monkeypatch, error_message, expected_error
):
    endpoint = "https://iris.example/proxy/t/scoped-token/eval/v1"
    shell = _local_pi_shell(
        temp_dir,
        monkeypatch,
        (
            "#!/bin/sh\n"
            "printf '%s\\n' "
            + shlex.quote(
                json.dumps(
                    {
                        "type": "message_end",
                        "message": {
                            "role": "assistant",
                            "stopReason": "error",
                            "errorMessage": error_message,
                        },
                    }
                )
            )
            + "\n"
        ),
    )
    agent = Pi(
        logs_dir=temp_dir,
        model_name="Qwen/Qwen3.6-35B-A3B",
        model_alias="hosted_vllm/served-model",
        api_base=endpoint,
        api_key="dummy-key",
        thinking_format="qwen-chat-template",
    )

    with pytest.raises(expected_error) as error:
        await agent.run("Fix the bug", shell.environment, AgentContext())

    assert type(error.value) is expected_error


@pytest.mark.asyncio
async def test_killed_output_validator_preserves_signal_error(temp_dir, monkeypatch):
    shell = _local_pi_shell(temp_dir, monkeypatch, "#!/bin/sh\nexit 0\n")
    node = shutil.which("node")
    assert node is not None
    (temp_dir / "fake-bin/node").write_text(
        "#!/bin/sh\n"
        'case "$2" in *readline.createInterface*) kill -KILL $$;; esac\n'
        f'exec {shlex.quote(node)} "$@"\n'
    )
    agent = Pi(
        logs_dir=temp_dir,
        model_name="Qwen/Qwen3.6-35B-A3B",
        model_alias="hosted_vllm/served-model",
        api_base="https://iris.example/proxy/t/scoped-token/eval/v1",
        api_key="dummy-key",
        thinking_format="qwen-chat-template",
    )

    with pytest.raises(AgentKilledBySignalError) as error:
        await agent.run("Fix the bug", shell.environment, AgentContext())

    assert error.value.signal_number == 9


@pytest.mark.asyncio
async def test_hosted_vllm_composed_provider_roundtrip_preserves_prefix_and_usage(
    temp_dir, monkeypatch
):
    received = {}

    class OpenAIHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            request_body = json.loads(
                self.rfile.read(int(self.headers["content-length"]))
            )
            received.update(
                path=self.path,
                authorization=self.headers.get("authorization"),
                model=request_body["model"],
            )
            response_body = json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "roundtrip complete",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, _format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), OpenAIHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    endpoint = f"http://127.0.0.1:{server.server_port}/proxy/t/redacted/endpoint/v1"
    shell = _local_pi_shell(
        temp_dir,
        monkeypatch,
        """#!/usr/bin/env node
const fs = require("fs");
const path = require("path");
const provider = JSON.parse(
  fs.readFileSync(path.join(process.env.HOME, ".pi/agent/models.json"))
).providers.hosted_vllm;
fetch(`${provider.baseUrl}/chat/completions`, {
  method: "POST",
  headers: {
    "authorization": `Bearer ${process.env.HARBOR_PI_API_KEY}`,
    "content-type": "application/json",
  },
  body: JSON.stringify({ model: provider.models[0].id, messages: [], stream: false }),
}).then(async (response) => {
  const body = await response.json();
  const choice = body.choices[0];
  console.log(JSON.stringify({
    type: "message_end",
    message: {
      role: "assistant",
      content: [{ type: "text", text: choice.message.content }],
      stopReason: choice.finish_reason,
      usage: {
        input: body.usage.prompt_tokens,
        output: body.usage.completion_tokens,
        cacheRead: 0,
        cacheWrite: 0,
      },
    },
  }));
});
""",
    )

    agent = Pi(
        logs_dir=temp_dir,
        model_name="Qwen/Qwen3.6-35B-A3B",
        model_alias="hosted_vllm/served-model",
        api_base=endpoint,
        api_key="dummy-key",
        thinking_format="qwen-chat-template",
    )

    try:
        await agent.run("Reply once", shell.environment, AgentContext())
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    output_path = shell.artifact_dir / "pi.txt"
    event = json.loads(output_path.read_text())
    assert received == {
        "path": "/proxy/t/redacted/endpoint/v1/chat/completions",
        "authorization": "Bearer dummy-key",
        "model": "served-model",
    }
    assert event["message"]["content"] == [
        {"type": "text", "text": "roundtrip complete"}
    ]
    assert event["message"]["stopReason"] == "stop"

    shutil.copyfile(output_path, temp_dir / "pi.txt")
    context = AgentContext()
    agent.populate_context_post_run(context)
    assert context.n_input_tokens == 7
    assert context.n_output_tokens == 2


@pytest.mark.asyncio
async def test_builtin_provider_does_not_replace_models_document(temp_dir):
    agent = Pi(logs_dir=temp_dir, model_name="anthropic/claude-sonnet-4-5")
    mock_env = AsyncMock()
    mock_env.agent_timeout_sec = None
    mock_env.scoped_exec_cancellation = MagicMock(return_value=contextlib.nullcontext())
    mock_env.exec.return_value = SimpleNamespace(return_code=0, stdout="", stderr="")

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=False):
        await agent.run("Fix the bug", mock_env, AgentContext())

    assert all(
        "models.json" not in call.kwargs["command"]
        for call in mock_env.exec.call_args_list
    )
    assert mock_env.exec.call_args_list[-1].kwargs["env"] == {
        "ANTHROPIC_API_KEY": "test-key"
    }


@pytest.mark.parametrize(
    "model_alias", ["hosted_vllm", "/served-model", "hosted_vllm/"]
)
@pytest.mark.asyncio
async def test_malformed_model_alias_fails_before_pi_starts(temp_dir, model_alias):
    agent = Pi(logs_dir=temp_dir, model_name=model_alias)
    mock_env = AsyncMock()
    mock_env.agent_timeout_sec = None
    mock_env.scoped_exec_cancellation = MagicMock(return_value=contextlib.nullcontext())

    with pytest.raises(ValueError, match="provider/model_name"):
        await agent.run("Fix the bug", mock_env, AgentContext())

    mock_env.exec.assert_not_awaited()


def test_hosted_vllm_rejects_unsupported_thinking_format(temp_dir):
    with pytest.raises(ValueError, match="thinking_format"):
        Pi(
            logs_dir=temp_dir,
            model_name="hosted_vllm/served-model",
            api_base="https://iris.example/v1",
            thinking_format="unsupported-template",
        )


def test_rollout_collection_requires_a_configurable_endpoint(temp_dir):
    with pytest.raises(ValueError, match="collect_rollout_details requires api_base"):
        Pi(
            logs_dir=temp_dir,
            model_name="openai/gpt-5",
            collect_rollout_details=True,
        )


def test_rollout_collection_stamps_a_unique_trial_header(temp_dir):
    agents = [
        Pi(
            logs_dir=temp_dir,
            model_name="hosted_vllm/served-model",
            api_base="https://iris.example/v1",
            thinking_format="qwen-chat-template",
            collect_rollout_details=True,
        )
        for _ in range(2)
    ]

    correlation_ids = []
    for agent in agents:
        command = agent._build_register_provider_command("hosted_vllm", "served-model")
        assert command is not None
        provider_document = json.loads(shlex.split(command)[3])
        correlation_id = provider_document["providers"]["hosted_vllm"]["headers"][
            CORRELATION_ID_HEADER
        ]
        assert correlation_id == agent.rollout_correlation_id
        correlation_ids.append(correlation_id)

    assert correlation_ids[0] != correlation_ids[1]


@pytest.mark.asyncio
async def test_hosted_vllm_failure_preserves_capability_url_and_redacts_api_key(
    temp_dir,
):
    endpoint = "https://iris.example/proxy/t/scoped-token/eval/v1"
    api_key = "secret-api-key"
    agent = Pi(
        logs_dir=temp_dir,
        model_name="hosted_vllm/served-model",
        api_base=endpoint,
        api_key=api_key,
        thinking_format="qwen-chat-template",
    )
    mock_env = AsyncMock()
    mock_env.agent_timeout_sec = None
    mock_env.scoped_exec_cancellation = MagicMock(return_value=contextlib.nullcontext())
    mock_env.exec.return_value = SimpleNamespace(
        return_code=1,
        stdout="",
        stderr=f"request to {endpoint} with key {api_key} failed",
    )

    with pytest.raises(RuntimeError) as error:
        await agent.run("Fix the bug", mock_env, AgentContext())

    assert endpoint in str(error.value)
    assert api_key not in str(error.value)


class TestPiPopulateContext:
    def _write_jsonl(self, path, events):
        path.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    def test_parses_token_usage(self, temp_dir):
        agent = Pi(logs_dir=temp_dir)
        context = AgentContext()

        self._write_jsonl(
            temp_dir / "pi.txt",
            [
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "usage": {
                            "input": 100,
                            "output": 50,
                            "cacheRead": 20,
                            "cacheWrite": 10,
                            "cost": {"total": 0.005},
                        },
                    },
                },
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "usage": {
                            "input": 200,
                            "output": 80,
                            "cacheRead": 30,
                            "cacheWrite": 5,
                            "cost": {"total": 0.008},
                        },
                    },
                },
            ],
        )

        agent.populate_context_post_run(context)

        assert context.n_input_tokens == 350  # input (100+200) + cacheRead (20+30)
        assert context.n_output_tokens == 130
        assert context.n_cache_tokens == 50  # cacheRead only (20 + 30)
        assert context.cost_usd == pytest.approx(0.013)

    def test_remote_output_uses_bounded_range_reads(self, monkeypatch):
        logs_dir = UPath("memory://pi-bounded-range-read")
        output_path = logs_dir / "pi.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "usage": {"input": 100, "output": 50},
                    },
                }
            )
            + "\n"
        )
        original_open = type(output_path).open

        def open_with_bounded_remote_reads(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if path == output_path and "r" in mode:
                block_size = kwargs.get("block_size", 50 * 1024 * 1024)
                if block_size > 1024 * 1024:
                    raise OSError("remote store rejects oversized range reads")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(type(output_path), "open", open_with_bounded_remote_reads)
        context = AgentContext()

        Pi(logs_dir=logs_dir).populate_context_post_run(context)

        assert context.n_input_tokens == 100
        assert context.n_output_tokens == 50

    def test_collects_exact_multiturn_literal_evidence(self, temp_dir):
        agent = Pi(
            logs_dir=temp_dir,
            model_name="hosted_vllm/served-model",
            api_base="https://iris.example/v1",
            thinking_format="qwen-chat-template",
            collect_rollout_details=True,
        )
        first_prompt = [1, 2]
        first_completion = [3, 4]
        second_prompt = [*first_prompt, *first_completion, 5, 6]
        self._write_jsonl(
            temp_dir / "literal.jsonl",
            [
                {
                    "literal": {
                        "prompt_token_ids": first_prompt,
                        "completion_token_ids": first_completion,
                        "logprobs": [-0.1, -0.2],
                    }
                },
                {
                    "literal": {
                        "prompt_token_ids": second_prompt,
                        "completion_token_ids": [7],
                        "logprobs": [-0.3],
                    }
                },
            ],
        )

        context = AgentContext()
        agent.populate_context_post_run(context)

        assert context.metadata == {
            "rollout_correlation_id": agent.rollout_correlation_id
        }
        assert context.rollout_details == [
            {
                "prompt_token_ids": [first_prompt, second_prompt],
                "completion_token_ids": [first_completion, [7]],
                "logprobs": [[-0.1, -0.2], [-0.3]],
            }
        ]

    def test_missing_output_file(self, temp_dir):
        agent = Pi(logs_dir=temp_dir)
        context = AgentContext()
        agent.populate_context_post_run(context)
        # Should not raise, context stays at defaults (None)
        assert context.n_input_tokens is None
        assert context.n_output_tokens is None

    def test_ignores_non_assistant_messages(self, temp_dir):
        agent = Pi(logs_dir=temp_dir)
        context = AgentContext()

        self._write_jsonl(
            temp_dir / "pi.txt",
            [
                {
                    "type": "message_end",
                    "message": {
                        "role": "user",
                        "usage": {"input": 999, "output": 999},
                    },
                },
                {"type": "tool_use", "name": "bash"},
            ],
        )

        agent.populate_context_post_run(context)
        assert context.n_input_tokens == 0
        assert context.n_output_tokens == 0

    def test_handles_malformed_jsonl(self, temp_dir):
        agent = Pi(logs_dir=temp_dir)
        context = AgentContext()

        (temp_dir / "pi.txt").write_text(
            "not json\n"
            + json.dumps(
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "usage": {"input": 10, "output": 5},
                    },
                }
            )
            + "\n"
        )

        agent.populate_context_post_run(context)
        assert context.n_input_tokens == 10
        assert context.n_output_tokens == 5

    @pytest.mark.parametrize(
        "truncated_record",
        [
            b'{"type":"message_end","message":{"content":"\xe2',
            b'{"type":"message_end","message":',
        ],
        ids=["utf8-codepoint", "json-record"],
    )
    def test_preserves_usage_before_truncated_final_record(
        self, temp_dir, truncated_record
    ):
        agent = Pi(logs_dir=temp_dir)
        context = AgentContext()
        valid_record = json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "usage": {"input": 10, "output": 5, "cacheRead": 2},
                },
            }
        ).encode()
        (temp_dir / "pi.txt").write_bytes(valid_record + b"\n" + truncated_record)

        agent.populate_context_post_run(context)

        assert context.n_input_tokens == 12
        assert context.n_output_tokens == 5
        assert context.n_cache_tokens == 2

    def test_preserves_usage_around_invalid_utf8_record(self, temp_dir):
        agent = Pi(logs_dir=temp_dir)
        context = AgentContext()
        records = [
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "usage": {"input": 10, "output": 5},
                },
            },
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "usage": {"input": 20, "output": 7},
                },
            },
        ]
        encoded = [json.dumps(record).encode() for record in records]
        (temp_dir / "pi.txt").write_bytes(b"\n\xff\n".join(encoded) + b"\n")

        agent.populate_context_post_run(context)

        assert context.n_input_tokens == 30
        assert context.n_output_tokens == 12

    def test_zero_cost_returns_none(self, temp_dir):
        agent = Pi(logs_dir=temp_dir)
        context = AgentContext()

        self._write_jsonl(
            temp_dir / "pi.txt",
            [
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "usage": {"input": 10, "output": 5},
                    },
                },
            ],
        )

        agent.populate_context_post_run(context)
        assert context.cost_usd is None

    def test_handles_null_nested_fields(self, temp_dir):
        agent = Pi(logs_dir=temp_dir)
        context = AgentContext()

        self._write_jsonl(
            temp_dir / "pi.txt",
            [
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "usage": {"input": 10, "output": 5, "cost": None},
                    },
                },
                {
                    "type": "message_end",
                    "message": {
                        "role": "assistant",
                        "usage": None,
                    },
                },
                {
                    "type": "message_end",
                    "message": None,
                },
            ],
        )

        agent.populate_context_post_run(context)
        assert context.n_input_tokens == 10
        assert context.n_output_tokens == 5
