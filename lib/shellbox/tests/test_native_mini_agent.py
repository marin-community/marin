# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native CLI behavior through the real guest command loop."""

import asyncio
import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("harbor")
pytest.importorskip("minisweagent")

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from minisweagent.environments.local import LocalEnvironment
from shellbox.backends.qemu.environment import QemuEnvironment
from shellbox.machine import Command
from shellbox.mini_agent import NativeMiniAgent

from lib.shellbox.tests.test_qemu_machine import local_guest


@contextmanager
def scripted_endpoint(commands, blocked_request=None, started=None, release=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            index = len(requests)
            requests.append(request)
            if index == blocked_request:
                started.set()
                release.wait(20)
            command = commands[index]
            response = {
                "id": f"response-{index}",
                "object": "chat.completion",
                "created": 0,
                "model": "scripted",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "Execute the next command.",
                            "tool_calls": [
                                {
                                    "id": f"action-{index}",
                                    "type": "function",
                                    "function": {
                                        "name": "bash",
                                        "arguments": json.dumps({"command": command}),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
            body = json.dumps(response).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def log_message(self, format: str, *args: object):  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def running_environment(tmp_path):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    environment = QemuEnvironment(
        environment_dir=environment_dir,
        environment_name="fixture",
        session_id="fixture",
        trial_paths=TrialPaths(trial_dir=tmp_path / "trial"),
        task_env_config=EnvironmentConfig(),
        guest_bundle=str(tmp_path),
        network_policy="deny",
    )
    environment.machine = await local_guest(tmp_path, env={"PATH": os.defpath})
    return environment


def native_agent(tmp_path, endpoint, *specs, model_name="openai/scripted"):
    return NativeMiniAgent(
        logs_dir=tmp_path / "logs",
        model_name=model_name,
        api_base=endpoint,
        config_specs=["mini.yaml", *specs],
        model_retry_attempts=1,
    )


@pytest.mark.parametrize("provider", ["openai", "hosted_vllm"])
def test_native_mini_cli_guest_commands_and_host_environment_isolation(tmp_path, monkeypatch, provider):
    monkeypatch.setenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "7")
    child_retry_attempts = []
    original = asyncio.create_subprocess_exec

    async def create_process(*args, **kwargs):
        if "minisweagent.run.mini" in args:
            child_retry_attempts.append(kwargs["env"]["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"])
        return await original(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-only")
    monkeypatch.setenv("HOST_ONLY_SENTINEL", "must-not-enter-guest-or-template")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.delenv("MSWEA_API_KEY", raising=False)
    commands = [
        "printf first; printf second >&2; printf third; printf file > answer",
        "printf partial; sleep 2",
        'printf "%s" "$HOST_ONLY_SENTINEL"; cat answer',
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT; echo final",
    ]
    instruction = "Fixture task " + "x" * 131072
    with scripted_endpoint(commands) as (endpoint, requests):

        async def scenario():
            environment = await running_environment(tmp_path)
            context = AgentContext(metadata={"caller": "preserved"})
            try:
                await native_agent(tmp_path, endpoint, "environment.timeout=1", model_name=f"{provider}/scripted").run(
                    instruction, environment, context
                )
                grade = await environment.machine.run(Command(("cat", "answer")))
                assert grade.stdout == b"file"
            finally:
                await environment.stop(True)
            return context

        context = asyncio.run(scenario())
    assert os.environ["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] == "7"
    assert child_retry_attempts == ["1"]
    native = json.loads((tmp_path / "logs/mini-swe-agent.trajectory.json").read_text())
    observations = [item for item in native["messages"] if item["role"] == "tool"]
    assert "firstsecondthird" in observations[0]["content"]
    assert "partial" in observations[1]["content"]
    assert observations[1]["extra"]["exception_type"] == "TimeoutExpired"
    assert observations[1]["extra"]["raw_output"] == "partial"
    assert "must-not-enter-guest-or-template" not in json.dumps(native)
    assert "file" in observations[2]["content"]
    assert native["info"]["exit_status"] == "Submitted"
    assert native["info"]["submission"] == "final\n"
    assert native["info"]["config"]["agent_type"] == "minisweagent.agents.interactive.InteractiveAgent"
    assert native["info"]["config"]["agent"]["mode"] == "yolo"
    assert native["info"]["config"]["agent"]["confirm_exit"] is False
    assert context.n_input_tokens == 40 and context.n_output_tokens == 20
    assert context.metadata is not None and context.metadata["caller"] == "preserved"
    atif = json.loads((tmp_path / "logs/trajectory.json").read_text())
    assert atif["session_id"] == "fixture"
    assert len([step for step in atif["steps"] if step["source"] == "agent"]) == 4
    assert len(requests) == 4
    assert requests[0]["tools"][0]["function"]["name"] == "bash"
    assert instruction in requests[0]["messages"][1]["content"]


def test_native_mini_limits_exceeded_gets_batch_eof_without_extra_request(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-only")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    with scripted_endpoint(["echo executed > answer"]) as (endpoint, requests):

        async def scenario():
            environment = await running_environment(tmp_path)
            try:
                with pytest.raises(NonZeroAgentExitCodeError, match="controller failed"):
                    await native_agent(tmp_path, endpoint, "agent.step_limit=1").run(
                        "Fixture task", environment, AgentContext()
                    )
                result = await environment.machine.run(Command(("cat", "answer")))
                assert result.stdout == b"executed\n"
            finally:
                await environment.stop(True)

        asyncio.run(scenario())
    native = json.loads((tmp_path / "logs/mini-swe-agent.trajectory.json").read_text())
    assert native["info"]["exit_status"] == "EOFError"
    assert native["info"]["model_stats"]["api_calls"] == 1
    assert len(requests) == 1


def test_native_mini_cancellation_reaps_controller_and_preserves_partial_native_record(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-only")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    started = threading.Event()
    release = threading.Event()
    controller = []
    original = asyncio.create_subprocess_exec

    async def create_process(*args, **kwargs):
        process = await original(*args, **kwargs)
        if "minisweagent.run.mini" in args:
            controller.append(process.pid)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    with scripted_endpoint(
        ["echo completed > answer", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"], 1, started, release
    ) as (endpoint, requests):

        async def scenario():
            environment = await running_environment(tmp_path)
            running = asyncio.create_task(
                native_agent(tmp_path, endpoint).run("Fixture task", environment, AgentContext())
            )
            try:
                assert await asyncio.to_thread(started.wait, 25)
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
                with pytest.raises(ProcessLookupError):
                    os.kill(controller[0], 0)
                result = await environment.machine.run(Command(("cat", "answer")))
                assert result.stdout == b"completed\n"
            finally:
                release.set()
                if not running.done():
                    running.cancel()
                    await asyncio.gather(running, return_exceptions=True)
                await environment.stop(True)

        asyncio.run(scenario())
    native = json.loads((tmp_path / "logs/mini-swe-agent.trajectory.json").read_text())
    assert native["info"]["model_stats"]["api_calls"] == 1
    assert [item["role"] for item in native["messages"]] == ["system", "user", "assistant", "tool"]
    assert len(requests) == 2


@pytest.mark.parametrize("cwd_kind", ["missing", "regular-file"])
def test_native_mini_guest_process_start_errors_match_local_environment(tmp_path, monkeypatch, cwd_kind):
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-only")
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    cwd = tmp_path / "invalid-cwd"
    if cwd_kind == "regular-file":
        cwd.write_text("not a directory")
    expected = LocalEnvironment(cwd=str(cwd)).execute({"command": "echo must-not-run"})
    with scripted_endpoint(["echo must-not-run"]) as (endpoint, requests):

        async def scenario():
            environment = await running_environment(tmp_path)
            try:
                with pytest.raises(NonZeroAgentExitCodeError, match="controller failed"):
                    await native_agent(tmp_path, endpoint, "agent.step_limit=1", f"environment.cwd={cwd}").run(
                        "Fixture task", environment, AgentContext()
                    )
            finally:
                await environment.stop(True)

        asyncio.run(scenario())
    native = json.loads((tmp_path / "logs/mini-swe-agent.trajectory.json").read_text())
    observed = next(item["extra"] for item in native["messages"] if item["role"] == "tool")
    assert observed["returncode"] == expected["returncode"]
    assert observed["exception_info"] == expected["exception_info"]
    assert observed["exception_type"] == expected["extra"]["exception_type"]
    assert observed["raw_output"] == expected["output"]
    assert len(requests) == 1
