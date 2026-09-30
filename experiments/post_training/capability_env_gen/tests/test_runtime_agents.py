import asyncio
import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def shellsim_snapshot(root, stored, modes, limits):
    import hashlib

    snapshot = {
        "schema_version": "taskcompendium-shellsim-vfs-snapshot-v1",
        "root": root,
        "limits": limits,
        "entries": [
            {"path": path.removeprefix(root + "/"), "kind": "file",
             "sha256": hashlib.sha256(data).hexdigest(), "size": len(data),
             "mode": modes[path]}
            for path, data in stored.items()
        ],
    }
    return {
        "snapshot": snapshot,
        "snapshot_sha256": hashlib.sha256(
            json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest(),
        "entry_count": len(snapshot["entries"]),
    }


@pytest.fixture
def agents(monkeypatch):
    # Exercise our transport boundary without installing Harbor in the controller.
    # Real inheritance/tool execution is additionally checked in the live probes.
    upstream = ModuleType("taskcompendium.harbor.agents")

    class Base:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.history = []

    class Replay(Base):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.steps = kwargs.get("steps")
            self.step_index = 0

        async def run(self, instruction, environment, context):
            context.metadata = {"ordinary_replay_called": True}

    class TurnCapExhaustedError(RuntimeError):
        pass

    def shell_tool_definition(binding):
        return {"type": "function", "function": {"name": binding.name}}

    def record(logs_dir, transcript, response, context, tools=None):
        context.metadata = {
            "assistant_final": response,
            "all_messages": list(transcript),
            "tools": tools or [],
        }

    upstream.DirectChatAgent = upstream.ShellToolAgent = Base
    upstream.ReplayAgent = Replay
    upstream.TurnCapExhaustedError = TurnCapExhaustedError
    upstream.shell_tool_definition = shell_tool_definition
    upstream._record = record
    monkeypatch.setitem(sys.modules, "taskcompendium", ModuleType("taskcompendium"))
    monkeypatch.setitem(
        sys.modules, "taskcompendium.harbor", ModuleType("taskcompendium.harbor")
    )
    monkeypatch.setitem(sys.modules, "taskcompendium.harbor.agents", upstream)
    source = Path(__file__).parents[1] / "capability_pipeline" / "runtime_agents.py"
    spec = importlib.util.spec_from_file_location("tested_runtime_agents", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("TEST_GLM_CREDENTIAL", "test-secret-never-export")
    return module


def make_agent(agents, tmp_path, **extra):
    max_tokens = extra.pop("max_tokens", 128)
    return agents.GLMChatAgent(
        api_key_env="TEST_GLM_CREDENTIAL",
        api_base="https://service.test/",
        model_name="glm-5.3",
        max_tokens=max_tokens,
        temperature=0,
        request_timeout=10,
        logs_dir=tmp_path,
        **extra,
    )


def test_trusted_workspace_stages_bytes_before_ordinary_replay(agents, tmp_path):
    import hashlib
    from types import SimpleNamespace

    source = tmp_path / "source"
    source.mkdir()
    data = b"fixed submission\n"
    (source / "answer.txt").write_bytes(data)
    manifest = {
        "schema_version": "capability-trusted-shellsim-workspace-v1",
        "workspace": "controls/reference",
        "source_root": str(source),
        "directories": [],
        "files": [{
            "path": "answer.txt", "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "executable": False,
        }],
    }
    stored, modes = {}, {}

    class Session:
        def mkdir(self, path):
            pass

        def write_file(self, path, content):
            stored[path] = content

        def read_file(self, path):
            return stored[path]

        def run(self, command, timeout=None):
            if command.startswith("chmod 0644"):
                modes["/app/answer.txt"] = 0o644
            else:
                assert command.startswith("if [ -L")
                assert "/app/answer.txt" in command
            return SimpleNamespace(return_code=0, stop_reason=None, usage=None)

        def _request(self, operation, *, path, limits):
            assert operation == "snapshot"
            return shellsim_snapshot(path, stored, modes, limits)

    class Environment:
        session = Session()
        task_env_config = SimpleNamespace(workdir="/app")

    context = SimpleNamespace(metadata={})
    agent = agents.TrustedWorkspaceReplayAgent(
        trusted_workspace=manifest, logs_dir=tmp_path,
    )
    asyncio.run(agent.run("task", Environment(), context))
    assert context.metadata["ordinary_replay_called"] is True
    assert context.metadata["trusted_workspace_staging"]["files"][0]["sha256"] == (
        manifest["files"][0]["sha256"]
    )
    assert context.metadata["trusted_workspace_staging"]["schema_version"] == (
        "capability-trusted-shellsim-workspace-staging-v2"
    )
    assert context.metadata["trusted_workspace_staging"]["files"][0]["mode_attestation"] == (
        "shellsim_vfs_snapshot"
    )
    assert json.loads((tmp_path / "trusted-workspace-staging.json").read_text())[
        "files"
    ][0]["sha256"] == manifest["files"][0]["sha256"]


def test_trusted_workspace_staging_failure_stops_before_ordinary_replay(
    agents, tmp_path,
):
    import hashlib
    from types import SimpleNamespace

    source = tmp_path / "source"
    source.mkdir()
    data = b"expected"
    (source / "answer.txt").write_bytes(data)
    manifest = {
        "schema_version": "capability-trusted-shellsim-workspace-v1",
        "workspace": "controls/reference", "source_root": str(source),
        "directories": [],
        "files": [{"path": "answer.txt", "size": len(data),
                   "sha256": hashlib.sha256(data).hexdigest(), "executable": False}],
    }

    class Session:
        def run(self, command, timeout=None):
            return SimpleNamespace(return_code=0, stop_reason=None, usage=None)

        def write_file(self, path, content):
            pass

        def read_file(self, path):
            return b"corrupted"

        def mkdir(self, path):
            pass

        def _request(self, operation, *, path, limits):
            pytest.fail("corrupted bytes must stop before snapshot attestation")

    class Environment:
        session = Session()
        task_env_config = SimpleNamespace(workdir="/app")

    environment = Environment()
    context = SimpleNamespace(metadata={})
    agent = agents.TrustedWorkspaceReplayAgent(
        trusted_workspace=manifest, logs_dir=tmp_path,
    )
    with pytest.raises(RuntimeError, match="readback hash differs"):
        asyncio.run(agent.run("task", environment, context))
    assert "ordinary_replay_called" not in context.metadata


@pytest.mark.parametrize("observed_mode, succeeds", [(0o755, True), (0o644, False)])
def test_trusted_workspace_attests_executable_mode_in_vfs_snapshot(
    agents, tmp_path, observed_mode, succeeds,
):
    import hashlib
    from types import SimpleNamespace

    source = tmp_path / "source"
    source.mkdir()
    data = b"#!/bin/sh\nexit 0\n"
    (source / "run.sh").write_bytes(data)
    manifest = {
        "schema_version": "capability-trusted-shellsim-workspace-v1",
        "workspace": "controls/reference", "source_root": str(source),
        "directories": [],
        "files": [{"path": "run.sh", "size": len(data),
                   "sha256": hashlib.sha256(data).hexdigest(), "executable": True}],
    }

    stored = {}

    class Session:
        def run(self, command, timeout=None):
            assert command.startswith(("if [ -L", "chmod 0755"))
            return SimpleNamespace(return_code=0, stop_reason=None, usage=None)

        def write_file(self, path, content):
            stored[path] = content

        def read_file(self, path):
            return stored[path]

        def mkdir(self, path):
            pass

        def _request(self, operation, *, path, limits):
            assert operation == "snapshot"
            return shellsim_snapshot(path, stored, {"/app/run.sh": observed_mode}, limits)

    class Environment:
        session = Session()
        task_env_config = SimpleNamespace(workdir="/app")

    context = SimpleNamespace(metadata={})
    agent = agents.TrustedWorkspaceReplayAgent(
        trusted_workspace=manifest, logs_dir=tmp_path,
    )
    if succeeds:
        asyncio.run(agent.run("task", Environment(), context))
        assert context.metadata["ordinary_replay_called"] is True
    else:
        with pytest.raises(RuntimeError, match="VFS snapshot differs"):
            asyncio.run(agent.run("task", Environment(), context))
        assert "ordinary_replay_called" not in context.metadata


def test_credentials_are_read_at_runtime_and_never_written_to_trace(
    agents, monkeypatch, tmp_path
):
    captured = []

    def respond(request, **kwargs):
        captured.append(request)
        return io.BytesIO(
            json.dumps(
                {
                    "model": "glm-5.3",
                    "usage": {"completion_tokens": 1},
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "answer"},
                        }
                    ],
                }
            ).encode()
        )

    monkeypatch.setattr(agents.urllib.request, "urlopen", respond)
    agent = make_agent(agents, tmp_path)
    assert (
        agent._completion([{"role": "user", "content": "public task"}])["content"]
        == "answer"
    )
    assert captured[0].full_url == "https://service.test/v1/chat/completions"
    assert captured[0].get_header("Authorization") == "Bearer test-secret-never-export"
    trace = (tmp_path / "glm-requests.jsonl").read_text()
    assert "test-secret-never-export" not in trace
    assert json.loads(trace)["request"]["chat_template_kwargs"] == {
        "reasoning_effort": "high"
    }


def test_literal_credentials_in_serialized_config_are_rejected(agents, tmp_path):
    with pytest.raises(ValueError, match="never a credential"):
        make_agent(agents, tmp_path, api_key="test-secret-never-export")


def test_truncated_response_never_becomes_final_solver_answer(
    agents, monkeypatch, tmp_path
):
    payload = {
        "choices": [{"finish_reason": "length", "message": {"content": "incomplete"}}]
    }
    monkeypatch.setattr(
        agents.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()),
    )
    with pytest.raises(RuntimeError, match="Incomplete GLM"):
        make_agent(agents, tmp_path)._completion([])
    records = [
        json.loads(line)
        for line in (tmp_path / "glm-requests.jsonl").read_text().splitlines()
    ]
    assert [record["request"]["max_tokens"] for record in records] == [128, 65536, 131072, None]


def test_length_stop_retries_once_and_only_returns_complete_response(
    agents, monkeypatch, tmp_path
):
    payloads = iter(
        [
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "truncated"},
                    }
                ]
            },
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "complete"}}
                ]
            },
        ]
    )
    monkeypatch.setattr(
        agents.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(next(payloads)).encode()),
    )
    assert make_agent(agents, tmp_path)._completion([])["content"] == "complete"
    records = [
        json.loads(line)
        for line in (tmp_path / "glm-requests.jsonl").read_text().splitlines()
    ]
    assert [record["finish_reason"] for record in records] == ["length", "stop"]
    assert [record["request_attempt"] for record in records] == [1, 2]


def test_adversary_policy_preserves_legacy_attempts_then_uses_remaining_context(
    agents, monkeypatch, tmp_path
):
    payloads = iter(
        [
            {"choices": [{"finish_reason": "length", "message": {}}]},
            {"choices": [{"finish_reason": "length", "message": {}}]},
            {"choices": [{"finish_reason": "length", "message": {}}]},
            {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]},
        ]
    )
    monkeypatch.setattr(
        agents.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(next(payloads)).encode()),
    )
    limits = agents.adversary_token_limits("32768,65536,131072,remaining_context")
    assert (
        make_agent(
            agents,
            tmp_path,
            max_tokens=32768,
            token_limits=limits,
        )._completion([])["content"]
        == "ok"
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "glm-requests.jsonl").read_text().splitlines()
    ]
    assert [record["request"]["max_tokens"] for record in records] == [
        32768,
        65536,
        131072,
        None,
    ]


@pytest.mark.parametrize(
    "value",
    ["32768,65536,131072,262144", "65536,32768,remaining_context", "bad"],
)
def test_adversary_token_policy_rejects_unsafe_or_partial_schedules(agents, value):
    with pytest.raises(ValueError):
        agents.adversary_token_limits(value)


def test_adversary_token_policy_allows_a_fresh_128k_then_remaining_replay(agents):
    assert agents.adversary_token_limits(None) == (131072, None)
    assert agents.adversary_token_limits("131072,remaining_context") == (
        131072,
        None,
    )
    assert agents.adversary_request_timeout("1800") == 1800


def test_request_trace_binds_boundary_phase(agents, monkeypatch, tmp_path):
    payload = {"choices": [{"finish_reason": "stop", "message": {"content": "draft"}}]}
    monkeypatch.setattr(
        agents.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(payload).encode()),
    )
    agent = make_agent(agents, tmp_path)
    agent._request_phase = "boundary-planning"
    assert agent._completion([])["content"] == "draft"
    record = json.loads((tmp_path / "glm-requests.jsonl").read_text())
    assert record["phase"] == "boundary-planning"


class _Binding:
    name = "shell"


class _Context:
    metadata = None


class _Result:
    def __init__(self, command):
        self.command = command

    def model_dump(self):
        return {"stdout": self.command, "exit_code": 0}


class _Environment:
    def __init__(self):
        self.commands = []
        self.timeouts = []

    async def exec(self, command, timeout_sec=None):
        # An unset deadline silently inherits ShellSim's 30 s default, which
        # failed a whole evaluation attempt on one contended command.
        assert isinstance(timeout_sec, (int, float)) and timeout_sec > 30
        self.commands.append(command)
        self.timeouts.append(timeout_sec)
        return _Result(command)


def make_shell_agent(agents, tmp_path, messages, *, max_turns=4):
    agent = agents.GLMShellToolAgent(
        api_key_env="TEST_GLM_CREDENTIAL",
        api_base="https://service.test/",
        model_name="glm-5.3",
        max_tokens=128,
        temperature=0,
        request_timeout=10,
        logs_dir=tmp_path,
        tool_binding=_Binding(),
        max_turns=max_turns,
    )
    responses = iter(messages)
    agent._completion = lambda transcript, tools: next(responses)
    return agent


def test_malformed_shell_arguments_are_returned_as_tool_error_for_correction(
    agents, tmp_path
):
    # Exact argument shape retained from the c32 GLM failure.
    bad_call_id = "chatcmpl-tool-af1353beb9c5df4f"
    agent = make_shell_agent(
        agents,
        tmp_path,
        [
            {
                "role": "assistant",
                "content": "Let me write the repair script.",
                "tool_calls": [
                    {
                        "id": bad_call_id,
                        "function": {"name": "shell", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "corrected-call",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "printf repaired"}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        ],
    )
    environment = _Environment()
    context = _Context()

    asyncio.run(agent.run("repair", environment, context))

    assert environment.commands == ["printf repaired"]
    error_message = agent.history[2]
    assert error_message["role"] == "tool"
    assert error_message["tool_call_id"] == bad_call_id
    assert json.loads(error_message["content"]) == {
        "error": {
            "type": "invalid_tool_call",
            "message": "Shell tool requires exactly one string command",
        }
    }
    assert context.metadata["assistant_final"] == "done"


def test_multiple_valid_shell_calls_execute_and_record_in_wire_order(agents, tmp_path):
    agent = make_shell_agent(
        agents,
        tmp_path,
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-one",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "first"}),
                        },
                    },
                    {
                        "id": "call-two",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "second"}),
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "complete"},
        ],
    )
    environment = _Environment()

    asyncio.run(agent.run("run both", environment, _Context()))

    assert environment.commands == ["first", "second"]
    tool_messages = [item for item in agent.history if item["role"] == "tool"]
    assert [item["tool_call_id"] for item in tool_messages] == [
        "call-one",
        "call-two",
    ]
    assert [json.loads(item["content"])["stdout"] for item in tool_messages] == [
        "first",
        "second",
    ]


def test_repeated_shell_result_prompts_a_change_of_method(agents, tmp_path):
    def call(identifier, command):
        return {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": identifier, "function": {
                "name": "shell", "arguments": json.dumps({"command": command}),
            }}],
        }

    agent = make_shell_agent(agents, tmp_path, [
        call("repeat-1", "awk probe"),
        call("repeat-2", "awk probe"),
        call("repeat-3", "awk probe"),
        call("new-method", "printf answer"),
        {"role": "assistant", "content": "done"},
    ], max_turns=8)
    environment = _Environment()

    asyncio.run(agent.run("solve", environment, _Context()))

    assert environment.commands == ["awk probe"] * 3 + ["printf answer"]
    nudge = [message for message in agent.history
             if message["role"] == "user" and "same tool call" in message["content"]]
    assert len(nudge) == 1
    assert agent.history.index(nudge[0]) < next(
        index for index, message in enumerate(agent.history)
        if message.get("tool_calls") and message["tool_calls"][0]["id"] == "new-method"
    )


def test_repeated_shell_result_fails_before_exhausting_all_turns(agents, tmp_path):
    calls = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": f"repeat-{index}", "function": {
             "name": "shell", "arguments": json.dumps({"command": "awk probe"}),
         }}]}
        for index in range(8)
    ]
    agent = make_shell_agent(agents, tmp_path, calls, max_turns=12)
    environment = _Environment()

    with pytest.raises(agents.TurnCapExhaustedError, match="identical call and result six times"):
        asyncio.run(agent.run("solve", environment, _Context()))

    assert environment.commands == ["awk probe"] * 6
    assert len(agent.history) == 14  # user, six pairs, and one recovery hint


@pytest.mark.parametrize("include_code", [True, False])
@pytest.mark.parametrize("configured", [True, False])
def test_context_rejection_uses_configured_remaining_budget_without_changing_prompt(
    agents, monkeypatch, tmp_path, include_code, configured
):
    captured = []
    secret = "server-text-must-not-be-logged"

    def respond(request, **kwargs):
        captured.append(json.loads(request.data))
        if len(captured) == 1:
            raise agents.urllib.error.HTTPError(
                request.full_url,
                400,
                "Bad Request",
                {},
                io.BytesIO(
                    json.dumps(
                        {
                            "error": {
                                "message": "This model's maximum context length is 262144; max_tokens exceeds remaining capacity. "
                                + secret,
                                **(
                                    {"code": "context_length_exceeded"}
                                    if include_code
                                    else {}
                                ),
                            }
                        }
                    ).encode()
                ),
            )
        return io.BytesIO(
            json.dumps(
                {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": "complete"}}
                    ]
                }
            ).encode()
        )

    monkeypatch.setattr(agents.urllib.request, "urlopen", respond)
    messages = [{"role": "user", "content": "immutable public task"}]
    tools = [{"type": "function", "function": {"name": "shell"}}]
    agent = make_agent(agents, tmp_path, max_tokens=32768, **(
        {"token_limits": (32768, 65536, 131072, None)} if configured else {}
    ))
    assert agent._completion(messages, tools)["content"] == "complete"
    assert [row["max_tokens"] for row in captured] == [32768, None]
    assert all(
        row["messages"] == messages and row["tools"] == tools for row in captured
    )
    trace = (tmp_path / "glm-requests.jsonl").read_text()
    assert secret not in trace
    records = [json.loads(line) for line in trace.splitlines()]
    assert records[0]["http_status"] == 400
    assert records[0]["retry"] == "remaining_context"
    assert len(records[0]["error_body_sha256"]) == 64
    assert [row["request_attempt"] for row in records] == [1, 2]


@pytest.mark.parametrize("code", ["invalid_request_error", "context_length_exceeded"])
def test_default_http_retries_only_context_rejection_once_with_remaining_budget(
    agents, monkeypatch, tmp_path, code
):
    calls = []

    def reject(request, **kwargs):
        calls.append(request)
        raise agents.urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            {},
            io.BytesIO(json.dumps({"error": {"code": code}}).encode()),
        )

    monkeypatch.setattr(agents.urllib.request, "urlopen", reject)
    extra = (
        {"max_tokens": 131072, "token_limits": (131072, None)}
        if code == "invalid_request_error"
        else {}
    )
    with pytest.raises(RuntimeError, match="HTTP 400"):
        make_agent(agents, tmp_path, **extra)._completion([])
    assert len(calls) == (2 if code == "context_length_exceeded" else 1)


def test_remaining_context_rejection_is_terminal(agents, monkeypatch, tmp_path):
    calls = []

    def reject(request, **kwargs):
        calls.append(json.loads(request.data)["max_tokens"])
        raise agents.urllib.error.HTTPError(
            request.full_url,
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"error":{"code":"context_length_exceeded"}}'),
        )

    monkeypatch.setattr(agents.urllib.request, "urlopen", reject)
    with pytest.raises(RuntimeError, match="category=context_length"):
        make_agent(
            agents, tmp_path, max_tokens=131072, token_limits=(131072, None)
        )._completion([])
    assert calls == [131072, None]


@pytest.mark.parametrize(
    ("tool_calls", "error"),
    [
        ({"id": "not-a-list"}, "tool_calls must be a list"),
        (
            [{"function": {"name": "shell", "arguments": "{}"}}],
            "nonempty string id",
        ),
        ([{"id": "missing-function"}], "function object"),
        (
            [
                {
                    "id": "non-string-arguments",
                    "function": {"name": "shell", "arguments": {}},
                }
            ],
            "name and arguments must be strings",
        ),
    ],
)
def test_invalid_tool_envelopes_fail_closed_without_provider_replay(
    agents, tmp_path, tool_calls, error
):
    agent = make_shell_agent(
        agents,
        tmp_path,
        [{"role": "assistant", "content": None, "tool_calls": tool_calls}],
    )
    environment = _Environment()

    with pytest.raises((TypeError, ValueError), match=error):
        asyncio.run(agent.run("repair", environment, _Context()))

    assert environment.commands == []
    assert len(agent.history) == 2


def _ok_payload(content="answer"):
    return io.BytesIO(json.dumps({
        "model": "glm-5.3",
        "system_fingerprint": "fp",
        "usage": {"completion_tokens": 1},
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    }).encode())


def _http_error(agents, code, body=b""):
    return agents.urllib.error.HTTPError("https://service.test/v1/chat/completions", code, "err", {}, io.BytesIO(body))


def test_transient_transport_failures_hold_then_succeed_without_polluting_requests(agents, monkeypatch, tmp_path):
    import urllib.error

    calls = []

    def respond(request, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError(TimeoutError(110, "Connection timed out"))
        if len(calls) == 2:
            raise _http_error(agents, 503)
        if len(calls) == 3:
            raise _http_error(agents, 404, b'{"error":"this relay has no route for model \'glm-5.3\'"}')
        return _ok_payload("held")

    monkeypatch.setattr(agents.urllib.request, "urlopen", respond)
    monkeypatch.setattr(agents.time, "sleep", lambda _s: None)
    assert make_agent(agents, tmp_path)._completion([])["content"] == "held"
    transport = [json.loads(line) for line in (tmp_path / "glm-transport.jsonl").read_text().splitlines()]
    assert [t["transport_attempt"] for t in transport] == [1, 2, 3]
    assert "transport_error" in transport[0] and transport[1]["http_status"] == 503
    assert transport[2]["http_status"] == 404
    # glm-requests.jsonl keeps its meaning: one record per request that got a response
    requests = (tmp_path / "glm-requests.jsonl").read_text().splitlines()
    assert len(requests) == 1 and json.loads(requests[0])["message"]["content"] == "held"


def test_non_transient_http_errors_are_not_held_and_body_stays_readable(agents, monkeypatch, tmp_path):
    def respond(request, **kwargs):
        raise _http_error(agents, 400, b'{"error":{"message":"bad request"}}')

    monkeypatch.setattr(agents.urllib.request, "urlopen", respond)
    monkeypatch.setattr(agents.time, "sleep", lambda _s: pytest.fail("a 400 must not be held"))
    with pytest.raises(RuntimeError, match="GLM completion HTTP 400"):
        make_agent(agents, tmp_path)._completion([])
    assert not (tmp_path / "glm-transport.jsonl").exists()


def test_plain_404_is_not_mistaken_for_a_route_outage(agents, monkeypatch, tmp_path):
    monkeypatch.setattr(agents.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(_http_error(agents, 404, b"not found")))
    monkeypatch.setattr(agents.time, "sleep", lambda _s: pytest.fail("a plain 404 must not be held"))
    with pytest.raises(RuntimeError, match="GLM completion HTTP 404"):
        make_agent(agents, tmp_path)._completion([])


def test_infrastructure_hold_is_bounded(agents, monkeypatch, tmp_path):
    import urllib.error

    monkeypatch.setattr(agents, "INFRASTRUCTURE_HOLD_SECONDS", 5.0)
    clock = [0.0]
    monkeypatch.setattr(agents.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(agents.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(agents.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("refused")))
    with pytest.raises(RuntimeError, match="GLM transport unavailable"):
        make_agent(agents, tmp_path)._completion([])
    assert len((tmp_path / "glm-transport.jsonl").read_text().splitlines()) >= 2
