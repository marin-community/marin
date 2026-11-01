# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Shellbox Daytona grading without provider credentials in the guest."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from daytona._utils.env import DaytonaEnvReader

from harbor.agents.terminus_2.tmux_session import TmuxCommandError, TmuxKeystrokeBatch, TmuxSession
from harbor.environments import daytona, model_relay
from harbor.environments.daytona import DaytonaEnvironment
from harbor.environments.local import LocalEnvironment
from harbor.environments.model_relay import host_model_authentication
from harbor.environments.qemu import QemuEnvironment
from harbor.environments.runtime import host_backend_credentials
from harbor.environments.shellsim import ShellSimEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.utils.env import task_environment
from harbor.verifier.verifier import Verifier
from shellbox.machine import MachineSpec, PackageSpec
from tests.unit.environments.transport import ProcessMachine, available_port


class ProcessFactory:
    def __init__(self, root: Path):
        self.root = root
        self.machine = None

    async def create(self, spec):
        self.machine = ProcessMachine(self.root, spec)
        return self.machine


@pytest.mark.parametrize(
    "environment_class",
    [DaytonaEnvironment, LocalEnvironment, QemuEnvironment, ShellSimEnvironment],
)
@pytest.mark.parametrize("capture_exit_code", [0, 42])
async def test_shellbox_environment_captures_terminus_scrollback_and_preserves_pipeline_failure(
    tmp_path, environment_class, capture_exit_code
):
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    tmux = binary_dir / "tmux"
    tmux.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = capture-pane ]; then\n'
        "  printf 'screen\\n'\n"
        '  case " $* " in\n'
        '    *" -S - "*) exit "$TMUX_CAPTURE_EXIT_CODE" ;;\n'
        "  esac\n"
        "fi\n"
    )
    tmux.chmod(0o755)
    machine = ProcessMachine(tmp_path / "guest", MachineSpec(PackageSpec(), workdir="/workspace"))
    # Substitute only the external transport; run the real adapter and Terminus scripts.
    adapter = object.__new__(environment_class)
    adapter.machine = machine
    adapter.session_id = "capture-test"
    adapter.default_user = None
    adapter._persistent_env = {
        "PATH": f"{binary_dir}:/usr/bin:/bin",
        "TMUX_CAPTURE_EXIT_CODE": str(capture_exit_code),
    }
    if environment_class is LocalEnvironment:
        adapter.execution = None
    session = TmuxSession("capture-test", adapter)
    try:
        if capture_exit_code:
            with pytest.raises(TmuxCommandError):
                await session.send_keys_and_capture([TmuxKeystrokeBatch("echo hello\n", 0)])
        else:
            output, alive = await session.send_keys_and_capture([TmuxKeystrokeBatch("echo hello\n", 0)])
            assert output == "Current Terminal Screen:\nscreen\n"
            assert alive is True
    finally:
        await machine.close()


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_cancelled_start_reclaims_a_late_daytona_machine(tmp_path, cleanup_failure):
    adapter = DaytonaEnvironment(
        environment_dir=tmp_path,
        environment_name="task",
        session_id="trial",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="test/image"),
        client_factory=lambda: pytest.fail("Unexpected provider access"),
    )

    async def run():
        submitting, submitted = asyncio.Event(), asyncio.Event()
        closing, closed = asyncio.Event(), asyncio.Event()
        finish_cleanup = asyncio.Event()

        async def close():
            closing.set()
            await finish_cleanup.wait()
            closed.set()
            if cleanup_failure:
                raise RuntimeError("Backend cleanup failed")

        async def create(spec):
            submitting.set()
            await submitted.wait()
            return SimpleNamespace(close=close)

        adapter.factory = SimpleNamespace(create=create)
        start = asyncio.create_task(adapter.start(False))
        await asyncio.wait_for(submitting.wait(), 5)
        start.cancel("initial cancellation")
        submitted.set()
        await asyncio.wait_for(closing.wait(), 5)
        start.cancel("repeated cancellation")
        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError) as error:
            await asyncio.wait_for(start, 5)
        assert error.value.args == ("initial cancellation",)
        assert closed.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("verifier_endpoint", ["https://api.together.xyz/v1", "https://other-judge.test/v1"])
def test_shellbox_daytona_runs_verifier_through_host_only_judge_relay(
    tmp_path, monkeypatch, verifier_endpoint
):
    port = available_port()
    monkeypatch.setattr(daytona, "MODEL_RELAY_PORT", port)
    monkeypatch.setenv("TOGETHER_API_KEY", "host-provider-private")
    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Answer the question.")
    (task_dir / "task.toml").write_text(
        '[environment]\ndocker_image="test/image@sha256:'
        + "a" * 64
        + '"\nallow_internet=false\nworkdir="/workspace"\n'
    )
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    # This remains a sandbox grading script. The model key it sees is a dummy.
    (task_dir / "tests" / "test.sh").write_text(
        "#!/bin/sh\nset -eu\npython3 - <<'PY'\n"
        "import json, os, urllib.request\n"
        "assert os.environ['OPENAI_API_KEY'] == 'EMPTY'\n"
        "assert 'TOGETHER_API_KEY' not in os.environ\n"
        "request=urllib.request.Request(os.environ['OPENAI_BASE_URL']+'/chat/completions',data=json.dumps({'model':'guest-override','messages':[]}).encode(),headers={'Authorization':'Bearer EMPTY'})\n"
        "assert json.load(urllib.request.urlopen(request))['choices'][0]['message']['content'] == 'correct'\n"
        f"open({str(trial_paths.reward_text_path)!r},'w').write('1')\nPY\n"
    )
    original_client = httpx.AsyncClient

    def provider(request):
        assert request.url == "https://api.together.xyz/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer host-provider-private"
        assert json.loads(request.content)["model"] == "openai/gpt-oss-120b"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "correct"}}]}
        )

    monkeypatch.setattr(
        model_relay.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(provider), **kwargs
        ),
    )
    factory = ProcessFactory(tmp_path / "guest")

    async def run():
        with (
            host_backend_credentials(
                daytona_api_key="host-daytona-private", daytona_target="test-target"
            ),
            task_environment({}),
            host_model_authentication(
                {"https://api.together.xyz/v1": "host-provider-private"}
            ),
        ):
            adapter = DaytonaEnvironment(
                environment_dir=task_dir / "environment",
                environment_name="task",
                session_id="trial",
                trial_paths=trial_paths,
                task_env_config=EnvironmentConfig(
                    docker_image="test/image@sha256:" + "a" * 64,
                    allow_internet=False,
                    workdir="/workspace",
                ),
                mounts=[
                    {
                        "source": str(trial_paths.verifier_dir),
                        "target": "/logs/verifier",
                    }
                ],
                model_endpoint="https://api.together.xyz/v1",
                model_name="together_ai/openai/gpt-oss-120b",
                verifier_model_endpoint=verifier_endpoint,
            )
            adapter.factory = factory
            if verifier_endpoint != "https://api.together.xyz/v1":
                with pytest.raises(ValueError, match="single host model endpoint"):
                    await adapter.start(False)
                assert factory.machine is None
                return
            await adapter.start(False)
            try:
                verifier = Verifier(
                    Task(task_dir),
                    trial_paths,
                    adapter,
                    override_env={
                        "OPENAI_API_KEY": "EMPTY",
                        "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
                    },
                )
                assert (await verifier.verify()).rewards == {"reward": 1.0}
            finally:
                await adapter.stop(True)
        assert factory.machine.closed

    asyncio.run(run())


def test_daytona_rejects_legacy_dind_and_unknown_runtime_options(tmp_path):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    arguments = dict(
        environment_dir=environment_dir,
        environment_name="task",
        session_id="trial",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="test/image"),
        client_factory=lambda: pytest.fail("Unexpected provider allocation"),
    )
    with pytest.raises(ValueError):
        DaytonaEnvironment(**arguments, dind_image="docker:dind")
    (environment_dir / "docker-compose.yaml").write_text("services: {}")
    with pytest.raises(ValueError):
        DaytonaEnvironment(**arguments)


async def test_installed_candidate_relay_buffers_sdk_events_and_keeps_judge_separate(tmp_path, monkeypatch):
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    candidate_port, judge_port = available_port(), available_port()
    monkeypatch.setattr(daytona, "CANDIDATE_MODEL_RELAY_PORT", candidate_port)
    monkeypatch.setattr(daytona, "MODEL_RELAY_PORT", judge_port)
    original_client = httpx.AsyncClient
    completion = {
        "id": "completion-one",
        "created": 1,
        "model": "candidate-model",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "reason",
                    "tool_calls": [
                        {
                            "id": "call-one",
                            "type": "function",
                            "function": {"name": "shell", "arguments": '{"cmd":"pwd"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }

    def provider(request):
        payload = json.loads(request.content)
        if request.url.host == "judge.example":
            assert request.headers["authorization"] == "Bearer judge-private"
            assert payload["model"] == "judge-model"
            return httpx.Response(200, json={"judge": True})
        assert str(request.url) in {
            "https://candidate.example/proxy/t/host-capability/serve/v1/chat/completions",
            "https://candidate.example/proxy/t/host-capability/serve/tokenize",
        }
        assert request.headers["authorization"] == "Bearer candidate-private"
        assert payload["model"] == "candidate-model"
        if request.url.path.endswith("/tokenize"):
            assert payload["tools"][0]["type"] == "function"
            return httpx.Response(
                200, json={"tokens": [1, 2], "count": 2, "max_model_len": 32}
            )
        assert payload["stream"] is False
        assert "stream_options" not in payload
        if payload.get("messages") == ["redirect"]:
            return httpx.Response(
                302,
                headers={"Location": "https://unauthorized.example/candidate-private"},
            )
        if payload.get("messages") == ["oversized"]:
            return httpx.Response(200, content=b"x" * (model_relay.MAX_BODY + 1))
        if payload.get("messages") == ["provider-failure"]:
            return httpx.Response(401, text="candidate-private host-capability")
        return httpx.Response(200, json=completion)

    monkeypatch.setattr(
        model_relay.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(provider), **kwargs
        ),
    )
    factory = ProcessFactory(tmp_path / "guest")
    with (
        host_model_authentication(
            {
                "https://candidate.example/proxy/t/host-capability/serve/v1": (
                    "candidate-private"
                ),
                "https://judge.example/v1": "judge-private",
            }
        ),
        task_environment({}),
    ):
        adapter = DaytonaEnvironment(
            environment_dir=environment_dir,
            environment_name="task",
            session_id="trial",
            trial_paths=TrialPaths(tmp_path / "trial"),
            task_env_config=EnvironmentConfig(
                docker_image="test/image", workdir="/workspace"
            ),
            client_factory=lambda: pytest.fail("Unexpected provider access"),
            candidate_model_endpoint="https://candidate.example/proxy/t/host-capability/serve/v1",
            candidate_model_name="candidate-model",
            model_endpoint="https://judge.example/v1",
            model_name="judge-model",
        )
        adapter.factory = factory
        await adapter.start(False)
        try:
            script = f"""
import json, os, urllib.request, urllib.error
assert os.environ['OPENAI_API_KEY'] == 'EMPTY'
assert os.environ['OPENAI_BASE_URL'] == 'http://127.0.0.1:{candidate_port}/v1'
assert not any('private' in value or 'host-capability' in value for value in os.environ.values())
def post(path, payload, port={candidate_port}):
    request = urllib.request.Request('http://127.0.0.1:'+str(port)+path, data=json.dumps(payload).encode(), headers={{'Authorization':'Bearer EMPTY'}})
    with urllib.request.urlopen(request) as response:
        return response.headers.get_content_type(), response.read().decode()
_, tokenized = post('/tokenize', {{'model':'guest-choice', 'messages':[], 'tools':[{{'type':'function'}}]}})
assert json.loads(tokenized)['tokens'] == [1, 2]
_, ordinary = post('/v1/chat/completions', {{'model':'guest-choice','messages':[]}})
assert json.loads(ordinary)['choices'][0]['message']['content'] == 'answer'
content_type, streamed = post('/v1/chat/completions', {{'model':'guest-choice','messages':[], 'stream':True, 'stream_options':{{'include_usage':True}}}})
assert content_type == 'text/event-stream'
events = [json.loads(event[6:]) for event in streamed.strip().split('\\n\\n')[:-1]]
assert streamed.endswith('data: [DONE]\\n\\n')
delta = events[0]['choices'][0]['delta']
assert delta['reasoning_content'] == 'reason' and delta['content'] == 'answer'
assert delta['tool_calls'][0]['index'] == 0 and delta['tool_calls'][0]['function']['arguments'] == '{{"cmd":"pwd"}}'
assert events[1]['choices'][0]['finish_reason'] == 'tool_calls'
assert events[2]['usage']['total_tokens'] == 8
assert json.loads(post('/v1/chat/completions', {{'model':'guest-choice','messages':[]}}, {judge_port})[1]) == {{'judge':True}}
try:
    post('/v1/responses', {{}})
    raise AssertionError('unsupported route accepted')
except urllib.error.HTTPError as error:
    assert error.code == 404
for failure in ('redirect', 'oversized', 'provider-failure'):
    try:
        post('/v1/chat/completions', {{'messages':[failure]}})
        raise AssertionError('upstream failure accepted')
    except urllib.error.HTTPError as error:
        assert error.code == 502
        body = error.read().decode()
        assert 'candidate-private' not in body and 'host-capability' not in body
print('ok')
"""
            result = await adapter.exec(
                "python3 - <<'PY'\n" + script + "\nPY",
                env={
                    "OPENAI_API_KEY": "EMPTY",
                    "OPENAI_BASE_URL": f"http://127.0.0.1:{candidate_port}/v1",
                },
            )
            assert result.return_code == 0, result.stderr
            assert result.stdout.strip() == "ok"
        finally:
            await adapter.stop(True)
    assert factory.machine.closed


def test_host_connection_prevents_sdk_ambient_auth_and_destination_fallback(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DAYTONA_JWT_TOKEN", "ambient-jwt-must-not-be-used")
    monkeypatch.setenv("DAYTONA_API_URL", "https://ambient.invalid/api")
    protected = {
        "DAYTONA_API_KEY",
        "DAYTONA_JWT_TOKEN",
        "DAYTONA_ORGANIZATION_ID",
        "DAYTONA_API_URL",
        "DAYTONA_SERVER_URL",
        "DAYTONA_TARGET",
    }

    def environment_value(reader, name):
        assert (
            name not in protected
        ), "Daytona SDK attempted ambient connection resolution"
        return None

    monkeypatch.setattr(DaytonaEnvReader, "get", environment_value)

    async def run():
        with host_backend_credentials(
            daytona_api_key="explicit-key", daytona_target="test-target"
        ):
            adapter = DaytonaEnvironment(
                environment_dir=tmp_path,
                environment_name="task",
                session_id="trial",
                trial_paths=TrialPaths(tmp_path / "trial"),
                task_env_config=EnvironmentConfig(docker_image="test/image"),
            )
            client = adapter.factory.client_factory()
            await client.close()

    asyncio.run(run())
