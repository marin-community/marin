# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise tau3's actor, runtime, model and verifier handoffs through real processes."""

import asyncio
import json
import os
import shlex
from pathlib import Path

import httpx

from harbor.environments import iris
from harbor.environments.iris import IrisEnvironment
from harbor.environments import model_relay
from harbor.environments.model_relay import host_model_authentication
from tests.unit.environments.transport import ProcessMachine, available_port
from harbor.models.task.config import EnvironmentConfig, MCPServerConfig
from harbor.models.trial.paths import TrialPaths
from harbor.utils.env import task_environment
from rigging.filesystem.storage_path import StoragePath
from shellbox.service import ServiceGroup, ServiceGroupSpec

_RUNTIME = """
import json, os, sys, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
state = Path(os.environ['STATE_PATH'])
state.parent.mkdir(parents=True, exist_ok=True)
state.write_text('{}')
class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        assert self.path == '/mcp'
        self.rfile.read(int(self.headers['Content-Length']))
        request = urllib.request.Request(
            os.environ['OPENAI_BASE_URL'] + '/chat/completions',
            data=json.dumps({'model':'untrusted-override','messages':[]}).encode(),
            headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['OPENAI_API_KEY']})
        answer = json.load(urllib.request.urlopen(request))['choices'][0]['message']['content']
        state.write_text(json.dumps({'answer':answer}))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({'answer':answer}).encode())
    def log_message(self, *args): pass
ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
"""


class ProcessGroupFactory:
    def __init__(self, root: Path, port: int):
        self.root, self.port = root, port
        self.closed = False

    async def create_group(self, spec: ServiceGroupSpec) -> ServiceGroup:
        actor = ProcessMachine(self.root / "actor", spec.task)
        (runtime_spec,) = spec.services
        runtime = ProcessMachine(self.root / "runtime", runtime_spec.machine)
        script = self.root / "runtime.py"
        script.write_text(_RUNTIME)
        process = await asyncio.create_subprocess_exec(
            "python3",
            str(script),
            str(self.port),
            env={
                "PATH": os.environ["PATH"],
                **runtime_spec.machine.env,
                "STATE_PATH": str(runtime.path("/logs/agent/tau3_runtime_state.json")),
            },
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        async def close() -> None:
            process.terminate()
            await process.wait()
            await actor.close()
            await runtime.close()
            self.closed = True

        return ServiceGroup(actor, {"tau3-runtime": runtime}, spec, close)


def test_tau3_runtime_model_relay_and_verifier_state_handoff(tmp_path, monkeypatch):
    runtime_port, relay_port = available_port(), available_port()
    monkeypatch.setattr(iris, "MODEL_RELAY_PORT", relay_port)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "storage-private-canary")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-ambient-provider-key")
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "iris-images.json").write_text(
        json.dumps(
            {
                "main": {"image": "test/main@sha256:" + "a" * 64, "workdir": "/app"},
                "tau3-runtime": {
                    "image": "test/runtime@sha256:" + "b" * 64,
                    "workdir": "/app",
                    "agent_logs": "publish",
                },
            }
        )
    )
    (environment_dir / "compose.yaml").write_text(
        json.dumps(
            {
                "services": {
                    "main": {
                        "depends_on": {"tau3-runtime": {"condition": "service_healthy"}}
                    },
                    "tau3-runtime": {
                        "environment": {
                            "OPENAI_BASE_URL": "${OPENAI_BASE_URL}",
                            "OPENAI_API_KEY": "${OPENAI_API_KEY}",
                        },
                        "expose": [runtime_port],
                        "volumes": ["${HOST_AGENT_LOGS_PATH}:${ENV_AGENT_LOGS_PATH}"],
                        "healthcheck": {
                            "test": [
                                "CMD",
                                "python3",
                                "-c",
                                f"import socket; socket.create_connection(('localhost',{runtime_port}),timeout=1).close()",
                            ],
                            "interval": "0.01s",
                            "timeout": "1s",
                            "retries": 100,
                        },
                    },
                }
            }
        )
    )
    original_client = httpx.AsyncClient
    model_calls = []

    def provider(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://judge.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer explicit-host-key"
        model_calls.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "42"}}]})

    monkeypatch.setattr(
        model_relay.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(provider), **kwargs
        ),
    )
    factory = ProcessGroupFactory(tmp_path, runtime_port)

    async def check() -> None:
        with (
            task_environment(
                {
                    "OPENAI_BASE_URL": f"http://127.0.0.1:{relay_port}/v1",
                    "OPENAI_API_KEY": "EMPTY",
                }
            ),
            host_model_authentication({"https://judge.test/v1": "explicit-host-key"}),
        ):
            adapter = IrisEnvironment(
                environment_dir=environment_dir,
                environment_name="tau3",
                session_id="trial",
                trial_paths=TrialPaths(tmp_path / "trial"),
                cluster="test",
                model_endpoint="https://judge.test/v1",
                model_name="judge-model",
                task_env_config=EnvironmentConfig(
                    allow_internet=False,
                    mcp_servers=[
                        MCPServerConfig(
                            name="tau3-runtime",
                            transport="streamable-http",
                            url=f"http://tau3-runtime:{runtime_port}/mcp",
                        )
                    ],
                ),
            )
            adapter.factory = factory
            await adapter.start(force_build=False)
            try:
                actor_code = f"import json,os,urllib.request; assert 'AWS_SECRET_ACCESS_KEY' not in os.environ; print(json.load(urllib.request.urlopen(urllib.request.Request('http://tau3-runtime:{runtime_port}/mcp',data=b'{{}}'))))"
                actor = await adapter.exec("python3 -c " + shlex.quote(actor_code))
                assert actor.return_code == 0, actor.stderr
                assert "42" in actor.stdout
                verifier = await adapter.exec(
                    "python3 -c \"import json; print(json.load(open('/logs/agent/tau3_runtime_state.json'))['answer'])\""
                )
                assert verifier.return_code == 0
                assert verifier.stdout.strip() == "42"
                archive = StoragePath(f"memory://tau3-{tmp_path.name}/logs")
                (archive / "previous.txt").write_text("previous artifact")
                await adapter.download_dir("/logs/agent", str(archive))
                assert json.loads(
                    (archive / "tau3_runtime_state.json").read_text()
                ) == {"answer": "42"}
                assert (archive / "previous.txt").read_text() == "previous artifact"
            finally:
                await adapter.stop(delete=True)
        assert factory.closed
        assert model_calls == [{"model": "judge-model", "messages": []}]

    asyncio.run(check())
