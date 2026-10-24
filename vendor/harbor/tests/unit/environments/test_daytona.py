# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Shellbox Daytona grading without provider credentials in the guest."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from harbor.environments import daytona, model_relay
from harbor.environments.daytona import DaytonaEnvironment
from harbor.environments.model_relay import host_model_authentication
from harbor.models.task.config import EnvironmentConfig
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.utils.env import task_environment
from harbor.verifier.verifier import Verifier
from tests.unit.environments.transport import ProcessMachine, available_port


class ProcessFactory:
    def __init__(self, root: Path):
        self.root = root
        self.machine = None

    async def create(self, spec):
        self.machine = ProcessMachine(self.root, spec)
        return self.machine


def test_shellbox_daytona_runs_verifier_through_host_only_judge_relay(
    tmp_path, monkeypatch
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
                verifier_model_endpoint="https://api.together.xyz/v1",
            )
            adapter.factory = factory
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
    )
    with pytest.raises(ValueError):
        DaytonaEnvironment(**arguments, dind_image="docker:dind")
    (environment_dir / "docker-compose.yaml").write_text("services: {}")
    with pytest.raises(ValueError):
        DaytonaEnvironment(**arguments)
