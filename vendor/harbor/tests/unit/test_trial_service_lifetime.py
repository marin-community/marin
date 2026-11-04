# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise trial deadlines and relay failures through real guest processes."""

import asyncio
import json
from pathlib import Path

import pytest
import httpx

from harbor.agents.base import BaseAgent
from harbor.environments import local, model_relay
from harbor.environments.runtime import HostCapabilities
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from shellbox.task import Task
from tests.unit.environments.transport import ProcessMachine, available_port


class PartialAnswerAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "partial-answer"

    def version(self) -> str:
        return "1"

    async def setup(self, environment) -> None:
        pass

    async def run(self, instruction, environment, context) -> None:
        await environment.exec("printf partial > answer; exec sleep 3600")


class WaitingAgent(PartialAnswerAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running = asyncio.Event()

    async def run(self, instruction, environment, context) -> None:
        self.running.set()
        await asyncio.Event().wait()


class JudgeProbeAgent(PartialAnswerAgent):
    async def run(self, instruction, environment, context) -> None:
        script = f"""
import os, urllib.error, urllib.request
assert 'OPENAI_BASE_URL' not in os.environ
assert 'OPENAI_API_KEY' not in os.environ
request = urllib.request.Request('http://127.0.0.1:{model_relay.MODEL_RELAY_PORT}/v1/chat/completions', data=b'{{}}')
try:
    urllib.request.urlopen(request, timeout=1)
except urllib.error.URLError:
    open('judge-blocked', 'w').write('blocked')
else:
    raise AssertionError('The candidate reached the verifier model')
"""
        with environment.scoped_exec_env(self.extra_env):
            result = await environment.exec("python3 - <<'PY'\n" + script + "\nPY")
        if result.return_code:
            raise RuntimeError(result.stderr)


class ProcessTaskFactory:
    def check_environment(self, requirements, spec) -> None:
        pass

    async def create_task(self, spec):
        machine = ProcessMachine(self.root, spec.primary_container.machine)
        self.machine = machine
        return Task({spec.primary: machine}, spec, machine.close)

    def __init__(self, root: Path):
        self.root = root


def trial_config(
    root: Path, name: str, *, guest_root: Path, waiting: bool = False
) -> TrialConfig:
    task = root / name
    environment = task / "environment"
    environment.mkdir(parents=True)
    (task / "instruction.md").write_text("Write an answer.")
    (task / "task.toml").write_text(
        '[agent]\ntimeout_sec=0.1\n[environment]\nallow_internet=false\nworkdir="/workspace"\n'
    )
    (environment / "shellbox-images.json").write_text(
        json.dumps(
            {"main": {"image": "test/actor@sha256:" + "a" * 64, "workdir": "/image"}}
        )
    )
    tests = task / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text(
        '#!/bin/sh\nset -eu\ntest "$(cat answer)" = partial\n'
        'printf 1 > "$TEST_REWARD_PATH"\n'
    )
    agent = "WaitingAgent" if waiting else "PartialAnswerAgent"
    return TrialConfig(
        task=TaskConfig(path=task),
        trial_name=name,
        trials_dir=root / "trials",
        agent=AgentConfig(
            import_path=f"tests.unit.test_trial_service_lifetime:{agent}"
        ),
        environment=EnvironmentConfig(
            type="local",
            kwargs=(
                {
                    "candidate_model_endpoint": "https://candidate.test/v1",
                    "candidate_model_name": "candidate",
                }
                if waiting
                else {}
            ),
        ),
        agent_timeout_multiplier=600 if waiting else 1,
        verifier=VerifierConfig(
            env={"TEST_REWARD_PATH": str(guest_root / "logs/verifier/reward.txt")}
        ),
        trial_cleanup_grace_sec=2,
    )


async def test_agent_deadline_preserves_partial_answer_for_actual_grading_script(
    tmp_path, monkeypatch
):
    factory = ProcessTaskFactory(tmp_path / "guest")
    monkeypatch.setattr(local, "LocalMachineFactory", lambda **kwargs: factory)
    trial = await Trial.create(
        trial_config(tmp_path, "deadline", guest_root=tmp_path / "guest")
    )

    result = await trial.run()

    assert result.exception_info.exception_type == "AgentTimeoutError"
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert (factory.machine.path("/workspace/answer")).read_text() == "partial"
    assert factory.machine.closed
    native = TrialResult.model_validate_json(trial.paths.result_path.read_text())
    assert native.completion is not None
    assert native.finished_at is not None


async def test_verifier_judge_is_unavailable_to_candidate_but_available_to_actual_grader(
    tmp_path, monkeypatch
):
    port = available_port()
    monkeypatch.setattr(model_relay, "MODEL_RELAY_PORT", port)
    factory = ProcessTaskFactory(tmp_path / "guest")
    monkeypatch.setattr(local, "LocalMachineFactory", lambda **kwargs: factory)
    original_client = httpx.AsyncClient

    def provider(request):
        assert request.headers["authorization"] == "Bearer verifier-private"
        assert json.loads(request.content)["model"] == "judge"
        return httpx.Response(200, json={"answer": "accepted"})

    monkeypatch.setattr(
        model_relay.httpx,
        "AsyncClient",
        lambda **kwargs: original_client(
            transport=httpx.MockTransport(provider), **kwargs
        ),
    )
    config = trial_config(tmp_path, "verifier-only", guest_root=tmp_path / "guest")
    config.agent = AgentConfig(
        import_path="tests.unit.test_trial_service_lifetime:JudgeProbeAgent",
        override_timeout_sec=10,
    )
    config.environment.kwargs = {
        "model_endpoint": "https://judge.test/v1",
        "model_name": "judge",
    }
    config.verifier.env.update(
        {"OPENAI_BASE_URL": "${OPENAI_BASE_URL}", "OPENAI_API_KEY": "${OPENAI_API_KEY}"}
    )
    (config.task.path / "tests/test.sh").write_text(
        "#!/bin/sh\nset -eu\npython3 - <<'PY'\n"
        "import json, os, urllib.request\n"
        "assert open('judge-blocked').read() == 'blocked'\n"
        "request=urllib.request.Request(os.environ['OPENAI_BASE_URL']+'/chat/completions',data=b'{}')\n"
        "assert json.load(urllib.request.urlopen(request))['answer'] == 'accepted'\n"
        "open(os.environ['TEST_REWARD_PATH'],'w').write('1')\nPY\n"
    )
    trial = await Trial.create(
        config,
        guest_environment={
            "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "OPENAI_API_KEY": "EMPTY",
        },
        host_capabilities=HostCapabilities(
            model_authentication={"https://judge.test/v1": "verifier-private"}
        ),
    )

    result = await trial.run()

    assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert factory.machine.closed


async def test_late_relay_failure_stops_active_trial_without_cancelling_neighbor(
    tmp_path, monkeypatch
):
    factories = iter(
        [
            ProcessTaskFactory(tmp_path / "victim"),
            ProcessTaskFactory(tmp_path / "neighbor"),
        ]
    )
    monkeypatch.setattr(local, "LocalMachineFactory", lambda **kwargs: next(factories))
    monkeypatch.setattr(model_relay, "CANDIDATE_MODEL_RELAY_PORT", available_port())
    victim = await Trial.create(
        trial_config(
            tmp_path, "relay-victim", guest_root=tmp_path / "victim", waiting=True
        )
    )
    neighbor = await Trial.create(
        trial_config(tmp_path, "healthy-neighbor", guest_root=tmp_path / "neighbor")
    )

    async with asyncio.TaskGroup() as trials:
        victim_run = trials.create_task(victim.run())
        neighbor_run = trials.create_task(neighbor.run())
        await asyncio.wait_for(victim.agent.running.wait(), 5)
        (relay,) = victim.agent_environment.machine.sessions
        relay.process.kill()
        result = await asyncio.wait_for(victim_run, 5)

    assert result.verifier_result is None
    assert result.exception_info.exception_type != "AgentTimeoutError"
    assert victim.agent_environment.machine is None
    assert neighbor_run.result().verifier_result.rewards == {"reward": 1.0}
    native = TrialResult.model_validate_json(victim.paths.result_path.read_text())
    assert native.verifier_result is None


@pytest.mark.parametrize("failure", ["collection", "teardown"])
async def test_runtime_failure_cannot_publish_a_success(tmp_path, monkeypatch, failure):
    factory = ProcessTaskFactory(tmp_path / "guest")
    monkeypatch.setattr(local, "LocalMachineFactory", lambda **kwargs: factory)
    trial = await Trial.create(
        trial_config(tmp_path, "runtime-failure", guest_root=tmp_path / "guest")
    )

    async def fail():
        raise RuntimeError("Required runtime output unavailable")

    if failure == "collection":
        monkeypatch.setattr(trial.agent_environment, "collect_runtime_logs", fail)
    else:
        original_stop = trial.agent_environment.stop

        async def failed_stop(delete):
            await original_stop(delete)
            await fail()

        monkeypatch.setattr(trial.agent_environment, "stop", failed_stop)

    result = await trial.run()

    assert result.verifier_result is None
    assert result.exception_info.exception_type == "RuntimeError"
    if failure == "teardown":
        assert result.completion is None
    else:
        assert result.completion is not None
