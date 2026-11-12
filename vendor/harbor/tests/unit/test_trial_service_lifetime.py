# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise trial deadlines and relay failures through real guest processes."""

import asyncio
import contextlib
import json
import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
from harbor_config.errors import ErrorCategory, error_category

from harbor.agents.base import BaseAgent
from harbor.agents.factory import AgentFactory
from harbor.environments import local, model_relay
from harbor.host_capabilities import HostCapabilities
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.models.trial.result import TrialResult
from harbor.job import Job
from harbor.trial.trial import Trial
from harbor.trial.single_step import SingleStepTrial
from harbor.trial.errors import AgentTimeoutError
from shellbox.task import Task
from tests.unit.environments.transport import ProcessMachine, available_port

_QEMU_TASK_REF = "sha256:60b7050b0e0aa51641208cf65766743d340e59575db2c4d2f8628240846c2a28"
_GIT_TASK_REF = "sha256:7759a92a373c967dd31535975bad8659410644025673fe62385444fd2605c5fa"


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
        await environment.exec(
            "sleep 3600 > /dev/null 2>&1 & echo $! > candidate-background.pid"
        )


class BackgroundServerAgent(PartialAnswerAgent):
    async def run(self, instruction, environment, context) -> None:
        port = self.extra_env["HTTP_PORT"]
        script = (
            "from http.server import HTTPServer, SimpleHTTPRequestHandler\n"
            "from pathlib import Path\n"
            f"server = HTTPServer(('127.0.0.1', {port}), SimpleHTTPRequestHandler)\n"
            "Path('server-ready').write_text('ready')\n"
            "server.serve_forever()\n"
        )
        result = await environment.exec(
            "printf partial > answer; python3 - <<'PY' > server.log 2>&1 &\n"
            + script
            + "PY\necho $! > server.pid; "
            "until test -f server-ready; do sleep 0.01; done",
            timeout_sec=5,
        )
        if result.return_code:
            raise RuntimeError(result.stderr)
        if self.extra_env["WAIT_FOR_TIMEOUT"] == "1":
            await environment.exec("echo $$ > foreground.pid; exec sleep 3600")

    async def cleanup(self, environment) -> None:
        await environment.exec("kill $(cat server.pid)")


@pytest.mark.parametrize(
    ("task_name", "task_ref", "deferred", "ending"),
    [
        ("terminal-bench/qemu-alpine-ssh", _QEMU_TASK_REF, True, "complete"),
        ("terminal-bench/git-multibranch", _GIT_TASK_REF, True, "complete"),
        ("terminal-bench/qemu-alpine-ssh", "sha256:" + "b" * 64, False, "complete"),
        ("terminal-bench/ordinary-task", "sha256:" + "a" * 64, False, "complete"),
        ("terminal-bench/qemu-alpine-ssh", _QEMU_TASK_REF, False, "timeout"),
        ("terminal-bench/qemu-alpine-ssh", _QEMU_TASK_REF, False, "cancel"),
    ],
)
async def test_server_dependent_tb2_tasks_keep_agent_service_until_verification(task_name, task_ref, deferred, ending):
    class Environment:
        requires_agent_quiescence_for_verifier = False

        def __init__(self):
            self.service_running = False
            self.cleanup_count = 0

        def with_default_user(self, user):
            return contextlib.nullcontext()

        def with_agent_timeout(self, timeout):
            return contextlib.nullcontext()

        async def kill_processes(self):
            self.cleanup_count += 1
            self.service_running = False

    class Agent:
        async def run(self, instruction, environment, context):
            environment.service_running = True
            if ending == "timeout":
                raise AgentTimeoutError("Agent timed out")
            if ending == "cancel":
                raise asyncio.CancelledError

        async def cleanup(self, environment):
            environment.cleanup_count += 1
            environment.service_running = False

    trial = object.__new__(SingleStepTrial)
    trial.config = TrialConfig(
        task=TaskConfig(
            name=task_name,
            ref=task_ref,
            source="terminal-bench/terminal-bench-2-1",
        ),
        agent=AgentConfig(name="terminus-2"),
        trial_name="server-lifetime",
    )
    trial.task = SimpleNamespace(name=task_name)
    trial.agent = Agent()
    trial.agent_environment = Environment()
    trial._hooks = defaultdict(list)
    trial._result = SimpleNamespace(secondary_exception_info=[])
    trial._pending_cleanup_tasks = set()
    trial.logger = logging.getLogger(__name__)
    target = SimpleNamespace(agent_result=None, agent_execution=None)

    if ending == "complete":
        await trial._run_agent_phase(target=target, instruction="keep server", timeout_sec=None, user=None)
    else:
        failure = AgentTimeoutError if ending == "timeout" else asyncio.CancelledError
        with pytest.raises(failure):
            await trial._run_agent_phase(target=target, instruction="keep server", timeout_sec=None, user=None)
    assert trial.agent_environment.service_running is deferred
    if deferred:
        await trial._cleanup_agents()
        await trial.agent_environment.kill_processes()
    assert trial.agent_environment.cleanup_count == 2
    assert not trial.agent_environment.service_running


@pytest.fixture(autouse=True)
def allow_trial_agents(monkeypatch):
    agents = AgentFactory._AGENT_MAP | {
        f"test-{agent.__name__}": agent
        for agent in (
            PartialAnswerAgent,
            WaitingAgent,
            JudgeProbeAgent,
            BackgroundServerAgent,
        )
    }
    monkeypatch.setattr(AgentFactory, "_AGENT_MAP", agents)


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
            name=f"test-{agent}"
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


@pytest.mark.parametrize("timeout", [False, True])
async def test_actual_grader_observes_agent_background_service_stopped(
    tmp_path, monkeypatch, timeout
):
    port = available_port()
    factory = ProcessTaskFactory(tmp_path / "guest")
    monkeypatch.setattr(local, "LocalMachineFactory", lambda **kwargs: factory)
    config = trial_config(tmp_path, "background", guest_root=tmp_path / "guest")
    config.agent = AgentConfig(
        name="test-BackgroundServerAgent",
        override_timeout_sec=1 if timeout else 5,
        env={"HTTP_PORT": str(port), "WAIT_FOR_TIMEOUT": "1" if timeout else "0"},
    )
    (config.task.path / "tests/test.sh").write_text(
        "#!/bin/sh\nset -eu\npython3 - <<'PY'\n"
        "import os, urllib.error, urllib.request\n"
        "try:\n"
        f"    urllib.request.urlopen('http://127.0.0.1:{port}/answer', timeout=1)\n"
        "except urllib.error.URLError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('Agent background service survived cleanup')\n"
        + (
            "pid = int(open('foreground.pid').read())\n"
            "try:\n    os.kill(pid, 0)\n"
            "except ProcessLookupError:\n    pass\n"
            "else:\n    raise AssertionError('Timed-out foreground command survived')\n"
            if timeout
            else ""
        )
        + "open(os.environ['TEST_REWARD_PATH'], 'w').write('1')\nPY\n"
    )
    trial = await Trial.create(config)

    result = await trial.run()

    if timeout:
        assert result.exception_info.exception_type == "AgentTimeoutError"
    else:
        assert result.exception_info is None
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert factory.machine.closed


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
        name="test-JudgeProbeAgent",
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
        "import json, os, urllib.request\nfrom pathlib import Path\n"
        "assert open('judge-blocked').read() == 'blocked'\n"
        "process = Path('/proc') / open('candidate-background.pid').read().strip() / 'stat'\n"
        "assert not process.exists() or process.read_text().split()[2] == 'Z'\n"
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
    assert (
        error_category(result.exception_info.exception_type)
        is ErrorCategory.INFRASTRUCTURE
    )
    assert Job._is_resume_retryable_result(result)
    assert victim.agent_environment.machine is None
    assert neighbor_run.result().verifier_result.rewards == {"reward": 1.0}
    native = TrialResult.model_validate_json(victim.paths.result_path.read_text())
    assert native.verifier_result is None
    assert (
        error_category(native.exception_info.exception_type)
        is ErrorCategory.INFRASTRUCTURE
    )
    assert Job._is_resume_retryable_result(native)


@pytest.mark.parametrize("failure", ["collection", "collection_deadline", "teardown"])
async def test_runtime_failure_cannot_publish_a_success(tmp_path, monkeypatch, failure):
    factory = ProcessTaskFactory(tmp_path / "guest")
    monkeypatch.setattr(local, "LocalMachineFactory", lambda **kwargs: factory)
    trial = await Trial.create(
        trial_config(tmp_path, "runtime-failure", guest_root=tmp_path / "guest")
    )

    async def fail():
        if failure == "collection_deadline":
            await asyncio.Event().wait()
        raise RuntimeError("Required runtime output unavailable")

    if failure.startswith("collection"):
        if failure == "collection_deadline":
            trial.config.trial_cleanup_grace_sec = 0.2
        monkeypatch.setattr(trial.agent_environment, "collect_runtime_logs", fail)
    else:
        original_stop = trial.agent_environment.stop

        async def failed_stop(delete):
            await original_stop(delete)
            await fail()

        monkeypatch.setattr(trial.agent_environment, "stop", failed_stop)

    result = await trial.run()

    assert result.verifier_result is None
    assert (
        error_category(result.exception_info.exception_type)
        is ErrorCategory.INFRASTRUCTURE
    )
    assert Job._is_resume_retryable_result(result)
    assert (
        "TimeoutError" if failure == "collection_deadline" else "RuntimeError"
    ) in result.exception_info.exception_traceback
    if failure == "teardown":
        assert result.completion is None
    else:
        assert result.completion is not None
