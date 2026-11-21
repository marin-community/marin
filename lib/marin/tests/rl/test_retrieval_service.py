# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import rolloutengine.harness.retrieval_service as retrieval_runtime
from fray.current_client import set_current_client
from fray.iris_backend import FrayIrisClient
from fray.types import ResourceConfig, create_environment
from iris.client.client import IrisClient
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.constraints import region_constraint
from iris.cluster.types import JobName
from rigging.secrets import SecretBehavior
from rigging.timing import Duration
from rolloutengine.harness.retrieval_service import OpenResearcherServiceProvider
from rolloutengine.harness.shell import ShellHarness, ToolContext
from rolloutengine.models import GradedRolloutInterrupted, RolloutOperation
from rolloutengine.test_support import ReplayModel, arithmetic_task, engine, machine_runtime, rollout_spec, tool_call
from shellbox.machine import Backend, Command, ExitReason, Result
from shellbox.process_session import SubprocessSession
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, CommandSemantics, EnvironmentRequirements, NoGrader, ProviderRequirement
from taskcompendium.openresearcher import OPENRESEARCHER_ACTION_INTERFACE, OpenResearcherState

from marin.rl.retrieval_backend import RetrievalServer, retrieval_handler
from marin.rl.retrieval_service import RetrievalServiceConfig, RetrievalServiceLaunch, retrieval_service


class FixtureIndex:
    def __init__(self, text: str):
        self.content = json.dumps({"docid": "7385", "text": text}, ensure_ascii=False).encode()

    def search(self, query: str) -> list[dict]:
        return [{"docid": "7385", "score": 1.0, "snippet": query}]

    def document(self, docid: str) -> bytes:
        assert docid == "7385"
        return self.content


class ProcessMachine:
    """Run the guest protocol in real local processes at the machine I/O boundary."""

    async def open_process(self, command: Command) -> SubprocessSession:
        process = await asyncio.create_subprocess_exec(
            *command.argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        async def stop():
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
            await process.wait()

        return SubprocessSession(process, stop=stop)

    async def run(self, command: Command) -> Result:
        result = await asyncio.to_thread(
            subprocess.run, command.argv, input=command.stdin, capture_output=True, timeout=command.timeout
        )
        return Result(result.returncode, result.stdout, result.stderr, False, False, ExitReason.EXITED)

    async def download(self, source: str, target: Path) -> None:
        shutil.copyfile(source, target)

    async def upload(self, source: Path, target: str) -> None:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    async def close(self) -> None:
        pass


@pytest.mark.timeout(120)
def test_retrieval_private_relay_preserves_documents_larger_than_capture_limit():
    state = OpenResearcherState(
        implementation_revision="a" * 40,
        corpus_revision="8367703babe6e080899a48782f365c284c32b668",
        index_manifest_sha256="b" * 64,
        tokenizer_revision="c" * 40,
        top_k=5,
        snippet_max_tokens=512,
        include_get_document=True,
    )
    text = "full document Ω\n" * 1_300_000
    expected = hashlib.sha256(text.encode()).hexdigest()
    with RetrievalServer(
        ("127.0.0.1", 0), retrieval_handler(FixtureIndex(text), state.model_dump(mode="json")), 4
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:

            async def check():
                provider = OpenResearcherServiceProvider(f"http://127.0.0.1:{server.server_port}")
                requirement = ProviderRequirement(
                    action_interface=OPENRESEARCHER_ACTION_INTERFACE,
                    network_access="service_only",
                    initial_state=state.model_dump(mode="json"),
                )
                session = await provider.open(ToolContext(ProcessMachine(), requirement))
                try:
                    search = json.loads((await session.call("search", {"query": "fixture query"})).content)
                    assert search == [{"docid": "7385", "score": 1.0, "snippet": "fixture query"}]
                    result = json.loads((await session.call("get_document", {"docid": "7385"})).content)
                    assert hashlib.sha256(result["text"].encode()).hexdigest() == expected
                    assert result["docid"] == "7385"
                finally:
                    await session.close()

            asyncio.run(check())
        finally:
            server.shutdown()
            thread.join(timeout=5)


def test_service_child_submission_forwards_mounted_storage_credentials(monkeypatch, tmp_path):
    for name, value in {"CW_KEY_ID": "mounted-id", "CW_KEY_SECRET": "mounted-secret"}.items():
        (tmp_path / name).write_text(value)
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "DAYTONA_RL_API_KEY").write_text("parent-only")
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path))

    class SubmissionCaptured(Exception):
        pass

    class ControllerStub:
        def __init__(self, *, address, **kwargs):
            assert address == "http://parent-controller:8080"

        def launch_job(self, request, *, timeout_ms=None):
            wire = request.environment
            assert dict(wire.secrets) == {"CW_KEY_ID": "mounted-id", "CW_KEY_SECRET": "mounted-secret"}
            assert not {"CW_KEY_ID", "CW_KEY_SECRET", "DAYTONA_RL_API_KEY"} & wire.env_vars.keys()
            assert not any(item.key in {"region", "cluster"} for item in request.constraints)
            assert request.name == "/alice/rollout/retrieval"
            raise SubmissionCaptured

        def close(self):
            pass

    monkeypatch.setattr("iris.cluster.client.remote_client.ControllerServiceClientSync", ControllerStub)
    monkeypatch.setenv("IRIS_CONTROLLER_URL", "http://parent-controller:8080")
    launch = RetrievalServiceLaunch(
        config=RetrievalServiceConfig(
            state=OpenResearcherState(
                implementation_revision="a" * 40,
                corpus_revision="8367703babe6e080899a48782f365c284c32b668",
                index_manifest_sha256="c" * 64,
                tokenizer_revision="d" * 40,
                top_k=5,
                snippet_max_tokens=512,
                include_get_document=True,
            ),
            index_uri="s3://fixture/index",
            region="US-EAST-02A",
            root="/opt/openresearcher",
            backend_python="/usr/local/bin/python",
            max_concurrent_requests=64,
            endpoint_name="retrieval",
        ),
        resources=ResourceConfig(cpu=16, ram="96GB", disk="128GB"),
        environment=create_environment(secret_behavior=SecretBehavior.ISOLATE),
        startup_timeout=60,
        lifetime=Duration.from_hours(1),
    )
    previous = get_job_info()
    info = JobInfo(
        task_id=JobName.from_wire("/alice/rollout/0"),
        worker_region="us-east-02a",
        constraints=[region_constraint(["us-east-02a"])],
    )
    set_job_info(info)
    try:
        with IrisClient.in_cluster("http://parent-controller:8080", workspace=tmp_path) as client:
            with set_current_client(FrayIrisClient.from_iris_client(client)):
                with pytest.raises(SubmissionCaptured), retrieval_service(launch):
                    pytest.fail("Submission capture must stop before service readiness")
    finally:
        set_job_info(previous)


def test_relay_pump_failure_preserves_the_partial_rollout_and_closes_the_guest(monkeypatch):
    class MalformedRelayMachine(ProcessMachine):
        def __init__(self):
            self.sessions = []
            self.closed = False

        async def open_process(self, command):
            script = (
                'print(\'{"type":"ready"}\', flush=True)\n'
                'print(\'{"type":"invalid"}\', flush=True)\n'
                "import sys; sys.stdin.buffer.read()\n"
            )
            session = await super().open_process(replace(command, argv=("python3", "-u", "-c", script)))
            self.sessions.append(session)
            return session

        async def close(self):
            self.closed = True

    machine = MalformedRelayMachine()

    class Factory:
        backend = Backend.LOCAL

        def check_environment(self, requirements, spec):
            pass

        async def create(self, spec):
            return machine

    state = OpenResearcherState(
        implementation_revision="a" * 40,
        corpus_revision="8367703babe6e080899a48782f365c284c32b668",
        index_manifest_sha256="b" * 64,
        tokenizer_revision="c" * 40,
        top_k=5,
        snippet_max_tokens=512,
        include_get_document=True,
    )
    requirement = ProviderRequirement(
        action_interface=OPENRESEARCHER_ACTION_INTERFACE,
        network_access="service_only",
        initial_state=state.model_dump(mode="json"),
    )
    provider = OpenResearcherServiceProvider("http://retrieval.fixture")
    task = arithmetic_task(
        answer_type=AnswerType.TEXT,
        grader=NoGrader(reason="relay failure fixture"),
        environment_requirements=EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            docker_image="fixture@sha256:" + "a" * 64,
            tool_providers={"retrieval": requirement},
        ),
        interaction_tools=provider.tools(requirement),
    )
    message = tool_call("search", {"query": "fixture query"}, call_id="search-1")
    model = ReplayModel([message])

    async def check():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"state": state.model_dump(mode="json")})
            )
        )
        monkeypatch.setattr(retrieval_runtime.httpx, "AsyncClient", lambda **kwargs: client)
        runner = engine(
            model,
            {Backend.LOCAL: Factory()},
            harnesses={"shell": ShellHarness(command_timeout=None, tools={OPENRESEARCHER_ACTION_INTERFACE: provider})},
        )
        with pytest.raises(GradedRolloutInterrupted) as caught:
            await runner.run_task(rollout_spec(task, machine=machine_runtime()))
        interrupted = caught.value
        assert interrupted.operation is RolloutOperation.ADVANCE
        assert interrupted.rollout.conversations[0].messages[-1] == message
        assert interrupted.grade.status is Outcome.UNAVAILABLE and interrupted.grade.reward is None
        assert client.is_closed and machine.closed
        assert len(machine.sessions) == 1 and machine.sessions[0].process.returncode is not None

    asyncio.run(check())
