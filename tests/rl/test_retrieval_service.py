# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
import rolloutengine.harness.retrieval_service as retrieval_runtime
from fray.current_client import set_current_client
from fray.iris_backend import FrayIrisClient
from fray.types import ResourceConfig, create_environment
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo, get_job_info, set_job_info
from iris.cluster.constraints import region_constraint
from iris.cluster.types import JobName
from iris.rpc import controller_pb2
from marin.rl.retrieval_backend import RetrievalServer, retrieval_handler
from marin.rl.retrieval_service import (
    RetrievalServiceConfig,
    RetrievalServiceLaunch,
    retrieval_service,
    run_retrieval_service,
)
from rigging.filesystem.storage_path import StoragePath
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


def test_service_child_preserves_credentials_and_region_and_resolves_its_run_endpoint(monkeypatch, tmp_path):
    for name, value in {"CW_KEY_ID": "mounted-id", "CW_KEY_SECRET": "mounted-secret"}.items():
        (tmp_path / name).write_text(value)
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "DAYTONA_RL_API_KEY").write_text("parent-only")
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path))

    endpoints = {}
    terminated = []

    class ControllerStub:
        def __init__(self, *, address, **kwargs):
            assert address == "http://parent-controller:8080"

        def launch_job(self, request, *, timeout_ms=None):
            wire = request.environment
            assert dict(wire.secrets) == {"CW_KEY_ID": "mounted-id", "CW_KEY_SECRET": "mounted-secret"}
            assert not {"CW_KEY_ID", "CW_KEY_SECRET", "DAYTONA_RL_API_KEY"} & wire.env_vars.keys()
            assert not any(item.key in {"region", "cluster"} for item in request.constraints)
            assert request.name == "/alice/rollout/retrieval"
            child = IrisContext.from_job_info(JobInfo(task_id=JobName.from_wire(request.name + "/0")), client=client)
            stack.enter_context(child.registry.registered(launch.config.endpoint_name, address))
            return controller_pb2.Controller.LaunchJobResponse(job_id=request.name)

        def register_endpoint(self, request, **kwargs):
            endpoints[request.name] = controller_pb2.Controller.Endpoint(
                endpoint_id=request.endpoint_id, name=request.name, address=request.address, task_id=request.task_id
            )
            return controller_pb2.Controller.RegisterEndpointResponse(
                endpoint_id=request.endpoint_id, lease_duration=request.lease_duration
            )

        def list_endpoints(self, request, **kwargs):
            return controller_pb2.Controller.ListEndpointsResponse(
                endpoints=[value for key, value in endpoints.items() if key == request.prefix]
            )

        def unregister_endpoint(self, request, **kwargs):
            endpoints.clear()

        def terminate_job(self, request, **kwargs):
            terminated.append(request.job_id)

        def get_job_status(self, request, **kwargs):
            pytest.fail("The registered service endpoint must resolve before waiting on job status")

        def close(self):
            pass

    monkeypatch.setattr("iris.cluster.client.remote_client.ControllerServiceClientSync", ControllerStub)
    monkeypatch.setattr("iris.cluster.client.remote_client.EndpointServiceClientSync", ControllerStub)
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
        with (
            RetrievalServer(
                ("127.0.0.1", 0),
                retrieval_handler(FixtureIndex("source document"), launch.config.state.model_dump(mode="json")),
                4,
            ) as server,
            ExitStack() as stack,
        ):
            address = f"http://127.0.0.1:{server.server_port}"
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = stack.enter_context(IrisClient.in_cluster("http://parent-controller:8080", workspace=tmp_path))
                context = IrisContext.from_job_info(info, client=client)
                with iris_ctx_scope(context), set_current_client(FrayIrisClient.from_iris_client(client)):
                    with retrieval_service(launch) as endpoint:
                        response = httpx.post(
                            endpoint + "/search",
                            json={
                                "name": "search",
                                "arguments": {"query": "quantum physics"},
                                "state": launch.config.state.model_dump(mode="json"),
                            },
                            trust_env=False,
                        )
                        response.raise_for_status()
                        assert response.json() == [{"docid": "7385", "score": 1.0, "snippet": "quantum physics"}]
                    assert terminated == ["/alice/rollout/retrieval"]
            finally:
                server.shutdown()
                thread.join(timeout=5)
        assert not endpoints
    finally:
        set_job_info(previous)


@pytest.mark.timeout(10)
def test_kubernetes_service_registers_the_actual_kernel_assigned_port(monkeypatch, tmp_path):
    # The image dependencies are external to this control-plane regression.
    package = tmp_path / "image" / "browsecomp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "tool.py").write_text("from contextlib import nullcontext\nsource_diagnostics = nullcontext\n")
    (package / "retrieval.py").write_text(
        "class BM25Searcher:\n"
        "    def __init__(self, args): pass\n"
        "    def search(self, query, k):\n"
        "        return [{'docid': '7385', 'score': 1.0, 'text': 'source document'}]\n"
        "class AutoTokenizer:\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, path, **kwargs): return cls()\n"
        "    def encode(self, text, **kwargs): return text.split()\n"
        "    def decode(self, tokens, **kwargs): return ' '.join(tokens)\n"
    )
    interpreter = tmp_path / "image-python"
    interpreter.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        f"os.environ['PYTHONPATH'] = {str(package.parent)!r}\n"
        "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    interpreter.chmod(0o755)
    manifest = b"[]\n"
    index_uri = "s3://marin-us-east-08a/retrieval-port-fixture"

    def read_manifest(path):
        assert str(path) == index_uri + "/index-manifest.json"
        return manifest

    monkeypatch.setattr(StoragePath, "read_bytes", read_manifest)
    config = RetrievalServiceConfig(
        state=OpenResearcherState(
            implementation_revision="a" * 40,
            corpus_revision="8367703babe6e080899a48782f365c284c32b668",
            index_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            tokenizer_revision="c" * 40,
            top_k=5,
            snippet_max_tokens=512,
            include_get_document=True,
        ),
        index_uri=index_uri,
        region="US-EAST-08A",
        root=str(tmp_path / "service"),
        backend_python=str(interpreter),
        max_concurrent_requests=4,
        endpoint_name="retrieval",
    )
    observed = []

    class ProbeFinished(RuntimeError):
        pass

    class ControllerStub:
        def __init__(self, **kwargs):
            pass

        def register_endpoint(self, request, **kwargs):
            response = httpx.post(
                request.address + "/search",
                json={
                    "name": "search",
                    "arguments": {"query": "physics"},
                    "state": config.state.model_dump(mode="json"),
                },
                trust_env=False,
            )
            response.raise_for_status()
            observed.append(response.json())
            # Stop the owned infinite service after verifying its published address.
            raise ProbeFinished

        def close(self):
            pass

    monkeypatch.setattr("iris.cluster.client.remote_client.ControllerServiceClientSync", ControllerStub)
    monkeypatch.setattr("iris.cluster.client.remote_client.EndpointServiceClientSync", ControllerStub)
    previous = get_job_info()
    info = JobInfo(
        task_id=JobName.from_wire("/alice/port-probe/retrieval/0"),
        advertise_host="127.0.0.1",
        worker_region="US-EAST-08A",
        ports={"retrieval": 0},
    )
    set_job_info(info)
    try:
        with IrisClient.in_cluster("http://controller.fixture", workspace=tmp_path) as client:
            with iris_ctx_scope(IrisContext.from_job_info(info, client=client)), pytest.raises(ProbeFinished):
                run_retrieval_service(config)
    finally:
        set_job_info(previous)
    assert observed == [[{"docid": "7385", "score": 1.0, "snippet": "source document"}]]


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
