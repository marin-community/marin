# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in proof run by an in-cluster driver against a deployed controller.

Select both ``integration`` and ``requires_cluster`` and explicitly configure
SHELLBOX_IRIS_CONTROLLER_URL, SHELLBOX_IRIS_PYTHON_IMAGE (digest pinned), and
SHELLBOX_IRIS_PUBLIC_ADDRESS / SHELLBOX_IRIS_PRIVATE_ADDRESS (known IP:port
canaries). The image needs Python 3 and standard POSIX transfer tools. This test
submits CPU-only sandbox jobs; it never deploys or restarts a controller.
"""

import asyncio
import ipaddress
import json
import os
import re
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import httpx
import pytest

pytest.importorskip("iris")

from iris.client import IrisClient
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, NetworkAttachment, ResourceSpec
from iris.resources.state import TERMINAL_TASK_STATES, JobState, TaskState, is_job_finished
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Duration, ExponentialBackoff
from shellbox.backends.iris.machine import IrisMachine, IrisMachineFactory
from shellbox.http_service import HttpRequest, HttpResponse, serve_http
from shellbox.image import RegistryImage
from shellbox.machine import Command, Machine, MachineSpec, NetworkPolicy
from shellbox.task import ContainerSpec, ReadinessProbe, ServiceEndpoint, TaskSpec

pytestmark = [pytest.mark.integration, pytest.mark.requires_cluster, pytest.mark.timeout(900)]

RUNTIME_PORT = 8123
RELAY_PORT = 8787
HOST_AUTH_NAME = "SHELLBOX_IRIS_LIVE_HOST_AUTH"
_FOREIGN_PROBE = """\
import json, socket, sys
control, broker = json.loads(sys.argv[1])
with socket.create_connection(control, timeout=10) as connection:
    assert connection.makefile('rb').read() == b'driver-control'
try:
    with socket.create_connection(broker, timeout=5):
        pass
except OSError:
    sys.exit(0)
sys.exit(1)
"""
_SERVER = """\
import http.server, pathlib, sys
state = pathlib.Path('/tmp/shellbox-live-state')
state.write_text(sys.argv[1])
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = state.read_bytes()
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
http.server.ThreadingHTTPServer(('127.0.0.1', 8123), Handler).serve_forever()
"""
_FETCH = """\
import sys, urllib.request
sys.stdout.buffer.write(urllib.request.urlopen(sys.argv[1], timeout=5).read())
"""
_ISOLATION = """\
import json, os, pathlib, socket, sys
names = {
    'GOOGLE_APPLICATION_CREDENTIALS', 'HF_TOKEN', 'HUGGINGFACE_API_KEY',
    'WANDB_API_KEY', 'KUBECONFIG', 'SSH_AUTH_SOCK', 'RIGGING_SECRETS_DIR',
    'IRIS_TASK_TOKEN', 'MARIN_CLUSTER_TOKEN', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY',
    'SHELLBOX_IRIS_LIVE_HOST_AUTH',
}
prefixes = ('AWS_', 'S3_', 'CW_KEY_', 'AZURE_', 'CLOUDSDK_AUTH_')
paths = (
    '/var/run/secrets/kubernetes.io/serviceaccount', '/run/rigging/secrets',
    '/run/rigging/storage-secrets', '/run/rigging/controller-secrets',
    '/var/run/docker.sock', '/run/docker.sock',
    '/root/.aws/credentials', '/root/.kube/config',
    '/root/.config/gcloud/application_default_credentials.json',
)
def blocked(address):
    try:
        with socket.create_connection(address, timeout=1):
            return False
    except OSError:
        return True
report = {
    'credential_names_absent': not any(key in names or key.startswith(prefixes) for key in os.environ),
    'credential_paths_absent': not any(pathlib.Path(path).exists() for path in paths),
}
for label, address in json.loads(sys.argv[1]).items():
    report[label + '_blocked'] = blocked(address)
print(json.dumps(report))
"""
_MODEL_REQUEST = """\
import json, sys, urllib.request
request = urllib.request.Request(sys.argv[1], data=b'{"messages":[]}', headers={'Content-Type':'application/json'})
sys.stdout.buffer.write(urllib.request.urlopen(request, timeout=30).read())
"""


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"Explicit live-test configuration is required: {name}")
    return value


def _address(name: str) -> tuple[str, int]:
    host, _, port = _required(name).rpartition(":")
    address = str(ipaddress.ip_address(host.strip("[]")))
    number = int(port)
    if not 0 < number < 65536:
        pytest.fail(f"Live-test port must be in range: {name}")
    return address, number


async def _output(machine: Machine, argv: tuple[str, ...]) -> bytes:
    result = await machine.run(Command(argv, timeout=60, output_limit_bytes=8192))
    succeeded = result.exit_code == 0
    bounded = not result.stdout_truncated and not result.stderr_truncated
    assert succeeded, "Live sandbox command failed; inspect task status without dumping its environment"
    assert bounded, "Live sandbox command exceeded the output bound"
    return result.stdout


def _jobs_terminal(observer: IrisClient, jobs: list[JobName]) -> bool:
    for name in jobs:
        job = observer.job(name)
        if not is_job_finished(job.state):
            return False
        tasks = job.tasks()
        if not tasks or any(task.state not in TERMINAL_TASK_STATES for task in tasks):
            return False
    return True


async def _direct_processes(machine: IrisMachine) -> None:
    payload = bytes(range(256)) * 4096
    process = await machine.open_process(
        Command(
            (
                "python3",
                "-u",
                "-c",
                "import sys; data=sys.stdin.buffer.read(); "
                "sys.stderr.buffer.write(b'e'*8192); sys.stdout.buffer.write(data); sys.exit(7)",
            ),
            output_limit_bytes=1024,
        )
    )
    try:
        await process.write(payload)
        await process.write_eof()
        output = bytearray()
        while chunk := await process.read(32768):
            output.extend(chunk)
        assert bytes(output) == payload
        assert await process.wait() == 7
        assert process.stderr == b"e" * 1024
    finally:
        await process.close()

    process = await machine.open_process(
        Command(("python3", "-u", "-c", "import os,time; print(os.getpid(), flush=True); time.sleep(300)"))
    )
    try:
        pid = int(await process.read(64))
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(1):
                await process.read()
    finally:
        await process.close()
    assert (
        await _output(
            machine,
            ("python3", "-c", "import pathlib,sys; assert not pathlib.Path('/proc',sys.argv[1]).exists()", str(pid)),
        )
        == b""
    )


async def _foreign_driver_blocked(observer: IrisClient, machine: IrisMachine, image: str) -> None:
    info = get_job_info()
    assert info is not None and info.attempt_uid
    attempt = await asyncio.to_thread(machine.task.current_attempt)
    assert attempt is not None
    status = await asyncio.to_thread(attempt.status)
    connection = await asyncio.to_thread(
        machine.rpc.get_task_connection,
        controller_pb2.Controller.GetTaskConnectionRequest(
            driver_task_id=info.task_id.to_wire(),
            driver_attempt_uid=info.attempt_uid,
            task_id=machine.task.task_id.to_wire(),
            attempt_uid=status.attempt_uid,
        ),
    )
    assert connection.attempt_uid == status.attempt_uid and connection.pod_uid == status.pod_uid

    async def control(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        del reader
        writer.write(b"driver-control")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(control, "0.0.0.0", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        addresses = ((info.advertise_host, port), (connection.pod_ip, connection.port))
        probe = await asyncio.to_thread(
            observer.submit,
            Entrypoint.from_command("python3", "-c", _FOREIGN_PROBE, json.dumps(addresses)),
            name="shellbox-foreign-driver-" + uuid.uuid4().hex[:12],
            resources=ResourceSpec(cpu=0.25, memory="128MB", disk="256MB"),
            environment=EnvironmentSpec(setup_scripts=[]),
            task_image=image,
            network_attachment=NetworkAttachment.POD,
            egress_policy=job_pb2.EGRESS_POLICY_CLUSTER,
            max_retries_failure=0,
            max_retries_preemption=0,
            scheduling_timeout=Duration.from_seconds(180),
            timeout=Duration.from_seconds(240),
        )
        print(f"Foreign driver probe: {probe.job_id.to_wire()}", flush=True)
        try:
            result = await asyncio.to_thread(probe.wait, timeout=300, poll_interval=2, raise_on_failure=False)
            assert result.state == JobState.SUCCEEDED, "A foreign Pod must reach the driver control but not its broker"
        finally:
            await asyncio.to_thread(probe.cancel)
            await asyncio.to_thread(
                ExponentialBackoff(initial=0.2, maximum=2).wait_until_or_raise,
                lambda: _jobs_terminal(observer, [probe.job_id]),
                timeout=Duration.from_seconds(120),
                error_message="Foreign driver probe did not terminate",
            )


def test_iris_container_groups_isolate_peers_relay_host_calls_and_reclaim_jobs(
    monkeypatch, record_property, tmp_path: Path
):
    info = get_job_info()
    assert info is not None and info.attempt_uid and info.task_token, "Run this proof inside an Iris driver Pod"
    controller = _required("SHELLBOX_IRIS_CONTROLLER_URL")
    endpoint = urlsplit(controller)
    if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username or endpoint.password:
        pytest.fail("Live-test controller must be an HTTP(S) URL without user information")
    image = _required("SHELLBOX_IRIS_PYTHON_IMAGE")
    if re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", image) is None:
        pytest.fail("A digest-pinned public Python image is required")
    public = _address("SHELLBOX_IRIS_PUBLIC_ADDRESS")
    private = _address("SHELLBOX_IRIS_PRIVATE_ADDRESS")
    assert ipaddress.ip_address(public[0]).is_global
    assert ipaddress.ip_address(private[0]).is_private and not ipaddress.ip_address(private[0]).is_loopback
    probes = {"public": public, "private": private, "metadata": ("169.254.169.254", 80)}
    host_auth = "synthetic-host-only-" + uuid.uuid4().hex
    monkeypatch.setenv(HOST_AUTH_NAME, host_auth)
    authenticated = []

    def upstream(request: httpx.Request) -> httpx.Response:
        authenticated.append(request.headers.get("Authorization") == "Bearer " + host_auth)
        return httpx.Response(200, json={"answer": 42})

    async def scenario(observer: IrisClient) -> None:
        factory = IrisMachineFactory(controller_url=controller, scheduling_timeout=180, job_ttl=600, disk_mb=512)
        machine = MachineSpec(
            RegistryImage(image), workdir="/tmp", network=NetworkPolicy.DENY, cpus=1, memory_mb=256, storage_mb=512
        )
        jobs: list[JobName] = []
        groups = []
        try:
            async with AsyncExitStack() as stack:
                for marker in ("first", "second"):
                    service = ContainerSpec(
                        "runtime",
                        machine,
                        endpoints=(ServiceEndpoint("http", RUNTIME_PORT),),
                        entrypoint=("python3", "-u", "-c", _SERVER),
                        arguments=(marker,),
                        readiness=ReadinessProbe(
                            Command(("python3", "-c", _FETCH, f"http://127.0.0.1:{RUNTIME_PORT}"), timeout=5),
                            timeout=60,
                            interval=1,
                        ),
                    )
                    group = await factory.create_task(
                        TaskSpec(
                            containers=(
                                ContainerSpec("task", machine),
                                service,
                            )
                        )
                    )
                    stack.push_async_callback(group.close)
                    groups.append(group)
                    jobs.append(cast(IrisMachine, group.primary).job.job_id)
                    record_property("iris_job_" + marker, jobs[-1].to_wire())
                    await group.wait_ready()
                    handoff = tmp_path / (marker + "-state")
                    await group.containers["runtime"].download("/tmp/shellbox-live-state", handoff)
                    assert handoff.read_bytes() == marker.encode()
                    await group.primary.upload(handoff, "/tmp/runtime-handoff")
                    assert await _output(group.primary, ("cat", "/tmp/runtime-handoff")) == marker.encode()
                    assert (
                        await _output(
                            group.primary, ("python3", "-c", _FETCH, "http://" + group.endpoint("runtime", "http"))
                        )
                        == marker.encode()
                    )
                    assert (
                        await _output(group.containers["runtime"], ("cat", "/tmp/shellbox-live-state"))
                        == marker.encode()
                    )
                    assert (
                        await _output(
                            group.primary,
                            (
                                "python3",
                                "-c",
                                "from pathlib import Path; p=Path('/tmp/shellbox-live-state'); "
                                "assert not p.exists(); p.write_text('actor'); print(p.read_text())",
                            ),
                        )
                        == b"actor\n"
                    )
                    for member in group.containers.values():
                        report = json.loads(await _output(member, ("python3", "-c", _ISOLATION, json.dumps(probes))))
                        assert report and all(report.values()), report

                for group, marker in zip(groups, ("first", "second"), strict=True):
                    assert (
                        await _output(
                            group.primary, ("python3", "-c", _FETCH, "http://" + group.endpoint("runtime", "http"))
                        )
                        == marker.encode()
                    )

                await _direct_processes(cast(IrisMachine, groups[0].primary))
                await _foreign_driver_blocked(observer, cast(IrisMachine, groups[0].primary), image)

                async with httpx.AsyncClient(transport=httpx.MockTransport(upstream), trust_env=False) as provider:

                    async def model(request: HttpRequest) -> HttpResponse:
                        if request.method != "POST" or request.path != "/model":
                            return HttpResponse(404, b"{}")
                        response = await provider.post(
                            "https://model.invalid/v1/chat/completions",
                            content=request.body,
                            headers={"Authorization": "Bearer " + host_auth},
                        )
                        return HttpResponse(response.status_code, response.content)

                    async with serve_http(groups[0].primary, model, port=RELAY_PORT) as relay:
                        answer = await _output(
                            groups[0].containers["runtime"], ("python3", "-c", _MODEL_REQUEST, relay + "/model")
                        )
                        assert json.loads(answer) == {"answer": 42}
                        peer = json.loads(
                            await _output(
                                groups[1].primary,
                                (
                                    "python3",
                                    "-c",
                                    _ISOLATION,
                                    json.dumps({"other_group_relay": ("127.0.0.1", RELAY_PORT)}),
                                ),
                            )
                        )
                        assert peer["other_group_relay_blocked"]
                        assert authenticated == [True]
        finally:
            if jobs:
                await asyncio.to_thread(
                    ExponentialBackoff(initial=0.2, maximum=2).wait_until_or_raise,
                    lambda: _jobs_terminal(observer, jobs),
                    timeout=Duration.from_seconds(120),
                    error_message="Live Shellbox jobs did not reach terminal job/task state after cleanup",
                )
        assert all(
            observer.job(job).state == JobState.KILLED
            and [task.state for task in observer.job(job).tasks()] == [TaskState.KILLED]
            for job in jobs
        ), "Successful live jobs must be killed by explicit group cleanup"

    with IrisClient.in_cluster(controller, workspace=None) as observer:
        asyncio.run(scenario(observer))
