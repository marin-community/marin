# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in proof of Iris service groups against an already deployed controller.

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
from typing import cast
from urllib.parse import urlsplit

import httpx
import pytest

pytest.importorskip("iris")

from iris.cli.connect import open_iris_client
from iris.client import IrisClient
from iris.cluster.types import JobName
from iris.resources.state import TERMINAL_TASK_STATES, JobState, TaskState, is_job_finished
from rigging.timing import Duration, ExponentialBackoff
from shellbox.backends.iris.machine import IrisMachine, IrisMachineFactory
from shellbox.http_service import HttpRequest, HttpResponse, serve_http
from shellbox.image import RegistryImage
from shellbox.machine import Command, Machine, MachineSpec, NetworkPolicy
from shellbox.service import ReadinessProbe, ServiceEndpoint, ServiceGroupSpec, ServiceSpec

pytestmark = [pytest.mark.integration, pytest.mark.requires_cluster, pytest.mark.timeout(900)]

RUNTIME_PORT = 8123
RELAY_PORT = 8787
HOST_AUTH_NAME = "SHELLBOX_IRIS_LIVE_HOST_AUTH"
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


def test_iris_service_groups_isolate_peers_relay_host_calls_and_reclaim_jobs(monkeypatch, record_property):
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
                    service = ServiceSpec(
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
                    group = await factory.create_group(ServiceGroupSpec(machine, (service,)))
                    stack.push_async_callback(group.close)
                    groups.append(group)
                    jobs.append(cast(IrisMachine, group.task).job.job_id)
                    record_property("iris_job_" + marker, jobs[-1].to_wire())
                    await group.wait_ready()
                    assert (
                        await _output(
                            group.task, ("python3", "-c", _FETCH, "http://" + group.endpoint("runtime", "http"))
                        )
                        == marker.encode()
                    )
                    assert (
                        await _output(group.services["runtime"], ("cat", "/tmp/shellbox-live-state")) == marker.encode()
                    )
                    assert (
                        await _output(
                            group.task,
                            (
                                "python3",
                                "-c",
                                "from pathlib import Path; p=Path('/tmp/shellbox-live-state'); "
                                "assert not p.exists(); p.write_text('actor'); print(p.read_text())",
                            ),
                        )
                        == b"actor\n"
                    )
                    for member in (group.task, *group.services.values()):
                        report = json.loads(await _output(member, ("python3", "-c", _ISOLATION, json.dumps(probes))))
                        assert report and all(report.values()), report

                for group, marker in zip(groups, ("first", "second"), strict=True):
                    assert (
                        await _output(
                            group.task, ("python3", "-c", _FETCH, "http://" + group.endpoint("runtime", "http"))
                        )
                        == marker.encode()
                    )

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

                    async with serve_http(groups[0].task, model, port=RELAY_PORT) as relay:
                        answer = await _output(
                            groups[0].services["runtime"], ("python3", "-c", _MODEL_REQUEST, relay + "/model")
                        )
                        assert json.loads(answer) == {"answer": 42}
                        peer = json.loads(
                            await _output(
                                groups[1].task,
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

    with open_iris_client(controller_url=controller, workspace=None) as observer:
        asyncio.run(scenario(observer))
