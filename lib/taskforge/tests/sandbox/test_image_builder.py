# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The registry credential's parsing and the environment of a publish's jobs, submitted through a recording fake."""

import hashlib
import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from iris.client import IrisClient, Job
from iris.client.client import IrisContext, iris_ctx_scope
from iris.client.workload import JobState
from iris.cluster.client import ClusterClient, JobInfo, set_job_info
from iris.cluster.secrets import task_secret_values
from iris.cluster.types import Entrypoint, JobName
from iris.rpc import job_pb2
from rigging.secrets import COREWEAVE_SECRET_NAMES, SECRETS_DIR_ENV
from rigging.timing import Duration

from taskforge.sandbox.image_builder import (
    BUILD_SCRIPT,
    BUILD_SETUP_FAILED_EXIT,
    REGISTRY_CREDENTIAL_ENV,
    BuildFailed,
    ImageJobFailed,
    ImageJobPolicy,
    RegistryCredential,
    manifest_readback,
    submit_job,
    wait_succeeded,
)
from taskforge.sandbox.image_push import ARCHIVE_URI_ENV, REGISTRY_AUTH_SECRET, archive_request

PASSWORD = "registry-password"
CREDENTIAL_JSON = json.dumps({"registry": "registry.example", "user": "publisher", "password": PASSWORD})
GLM_TOKEN = "glm-token-value"
PARENT_ENV = {REGISTRY_CREDENTIAL_ENV: CREDENTIAL_JSON, "GLM_HIGH_TOKEN": GLM_TOKEN}
STORAGE = {"CW_KEY_ID": "storage-id", "CW_KEY_SECRET": "storage-secret"}
POLICY = ImageJobPolicy(
    priority_band=job_pb2.PRIORITY_BAND_BATCH,
    timeout=Duration.from_seconds(259200),
    scheduling_timeout=Duration.from_seconds(259200),
)


class RecordingCluster:
    """A ``ClusterClient``'s ``submit_job``, the only method ``IrisClient.submit`` calls.

    It records each job's environment and the secrets Iris will mount for it (``task_secret_values``).
    """

    def __init__(self):
        self.env_vars: dict[str, dict[str, str]] = {}
        self.secrets: dict[str, dict[str, str]] = {}
        self.policies: dict[str, dict[str, object]] = {}

    def submit_job(
        self,
        job_id: JobName,
        entrypoint: Entrypoint,
        resources: job_pb2.ResourceSpecProto,
        environment: job_pb2.EnvironmentConfig | None = None,
        *args: object,
        **kwargs: object,
    ) -> JobName:
        assert environment is not None
        self.env_vars[job_id.name] = dict(environment.env_vars)
        self.secrets[job_id.name] = task_secret_values(environment)
        self.policies[job_id.name] = {key: kwargs[key] for key in ("priority_band", "timeout", "scheduling_timeout")}
        return job_id


@pytest.fixture
def in_a_coordinator_task(request) -> Iterator[dict[str, str]]:
    """Run as task 0 of a job submitted with ``PARENT_ENV``, as a ladder host's build child does."""
    info = JobInfo(task_id=JobName.from_wire("/alice/ladder/0"), env=request.param)
    set_job_info(info)
    with iris_ctx_scope(IrisContext.from_job_info(info)):
        yield request.param
    set_job_info(None)


def submit(
    cluster: RecordingCluster, name: str, env: dict[str, str], secrets: dict[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Submit an image job through ``cluster`` and return the environment and secrets it recorded."""
    submit_job(
        IrisClient(cast(ClusterClient, cluster)),
        name=name,
        entrypoint=Entrypoint(command=["bash", "-c", "true"]),
        image="docker.io/library/bash:5",
        env=env,
        secrets=secrets,
        target_cluster="cw-us-east-02a",
        policy=POLICY,
    )
    return cluster.env_vars[name], cluster.secrets[name]


def test_the_credential_parses_from_its_secret_json_and_hides_its_password():
    credential = RegistryCredential.from_json(CREDENTIAL_JSON)

    assert credential == RegistryCredential(registry="registry.example", user="publisher", password=PASSWORD)
    assert PASSWORD not in repr(credential)


@pytest.mark.parametrize(
    "text",
    [
        PASSWORD,
        json.dumps({"registry": "registry.example", "user": "publisher", "secret": PASSWORD}),
        json.dumps([PASSWORD]),
    ],
)
def test_a_malformed_credential_fails_without_quoting_it(text):
    with pytest.raises(ValueError) as error:
        RegistryCredential.from_json(text)

    assert PASSWORD not in str(error.value)


@pytest.mark.parametrize(
    "in_a_coordinator_task",
    [PARENT_ENV, {"GLM_HIGH_TOKEN": GLM_TOKEN}, {**PARENT_ENV, **STORAGE}],
    indirect=True,
    ids=["env-and-mounted", "mounted-only", "storage-in-env"],
)
def test_image_jobs_only_receive_explicit_secrets_and_storage_credentials(in_a_coordinator_task, tmp_path, monkeypatch):
    (tmp_path / "UNRELATED_SERVICE_TOKEN").write_text("private-parent-token")
    (tmp_path / REGISTRY_CREDENTIAL_ENV).write_text(CREDENTIAL_JSON)
    (tmp_path / "CW_KEY_ID").write_text("storage-id")
    (tmp_path / "CW_KEY_SECRET").write_text("storage-secret")
    monkeypatch.setenv(SECRETS_DIR_ENV, str(tmp_path))
    cluster = RecordingCluster()
    build_env, build_secrets = submit(cluster, "taskforge-image-build-x", {"DESTINATION": "registry.example/r:t"}, {})
    push_env, push_secrets = submit(cluster, "taskforge-image-push-x", {}, {REGISTRY_AUTH_SECRET: "push-auth"})

    blanked = [name for name in in_a_coordinator_task if name not in COREWEAVE_SECRET_NAMES]
    for env in (build_env, push_env):
        assert {name: env[name] for name in blanked} == dict.fromkeys(blanked, "")
        assert not any(PASSWORD in value or GLM_TOKEN in value for value in env.values())
        # A CoreWeave key reaches a job's environment only as the parent's own copy, never blanked.
        assert {name: value for name, value in env.items() if name in COREWEAVE_SECRET_NAMES} == {
            name: value for name, value in in_a_coordinator_task.items() if name in COREWEAVE_SECRET_NAMES
        }
    assert build_secrets == STORAGE
    assert push_secrets == {**STORAGE, REGISTRY_AUTH_SECRET: "push-auth"}
    for name in ("taskforge-image-build-x", "taskforge-image-push-x"):
        assert cluster.policies[name] == {
            "priority_band": job_pb2.PRIORITY_BAND_BATCH,
            "timeout": Duration.from_seconds(259200),
            "scheduling_timeout": Duration.from_seconds(259200),
        }


@dataclass(frozen=True)
class EndedStatus:
    """What the controller's ``GetJobStatus`` returns: aggregates only, no tasks and a job exit code of 0."""

    state: JobState
    error_message: str
    exit_code: int = 0
    tasks: tuple[object, ...] = ()


class EndedJob:
    """A ``Job`` whose ``wait`` returns at once with ``state``, or raises ``TimeoutError`` for ``None``.

    The task's exit code is only visible through ``task(0).status()``, as on a real controller.
    """

    job_id = "/alice/taskforge-image-build-x"

    def __init__(self, state: JobState | None, exit_code: int = 1):
        self.state = state
        self.exit_code = exit_code

    def wait(self, **kwargs: object) -> EndedStatus:
        if self.state is None:
            raise TimeoutError("still running")
        return EndedStatus(self.state, "the job's log tail")

    def task(self, index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(status=lambda: SimpleNamespace(exit_code=self.exit_code))


@pytest.mark.parametrize("state", [JobState.UNSCHEDULABLE, JobState.WORKER_FAILED, JobState.KILLED, None])
@pytest.mark.parametrize("failed", [BuildFailed, ImageJobFailed])
def test_a_job_the_host_ended_is_an_image_job_failure_whichever_job_it_is(state, failed):
    with pytest.raises(ImageJobFailed):
        wait_succeeded(cast(Job, EndedJob(state)), failed, POLICY)


@pytest.mark.parametrize("failed", [BuildFailed, ImageJobFailed])
def test_a_job_that_ran_and_failed_raises_the_callers_failure(failed):
    with pytest.raises(failed, match="ended failed: the job's log tail"):
        wait_succeeded(cast(Job, EndedJob(JobState.FAILED)), failed, POLICY)


def test_build_bootstrap_failure_is_a_host_failure(tmp_path):
    mkdir = tmp_path / "mkdir"
    mkdir.write_text("#!/bin/sh\nexit 42\n")
    mkdir.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", "-c", BUILD_SCRIPT],
        env={
            "PATH": str(tmp_path),
            "IRIS_OUTPUT_DIR": str(tmp_path / "output"),
            "BUILD_SETUP_FAILED_EXIT": str(BUILD_SETUP_FAILED_EXIT),
        },
        capture_output=True,
        check=False,
    )
    with pytest.raises(ImageJobFailed):
        wait_succeeded(cast(Job, EndedJob(JobState.FAILED, result.returncode)), BuildFailed, POLICY)


@pytest.mark.parametrize("failure", [401, 404, 503, "transport", "malformed"])
def test_registry_readback_failure_is_a_host_failure(monkeypatch, failure):
    def respond(request):
        if failure == "transport":
            raise httpx.ConnectError("connection refused", request=request)
        if failure == "malformed":
            return httpx.Response(200, content=b"not a manifest")
        return httpx.Response(failure)

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    with pytest.raises(ImageJobFailed):
        manifest_readback(RegistryCredential.from_json(CREDENTIAL_JSON), "task", "build-tag")


@pytest.mark.parametrize("authentication", ["basic", "bearer"])
def test_registry_readback_resolves_authenticated_manifest_and_platform(monkeypatch, authentication):
    manifest = json.dumps({"config": {"digest": "sha256:config"}, "layers": []}).encode()
    digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    token_requests = []

    def respond(request):
        if request.url.path == "/token":
            token_requests.append(request)
            assert request.headers["authorization"].startswith("Basic ")
            assert request.url.params == httpx.QueryParams(service="registry.example", scope="repository:task:pull")
            return httpx.Response(200, json={"token": "scoped-read-token"})
        if authentication == "bearer" and request.headers["authorization"] != "Bearer scoped-read-token":
            return httpx.Response(
                401,
                headers={
                    "www-authenticate": (
                        'Bearer realm="https://registry.example/token",service="registry.example",scope="repository:task:pull"'
                    )
                },
            )
        if request.url.path == "/v2/task/blobs/sha256:config":
            return httpx.Response(200, json={"os": "linux", "architecture": "amd64"})
        assert request.url.path in ("/v2/task/manifests/build-tag", f"/v2/task/manifests/{digest}")
        return httpx.Response(200, content=manifest)

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    assert manifest_readback(RegistryCredential.from_json(CREDENTIAL_JSON), "task", "build-tag") == digest
    assert bool(token_requests) is (authentication == "bearer")


def test_registry_challenge_cannot_forward_credentials_to_another_origin(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            401,
            headers={
                "www-authenticate": (
                    'Bearer realm="https://other.example/token",service="registry.example",scope="repository:task:pull"'
                )
            },
        )

    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(respond), **kwargs))
    with pytest.raises(ImageJobFailed):
        manifest_readback(RegistryCredential.from_json(CREDENTIAL_JSON), "task", "build-tag")
    assert [request.url.host for request in requests] == ["registry.example"]


def test_archive_request_uses_the_coreweave_credentials_of_the_jobs_secret_files(monkeypatch, tmp_path):
    (tmp_path / "CW_KEY_ID").write_text("storage-id")
    (tmp_path / "CW_KEY_SECRET").write_text("storage-secret")
    monkeypatch.setenv(SECRETS_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(ARCHIVE_URI_ENV, "s3://task-bucket/build/output.tar.zst")
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL", "AWS_REGION"):
        monkeypatch.delenv(name, raising=False)

    request = archive_request()

    assert request.full_url == "https://task-bucket.cwobject.com/build/output.tar.zst"
    authorization = request.get_header("Authorization")
    assert authorization is not None
    assert "Credential=storage-id/" in authorization
    assert "/auto/s3/aws4_request" in authorization
