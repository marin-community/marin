# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The registry credential's parsing and the environment of a publish's jobs, submitted through a recording fake."""

import json
from collections.abc import Iterator
from typing import cast

import pytest
from iris.client import IrisClient
from iris.client.client import IrisContext, iris_ctx_scope
from iris.cluster.client import ClusterClient, JobInfo, set_job_info
from iris.cluster.types import Entrypoint, JobName
from iris.rpc import job_pb2

from taskforge.sandbox.image_builder import (
    BLANKED_ENV,
    REGISTRY_CREDENTIAL_ENV,
    RegistryCredential,
    submit_job,
)
from taskforge.sandbox.image_push import REGISTRY_AUTH_ENV

PASSWORD = "registry-password"
CREDENTIAL_JSON = json.dumps({"registry": "registry.example", "user": "publisher", "password": PASSWORD})
GLM_TOKEN = "glm-token-value"
PARENT_ENV = {REGISTRY_CREDENTIAL_ENV: CREDENTIAL_JSON, "GLM_HIGH_TOKEN": GLM_TOKEN}


class RecordingCluster:
    """A ``ClusterClient``'s ``submit_job``, the only method ``IrisClient.submit`` calls, recording environments."""

    def __init__(self):
        self.env_vars: dict[str, dict[str, str]] = {}

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
        return job_id


@pytest.fixture
def in_a_coordinator_task() -> Iterator[None]:
    """Run as task 0 of a job submitted with ``PARENT_ENV``, as a ladder host's build child does."""
    info = JobInfo(task_id=JobName.from_wire("/alice/ladder/0"), env=PARENT_ENV)
    set_job_info(info)
    with iris_ctx_scope(IrisContext.from_job_info(info)):
        yield
    set_job_info(None)


def submitted_env(cluster: RecordingCluster, name: str, env: dict[str, str]) -> dict[str, str]:
    submit_job(
        IrisClient(cast(ClusterClient, cluster)),
        name=name,
        entrypoint=Entrypoint(command=["bash", "-c", "true"]),
        image="docker.io/library/bash:5",
        env=env,
        target_cluster="cw-us-east-02a",
    )
    return cluster.env_vars[name]


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


def test_jobs_submitted_from_a_coordinator_task_inherit_none_of_its_environment(in_a_coordinator_task):
    cluster = RecordingCluster()
    build = submitted_env(cluster, "taskforge-image-build-x", {"DESTINATION": "registry.example/r:t"})
    push = submitted_env(cluster, "taskforge-image-push-x", {REGISTRY_AUTH_ENV: "push-auth"})

    for env in (build, push):
        assert {name: env[name] for name in [*PARENT_ENV, *BLANKED_ENV]} == dict.fromkeys(
            [*PARENT_ENV, *BLANKED_ENV], ""
        )
        assert not any(PASSWORD in value or GLM_TOKEN in value for value in env.values())
    assert build["DESTINATION"] == "registry.example/r:t"
    assert REGISTRY_AUTH_ENV not in build
    assert push[REGISTRY_AUTH_ENV] == "push-auth"
