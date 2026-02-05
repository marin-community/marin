# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for environment variable propagation from parent jobs to child jobs.

Env inheritance uses JobInfo.env (populated from IRIS_JOB_ENV) which contains
only the explicit vars from the parent's EnvironmentConfig — not infrastructure
vars like TPU_NAME or PATH that happen to be in os.environ.

Extras and pip_packages are conveyed through the inherited dockerfile
(IRIS_DOCKERFILE), not through IRIS_JOB_ENV.
"""

import json
from unittest.mock import patch

import pytest

from iris.client import IrisClient, IrisContext, LocalClientConfig, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec


def dummy_entrypoint():
    pass


@pytest.fixture
def local_client():
    config = LocalClientConfig(max_workers=2)
    with IrisClient.local(config) as client:
        yield client


@pytest.fixture
def parent_context(local_client):
    """Simulate running inside a parent Iris job."""
    return IrisContext(
        job_id=JobName.root("parent-job"),
        client=local_client,
    )


def _parent_job_info(env: dict[str, str]) -> JobInfo:
    return JobInfo(
        task_id=JobName.from_wire("/parent-job/0"),
        env=env,
    )


def test_child_job_inherits_parent_env(local_client, parent_context):
    """Child jobs inherit the parent's explicit env vars from JobInfo.env."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")
    parent_env = {"MY_CUSTOM_VAR": "hello", "WANDB_API_KEY": "secret"}

    with (
        iris_ctx_scope(parent_context),
        patch("iris.client.client.get_job_info", return_value=_parent_job_info(parent_env)),
    ):
        job = local_client.submit(entrypoint, "child-job", resources)

    job.wait(timeout=30)
    assert job.job_id == JobName.root("parent-job").child("child-job")


def test_child_job_does_not_inherit_os_environ(local_client, parent_context):
    """Infrastructure vars in os.environ (TPU_NAME, PATH, etc.) should NOT
    be inherited — only the explicit vars from JobInfo.env."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    parent_env = {"MY_VAR": "keep"}

    captured_env = {}
    original_submit = local_client._cluster_client.submit_job

    def capturing_submit(*, environment=None, **kwargs):
        if environment:
            captured_env.update(dict(environment.env_vars))
        return original_submit(environment=environment, **kwargs)

    with (
        iris_ctx_scope(parent_context),
        patch("iris.client.client.get_job_info", return_value=_parent_job_info(parent_env)),
    ):
        local_client._cluster_client.submit_job = capturing_submit
        try:
            local_client.submit(entrypoint, "infra-test", resources)
        finally:
            local_client._cluster_client.submit_job = original_submit

    assert captured_env["MY_VAR"] == "keep"
    assert "PATH" not in captured_env
    assert "HOME" not in captured_env


def test_child_explicit_env_overrides_inherited(local_client, parent_context):
    """Explicit env_vars in EnvironmentSpec override inherited parent env vars."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")
    env = EnvironmentSpec(env_vars={"MY_VAR": "child_override", "CHILD_ONLY": "yes"})

    parent_env = {"MY_VAR": "parent_value", "PARENT_ONLY": "yes"}

    captured_env = {}
    original_submit = local_client._cluster_client.submit_job

    def capturing_submit(*, environment=None, **kwargs):
        if environment:
            captured_env.update(dict(environment.env_vars))
        return original_submit(environment=environment, **kwargs)

    with (
        iris_ctx_scope(parent_context),
        patch("iris.client.client.get_job_info", return_value=_parent_job_info(parent_env)),
    ):
        local_client._cluster_client.submit_job = capturing_submit
        try:
            local_client.submit(entrypoint, "override-test", resources, environment=env)
        finally:
            local_client._cluster_client.submit_job = original_submit

    assert captured_env["MY_VAR"] == "child_override"
    assert captured_env["CHILD_ONLY"] == "yes"
    assert captured_env["PARENT_ONLY"] == "yes"


def test_no_env_inheritance_without_parent_context(local_client):
    """Without a parent job context, no env inheritance should occur."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    captured_env = {}
    original_submit = local_client._cluster_client.submit_job

    def capturing_submit(*, environment=None, **kwargs):
        if environment:
            captured_env.update(dict(environment.env_vars))
        return original_submit(environment=environment, **kwargs)

    local_client._cluster_client.submit_job = capturing_submit
    try:
        local_client.submit(entrypoint, "no-parent-test", resources)
    finally:
        local_client._cluster_client.submit_job = original_submit

    assert captured_env == {}


# ---------------------------------------------------------------------------
# E2E chain test: A → B → C
# ---------------------------------------------------------------------------


def _chain_job(output_file: str, child_spec: dict | None = None):
    """Job that dumps its JobInfo state and optionally submits a child.

    Args:
        output_file: Path to write JSON with {"env": ..., "dockerfile": ...}
        child_spec: If not None, submit a child job with keys:
            - output_file: str — child's output path
            - extras: list[str] | None — extras for the child's EnvironmentSpec
            - child_spec: dict | None — recursive spec for the grandchild
    """
    import json

    from iris.client.client import iris_ctx
    from iris.cluster.client.job_info import get_job_info
    from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

    info = get_job_info()
    state = {
        "env": dict(info.env) if info else {},
        "dockerfile": info.dockerfile if info else None,
    }
    with open(output_file, "w") as f:
        json.dump(state, f)

    if child_spec is not None:
        ctx = iris_ctx()
        env_spec = EnvironmentSpec(extras=child_spec["extras"]) if child_spec.get("extras") else None
        entrypoint = Entrypoint.from_callable(
            _chain_job,
            child_spec["output_file"],
            child_spec.get("child_spec"),
        )
        resources = ResourceSpec(cpu=1, memory="1g")
        job = ctx.client.submit(entrypoint, "child", resources, environment=env_spec)
        job.wait(timeout=60, raise_on_failure=True)


@pytest.mark.timeout(120)
def test_env_propagates_through_job_chain(tmp_path):
    """E2E: env vars propagate A → B → C; extras conveyed via dockerfile, child overrides parent."""
    out_a = str(tmp_path / "a.json")
    out_b = str(tmp_path / "b.json")
    out_c = str(tmp_path / "c.json")

    # Chain: A → B → C
    # C: leaf job, no children (inherits B's dockerfile)
    # B: submits C with extras=["extra-from-b"] (generates new dockerfile)
    # A: submits B with no explicit extras (B inherits A's dockerfile)
    chain_spec = {
        "output_file": out_b,
        "extras": None,
        "child_spec": {
            "output_file": out_c,
            "extras": ["extra-from-b"],
            "child_spec": None,
        },
    }

    config = LocalClientConfig(max_workers=4)
    with IrisClient.local(config) as client:
        entrypoint = Entrypoint.from_callable(_chain_job, out_a, chain_spec)
        resources = ResourceSpec(cpu=1, memory="1g")
        environment = EnvironmentSpec(
            env_vars={"TEST_PROPAGATION_KEY": "hello_chain"},
            extras=["extra-from-a"],
        )
        job = client.submit(entrypoint, "job-a", resources, environment=environment)
        job.wait(timeout=120, raise_on_failure=True, stream_logs=True)

    state_a = json.loads(open(out_a).read())
    state_b = json.loads(open(out_b).read())
    state_c = json.loads(open(out_c).read())

    # env_vars propagate through the full chain
    assert state_a["env"]["TEST_PROPAGATION_KEY"] == "hello_chain"
    assert state_b["env"]["TEST_PROPAGATION_KEY"] == "hello_chain"
    assert state_c["env"]["TEST_PROPAGATION_KEY"] == "hello_chain"

    # Infrastructure vars from os.environ are NOT in JobInfo.env
    for state in [state_a, state_b, state_c]:
        assert "PATH" not in state["env"]
        assert "HOME" not in state["env"]

    # A was launched with extras=["extra-from-a"], its dockerfile should contain that
    assert "--extra extra-from-a" in state_a["dockerfile"]

    # B was launched without explicit extras, so it inherits A's dockerfile
    assert "--extra extra-from-a" in state_b["dockerfile"]

    # C was launched by B with extras=["extra-from-b"], which generates a NEW dockerfile
    assert "--extra extra-from-b" in state_c["dockerfile"]
    assert "--extra extra-from-a" not in state_c["dockerfile"]
