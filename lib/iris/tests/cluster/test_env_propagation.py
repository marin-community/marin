# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for environment variable propagation from parent jobs to child jobs.

Env inheritance uses JobInfo.env (populated from IRIS_JOB_ENV) which contains
only the explicit vars from the parent's EnvironmentConfig — not infrastructure
vars like TPU_NAME or PATH that happen to be in os.environ.

Extras and pip_packages are inherited via IRIS_JOB_EXTRAS and IRIS_JOB_PIP_PACKAGES.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import patch

import pytest
from iris.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute
from iris.cluster.types import (
    Entrypoint,
    EnvironmentSpec,
    JobName,
    ResourceSpec,
)


def dummy_entrypoint():
    pass


@dataclass
class _RecordingClusterClient:
    """Records submit_job kwargs without running anything."""

    captured_env: dict = field(default_factory=dict)
    captured_constraints: list = field(default_factory=list)

    def submit_job(self, *, job_id=None, environment=None, constraints=None, **kwargs) -> JobName:
        if environment:
            self.captured_env = dict(environment.env_vars)
        if constraints:
            self.captured_constraints = list(constraints)
        return job_id or JobName.root("test", "dummy")

    def shutdown(self, wait: bool = True) -> None:
        pass


@pytest.fixture
def capturing_client():
    """IrisClient wired to a recording stub — no cluster boots."""
    stub = _RecordingClusterClient()
    client = IrisClient(cluster=stub)
    return client, stub


@pytest.fixture
def parent_context(capturing_client):
    """Simulate running inside a parent Iris job."""
    client, _ = capturing_client
    return IrisContext(
        job_id=JobName.root("test-user", "parent-job"),
        client=client,
    )


def _parent_job_info(
    env: dict[str, str],
    constraints: list[Constraint] | None = None,
    worker_region: str | None = None,
) -> JobInfo:
    return JobInfo(
        task_id=JobName.from_wire("/parent-job/0"),
        env=env,
        constraints=constraints or [],
        worker_region=worker_region,
    )


def test_child_job_does_not_inherit_os_environ(capturing_client, parent_context):
    """Infrastructure vars in os.environ (TPU_NAME, PATH, etc.) should NOT
    be inherited — only the explicit vars from JobInfo.env."""
    client, stub = capturing_client
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")
    parent_env = {"MY_VAR": "keep"}

    with (
        iris_ctx_scope(parent_context),
        patch("iris.client.client.get_job_info", return_value=_parent_job_info(parent_env)),
    ):
        client.submit(entrypoint, "infra-test", resources)

    assert stub.captured_env["MY_VAR"] == "keep"
    assert "PATH" not in stub.captured_env
    assert "HOME" not in stub.captured_env


def test_child_explicit_env_overrides_inherited(capturing_client, parent_context):
    """Explicit env_vars in EnvironmentSpec override inherited parent env vars."""
    client, stub = capturing_client
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")
    env = EnvironmentSpec(env_vars={"MY_VAR": "child_override", "CHILD_ONLY": "yes"})
    parent_env = {"MY_VAR": "parent_value", "PARENT_ONLY": "yes"}

    with (
        iris_ctx_scope(parent_context),
        patch("iris.client.client.get_job_info", return_value=_parent_job_info(parent_env)),
    ):
        client.submit(entrypoint, "override-test", resources, environment=env)

    assert stub.captured_env["MY_VAR"] == "child_override"
    assert stub.captured_env["CHILD_ONLY"] == "yes"
    assert stub.captured_env["PARENT_ONLY"] == "yes"


def test_no_env_inheritance_without_parent_context(capturing_client):
    """Without a parent job context, no env inheritance should occur."""
    client, stub = capturing_client
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    client.submit(entrypoint, "no-parent-test", resources)

    assert stub.captured_env == {}


def test_child_job_inherits_parent_constraints(capturing_client, parent_context):
    client, stub = capturing_client
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")
    parent_constraints = [
        Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.EQ, value="us-west4"),
        Constraint.create(key=WellKnownAttribute.PREEMPTIBLE, op=ConstraintOp.EQ, value="true"),
    ]

    with (
        iris_ctx_scope(parent_context),
        patch("iris.client.client.get_job_info", return_value=_parent_job_info({}, constraints=parent_constraints)),
    ):
        client.submit(entrypoint, "child-inherit-constraints", resources)

    assert any(
        c.key == WellKnownAttribute.REGION and c.value.string_value == "us-west4" for c in stub.captured_constraints
    )
    assert any(
        c.key == WellKnownAttribute.PREEMPTIBLE and c.value.string_value == "true" for c in stub.captured_constraints
    )


def test_child_explicit_constraints_override_parent(capturing_client, parent_context):
    client, stub = capturing_client
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")
    parent_constraints = [Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.EQ, value="us-west4")]
    child_constraints = [Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.EQ, value="europe-west4")]

    with (
        iris_ctx_scope(parent_context),
        patch("iris.client.client.get_job_info", return_value=_parent_job_info({}, constraints=parent_constraints)),
    ):
        client.submit(entrypoint, "child-override-constraints", resources, constraints=child_constraints)

    assert any(
        c.key == WellKnownAttribute.REGION and c.value.string_value == "europe-west4" for c in stub.captured_constraints
    )
    assert not any(
        c.key == WellKnownAttribute.REGION and c.value.string_value == "us-west4" for c in stub.captured_constraints
    )


def test_child_does_not_inherit_worker_region(capturing_client, parent_context):
    """A parent's worker_region must NOT auto-pin a child job's region.

    Region inheritance was removed so that an Iris job submitting another Iris
    job from inside a region-pinned worker does not automatically force the
    child into the same region. Region pinning is now opt-in via explicit
    `regions=[...]` on the child's resources or via the executor's GCS-path
    inference.
    """
    client, stub = capturing_client
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")

    with (
        iris_ctx_scope(parent_context),
        patch(
            "iris.client.client.get_job_info",
            return_value=_parent_job_info({}, worker_region="us-central1"),
        ),
    ):
        client.submit(entrypoint, "child-no-region-inherit", resources)

    region_constraints = [c for c in stub.captured_constraints if c.key == WellKnownAttribute.REGION]
    assert region_constraints == []


def test_child_explicit_region_preserved(capturing_client, parent_context):
    """A child's explicit region constraint is preserved verbatim — independent
    of whatever region the parent worker happens to be in."""
    client, stub = capturing_client
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")
    child_constraints = [Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.EQ, value="europe-west4")]

    with (
        iris_ctx_scope(parent_context),
        patch(
            "iris.client.client.get_job_info",
            return_value=_parent_job_info({}, worker_region="us-central1"),
        ),
    ):
        client.submit(entrypoint, "child-explicit-region", resources, constraints=child_constraints)

    assert any(
        c.key == WellKnownAttribute.REGION and c.value.string_value == "europe-west4" for c in stub.captured_constraints
    )
    assert not any(
        c.key == WellKnownAttribute.REGION and c.value.string_value == "us-central1" for c in stub.captured_constraints
    )
