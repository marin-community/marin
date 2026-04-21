# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for environment variable propagation across real job execution.

These tests boot a real local cluster and execute jobs to verify that env vars,
extras, and pip_packages propagate correctly through job hierarchies.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from iris.client import IrisContext, iris_ctx_scope
from iris.client.client import IrisClient, LocalClientConfig
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import (
    Entrypoint,
    EnvironmentSpec,
    JobName,
    ResourceSpec,
)

pytestmark = pytest.mark.e2e


def _parent_job_info(env: dict[str, str]) -> JobInfo:
    return JobInfo(
        task_id=JobName.from_wire("/parent-job/0"),
        env=env,
        constraints=[],
        worker_region=None,
    )


def dummy_entrypoint():
    pass


def _sleep_entrypoint():
    import time

    time.sleep(300)


@pytest.mark.timeout(60)
def test_child_job_inherits_parent_env(cluster):
    """Child jobs inherit the parent's explicit env vars from JobInfo.env."""
    entrypoint = Entrypoint.from_callable(dummy_entrypoint)
    resources = ResourceSpec(cpu=1, memory="1g")
    parent_env = {"MY_CUSTOM_VAR": "hello", "WANDB_API_KEY": "secret"}

    # Submit a long-running parent so the controller has a live row for its
    # hierarchy. Child submissions are rejected with FAILED_PRECONDITION when
    # the parent row is missing or terminated, so the parent must stay alive
    # until the child has been submitted.
    parent_job = cluster.client.submit(Entrypoint.from_callable(_sleep_entrypoint), "parent-job", resources)
    try:
        parent_context = IrisContext(
            job_id=parent_job.job_id,
            client=cluster.client,
        )

        with (
            iris_ctx_scope(parent_context),
            patch("iris.client.client.get_job_info", return_value=_parent_job_info(parent_env)),
        ):
            job = cluster.client.submit(entrypoint, "child-job", resources)

        job.wait(timeout=30)
        assert job.job_id == parent_job.job_id.child("child-job")
    finally:
        cluster.kill(parent_job)


def _chain_job(output_file: str, child_spec: dict | None = None):
    """Job that dumps its JobInfo state and optionally submits a child.

    Args:
        output_file: Path to write JSON with {"env": ..., "extras": ..., "pip_packages": ...}
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
        "extras": list(info.extras) if info else [],
        "pip_packages": list(info.pip_packages) if info else [],
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
    """E2E: env vars and extras propagate A → B → C; child overrides parent."""
    out_a = str(tmp_path / "a.json")
    out_b = str(tmp_path / "b.json")
    out_c = str(tmp_path / "c.json")

    # Chain: A → B → C
    # C: leaf job, no children (inherits B's extras)
    # B: submits C with extras=["extra-from-b"] (overrides parent extras)
    # A: submits B with no explicit extras (B inherits A's extras)
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

    # A was launched with extras=["extra-from-a"]
    assert state_a["extras"] == ["extra-from-a"]

    # B was launched without explicit extras, so it inherits A's extras
    assert state_b["extras"] == ["extra-from-a"]

    # C was launched by B with extras=["extra-from-b"], which overrides parent extras
    assert state_c["extras"] == ["extra-from-b"]
