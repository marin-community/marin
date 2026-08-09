# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for environment variable propagation across real job execution.

These tests boot a real local cluster and execute jobs to verify that env vars
and the parent's resolved setup propagate correctly through job hierarchies.
"""

import json

import pytest
from iris.client.client import LocalClientConfig, iris_ctx
from iris.client.local_client import local_client
from iris.cluster.client.job_info import get_job_info
from iris.cluster.resources.execution import Entrypoint, EnvironmentSpec, ResourceSpec

pytestmark = pytest.mark.requires_cluster


def _chain_job(output_file: str, child_spec: dict | None = None):
    """Job that dumps its JobInfo state and optionally submits a child.

    Args:
        output_file: Path to write JSON with {"env": ..., "setup_scripts": ...}.
        child_spec: If not None, submit a child job with keys:
            - output_file: str — child's output path
            - extras: list[str] | None — if set, the child builds its own setup
              with these extras (taking control of its environment); otherwise the
              child inherits the parent's resolved setup
            - child_spec: dict | None — recursive spec for the grandchild
    """

    info = get_job_info()
    state = {
        "env": dict(info.env) if info else {},
        "setup_scripts": list(info.setup_scripts) if info and info.setup_scripts is not None else None,
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
    """E2E: env vars and the parent's resolved setup propagate A → B → C."""
    out_a = str(tmp_path / "a.json")
    out_b = str(tmp_path / "b.json")
    out_c = str(tmp_path / "c.json")

    # Chain: A → B → C
    # A: built with extras=["extra-from-a"], so its setup references that extra.
    # B: submitted with no environment, so it inherits A's resolved setup verbatim.
    # C: submitted with extras=["extra-from-b"], so it builds its own setup and
    #    does not inherit A's.
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
    with local_client(config) as client:
        entrypoint = Entrypoint.from_callable(_chain_job, out_a, chain_spec)
        resources = ResourceSpec(cpu=1, memory="1g")
        environment = EnvironmentSpec(
            env_vars={"TEST_PROPAGATION_KEY": "hello_chain"},
            extras=["extra-from-a"],
        )
        job = client.submit(entrypoint, "job-a", resources, environment=environment)
        job.wait(timeout=120, raise_on_failure=True, stream_logs=True)

    with open(out_a) as f:
        state_a = json.load(f)
    with open(out_b) as f:
        state_b = json.load(f)
    with open(out_c) as f:
        state_c = json.load(f)

    # env_vars propagate through the full chain
    assert state_a["env"]["TEST_PROPAGATION_KEY"] == "hello_chain"
    assert state_b["env"]["TEST_PROPAGATION_KEY"] == "hello_chain"
    assert state_c["env"]["TEST_PROPAGATION_KEY"] == "hello_chain"

    # Infrastructure vars from os.environ are NOT in JobInfo.env
    for state in [state_a, state_b, state_c]:
        assert "PATH" not in state["env"]
        assert "HOME" not in state["env"]

    # A built its setup from extras=["extra-from-a"]
    assert state_a["setup_scripts"] is not None
    assert any("extra-from-a" in s for s in state_a["setup_scripts"])

    # B specified no environment, so it inherits A's resolved setup verbatim
    assert state_b["setup_scripts"] == state_a["setup_scripts"]

    # C specified its own extras, so it builds its own setup instead of inheriting A's
    assert state_c["setup_scripts"] is not None
    assert any("extra-from-b" in s for s in state_c["setup_scripts"])
    assert not any("extra-from-a" in s for s in state_c["setup_scripts"])
