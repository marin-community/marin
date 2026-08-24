# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for environment variable propagation across real job execution.

These tests boot a real local cluster and execute jobs to verify that env vars
and the parent's resolved setup propagate correctly through job hierarchies.
"""

import json
import os
import shlex

import pytest
from iris.client.client import LocalClientConfig, iris_ctx
from iris.client.local_client import local_client
from iris.cluster.client.job_info import get_job_info
from iris.cluster.setup_scripts import EnvironmentLayer, SetupPlan
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

pytestmark = pytest.mark.requires_cluster
_ACTIVE_LAYER_ENV = "TEST_ACTIVE_ENVIRONMENT_LAYER"


def _chain_job(output_file: str, child_spec: dict | None = None):
    """Job that dumps its JobInfo state and optionally submits a child.

    Args:
        output_file: Path to write the observed environment as JSON.
        child_spec: If not None, submit a child job with keys:
            - output_file: str — child's output path
            - layer: str | None — if set, the child replaces its environment
              layer; otherwise it inherits the parent's resolved layers
            - child_spec: dict | None — recursive spec for the grandchild
    """

    info = get_job_info()
    state = {
        "env": dict(info.env) if info else {},
        "active_layer": os.environ.get(_ACTIVE_LAYER_ENV),
    }
    with open(output_file, "w") as f:
        json.dump(state, f)

    if child_spec is not None:
        ctx = iris_ctx()
        layer_name = child_spec.get("layer")
        env_spec = None
        if layer_name:
            layer = EnvironmentLayer.environment(
                activate=f"export {_ACTIVE_LAYER_ENV}={shlex.quote(layer_name)}",
            )
            env_spec = EnvironmentSpec(setup=SetupPlan.custom([]).with_layer(layer))
        entrypoint = Entrypoint.from_callable(
            _chain_job,
            child_spec["output_file"],
            child_spec.get("child_spec"),
        )
        resources = ResourceSpec(cpu=1, memory="1g")
        job = ctx.client.submit(entrypoint, "child", resources, environment=env_spec)
        job.wait(timeout=60, raise_on_failure=True)


@pytest.mark.timeout(120)
def test_env_and_activation_layers_propagate_through_job_chain(tmp_path):
    """E2E: env vars and activation layers propagate A → B → C."""
    out_a = str(tmp_path / "a.json")
    out_b = str(tmp_path / "b.json")
    out_c = str(tmp_path / "c.json")

    # Chain: A → B → C
    # B inherits A's activation. C replaces it with its own environment layer.
    chain_spec = {
        "output_file": out_b,
        "layer": None,
        "child_spec": {
            "output_file": out_c,
            "layer": "c",
            "child_spec": None,
        },
    }

    config = LocalClientConfig(max_workers=4)
    with local_client(config) as client:
        entrypoint = Entrypoint.from_callable(_chain_job, out_a, chain_spec)
        resources = ResourceSpec(cpu=1, memory="1g")
        layer = EnvironmentLayer.environment(activate=f"export {_ACTIVE_LAYER_ENV}=a")
        environment = EnvironmentSpec(
            env_vars={"TEST_PROPAGATION_KEY": "hello_chain"},
            setup=SetupPlan.custom([]).with_layer(layer),
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

    assert state_a["active_layer"] == "a"
    assert state_b["active_layer"] == "a"
    assert state_c["active_layer"] == "c"
