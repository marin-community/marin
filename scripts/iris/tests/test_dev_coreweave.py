# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from scripts.iris.dev_coreweave import CoreweaveTarget, DevCoreweaveState, PodRef


def test_state_round_trip():
    state = DevCoreweaveState(
        session_name="matt",
        config_file="/abs/coreweave.yaml",
        job_id="/matt/dev-cw-matt",
        gpu_count=8,
        target=CoreweaveTarget(namespace="iris", kubeconfig_path="/k/cfg"),
        pod=PodRef(namespace="iris", pod_name="dev-cw-matt-abc", container="task"),
    )
    assert DevCoreweaveState.from_json(state.to_json()) == state


import click
import pytest
from iris.rpc import config_pb2

from scripts.iris.dev_coreweave import require_coreweave_platform


def _coreweave_config(namespace: str = "iris", kubeconfig: str = "") -> config_pb2.IrisClusterConfig:
    c = config_pb2.IrisClusterConfig()
    c.platform.coreweave.SetInParent()
    if namespace:
        c.platform.coreweave.namespace = namespace
    if kubeconfig:
        c.platform.coreweave.kubeconfig_path = kubeconfig
    return c


def test_require_coreweave_accepts_and_expands_kubeconfig():
    target = require_coreweave_platform(_coreweave_config(kubeconfig="~/.kube/coreweave-iris"))
    assert target.namespace == "iris"
    assert target.kubeconfig_path.endswith("/.kube/coreweave-iris")
    assert "~" not in target.kubeconfig_path


def test_require_coreweave_default_namespace_when_unset():
    c = config_pb2.IrisClusterConfig()
    c.platform.coreweave.SetInParent()
    assert require_coreweave_platform(c).namespace == "iris"


def test_require_coreweave_rejects_gcp_with_pointer_to_dev_tpu():
    g = config_pb2.IrisClusterConfig()
    g.platform.gcp.SetInParent()
    with pytest.raises(click.ClickException, match="dev_tpu.py"):
        require_coreweave_platform(g)
