# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re

import click
import pytest
from iris.cluster.backends.k8s.tasks import _LABEL_TASK_ID, _sanitize_label_value
from iris.rpc import config_pb2

from scripts.iris.dev_coreweave import (
    CoreweaveTarget,
    DevCoreweaveState,
    PodRef,
    kubectl_base,
    kubectl_connect_cmd,
    kubectl_get_pods_cmd,
    parse_running_pod,
    pod_label_selector,
    require_coreweave_platform,
)


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


def test_from_json_defaults_container_when_absent():
    raw = """
    {
        "session_name": "matt",
        "config_file": "/abs/coreweave.yaml",
        "job_id": "/matt/dev-cw-matt",
        "gpu_count": 8,
        "target": {"namespace": "iris", "kubeconfig_path": "/k/cfg"},
        "pod": {"namespace": "iris", "pod_name": "dev-cw-matt-abc"}
    }
    """
    state = DevCoreweaveState.from_json(raw)
    assert state.pod.container == "task"
    assert state.pod.namespace == "iris"
    assert state.pod.pod_name == "dev-cw-matt-abc"
    assert state.session_name == "matt"
    assert state.job_id == "/matt/dev-cw-matt"
    assert state.gpu_count == 8
    assert state.target == CoreweaveTarget(namespace="iris", kubeconfig_path="/k/cfg")


def _coreweave_config(namespace: str = "iris", kubeconfig: str = "") -> config_pb2.IrisClusterConfig:
    c = config_pb2.IrisClusterConfig()
    c.platform.coreweave.SetInParent()
    if namespace:
        c.kubernetes_provider.namespace = namespace
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


def test_require_coreweave_namespace_comes_from_kubernetes_provider():
    # Pods are created/listed in kubernetes_provider.namespace; platform.coreweave.namespace
    # must not be the source (the two are independent proto fields that can diverge).
    c = config_pb2.IrisClusterConfig()
    c.platform.coreweave.namespace = "platform-ns"
    c.kubernetes_provider.namespace = "pods-live-here"
    assert require_coreweave_platform(c).namespace == "pods-live-here"


def test_require_coreweave_rejects_gcp_with_pointer_to_dev_tpu():
    g = config_pb2.IrisClusterConfig()
    g.platform.gcp.SetInParent()
    with pytest.raises(click.ClickException, match=re.escape("dev_tpu.py")):
        require_coreweave_platform(g)


def test_pod_label_selector_matches_iris_label():
    sel = pod_label_selector("/matt/dev-cw-matt/0")
    assert sel == f"{_LABEL_TASK_ID}={_sanitize_label_value('/matt/dev-cw-matt/0')}"
    assert sel == "iris.task_id=matt.dev-cw-matt.0"


def test_kubectl_base_with_kubeconfig():
    t = CoreweaveTarget(namespace="iris", kubeconfig_path="/k/cfg")
    assert kubectl_base(t) == ["kubectl", "--kubeconfig=/k/cfg", "--namespace=iris"]


def test_kubectl_base_without_kubeconfig():
    t = CoreweaveTarget(namespace="iris", kubeconfig_path="")
    assert kubectl_base(t) == ["kubectl", "--namespace=iris"]


def test_kubectl_get_pods_cmd():
    t = CoreweaveTarget(namespace="iris", kubeconfig_path="")
    assert kubectl_get_pods_cmd(t, "iris.task_id=x") == [
        "kubectl",
        "--namespace=iris",
        "get",
        "pods",
        "-l",
        "iris.task_id=x",
        "-o",
        "json",
    ]


def test_kubectl_connect_cmd():
    t = CoreweaveTarget(namespace="iris", kubeconfig_path="/k/cfg")
    pod = PodRef(namespace="iris", pod_name="dev-cw-matt-abc", container="task")
    assert kubectl_connect_cmd(t, pod) == [
        "kubectl",
        "--kubeconfig=/k/cfg",
        "--namespace=iris",
        "exec",
        "-it",
        "dev-cw-matt-abc",
        "-c",
        "task",
        "--",
        "bash",
        "-l",
    ]


def test_parse_running_pod_picks_running():
    pods = {
        "items": [
            {"metadata": {"name": "b"}, "status": {"phase": "Pending"}},
            {"metadata": {"name": "a"}, "status": {"phase": "Running"}},
        ]
    }
    assert parse_running_pod(pods) == "a"


def test_parse_running_pod_is_deterministic_by_name():
    pods = {
        "items": [
            {"metadata": {"name": "z"}, "status": {"phase": "Running"}},
            {"metadata": {"name": "a"}, "status": {"phase": "Running"}},
        ]
    }
    assert parse_running_pod(pods) == "a"


def test_parse_running_pod_none_when_no_running():
    pods = {"items": [{"metadata": {"name": "a"}, "status": {"phase": "Pending"}}]}
    assert parse_running_pod(pods) is None
