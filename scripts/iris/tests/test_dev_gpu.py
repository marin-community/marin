# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import click
import pytest
from click.testing import CliRunner
from iris.rpc import config_pb2

from scripts.iris.dev_gpu import (
    CoreweaveTarget,
    DevGpuState,
    PodRef,
    cli,
    kubectl_connect_cmd,
    parse_running_pod,
    pod_label_selector,
    require_coreweave_platform,
)


def test_state_round_trip():
    # The session file is a persisted contract: `status` and `release` read it back.
    state = DevGpuState(
        session_name="matt",
        config_file="/abs/coreweave.yaml",
        job_id="/matt/dev-gpu-matt",
        gpu_count=8,
        target=CoreweaveTarget(namespace="iris", kubeconfig_path="/k/cfg"),
        pod=PodRef(namespace="iris", pod_name="dev-gpu-matt-abc", container="task"),
    )
    assert DevGpuState.from_json(state.to_json()) == state


def test_require_coreweave_namespace_comes_from_kubernetes_provider():
    # Regression: pods are created/listed in kubernetes_provider.namespace, NOT
    # platform.coreweave.namespace (independent proto fields that can diverge).
    c = config_pb2.IrisClusterConfig()
    c.platform.coreweave.namespace = "platform-ns"
    c.kubernetes_provider.namespace = "pods-live-here"
    assert require_coreweave_platform(c).namespace == "pods-live-here"


def test_require_coreweave_rejects_gcp():
    # The platform gate is the guard that stops pointing this CoreWeave tool at a
    # TPU cluster; rejecting non-CoreWeave platforms is the contract.
    g = config_pb2.IrisClusterConfig()
    g.platform.gcp.SetInParent()
    with pytest.raises(click.ClickException):
        require_coreweave_platform(g)


def test_pod_label_selector_matches_iris_label():
    # Cross-system contract: the selector must equal the label Iris's k8s backend
    # stamps on the pod (task id sanitized: leading slash dropped, slashes -> dots).
    assert pod_label_selector("/matt/dev-gpu-matt/0") == "iris.task_id=matt.dev-gpu-matt.0"


def test_kubectl_connect_cmd():
    # -it + -c task + bash -l is the core behavior: an interactive TTY into the
    # task container. Dropping -it silently breaks `connect`.
    t = CoreweaveTarget(namespace="iris", kubeconfig_path="/k/cfg")
    pod = PodRef(namespace="iris", pod_name="dev-gpu-matt-abc", container="task")
    assert kubectl_connect_cmd(t, pod) == [
        "kubectl",
        "--kubeconfig=/k/cfg",
        "--namespace=iris",
        "exec",
        "-it",
        "dev-gpu-matt-abc",
        "-c",
        "task",
        "--",
        "bash",
        "-l",
    ]


@pytest.mark.parametrize(
    "pods, expected",
    [
        # picks the Running pod, ignoring Pending
        (
            {
                "items": [
                    {"metadata": {"name": "b"}, "status": {"phase": "Pending"}},
                    {"metadata": {"name": "a"}, "status": {"phase": "Running"}},
                ]
            },
            "a",
        ),
        # deterministic tie-break: lexicographically-first among multiple Running
        (
            {
                "items": [
                    {"metadata": {"name": "z"}, "status": {"phase": "Running"}},
                    {"metadata": {"name": "a"}, "status": {"phase": "Running"}},
                ]
            },
            "a",
        ),
        # nothing Running -> None (so the caller keeps polling)
        ({"items": [{"metadata": {"name": "a"}, "status": {"phase": "Pending"}}]}, None),
    ],
)
def test_parse_running_pod(pods, expected):
    assert parse_running_pod(pods) == expected


@pytest.mark.parametrize("bad_count", ["0", "-1"])
def test_allocate_rejects_non_positive_gpu_count(bad_count):
    # Regression: zero (which Iris helpers treat inconsistently as one GPU) and
    # negatives (which can corrupt scheduler accounting) must fail before submit.
    result = CliRunner().invoke(cli, ["--config", "x.yaml", "allocate", "--gpu-count", bad_count])
    assert result.exit_code != 0
    assert "gpu-count" in result.output.lower()
