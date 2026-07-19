# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.cluster.config import (
    CoreweavePlatformConfig,
    IrisClusterConfig,
    KubernetesProviderConfig,
    PlatformConfig,
)

from scripts.iris.dev_gpu import (
    CoreweaveTarget,
    DevGpuState,
    PodRef,
    Priority,
    parse_running_pod,
    require_coreweave_platform,
)


def test_state_round_trip():
    # The session file is a persisted contract: `status` and `release` read it back.
    state = DevGpuState(
        session_name="matt",
        config_file="/abs/coreweave.yaml",
        job_id="/matt/dev-gpu-matt",
        gpu_count=8,
        gpu_variant="H100",
        priority=Priority.BATCH,
        target=CoreweaveTarget(namespace="iris", kubeconfig_path="/k/cfg"),
        pod=PodRef(namespace="iris", pod_name="dev-gpu-matt-abc", container="task"),
    )
    assert DevGpuState.from_json(state.to_json()) == state


def test_require_coreweave_namespace_comes_from_kubernetes_provider():
    # Regression: pods are created/listed in kubernetes_provider.namespace, NOT
    # platform.coreweave.namespace (independent config fields that can diverge).
    c = IrisClusterConfig(
        platform=PlatformConfig(coreweave=CoreweavePlatformConfig(namespace="platform-ns")),
        kubernetes_provider=KubernetesProviderConfig(namespace="pods-live-here"),
    )
    assert require_coreweave_platform(c).namespace == "pods-live-here"


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
