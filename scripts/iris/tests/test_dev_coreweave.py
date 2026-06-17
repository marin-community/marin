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
