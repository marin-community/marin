# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from shellbox.machine import NetworkPolicy

from taskcompendium.pipeline.execution_binding import verification_machine


@pytest.mark.parametrize("network", [NetworkPolicy.DENY, NetworkPolicy.ALLOW])
def test_iris_grader_preserves_source_network_policy(network):
    # Iris previously granted Internet egress even for network-denied sources.
    _, spec = verification_machine(
        image="example.org/grader@sha256:" + "a" * 64,
        verification_runtime="iris-gvisor",
        controller_url="http://unused-controller.invalid",
        memory_mb=512,
        network=network,
    )
    assert spec.network == network
