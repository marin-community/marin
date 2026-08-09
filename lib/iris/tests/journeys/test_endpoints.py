# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Endpoint ownership across Task replacement Attempts."""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError


def test_retry_endpoint_survives_then_new_attempt_atomically_replaces_it(journey):
    job = journey.submit("endpoint-retry", failure_retries=1)
    journey.settle()
    journey.register_endpoint(job[0], "service", "old:1", endpoint_id="old")

    journey.fail(job[0])
    journey.settle()

    assert [(endpoint.endpoint_id, endpoint.address) for endpoint in journey.endpoints(name="service")] == [
        ("old", "old:1")
    ]

    journey.register_endpoint(job[0], "service", "new:1", endpoint_id="new")

    assert [(endpoint.endpoint_id, endpoint.address) for endpoint in journey.endpoints(name="service")] == [
        ("new", "new:1")
    ]
    with pytest.raises(ConnectError) as excinfo:
        journey.register_endpoint(job[0], "service", "old:1", endpoint_id="old", attempt_id=0)
    assert excinfo.value.code is Code.FAILED_PRECONDITION
    assert [endpoint.endpoint_id for endpoint in journey.endpoints(name="service")] == ["new"]


def test_cancelling_a_job_tree_removes_its_endpoints(journey):
    root = journey.submit("endpoint-tree")
    journey.settle()
    child = journey.submit_child(root, "child")
    journey.settle()
    journey.register_endpoint(root[0], "root-service", "root:1", endpoint_id="root")
    journey.register_endpoint(child[0], "child-service", "child:1", endpoint_id="child")

    journey.cancel(root)
    journey.settle()

    assert journey.endpoints() == []


def test_coordinator_endpoint_survives_controller_restart_and_moves_to_the_replacement_attempt(journey):
    job = journey.submit("coordinator", tasks=4, failure_retries=1)
    journey.settle()
    journey.register_endpoint(job[0], "jax-coordinator", "task-0:1234", endpoint_id="first")

    journey.restart()

    assert [(endpoint.endpoint_id, endpoint.address) for endpoint in journey.endpoints(name="jax-coordinator")] == [
        ("first", "task-0:1234")
    ]

    journey.fail(job[0])
    journey.settle()
    journey.register_endpoint(job[0], "jax-coordinator", "task-0:5678", endpoint_id="replacement")

    assert [(endpoint.endpoint_id, endpoint.address) for endpoint in journey.endpoints(name="jax-coordinator")] == [
        ("replacement", "task-0:5678")
    ]
