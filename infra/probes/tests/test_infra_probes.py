# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resource-boundary behavior for the Iris canary probes."""

from types import SimpleNamespace
from typing import cast

import pytest
from infra_probes import iris_job_succeeds, resolve_finelog_address
from iris.client.client import IrisClient
from iris.cluster.resources.endpoint import EndpointAccess, EndpointDetail, EndpointSummary
from iris.cluster.resources.identity import ResourceKey, ResourceKind
from iris.cluster.resources.source import Page
from iris.rpc import job_pb2


def _endpoint(endpoint_id: str, name: str, address: str) -> tuple[EndpointSummary, EndpointDetail]:
    summary = EndpointSummary(
        key=ResourceKey("marin", ResourceKind.ENDPOINT, endpoint_id),
        endpoint_id=endpoint_id,
        name=name,
        task=None,
        execution_cluster_id="marin",
        access=EndpointAccess.PRIVATE,
        lease_deadline=None,
    )
    return summary, EndpointDetail(summary=summary, address=address, metadata={})


def test_finelog_resolution_finds_exact_endpoint_beyond_first_page():
    sibling, sibling_detail = _endpoint("sibling", "/system/log-server-canary", "http://wrong")
    target, target_detail = _endpoint("target", "/system/log-server", "http://finelog:10000")
    details = {sibling.key: sibling_detail, target.key: target_detail}

    class EndpointResourceFake:
        def list_endpoints(self, query):
            if query.page_token is None:
                return Page((sibling,), "next", ())
            return Page((target,), None, ())

        def describe_endpoint(self, key):
            return details[key]

    address = resolve_finelog_address(cast(IrisClient, EndpointResourceFake()), "/system/log-server")

    assert address == "http://finelog:10000"


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (job_pb2.JOB_STATE_SUCCEEDED, True),
        (job_pb2.JOB_STATE_FAILED, False),
    ],
)
def test_scheduling_probe_observes_exact_job_handle(state, expected):
    class JobHandleFake:
        def wait(self, **_kwargs):
            return SimpleNamespace(state=state)

    class JobClientFake:
        def submit(self, **_kwargs):
            return JobHandleFake()

    assert iris_job_succeeds(cast(IrisClient, JobClientFake()), "us-west4-a", []) is expected
