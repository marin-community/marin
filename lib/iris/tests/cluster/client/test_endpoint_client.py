# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.client.endpoint_client import EndpointLeaseRenewer, renew_interval
from iris.rpc import controller_pb2
from iris.time_proto import duration_to_proto
from rigging.timing import Duration, Timestamp


class _RegisterStub:
    def __init__(self, lease: Duration):
        self.lease = lease
        self.requests: list[controller_pb2.Controller.RegisterEndpointRequest] = []

    def __call__(
        self, request: controller_pb2.Controller.RegisterEndpointRequest
    ) -> controller_pb2.Controller.RegisterEndpointResponse:
        self.requests.append(request)
        return controller_pb2.Controller.RegisterEndpointResponse(
            endpoint_id=request.endpoint_id,
            lease_duration=duration_to_proto(self.lease),
        )


def _request(endpoint_id: str = "endpoint") -> controller_pb2.Controller.RegisterEndpointRequest:
    return controller_pb2.Controller.RegisterEndpointRequest(
        name="/user/job/service",
        address="127.0.0.1:1234",
        task_id="/user/job/0",
        endpoint_id=endpoint_id,
    )


def test_endpoint_lease_renewer_reregisters_due_endpoint():
    lease = Duration.from_minutes(10)
    stub = _RegisterStub(lease)
    renewer = EndpointLeaseRenewer(stub)
    start = Timestamp.now()
    request = _request()
    renewer.track(request, lease, now=start)

    interval = renew_interval(lease)
    renewer.tick(now=start.add(interval).add(Duration.from_ms(-1)))
    assert stub.requests == []

    renewer.tick(now=start.add(interval).add(Duration.from_ms(1)))
    assert stub.requests == [request]


def test_endpoint_lease_renewer_drops_untracked_endpoint():
    lease = Duration.from_minutes(10)
    stub = _RegisterStub(lease)
    renewer = EndpointLeaseRenewer(stub)
    start = Timestamp.now()
    renewer.track(_request(), lease, now=start)

    renewer.untrack("endpoint")
    renewer.tick(now=start.add(lease))

    assert stub.requests == []
