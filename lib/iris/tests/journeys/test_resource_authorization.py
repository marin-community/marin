# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.controller.resources.rpc import ResourceServiceImpl
from iris.rpc import resource_pb2
from rigging.server_auth import VerifiedIdentity, identity_scope


def _key(kind: int, resource_id: str) -> resource_pb2.ResourceKey:
    return resource_pb2.ResourceKey(cluster_id="journey", kind=kind, resource_id=resource_id)


def test_resource_reads_scope_users_before_rows_cross_the_rpc_boundary(journey) -> None:
    alice = journey.submit("alice-job", user="alice")
    bob = journey.submit("bob-job", user="bob")
    journey.settle()
    service = ResourceServiceImpl(journey.controller.resources)

    with identity_scope(VerifiedIdentity(user_id="alice", role="user")):
        jobs = service.list_jobs(resource_pb2.ListJobsRequest(), None)
        tasks = service.list_tasks(resource_pb2.ListTasksRequest(), None)
        assert [job.identity.key.resource_id for job in jobs.jobs] == [alice.wire_id]
        assert [task.identity.key.resource_id for task in tasks.tasks] == [f"{alice.wire_id}/0"]

        with pytest.raises(ConnectError) as denied:
            service.describe_job(
                resource_pb2.DescribeJobRequest(job=_key(resource_pb2.RESOURCE_KIND_JOB, bob.wire_id)),
                None,
            )
        assert denied.value.code is Code.PERMISSION_DENIED

        with pytest.raises(ConnectError) as denied:
            service.describe_task(
                resource_pb2.DescribeTaskRequest(task=_key(resource_pb2.RESOURCE_KIND_TASK, f"{bob.wire_id}/0")),
                None,
            )
        assert denied.value.code is Code.PERMISSION_DENIED
