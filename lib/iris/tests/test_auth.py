# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Auth tests for the Iris controller: loopback trust and identity attribution.

A local controller boots without a persistent signing key, so it runs in
null-auth (permissive) mode; these integration tests exercise loopback trust and
job-owner attribution against it. The lower-level loopback, token-verification
and authenticator-chain logic is unit-tested in ``lib/rigging/tests/test_server_auth.py``.
"""

import pytest
from iris.cluster.local_cluster import LocalCluster
from iris.cluster.types import Entrypoint, ResourceSpec
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.version import client_revision_date


def _quick():
    return 1


@pytest.mark.requires_cluster
def test_loopback_resolves_as_admin(controller_only_config):
    """End-to-end: a tokenless loopback client is resolved as the anonymous admin."""

    controller = LocalCluster(controller_only_config)
    url = controller.start()

    try:
        client = ControllerServiceClientSync(address=url, timeout_ms=10000)
        resp = client.get_current_user(job_pb2.GetCurrentUserRequest())
        assert resp.user_id == "anonymous"
        assert resp.role == "admin"
        client.close()
    finally:
        controller.close()


@pytest.mark.requires_cluster
def test_loopback_admin_submits_as_named_user(controller_only_config):
    """A tokenless loopback caller is admin, so the job name's owner segment is
    authoritative — jobs are attributed to the user the client names (the CLI
    fills this with $USER), matching the null-auth behaviour SSH users rely on."""

    controller = LocalCluster(controller_only_config)
    url = controller.start()

    try:
        client = ControllerServiceClientSync(address=url, timeout_ms=10000)
        launch_req = controller_pb2.Controller.LaunchJobRequest(
            name="/bob/acted-job",
            entrypoint=Entrypoint.from_callable(_quick).to_proto(),
            resources=ResourceSpec(cpu=1, memory="1g").to_proto(),
            client_revision_date=client_revision_date(),
        )
        resp = client.launch_job(launch_req)
        assert resp.job_id.startswith("/bob/"), resp.job_id
        client.close()
    finally:
        controller.close()
