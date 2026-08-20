# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Live-process coverage for the controller rollout smoke suite."""

import pytest
from iris.client import IrisClient, LocalClientConfig
from iris.cluster.local_cluster import LocalCluster, make_local_cluster_config
from iris.rpc.resource_client import ResourceRpcClient
from iris.testing.rollout_smoke import run_smoke_suite


@pytest.mark.requires_cluster
def test_rollout_smoke_suite_exercises_resource_lifecycle_on_local_cluster():
    config = LocalClientConfig(max_workers=1)
    cluster = LocalCluster(make_local_cluster_config(config.max_workers))
    address = cluster.start()
    try:
        with IrisClient.remote(address) as client, ResourceRpcClient(address) as resources:
            result = run_smoke_suite(client, resources, timeout=120)
    finally:
        cluster.close()

    assert len({result.completed_job, result.cancelled_job, result.followup_job}) == 3
    assert result.cancel_action_id
