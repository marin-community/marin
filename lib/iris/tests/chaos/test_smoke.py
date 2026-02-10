# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.rpc import cluster_pb2
from .conftest import submit, wait, _quick


@pytest.mark.chaos
def test_smoke(cluster):
    _url, client = cluster
    job = submit(client, _quick, "chaos-smoke")
    status = wait(client, job, timeout=30)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED
