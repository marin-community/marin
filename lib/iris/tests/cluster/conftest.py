# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for cluster tests."""

import pytest
from iris.testing.cluster import make_cpu_resource_spec, make_gpu_resource_spec, make_service_test_harness


@pytest.fixture
def cpu_resource_spec():
    return make_cpu_resource_spec()


@pytest.fixture
def gpu_resource_spec():
    return make_gpu_resource_spec()


@pytest.fixture(params=["gcp", "k8s"])
def harness(request, tmp_path, embedded_log_server):
    """Run service tests against both provider implementations."""
    result = make_service_test_harness(request.param, tmp_path, embedded_log_server.address)
    yield result
    result.db.close()
